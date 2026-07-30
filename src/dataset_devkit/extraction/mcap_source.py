"""Read MCAP records and dynamically decode their embedded protobuf descriptors."""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from google.protobuf import descriptor_pb2, descriptor_pool, json_format, message_factory
from google.protobuf.descriptor import Descriptor, FieldDescriptor
from google.protobuf.message import DecodeError, Message
from google.protobuf.timestamp_pb2 import Timestamp
from mcap.reader import make_reader

from dataset_devkit.extraction.camera import (
    validate_annex_b_hevc_access_unit,
    validate_camera_batch,
)
from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.gnss import index_gnss_samples
from dataset_devkit.extraction.models import (
    CameraAccessUnit,
    CameraCalibration,
    CameraExtrinsic,
    CameraIntrinsic,
    CameraStructure,
    GnssSample,
    RawCameraBatch,
    RawCameraFrame,
    RawRecording,
    SourceIdentity,
    TimestampObservation,
)
from dataset_devkit.extraction.uncertainty import TraversalBudget, bounded_freeze

CAMERA_SCHEMA_NAME = "autonome.CompressedVideos"
_GNSS_NUMERIC_FIELDS = {
    "lat_lon_ht": ("latitude_deg", "longitude_deg", "height_m"),
    "orientation": ("roll_rad", "pitch_rad", "yaw_rad"),
    "position_error": ("east_sigma_m", "north_sigma_m", "up_sigma_m", "hdop"),
}
_PROTOBUF_NUMERIC_TYPES = {
    FieldDescriptor.TYPE_DOUBLE,
    FieldDescriptor.TYPE_FLOAT,
    FieldDescriptor.TYPE_INT64,
    FieldDescriptor.TYPE_UINT64,
    FieldDescriptor.TYPE_INT32,
    FieldDescriptor.TYPE_FIXED64,
    FieldDescriptor.TYPE_FIXED32,
    FieldDescriptor.TYPE_UINT32,
    FieldDescriptor.TYPE_SFIXED32,
    FieldDescriptor.TYPE_SFIXED64,
    FieldDescriptor.TYPE_SINT32,
    FieldDescriptor.TYPE_SINT64,
}


def _message_names(
    file_proto: descriptor_pb2.FileDescriptorProto,
    *,
    budget: TraversalBudget,
) -> list[str]:
    names: list[str] = []

    def visit(messages: Any, prefix: str, depth: int) -> None:
        for message in messages:
            full_name = f"{prefix}.{message.name}" if prefix else message.name
            budget.check_depth(full_name, depth)
            budget.visit(full_name, leaf=False, work=len(message.name))
            names.append(full_name)
            visit(message.nested_type, full_name, depth + 1)

    visit(file_proto.message_type, file_proto.package, 0)
    return names


def build_message_classes(data: bytes) -> dict[str, type[Message]]:
    """Build dynamic classes from a FileDescriptorSet regardless of file order."""
    file_set = descriptor_pb2.FileDescriptorSet()
    try:
        file_set.ParseFromString(data)
    except DecodeError as error:
        raise StructuralExtractionError(
            "schema data is not a valid protobuf FileDescriptorSet"
        ) from error
    if not file_set.file:
        raise StructuralExtractionError("protobuf FileDescriptorSet contains no files")

    name_budget = TraversalBudget()
    declared_names = tuple(
        name
        for file_proto in file_set.file
        for name in _message_names(file_proto, budget=name_budget)
    )

    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(Timestamp.DESCRIPTOR.file.serialized_pb)  # type: ignore[no-untyped-call]
    pending = {
        file_proto.name: file_proto
        for file_proto in file_set.file
        if file_proto.name != Timestamp.DESCRIPTOR.file.name
    }
    if len(pending) != sum(
        file_proto.name != Timestamp.DESCRIPTOR.file.name for file_proto in file_set.file
    ):
        raise StructuralExtractionError("FileDescriptorSet contains duplicate file names")
    while pending:
        progress = False
        for name, file_proto in tuple(pending.items()):
            if any(
                dependency in pending
                for dependency in file_proto.dependency
                if dependency != name
            ):
                continue
            try:
                pool.Add(file_proto)  # type: ignore[no-untyped-call]
            except Exception as error:
                raise StructuralExtractionError(
                    f"malformed protobuf descriptor file {name!r}: {error}"
                ) from error
            del pending[name]
            progress = True
        if not progress:
            unresolved = ", ".join(sorted(pending))
            raise StructuralExtractionError(
                f"protobuf descriptor dependencies cannot be resolved: {unresolved}"
            )

    classes: dict[str, type[Message]] = {}
    for name in declared_names:
        try:
            descriptor = pool.FindMessageTypeByName(name)  # type: ignore[no-untyped-call]
        except KeyError as error:
            raise StructuralExtractionError(
                f"protobuf descriptor does not define declared message {name!r}"
            ) from error
        classes[name] = message_factory.GetMessageClass(descriptor)
    return classes


def _timestamp_ns(message: Message, field_name: str, context: str) -> int:
    descriptor = message.DESCRIPTOR.fields_by_name.get(field_name)
    if (
        descriptor is None
        or descriptor.message_type is None
        or descriptor.message_type.full_name != "google.protobuf.Timestamp"
    ):
        raise StructuralExtractionError(f"{context} lacks required Timestamp field {field_name!r}")
    try:
        if not message.HasField(field_name):
            raise StructuralExtractionError(f"{context} has no value for Timestamp {field_name!r}")
        value: Any = getattr(message, field_name)
        timestamp = Timestamp(seconds=int(value.seconds), nanos=int(value.nanos))
        return timestamp.ToNanoseconds()
    except (AttributeError, TypeError, ValueError) as error:
        raise StructuralExtractionError(
            f"{context} has invalid protobuf Timestamp {field_name!r}"
        ) from error


def _parse_message(message_type: type[Message], payload: bytes, context: str) -> Message:
    message = message_type()
    try:
        message.ParseFromString(payload)
    except DecodeError as error:
        raise StructuralExtractionError(f"malformed protobuf {context} message") from error
    return message


def _require_descriptor_field(
    descriptor: Descriptor,
    *,
    context: str,
    path: str,
    name: str,
    field_type: int,
    repeated: bool,
    number: int | None = None,
    message_type: str | None = None,
) -> FieldDescriptor:
    field = descriptor.fields_by_name.get(name)
    valid = (
        field is not None
        and field.type == field_type
        and field.is_repeated is repeated
        and (repeated or not field.is_required)
        and (number is None or field.number == number)
        and (
            message_type is None
            or (field.message_type is not None and field.message_type.full_name == message_type)
        )
    )
    if not valid or field is None:
        raise StructuralExtractionError(
            f"{context} schema field {path!r} has the wrong number, type, or cardinality"
        )
    return cast(FieldDescriptor, field)


def _validate_camera_schema(descriptor: Descriptor) -> None:
    scalar_specs = (
        ("rec_frame_id", 1, FieldDescriptor.TYPE_INT64),
        ("format", 4, FieldDescriptor.TYPE_STRING),
        ("frame_id", 5, FieldDescriptor.TYPE_INT64),
        ("width", 8, FieldDescriptor.TYPE_INT32),
        ("height", 9, FieldDescriptor.TYPE_INT32),
        ("number_of_cameras", 10, FieldDescriptor.TYPE_INT32),
    )
    for name, number, field_type in scalar_specs:
        _require_descriptor_field(
            descriptor,
            context="camera",
            path=name,
            name=name,
            field_type=field_type,
            repeated=False,
            number=number,
        )
    for name, number in (("rec_timestamp", 2), ("timestamp", 3)):
        _require_descriptor_field(
            descriptor,
            context="camera",
            path=name,
            name=name,
            field_type=FieldDescriptor.TYPE_MESSAGE,
            repeated=False,
            number=number,
            message_type="google.protobuf.Timestamp",
        )
    for name, number, field_type in (
        ("data", 6, FieldDescriptor.TYPE_BYTES),
        ("name", 7, FieldDescriptor.TYPE_STRING),
    ):
        _require_descriptor_field(
            descriptor,
            context="camera",
            path=name,
            name=name,
            field_type=field_type,
            repeated=True,
            number=number,
        )
    _require_descriptor_field(
        descriptor,
        context="camera",
        path="camera_timestamp",
        name="camera_timestamp",
        field_type=FieldDescriptor.TYPE_MESSAGE,
        repeated=True,
        number=14,
        message_type="google.protobuf.Timestamp",
    )
    intrinsic_field = _require_descriptor_field(
        descriptor,
        context="camera",
        path="camera_intrinsic",
        name="camera_intrinsic",
        field_type=FieldDescriptor.TYPE_MESSAGE,
        repeated=True,
        number=11,
        message_type="autonome.CompressedVideos.CameraIntrinsic",
    )
    extrinsic_field = _require_descriptor_field(
        descriptor,
        context="camera",
        path="camera_extrinsic",
        name="camera_extrinsic",
        field_type=FieldDescriptor.TYPE_MESSAGE,
        repeated=True,
        number=12,
        message_type="autonome.CompressedVideos.CameraExtrinsic",
    )
    intrinsic = cast(Descriptor, intrinsic_field.message_type)
    for name, number in (
        ("focal_length_x", 1),
        ("focal_length_y", 2),
        ("optical_center_x", 3),
        ("optical_center_y", 4),
        ("rmse", 5),
        ("skew", 6),
    ):
        _require_descriptor_field(
            intrinsic,
            context="camera",
            path=f"camera_intrinsic.{name}",
            name=name,
            field_type=FieldDescriptor.TYPE_FLOAT,
            repeated=False,
            number=number,
        )
    _require_descriptor_field(
        intrinsic,
        context="camera",
        path="camera_intrinsic.distortion_coeffs",
        name="distortion_coeffs",
        field_type=FieldDescriptor.TYPE_DOUBLE,
        repeated=True,
        number=7,
    )
    for name, number in (("width", 8), ("height", 9)):
        _require_descriptor_field(
            intrinsic,
            context="camera",
            path=f"camera_intrinsic.{name}",
            name=name,
            field_type=FieldDescriptor.TYPE_FLOAT,
            repeated=False,
            number=number,
        )
    extrinsic = cast(Descriptor, extrinsic_field.message_type)
    for name, number in (("rotation_vector", 1), ("translation_vector", 2)):
        _require_descriptor_field(
            extrinsic,
            context="camera",
            path=f"camera_extrinsic.{name}",
            name=name,
            field_type=FieldDescriptor.TYPE_FLOAT,
            repeated=True,
            number=number,
        )


def _parse_camera(message: Message) -> tuple[RawCameraBatch, tuple[bytes, ...]]:
    dynamic: Any = message
    try:
        number_of_cameras = int(dynamic.number_of_cameras)
        if number_of_cameras <= 0:
            raise StructuralExtractionError("number_of_cameras must be greater than zero")
        arrays = {
            "data": dynamic.data,
            "name": dynamic.name,
            "camera_timestamp": dynamic.camera_timestamp,
            "camera_intrinsic": dynamic.camera_intrinsic,
            "camera_extrinsic": dynamic.camera_extrinsic,
        }
        mismatched = [name for name, values in arrays.items() if len(values) != number_of_cameras]
        if mismatched:
            raise StructuralExtractionError(
                "camera indexed-array length mismatch for " + ", ".join(mismatched)
            )
        frames: list[RawCameraFrame] = []
        payloads: list[bytes] = []
        for index in range(number_of_cameras):
            timestamp_holder = dynamic.camera_timestamp[index]
            timestamp = Timestamp(
                seconds=int(timestamp_holder.seconds), nanos=int(timestamp_holder.nanos)
            )
            try:
                camera_timestamp_ns = timestamp.ToNanoseconds()
            except ValueError as error:
                raise StructuralExtractionError(
                    f"camera[{index}] has invalid protobuf Timestamp camera_timestamp"
                ) from error
            intrinsic = dynamic.camera_intrinsic[index]
            extrinsic = dynamic.camera_extrinsic[index]
            calibration = CameraCalibration(
                CameraIntrinsic(
                    float(intrinsic.focal_length_x),
                    float(intrinsic.focal_length_y),
                    float(intrinsic.optical_center_x),
                    float(intrinsic.optical_center_y),
                    float(intrinsic.rmse),
                    float(intrinsic.skew),
                    tuple(float(value) for value in intrinsic.distortion_coeffs),
                    float(intrinsic.width),
                    float(intrinsic.height),
                ),
                CameraExtrinsic(
                    tuple(float(value) for value in extrinsic.rotation_vector),
                    tuple(float(value) for value in extrinsic.translation_vector),
                ),
            )
            frames.append(
                RawCameraFrame(
                    index,
                    str(dynamic.name[index]),
                    camera_timestamp_ns,
                    calibration,
                )
            )
            payload = bytes(dynamic.data[index])
            validate_annex_b_hevc_access_unit(payload)
            payloads.append(payload)
        return (
            RawCameraBatch(
                rec_timestamp_ns=_timestamp_ns(message, "rec_timestamp", "camera message"),
                recorded_timestamp_ns=_timestamp_ns(message, "timestamp", "camera message"),
                frame_id=int(dynamic.frame_id),
                rec_frame_id=int(dynamic.rec_frame_id),
                format=str(dynamic.format),
                width=int(dynamic.width),
                height=int(dynamic.height),
                frames=tuple(frames),
            ),
            tuple(payloads),
        )
    except StructuralExtractionError:
        raise
    except (AttributeError, TypeError, ValueError) as error:
        raise StructuralExtractionError(
            f"camera message has impossible indexed arrays: {error}"
        ) from error


def _validate_gnss_schema(descriptor: Descriptor) -> None:
    for name in ("timestamp", "rec_timestamp"):
        _require_descriptor_field(
            descriptor,
            context="GNSS",
            path=name,
            name=name,
            field_type=FieldDescriptor.TYPE_MESSAGE,
            repeated=False,
            message_type="google.protobuf.Timestamp",
        )
    _require_descriptor_field(
        descriptor,
        context="GNSS",
        path="is_valid",
        name="is_valid",
        field_type=FieldDescriptor.TYPE_BOOL,
        repeated=False,
    )
    for parent_name, child_names in _GNSS_NUMERIC_FIELDS.items():
        parent = _require_descriptor_field(
            descriptor,
            context="GNSS",
            path=parent_name,
            name=parent_name,
            field_type=FieldDescriptor.TYPE_MESSAGE,
            repeated=False,
        )
        nested = cast(Descriptor, parent.message_type)
        for child_name in child_names:
            _require_descriptor_field(
                nested,
                context="GNSS",
                path=f"{parent_name}.{child_name}",
                name=child_name,
                field_type=FieldDescriptor.TYPE_DOUBLE,
                repeated=False,
            )
    orientation_error = _require_descriptor_field(
        descriptor,
        context="GNSS",
        path="orientation_error",
        name="orientation_error",
        field_type=FieldDescriptor.TYPE_MESSAGE,
        repeated=False,
    )
    if orientation_error.message_type is None:
        raise StructuralExtractionError(
            "GNSS schema field 'orientation_error' has no message descriptor"
        )
    _validate_orientation_error_schema(
        cast(Descriptor, orientation_error.message_type),
        path="orientation_error",
        ancestors=frozenset(),
        depth=0,
        budget=TraversalBudget(),
    )


def _validate_orientation_error_schema(
    descriptor: Descriptor,
    *,
    path: str,
    ancestors: frozenset[str],
    depth: int,
    budget: TraversalBudget,
) -> None:
    budget.check_depth(path, depth)
    budget.visit(path, leaf=False, work=len(descriptor.fields))
    if descriptor.full_name in ancestors:
        raise StructuralExtractionError(
            f"GNSS schema field {path!r} has a recursive message shape"
        )
    nested_ancestors = ancestors | {descriptor.full_name}
    for field in descriptor.fields:
        field_path = f"{path}.{field.name}"
        if field.type == FieldDescriptor.TYPE_MESSAGE:
            if field.message_type is None:
                raise StructuralExtractionError(
                    f"GNSS schema field {field_path!r} has no message descriptor"
                )
            _validate_orientation_error_schema(
                cast(Descriptor, field.message_type),
                path=field_path,
                ancestors=nested_ancestors,
                depth=depth + 1,
                budget=budget,
            )
        elif field.type not in _PROTOBUF_NUMERIC_TYPES:
            raise StructuralExtractionError(
                f"GNSS schema field {field_path!r} must be numeric or a nested numeric message"
            )


def _validate_orientation_error_values(
    message: Message,
    *,
    path: str,
    depth: int = 0,
    budget: TraversalBudget | None = None,
) -> None:
    active_budget = TraversalBudget() if budget is None else budget
    active_budget.check_depth(path, depth)
    active_budget.visit(path, leaf=False, work=len(message.ListFields()))
    for field, value in message.ListFields():
        field_path = f"{path}.{field.name}"
        values = value if field.is_repeated else (value,)
        if field.type == FieldDescriptor.TYPE_MESSAGE:
            for nested in values:
                _validate_orientation_error_values(
                    cast(Message, nested),
                    path=field_path,
                    depth=depth + 1,
                    budget=active_budget,
                )
            continue
        for index, numeric in enumerate(values):
            numeric_path = f"{field_path}[{index}]" if field.is_repeated else field_path
            active_budget.visit(
                numeric_path,
                leaf=True,
                work=len(field.name) + (len(str(index)) + 2 if field.is_repeated else 0),
            )
            if not math.isfinite(float(numeric)):
                raise StructuralExtractionError(
                    f"GNSS {numeric_path} numeric value must be finite"
                )


def _message_dict(message: Message, *, root_path: str) -> dict[str, Any]:
    converted = json_format.MessageToDict(
        message,
        preserving_proto_field_name=True,
        always_print_fields_with_no_presence=True,
    )
    frozen = bounded_freeze(converted, root_path=root_path)
    return dict(cast(dict[str, Any], frozen))


def _parse_gnss(message: Message) -> GnssSample:
    dynamic: Any = message
    try:
        for field_name in (
            "lat_lon_ht",
            "orientation",
            "position_error",
            "orientation_error",
        ):
            if not message.HasField(field_name):
                raise StructuralExtractionError(
                    f"GNSS message has no value for required field {field_name!r}"
                )
        for parent_name, child_names in _GNSS_NUMERIC_FIELDS.items():
            nested_message: Message = getattr(message, parent_name)
            for child_name in child_names:
                child = nested_message.DESCRIPTOR.fields_by_name[child_name]
                if child.has_presence and not nested_message.HasField(child_name):
                    raise StructuralExtractionError(
                        f"GNSS message has no value for required field "
                        f"{parent_name}.{child_name}"
                    )
        lat_lon_ht = dynamic.lat_lon_ht
        orientation = dynamic.orientation
        position_error = dynamic.position_error
        numeric_values = (
            float(lat_lon_ht.latitude_deg),
            float(lat_lon_ht.longitude_deg),
            float(lat_lon_ht.height_m),
            float(orientation.roll_rad),
            float(orientation.pitch_rad),
            float(orientation.yaw_rad),
            float(position_error.east_sigma_m),
            float(position_error.north_sigma_m),
            float(position_error.up_sigma_m),
            float(position_error.hdop),
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise StructuralExtractionError("GNSS message contains non-finite required values")
        raw = _message_dict(message, root_path="gnss")
        for known in (
            "timestamp",
            "rec_timestamp",
            "is_valid",
            "lat_lon_ht",
            "orientation",
            "position_error",
            "orientation_error",
        ):
            raw.pop(known, None)
        _validate_orientation_error_values(
            cast(Message, dynamic.orientation_error), path="orientation_error"
        )
        orientation_uncertainty = _message_dict(
            dynamic.orientation_error, root_path="orientation_error"
        )
        return GnssSample(
            timestamp_ns=_timestamp_ns(message, "timestamp", "GNSS message"),
            rec_timestamp_ns=_timestamp_ns(message, "rec_timestamp", "GNSS message"),
            is_valid=bool(dynamic.is_valid),
            latitude_deg=numeric_values[0],
            longitude_deg=numeric_values[1],
            height_m=numeric_values[2],
            roll_rad=numeric_values[3],
            pitch_rad=numeric_values[4],
            yaw_rad=numeric_values[5],
            position_uncertainty={
                "east_sigma_m": numeric_values[6],
                "north_sigma_m": numeric_values[7],
                "up_sigma_m": numeric_values[8],
                "hdop": numeric_values[9],
            },
            orientation_uncertainty=orientation_uncertainty,
            raw_identifiers=raw,
        )
    except StructuralExtractionError:
        raise
    except (AttributeError, TypeError, ValueError) as error:
        raise StructuralExtractionError(
            f"GNSS message does not have a usable field shape: {error}"
        ) from error


def _timestamp_observations(batches: list[RawCameraBatch]) -> tuple[TimestampObservation, ...]:
    observations: list[TimestampObservation] = []
    previous_batch: int | None = None
    previous_cameras: dict[int, int] = {}
    for batch in batches:
        if previous_batch is not None:
            observations.append(
                TimestampObservation(
                    "camera_batch_rec_timestamp",
                    previous_batch,
                    batch.rec_timestamp_ns,
                    batch.rec_timestamp_ns - previous_batch,
                )
            )
        previous_batch = batch.rec_timestamp_ns
        for frame in batch.frames:
            previous = previous_cameras.get(frame.camera_index)
            if previous is not None:
                observations.append(
                    TimestampObservation(
                        f"camera[{frame.camera_index}]",
                        previous,
                        frame.camera_timestamp_ns,
                        frame.camera_timestamp_ns - previous,
                    )
                )
            previous_cameras[frame.camera_index] = frame.camera_timestamp_ns
    return tuple(observations)


def _source_identity(file_stat: os.stat_result) -> SourceIdentity:
    return SourceIdentity(
        device=file_stat.st_dev,
        inode=file_stat.st_ino,
        size=file_stat.st_size,
        modified_ns=file_stat.st_mtime_ns,
        changed_ns=file_stat.st_ctime_ns,
    )


def _assert_source_identity(path: Path, expected: SourceIdentity) -> None:
    try:
        actual = _source_identity(path.stat(follow_symlinks=False))
    except OSError as error:
        raise StructuralExtractionError(
            "recording changed between extraction passes: source is unavailable"
        ) from error
    if actual != expected:
        raise StructuralExtractionError(
            "recording changed between extraction passes; reacquire and retry"
        )


def read_recording(path: Path, camera_topic: str, gnss_topic: str) -> RawRecording:
    """Index configured topics without retaining compressed camera payload bytes."""
    camera_batches: list[RawCameraBatch] = []
    gnss_samples: list[GnssSample] = []
    camera_seen = False
    gnss_seen = False
    camera_state: CameraStructure | None = None
    schema_cache: dict[int, tuple[str, type[Message]]] = {}
    source_identity: SourceIdentity | None = None
    try:
        with path.open("rb") as stream:
            source_identity = _source_identity(os.fstat(stream.fileno()))
            reader = make_reader(stream)
            for schema, channel, record in reader.iter_messages(
                topics=[camera_topic, gnss_topic], log_time_order=True
            ):
                if schema is None:
                    raise StructuralExtractionError(
                        f"topic {channel.topic!r} message has no protobuf schema"
                    )
                if schema.encoding != "protobuf" or channel.message_encoding != "protobuf":
                    raise StructuralExtractionError(
                        f"topic {channel.topic!r} must use protobuf encoding"
                    )
                if channel.topic == camera_topic and schema.name != CAMERA_SCHEMA_NAME:
                    raise StructuralExtractionError(
                        f"camera topic requires exact schema {CAMERA_SCHEMA_NAME!r}"
                    )
                cached = schema_cache.get(schema.id)
                if cached is None:
                    classes = build_message_classes(schema.data)
                    message_type = classes.get(schema.name)
                    if message_type is None:
                        raise StructuralExtractionError(
                            f"embedded descriptor does not define MCAP schema {schema.name!r}"
                        )
                    cached = (schema.name, message_type)
                    schema_cache[schema.id] = cached
                schema_name, message_type = cached
                if channel.topic == camera_topic:
                    camera_seen = True
                    _validate_camera_schema(cast(Descriptor, message_type.DESCRIPTOR))
                    batch, _ = _parse_camera(
                        _parse_message(message_type, record.data, "camera")
                    )
                    camera_state = validate_camera_batch(batch, camera_state)
                    camera_batches.append(batch)
                elif channel.topic == gnss_topic:
                    gnss_seen = True
                    _validate_gnss_schema(cast(Descriptor, message_type.DESCRIPTOR))
                    gnss_samples.append(
                        _parse_gnss(_parse_message(message_type, record.data, "GNSS"))
                    )
            if _source_identity(os.fstat(stream.fileno())) != source_identity:
                raise StructuralExtractionError("recording changed while building extraction index")
    except StructuralExtractionError:
        raise
    except Exception as error:
        raise StructuralExtractionError(f"failed to read MCAP recording {path}: {error}") from error
    if not camera_seen:
        raise StructuralExtractionError(f"missing required camera topic {camera_topic!r}")
    if not gnss_seen:
        raise StructuralExtractionError(f"missing required GNSS topic {gnss_topic!r}")
    assert source_identity is not None
    _assert_source_identity(path, source_identity)
    indexed_gnss = index_gnss_samples(gnss_samples)
    return RawRecording(
        source_identity,
        tuple(camera_batches),
        indexed_gnss,
        _timestamp_observations(camera_batches),
    )


def iter_camera_access_units(
    path: Path,
    camera_topic: str,
    recording: RawRecording,
) -> Iterator[CameraAccessUnit]:
    """Reopen and stream structurally verified camera AUs for the decode pass."""
    _assert_source_identity(path, recording.source_identity)
    camera_state: CameraStructure | None = None
    schema_cache: dict[int, type[Message]] = {}
    batch_ordinal = 0
    try:
        with path.open("rb") as stream:
            if _source_identity(os.fstat(stream.fileno())) != recording.source_identity:
                raise StructuralExtractionError(
                    "recording changed between extraction passes; reacquire and retry"
                )
            reader = make_reader(stream)
            for schema, channel, record in reader.iter_messages(
                topics=[camera_topic], log_time_order=True
            ):
                if schema is None:
                    raise StructuralExtractionError(
                        f"topic {channel.topic!r} message has no protobuf schema"
                    )
                if schema.encoding != "protobuf" or channel.message_encoding != "protobuf":
                    raise StructuralExtractionError(
                        f"topic {channel.topic!r} must use protobuf encoding"
                    )
                if schema.name != CAMERA_SCHEMA_NAME:
                    raise StructuralExtractionError(
                        f"camera topic requires exact schema {CAMERA_SCHEMA_NAME!r}"
                    )
                message_type = schema_cache.get(schema.id)
                if message_type is None:
                    message_type = build_message_classes(schema.data).get(schema.name)
                    if message_type is None:
                        raise StructuralExtractionError(
                            f"embedded descriptor does not define MCAP schema {schema.name!r}"
                        )
                    _validate_camera_schema(cast(Descriptor, message_type.DESCRIPTOR))
                    schema_cache[schema.id] = message_type
                batch, payloads = _parse_camera(
                    _parse_message(message_type, record.data, "camera")
                )
                camera_state = validate_camera_batch(batch, camera_state)
                if batch_ordinal >= len(recording.camera_batches):
                    raise StructuralExtractionError(
                        "camera stream changed between extraction passes: extra batch"
                    )
                if batch != recording.camera_batches[batch_ordinal]:
                    raise StructuralExtractionError(
                        "camera metadata changed between extraction passes"
                    )
                for frame, payload in zip(batch.frames, payloads, strict=True):
                    yield CameraAccessUnit(batch_ordinal, batch, frame, payload)
                batch_ordinal += 1
            if batch_ordinal != len(recording.camera_batches):
                raise StructuralExtractionError(
                    "camera stream changed between extraction passes: missing batch"
                )
            if _source_identity(os.fstat(stream.fileno())) != recording.source_identity:
                raise StructuralExtractionError("recording changed during camera decode pass")
    except StructuralExtractionError:
        raise
    except Exception as error:
        raise StructuralExtractionError(
            f"failed to stream MCAP recording {path}: {error}"
        ) from error
    _assert_source_identity(path, recording.source_identity)
