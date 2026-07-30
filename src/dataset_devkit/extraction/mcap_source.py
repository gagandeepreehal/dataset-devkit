"""Read MCAP records and dynamically decode their embedded protobuf descriptors."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

from google.protobuf import descriptor_pb2, descriptor_pool, json_format, message_factory
from google.protobuf.descriptor import Descriptor
from google.protobuf.message import DecodeError, Message
from google.protobuf.timestamp_pb2 import Timestamp
from mcap.reader import make_reader

from dataset_devkit.extraction.camera import validate_camera_batch
from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.gnss import index_gnss_samples
from dataset_devkit.extraction.models import (
    CameraCalibration,
    CameraExtrinsic,
    CameraIntrinsic,
    CameraStructure,
    GnssSample,
    RawCameraBatch,
    RawCameraFrame,
    RawRecording,
    TimestampObservation,
)

CAMERA_SCHEMA_NAME = "autonome.CompressedVideos"


def _message_names(
    file_proto: descriptor_pb2.FileDescriptorProto,
) -> list[str]:
    names: list[str] = []

    def visit(messages: Any, prefix: str) -> None:
        for message in messages:
            full_name = f"{prefix}.{message.name}" if prefix else message.name
            names.append(full_name)
            visit(message.nested_type, full_name)

    visit(file_proto.message_type, file_proto.package)
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
    for file_proto in file_set.file:
        for name in _message_names(file_proto):
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


def _parse_camera(message: Message) -> RawCameraBatch:
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
                    int(intrinsic.width),
                    int(intrinsic.height),
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
                    bytes(dynamic.data[index]),
                    calibration,
                )
            )
        return RawCameraBatch(
            rec_timestamp_ns=_timestamp_ns(message, "rec_timestamp", "camera message"),
            recorded_timestamp_ns=_timestamp_ns(message, "timestamp", "camera message"),
            frame_id=int(dynamic.frame_id),
            rec_frame_id=int(dynamic.rec_frame_id),
            format=str(dynamic.format),
            width=int(dynamic.width),
            height=int(dynamic.height),
            frames=tuple(frames),
        )
    except StructuralExtractionError:
        raise
    except (AttributeError, TypeError, ValueError) as error:
        raise StructuralExtractionError(
            f"camera message has impossible indexed arrays: {error}"
        ) from error


def _require_nested_shape(
    descriptor: Descriptor, field_name: str, required_fields: set[str]
) -> None:
    field = descriptor.fields_by_name.get(field_name)
    if field is None or field.message_type is None:
        raise StructuralExtractionError(
            f"GNSS schema lacks required message field {field_name!r}"
        )
    missing = required_fields - set(field.message_type.fields_by_name)
    if missing:
        raise StructuralExtractionError(
            f"GNSS schema field {field_name!r} lacks required fields: {', '.join(sorted(missing))}"
        )


def _validate_gnss_shape(descriptor: Descriptor) -> None:
    for field_name in ("timestamp", "rec_timestamp", "is_valid"):
        if field_name not in descriptor.fields_by_name:
            raise StructuralExtractionError(f"GNSS schema lacks required field {field_name!r}")
    _require_nested_shape(
        descriptor, "lat_lon_ht", {"latitude_deg", "longitude_deg", "height_m"}
    )
    _require_nested_shape(descriptor, "orientation", {"roll_rad", "pitch_rad", "yaw_rad"})
    _require_nested_shape(
        descriptor,
        "position_error",
        {"east_sigma_m", "north_sigma_m", "up_sigma_m", "hdop"},
    )
    _require_nested_shape(descriptor, "orientation_error", set())


def _message_dict(message: Message) -> dict[str, Any]:
    return json_format.MessageToDict(
        message,
        preserving_proto_field_name=True,
        always_print_fields_with_no_presence=True,
    )


def _parse_gnss(message: Message) -> GnssSample:
    _validate_gnss_shape(cast(Descriptor, message.DESCRIPTOR))
    dynamic: Any = message
    try:
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
        raw = _message_dict(message)
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
        orientation_uncertainty = _message_dict(dynamic.orientation_error)
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


def read_recording(path: Path, camera_topic: str, gnss_topic: str) -> RawRecording:
    """Decode configured topics in deterministic MCAP log-time order."""
    camera_batches: list[RawCameraBatch] = []
    gnss_samples: list[GnssSample] = []
    camera_seen = False
    gnss_seen = False
    camera_state: CameraStructure | None = None
    schema_cache: dict[int, tuple[str, type[Message]]] = {}
    try:
        with path.open("rb") as stream:
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
                    batch = _parse_camera(
                        _parse_message(message_type, record.data, "camera")
                    )
                    camera_state = validate_camera_batch(batch, camera_state)
                    camera_batches.append(batch)
                elif channel.topic == gnss_topic:
                    gnss_seen = True
                    gnss_samples.append(
                        _parse_gnss(_parse_message(message_type, record.data, "GNSS"))
                    )
    except StructuralExtractionError:
        raise
    except Exception as error:
        raise StructuralExtractionError(f"failed to read MCAP recording {path}: {error}") from error
    if not camera_seen:
        raise StructuralExtractionError(f"missing required camera topic {camera_topic!r}")
    if not gnss_seen:
        raise StructuralExtractionError(f"missing required GNSS topic {gnss_topic!r}")
    indexed_gnss = index_gnss_samples(gnss_samples)
    return RawRecording(
        tuple(camera_batches), indexed_gnss, _timestamp_observations(camera_batches)
    )
