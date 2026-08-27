from __future__ import annotations

import re
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import pytest
from google.protobuf import descriptor_pb2

from dataset_devkit.extraction import mcap_source
from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.mcap_source import (
    build_message_classes,
    iter_camera_access_units,
    read_recording,
)
from mcap_fixture import (
    HEVC_AU,
    camera_message,
    descriptor_set_bytes,
    gnss_message,
    message_classes,
    write_mcap,
)


def iter_descriptor_messages(
    messages: Any,
) -> Any:
    for message in messages:
        yield message
        yield from iter_descriptor_messages(message.nested_type)


def descriptor_with_field_change(
    message_name: str,
    field_name: str,
    *,
    field_type: int | None = None,
    label: int | None = None,
    type_name: str | None = None,
) -> bytes:
    file_set = descriptor_pb2.FileDescriptorSet.FromString(descriptor_set_bytes())
    target: Any = None
    for file_proto in file_set.file:
        for message in iter_descriptor_messages(file_proto.message_type):
            if message.name == message_name:
                target = next(field for field in message.field if field.name == field_name)
    assert target is not None
    if field_type is not None:
        target.type = field_type
        if field_type != descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE:
            target.ClearField("type_name")
    if label is not None:
        target.label = label
    if type_name is not None:
        target.type_name = type_name
    return file_set.SerializeToString()


def descriptor_with_optional_numeric(message_name: str, field_name: str) -> bytes:
    file_set = descriptor_pb2.FileDescriptorSet.FromString(descriptor_set_bytes())
    for file_proto in file_set.file:
        for message in iter_descriptor_messages(file_proto.message_type):
            if message.name != message_name:
                continue
            oneof_index = len(message.oneof_decl)
            message.oneof_decl.add(name=f"_{field_name}")
            field = next(field for field in message.field if field.name == field_name)
            field.proto3_optional = True
            field.oneof_index = oneof_index
            return file_set.SerializeToString()
    raise AssertionError(f"message {message_name!r} not found")


def descriptor_without_field(message_name: str, field_name: str) -> bytes:
    file_set = descriptor_pb2.FileDescriptorSet.FromString(descriptor_set_bytes())
    for file_proto in file_set.file:
        for message in iter_descriptor_messages(file_proto.message_type):
            if message.name != message_name:
                continue
            retained = [field for field in message.field if field.name != field_name]
            if len(retained) == len(message.field):
                raise AssertionError(f"field {message_name}.{field_name} not found")
            message.ClearField("field")
            message.field.extend(retained)
            return file_set.SerializeToString()
    raise AssertionError(f"message {message_name!r} not found")


def descriptor_with_nested_orientation_variances() -> bytes:
    file_set = descriptor_pb2.FileDescriptorSet.FromString(descriptor_set_bytes())
    calibration = next(file for file in file_set.file if file.name == "calibration.proto")
    orientation_error = next(
        message for message in calibration.message_type if message.name == "OrientationError"
    )
    axis = orientation_error.nested_type.add(name="AxisVariance")
    variance = axis.field.add(
        name="variance",
        number=1,
        type=descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE,
    )
    variance.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    samples = orientation_error.field.add(
        name="samples",
        number=4,
        type=descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".autonome.OrientationError.AxisVariance",
    )
    samples.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    return file_set.SerializeToString()


def descriptor_with_top_level_camera_types() -> bytes:
    file_set = descriptor_pb2.FileDescriptorSet.FromString(descriptor_set_bytes())
    telemetry = next(file for file in file_set.file if file.name == "telemetry.proto")
    camera = next(
        message for message in telemetry.message_type if message.name == "CompressedVideos"
    )
    nested: dict[str, descriptor_pb2.DescriptorProto] = {}
    for message in camera.nested_type:
        copied = descriptor_pb2.DescriptorProto()
        copied.CopyFrom(message)
        nested[message.name] = copied
        telemetry.message_type.add().CopyFrom(copied)
    camera.ClearField("nested_type")
    for field in camera.field:
        if field.name == "camera_intrinsic":
            field.type_name = ".autonome.CameraIntrinsic"
        elif field.name == "camera_extrinsic":
            field.type_name = ".autonome.CameraExtrinsic"
    top_level = {message.name: message for message in telemetry.message_type}
    top_level["CameraIntrinsic"].CopyFrom(nested["CameraIntrinsic"])
    top_level["CameraExtrinsic"].CopyFrom(nested["CameraExtrinsic"])
    return file_set.SerializeToString()


_PROTO_SCALARS = {
    "bytes": descriptor_pb2.FieldDescriptorProto.TYPE_BYTES,
    "double": descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE,
    "float": descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT,
    "int32": descriptor_pb2.FieldDescriptorProto.TYPE_INT32,
    "int64": descriptor_pb2.FieldDescriptorProto.TYPE_INT64,
    "string": descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
}


def _camera_descriptor_from_golden_proto() -> descriptor_pb2.DescriptorProto:
    """Parse the checked-in sanitized proto into a descriptor independently of fixtures."""
    path = Path(__file__).parent / "fixtures" / "CompressedVideos.proto"
    without_comments = re.sub(r"//[^\n]*", "", path.read_text(encoding="utf-8"))
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_.]*|\d+|[{}=;]", without_comments)
    package_position = tokens.index("package")
    package = tokens[package_position + 1]
    position = tokens.index("message")

    def consume(expected: str) -> None:
        nonlocal position
        assert tokens[position] == expected
        position += 1

    def parse_message(scope: str) -> descriptor_pb2.DescriptorProto:
        nonlocal position
        consume("message")
        name = tokens[position]
        position += 1
        consume("{")
        descriptor = descriptor_pb2.DescriptorProto(name=name)
        full_scope = f"{scope}.{name}"
        while tokens[position] != "}":
            if tokens[position] == "message":
                descriptor.nested_type.add().CopyFrom(parse_message(full_scope))
                continue
            repeated = tokens[position] == "repeated"
            if repeated:
                position += 1
            type_name = tokens[position]
            field_name = tokens[position + 1]
            position += 2
            consume("=")
            number = int(tokens[position])
            position += 1
            consume(";")
            field = descriptor.field.add(name=field_name, number=number)
            field.label = (
                descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
                if repeated
                else descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
            )
            scalar_type = _PROTO_SCALARS.get(type_name)
            if scalar_type is not None:
                field.type = scalar_type
            else:
                field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
                if "." in type_name:
                    field.type_name = f".{type_name}"
                elif any(nested.name == type_name for nested in descriptor.nested_type):
                    field.type_name = f"{full_scope}.{type_name}"
                else:
                    field.type_name = f"{scope}.{type_name}"
        consume("}")
        return descriptor

    return parse_message(f".{package}")


def _normalized_descriptor_contract(
    descriptor: descriptor_pb2.DescriptorProto, scope: str
) -> tuple[object, ...]:
    full_name = f"{scope}.{descriptor.name}"
    return (
        full_name,
        tuple(
            (field.name, field.number, field.type, field.label, field.type_name)
            for field in descriptor.field
        ),
        tuple(
            _normalized_descriptor_contract(nested, full_name) for nested in descriptor.nested_type
        ),
    )


def test_dynamic_descriptor_loader_resolves_reverse_dependencies() -> None:
    classes = build_message_classes(descriptor_set_bytes())
    assert "autonome.CompressedVideos" in classes
    assert "autonome.GnssFix" in classes

    with pytest.raises(StructuralExtractionError, match="FileDescriptorSet"):
        build_message_classes(b"not-a-descriptor")


def test_exact_real_nested_camera_descriptor_shape_is_accepted(tmp_path: Path) -> None:
    descriptor_data = descriptor_set_bytes()
    descriptor_files = descriptor_pb2.FileDescriptorSet.FromString(descriptor_data)
    telemetry = next(file for file in descriptor_files.file if file.name == "telemetry.proto")
    fixture_camera = next(
        message for message in telemetry.message_type if message.name == "CompressedVideos"
    )
    golden_camera = _camera_descriptor_from_golden_proto()
    assert _normalized_descriptor_contract(
        fixture_camera, ".autonome"
    ) == _normalized_descriptor_contract(golden_camera, ".autonome")

    camera_type = build_message_classes(descriptor_data)["autonome.CompressedVideos"]
    descriptor = camera_type.DESCRIPTOR
    intrinsic = descriptor.fields_by_name["camera_intrinsic"].message_type
    extrinsic = descriptor.fields_by_name["camera_extrinsic"].message_type

    assert intrinsic.full_name == "autonome.CompressedVideos.CameraIntrinsic"
    assert extrinsic.full_name == "autonome.CompressedVideos.CameraExtrinsic"
    assert intrinsic.fields_by_name["focal_length_x"].type == (
        descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT
    )
    assert intrinsic.fields_by_name["distortion_coeffs"].type == (
        descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE
    )
    assert extrinsic.fields_by_name["rotation_vector"].type == (
        descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT
    )

    path = tmp_path / "real-nested-schema.mcap"
    write_mcap(path, descriptor_data=descriptor_data)
    recording = read_recording(path, "rec_cameras", "gnss")
    assert len(recording.camera_batches[0].frames) == 2


def test_real_mcap_reader_decodes_exact_topics_and_per_camera_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "recording.mcap"
    write_mcap(path)
    recording = read_recording(path, "rec_cameras", "gnss")

    assert len(recording.camera_batches) == 1
    assert [frame.camera_timestamp_ns for frame in recording.camera_batches[0].frames] == [
        1_000_000_010,
        1_000_000_020,
    ]
    assert [sample.timestamp_ns for sample in recording.gnss_samples] == [
        900_000_000,
        2_100_000_000,
    ]
    assert recording.gnss_samples[0].raw_identifiers["receiver_id"] == "rx-1"


@pytest.mark.parametrize(
    "camera_name",
    ["", "../rear", "cam/front", r"cam\rear", "CON", "rear\x00bad", "rear\x7fbad"],
)
def test_actual_mcap_rejects_unsafe_camera_channel_names(tmp_path: Path, camera_name: str) -> None:
    path = tmp_path / "unsafe-camera-name.mcap"
    write_mcap(
        path,
        camera_payloads=(camera_message(camera_names=(camera_name, "CAM_FRONT")),),
    )

    with pytest.raises(StructuralExtractionError, match="camera.*name|safe path segment"):
        read_recording(path, "rec_cameras", "gnss")


def test_actual_mcap_preserves_exact_safe_unicode_and_uppercase_camera_names(
    tmp_path: Path,
) -> None:
    path = tmp_path / "safe-camera-names.mcap"
    write_mcap(
        path,
        camera_payloads=(camera_message(camera_names=("CAM_FRONT", "Cámara_Rear")),),
    )

    recording = read_recording(path, "rec_cameras", "gnss")

    assert tuple(frame.camera_name for frame in recording.camera_batches[0].frames) == (
        "CAM_FRONT",
        "Cámara_Rear",
    )


def _contains_bytes(value: object, seen: set[int] | None = None) -> bool:
    if isinstance(value, bytes):
        return True
    visited = seen if seen is not None else set()
    if id(value) in visited:
        return False
    visited.add(id(value))
    if is_dataclass(value) and not isinstance(value, type):
        return any(_contains_bytes(getattr(value, field.name), visited) for field in fields(value))
    if isinstance(value, dict):
        return any(
            _contains_bytes(key, visited) or _contains_bytes(item, visited)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_bytes(item, visited) for item in value)
    return False


def test_index_pass_does_not_retain_many_large_camera_payload_objects(tmp_path: Path) -> None:
    path = tmp_path / "large-recording.mcap"
    large_au = HEVC_AU + b"x" * (256 * 1024)
    batches = tuple(
        camera_message(
            1_000_000_000 + index * 1_000_000,
            (
                1_000_000_010 + index * 1_000_000,
                1_000_000_020 + index * 1_000_000,
            ),
            payloads=(large_au, large_au),
        )
        for index in range(24)
    )
    write_mcap(path, camera_payloads=batches)

    recording = read_recording(path, "rec_cameras", "gnss")

    assert not _contains_bytes(recording)
    assert len(recording.camera_batches) == 24


def test_streaming_pass_rejects_recording_changed_after_index(tmp_path: Path) -> None:
    path = tmp_path / "changed-between-passes.mcap"
    write_mcap(path)
    recording = read_recording(path, "rec_cameras", "gnss")
    path.write_bytes(path.read_bytes() + b"changed")

    stream = getattr(mcap_source, "iter_camera_access_units", None)
    assert callable(stream), "two-pass camera streaming API is required"
    with pytest.raises(StructuralExtractionError, match="changed between extraction passes"):
        list(stream(path, "rec_cameras", recording))


def test_backward_camera_timestamp_is_observed_without_structural_rejection(
    tmp_path: Path,
) -> None:
    path = tmp_path / "backward.mcap"
    write_mcap(
        path,
        camera_payloads=(
            camera_message(1_000, (1_010, 1_020)),
            camera_message(2_000, (900, 910)),
        ),
    )

    recording = read_recording(path, "rec_cameras", "gnss")

    camera_deltas = [
        item.delta_ns for item in recording.timestamp_observations if item.stream == "camera[0]"
    ]
    assert camera_deltas == [-110]


def test_invalid_indexed_camera_timestamp_is_structural(tmp_path: Path) -> None:
    camera_type, _ = message_classes()
    message = camera_type.FromString(camera_message())
    message.camera_timestamp[0].nanos = -1
    path = tmp_path / "invalid-timestamp.mcap"
    write_mcap(path, camera_payloads=(bytes(message.SerializeToString()),))

    with pytest.raises(StructuralExtractionError, match="invalid protobuf Timestamp"):
        read_recording(path, "rec_cameras", "gnss")


@pytest.mark.parametrize("payload", [b"not-annex-b", b"\x00\x00\x00\x01\x40\x01\xaa"])
def test_actual_mcap_rejects_non_annex_b_or_non_vcl_camera_payload(
    tmp_path: Path, payload: bytes
) -> None:
    path = tmp_path / "corrupt-camera.mcap"
    write_mcap(path, camera_payloads=(camera_message(payloads=(payload, payload)),))

    with pytest.raises(StructuralExtractionError, match="Annex-B|VCL"):
        read_recording(path, "rec_cameras", "gnss")


@pytest.mark.parametrize(
    ("descriptor_data", "field_name"),
    [
        (
            descriptor_with_field_change(
                "CompressedVideos",
                "data",
                label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
            ),
            "data",
        ),
        (
            descriptor_with_field_change(
                "CompressedVideos",
                "format",
                field_type=descriptor_pb2.FieldDescriptorProto.TYPE_BYTES,
            ),
            "format",
        ),
        (
            descriptor_with_field_change(
                "CompressedVideos",
                "rec_timestamp",
                type_name=".autonome.CompressedVideos.CameraIntrinsic",
            ),
            "rec_timestamp",
        ),
        (
            descriptor_with_field_change(
                "CameraIntrinsic",
                "width",
                field_type=descriptor_pb2.FieldDescriptorProto.TYPE_UINT32,
            ),
            "camera_intrinsic.width",
        ),
    ],
)
def test_camera_descriptor_rejects_wrong_type_or_cardinality_lookalikes(
    tmp_path: Path, descriptor_data: bytes, field_name: str
) -> None:
    path = tmp_path / f"bad-camera-{field_name.replace('.', '-')}.mcap"
    write_mcap(path, descriptor_data=descriptor_data)

    with pytest.raises(StructuralExtractionError, match=f"camera schema field.*{field_name}"):
        read_recording(path, "rec_cameras", "gnss")


def test_camera_descriptor_rejects_top_level_calibration_types(
    tmp_path: Path,
) -> None:
    descriptor_data = descriptor_with_top_level_camera_types()
    path = tmp_path / "legacy-top-level-camera-types.mcap"
    write_mcap(
        path,
        descriptor_data=descriptor_data,
        camera_payloads=(camera_message(descriptor_data=descriptor_data),),
    )

    with pytest.raises(StructuralExtractionError, match="camera schema field.*camera_intrinsic"):
        read_recording(path, "rec_cameras", "gnss")


def test_missing_per_camera_timestamps_use_auditable_batch_fallback(
    tmp_path: Path,
) -> None:
    descriptor_data = descriptor_without_field("CompressedVideos", "camera_timestamp")
    path = tmp_path / "missing-camera-timestamps.mcap"
    write_mcap(
        path,
        descriptor_data=descriptor_data,
        camera_payloads=(camera_message(descriptor_data=descriptor_data),),
    )

    with pytest.raises(StructuralExtractionError, match="camera_timestamp"):
        read_recording(path, "rec_cameras", "gnss")

    recording = read_recording(
        path,
        "rec_cameras",
        "gnss",
        allow_camera_timestamp_batch_fallback=True,
    )
    access_units = tuple(
        iter_camera_access_units(
            path,
            "rec_cameras",
            recording,
            allow_camera_timestamp_batch_fallback=True,
        )
    )

    frames = recording.camera_batches[0].frames
    assert {frame.camera_timestamp_source for frame in frames} == {"batch_timestamp"}
    assert {frame.camera_timestamp_ns for frame in frames} == {1_000_000_005}
    assert len(access_units) == 2


def test_camera_native_calibration_resolution_mode_is_opt_in_and_uniform(
    tmp_path: Path,
) -> None:
    payload = camera_message(
        dimensions=(1280, 720),
        intrinsic_dimensions=(1920, 1080),
    )
    path = tmp_path / "native-resolution-calibration.mcap"
    write_mcap(path, camera_payloads=(payload,))

    with pytest.raises(StructuralExtractionError, match="dimensions differ"):
        read_recording(path, "rec_cameras", "gnss")

    recording = read_recording(
        path,
        "rec_cameras",
        "gnss",
        allow_native_camera_calibration_resolution=True,
    )
    access_units = tuple(
        iter_camera_access_units(
            path,
            "rec_cameras",
            recording,
            allow_native_camera_calibration_resolution=True,
        )
    )

    assert len(access_units) == 2
    assert recording.camera_batches[0].frames[0].calibration.intrinsic.width == 1920


def test_camera_native_calibration_resolution_rejects_aspect_ratio_change(
    tmp_path: Path,
) -> None:
    payload = camera_message(
        dimensions=(1280, 720),
        intrinsic_dimensions=(1920, 1200),
    )
    path = tmp_path / "bad-native-resolution-calibration.mcap"
    write_mcap(path, camera_payloads=(payload,))

    with pytest.raises(StructuralExtractionError, match="aspect ratio"):
        read_recording(
            path,
            "rec_cameras",
            "gnss",
            allow_native_camera_calibration_resolution=True,
        )


@pytest.mark.parametrize(
    ("message_name", "field_name"),
    [("GnssFix", "is_valid"), ("PositionError", "hdop")],
)
def test_gnss_descriptor_rejects_wrong_scalar_types(
    tmp_path: Path, message_name: str, field_name: str
) -> None:
    descriptor_data = descriptor_with_field_change(
        message_name,
        field_name,
        field_type=descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
    )
    path = tmp_path / f"bad-gnss-{field_name}.mcap"
    write_mcap(path, descriptor_data=descriptor_data)

    with pytest.raises(StructuralExtractionError, match=f"GNSS schema field.*{field_name}"):
        read_recording(path, "rec_cameras", "gnss")


def test_gnss_compatible_numeric_mode_accepts_float_hdop(tmp_path: Path) -> None:
    descriptor_data = descriptor_with_field_change(
        "PositionError",
        "hdop",
        field_type=descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT,
    )
    path = tmp_path / "float-hdop.mcap"
    write_mcap(
        path,
        descriptor_data=descriptor_data,
        gnss_payloads=(gnss_message(900_000_000, 0, descriptor_data=descriptor_data),),
    )

    with pytest.raises(StructuralExtractionError, match="position_error.hdop"):
        read_recording(path, "rec_cameras", "gnss")

    recording = read_recording(
        path,
        "rec_cameras",
        "gnss",
        compatible_gnss_numeric_types=True,
    )

    assert recording.gnss_samples[0].position_uncertainty["hdop"] == pytest.approx(1.0)


def test_missing_gnss_rec_timestamp_log_time_fallback_is_opt_in_and_auditable(
    tmp_path: Path,
) -> None:
    _, gnss_type = message_classes()
    message = gnss_type.FromString(gnss_message(900_000_000, 0))
    message.ClearField("rec_timestamp")
    path = tmp_path / "missing-gnss-rec-timestamp.mcap"
    write_mcap(path, gnss_payloads=(bytes(message.SerializeToString()),))

    with pytest.raises(
        StructuralExtractionError,
        match="has no value for Timestamp 'rec_timestamp'",
    ):
        read_recording(path, "rec_cameras", "gnss")

    recording = read_recording(
        path,
        "rec_cameras",
        "gnss",
        allow_gnss_rec_timestamp_log_time_fallback=True,
    )

    sample = recording.gnss_samples[0]
    assert sample.rec_timestamp_ns == 50
    assert sample.raw_identifiers["dataset_devkit_rec_timestamp_fallback"] == {
        "source": "mcap_log_time",
        "timestamp_ns": 50,
    }


def test_gnss_descriptor_is_validated_before_message_decode(tmp_path: Path) -> None:
    descriptor_data = descriptor_with_field_change(
        "GnssFix",
        "is_valid",
        field_type=descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
    )
    path = tmp_path / "bad-gnss-descriptor-and-message.mcap"
    write_mcap(path, descriptor_data=descriptor_data, gnss_payloads=(b"\x80",))

    with pytest.raises(StructuralExtractionError, match="GNSS schema field.*is_valid"):
        read_recording(path, "rec_cameras", "gnss")


@pytest.mark.parametrize(
    "field_name",
    [
        "timestamp",
        "rec_timestamp",
        "lat_lon_ht",
        "orientation",
        "position_error",
        "orientation_error",
    ],
)
def test_gnss_message_requires_present_timestamp_and_nested_messages(
    tmp_path: Path, field_name: str
) -> None:
    _, gnss_type = message_classes()
    message = gnss_type.FromString(gnss_message(900_000_000, 0))
    message.ClearField(field_name)
    path = tmp_path / f"missing-{field_name}.mcap"
    write_mcap(path, gnss_payloads=(bytes(message.SerializeToString()),))

    with pytest.raises(StructuralExtractionError, match=f"GNSS message.*{field_name}"):
        read_recording(path, "rec_cameras", "gnss")


def test_gnss_message_requires_nested_numeric_when_descriptor_exposes_presence(
    tmp_path: Path,
) -> None:
    descriptor_data = descriptor_with_optional_numeric("PositionError", "hdop")
    _, gnss_type = message_classes(descriptor_data)
    message = gnss_type.FromString(gnss_message(900_000_000, 0))
    message.ClearField("position_error")
    message.position_error.east_sigma_m = 0.1
    message.position_error.north_sigma_m = 0.2
    message.position_error.up_sigma_m = 0.3
    path = tmp_path / "missing-numeric-presence.mcap"
    write_mcap(
        path,
        descriptor_data=descriptor_data,
        gnss_payloads=(bytes(message.SerializeToString()),),
    )

    with pytest.raises(StructuralExtractionError, match="position_error.hdop"):
        read_recording(path, "rec_cameras", "gnss")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_gnss_orientation_error_nonfinite_value_is_structural(tmp_path: Path, value: float) -> None:
    _, gnss_type = message_classes()
    message = gnss_type.FromString(gnss_message(900_000_000, 0))
    message.orientation_error.yaw_variance = value
    path = tmp_path / "nonfinite-orientation-error.mcap"
    write_mcap(path, gnss_payloads=(bytes(message.SerializeToString()),))

    with pytest.raises(StructuralExtractionError, match="orientation_error.*finite"):
        read_recording(path, "rec_cameras", "gnss")


def test_gnss_orientation_error_descriptor_rejects_nonnumeric_field(
    tmp_path: Path,
) -> None:
    descriptor_data = descriptor_with_field_change(
        "OrientationError",
        "yaw_variance",
        field_type=descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
    )
    path = tmp_path / "string-orientation-error.mcap"
    write_mcap(path, descriptor_data=descriptor_data, gnss_payloads=(b"",))

    with pytest.raises(
        StructuralExtractionError,
        match="GNSS schema field.*orientation_error.yaw_variance.*numeric",
    ):
        read_recording(path, "rec_cameras", "gnss")


def test_gnss_repeated_orientation_error_nonfinite_value_is_structural(
    tmp_path: Path,
) -> None:
    descriptor_data = descriptor_with_field_change(
        "OrientationError",
        "yaw_variance",
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
    )
    _, gnss_type = message_classes(descriptor_data)
    message = gnss_type.FromString(gnss_message(900_000_000, 0))
    message.orientation_error.yaw_variance.extend([0.1, float("inf")])
    path = tmp_path / "repeated-nonfinite-orientation-error.mcap"
    write_mcap(
        path,
        descriptor_data=descriptor_data,
        gnss_payloads=(bytes(message.SerializeToString()),),
    )

    with pytest.raises(StructuralExtractionError, match="orientation_error.yaw_variance.*finite"):
        read_recording(path, "rec_cameras", "gnss")


def test_gnss_nested_orientation_error_nonfinite_value_is_structural(
    tmp_path: Path,
) -> None:
    descriptor_data = descriptor_with_nested_orientation_variances()
    _, gnss_type = message_classes(descriptor_data)
    message = gnss_type.FromString(gnss_message(900_000_000, 0))
    message.orientation_error.samples.add().variance = float("nan")
    path = tmp_path / "nested-nonfinite-orientation-error.mcap"
    write_mcap(
        path,
        descriptor_data=descriptor_data,
        gnss_payloads=(bytes(message.SerializeToString()),),
    )

    with pytest.raises(
        StructuralExtractionError,
        match="orientation_error.samples.variance.*finite",
    ):
        read_recording(path, "rec_cameras", "gnss")


def test_deep_protobuf_descriptor_shape_is_bounded_structurally() -> None:
    file_proto = descriptor_pb2.FileDescriptorProto(
        name="deep.proto", package="deep", syntax="proto3"
    )
    current = file_proto.message_type.add(name="Root")
    for index in range(33):
        current = current.nested_type.add(name=f"Level{index}")
    file_set = descriptor_pb2.FileDescriptorSet()
    file_set.file.add().CopyFrom(file_proto)

    with pytest.raises(
        StructuralExtractionError,
        match=r"deep\.Root(?:\.Level\d+)+.*maximum depth 32",
    ):
        build_message_classes(file_set.SerializeToString())


def test_wide_repeated_protobuf_uncertainty_is_bounded_structurally(
    tmp_path: Path,
) -> None:
    descriptor_data = descriptor_with_field_change(
        "OrientationError",
        "yaw_variance",
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
    )
    _, gnss_type = message_classes(descriptor_data)
    message = gnss_type.FromString(gnss_message(900_000_000, 0))
    message.orientation_error.yaw_variance.extend(float(index) for index in range(8_001))
    path = tmp_path / "wide-orientation-error.mcap"
    write_mcap(
        path,
        descriptor_data=descriptor_data,
        gnss_payloads=(bytes(message.SerializeToString()),),
    )

    with pytest.raises(StructuralExtractionError, match="maximum leaves 8000"):
        read_recording(path, "rec_cameras", "gnss")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"camera_schema_name": "autonome.Other"}, "exact schema"),
        ({"camera_encoding": "jsonschema"}, "encoding"),
        ({"include_camera": False}, "missing required camera topic"),
        ({"include_gnss": False}, "missing required GNSS topic"),
        ({"descriptor_data": b"not-a-descriptor"}, "FileDescriptorSet"),
        ({"camera_payloads": (b"\x80",)}, "protobuf camera message"),
        ({"camera_payloads": (camera_message(format_name="H265"),)}, "exact format"),
        ({"camera_payloads": (camera_message(camera_timestamps_ns=(1,)),)}, "array length"),
    ],
)
def test_mcap_structural_failures_are_clear(
    tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    path = tmp_path / "bad.mcap"
    write_mcap(path, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(StructuralExtractionError, match=message):
        read_recording(path, "rec_cameras", "gnss")
