from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from google.protobuf import descriptor_pb2

from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.mcap_source import build_message_classes, read_recording
from mcap_fixture import (
    camera_message,
    descriptor_set_bytes,
    gnss_message,
    message_classes,
    write_mcap,
)


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
        for message in file_proto.message_type:
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
        for message in file_proto.message_type:
            if message.name != message_name:
                continue
            oneof_index = len(message.oneof_decl)
            message.oneof_decl.add(name=f"_{field_name}")
            field = next(field for field in message.field if field.name == field_name)
            field.proto3_optional = True
            field.oneof_index = oneof_index
            return file_set.SerializeToString()
    raise AssertionError(f"message {message_name!r} not found")


def test_dynamic_descriptor_loader_resolves_reverse_dependencies() -> None:
    classes = build_message_classes(descriptor_set_bytes())
    assert "autonome.CompressedVideos" in classes
    assert "autonome.GnssFix" in classes

    with pytest.raises(StructuralExtractionError, match="FileDescriptorSet"):
        build_message_classes(b"not-a-descriptor")


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


@pytest.mark.parametrize(
    "payload", [b"not-annex-b", b"\x00\x00\x00\x01\x40\x01\xaa"]
)
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
                type_name=".autonome.CameraIntrinsic",
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
