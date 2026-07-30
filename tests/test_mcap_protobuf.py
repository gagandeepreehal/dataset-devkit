from __future__ import annotations

from pathlib import Path

import pytest

from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.mcap_source import build_message_classes, read_recording
from mcap_fixture import camera_message, descriptor_set_bytes, message_classes, write_mcap


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
    ("kwargs", "message"),
    [
        ({"camera_schema_name": "autonome.Other"}, "exact schema"),
        ({"camera_encoding": "jsonschema"}, "encoding"),
        ({"include_camera": False}, "missing required camera topic"),
        ({"include_gnss": False}, "missing required GNSS topic"),
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
