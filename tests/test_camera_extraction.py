from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from dataset_devkit.extraction.camera import (
    CameraDecoderSet,
    validate_annex_b_hevc_access_unit,
    validate_camera_batch,
)
from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.models import (
    CameraCalibration,
    CameraExtrinsic,
    CameraIntrinsic,
    RawCameraBatch,
    RawCameraFrame,
)
from dataset_devkit.extraction.staging import stage_jpeg

HEVC_AU = b"\x00\x00\x00\x01\x26\x01\xaa"  # type 19 VCL NAL


def calibration(width: int = 4, height: int = 3) -> CameraCalibration:
    return CameraCalibration(
        intrinsic=CameraIntrinsic(1, 1, 2, 2, 0, 0, (0.1, 0.2), width, height),
        extrinsic=CameraExtrinsic((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)),
    )


def batch(*, timestamp: int = 10, name: str = "cam/front") -> RawCameraBatch:
    return RawCameraBatch(
        rec_timestamp_ns=timestamp,
        recorded_timestamp_ns=timestamp + 1,
        frame_id=1,
        rec_frame_id=2,
        format="h265",
        width=4,
        height=3,
        frames=(RawCameraFrame(0, name, timestamp + 5, HEVC_AU, calibration()),),
    )


def test_camera_batch_requires_exact_format_valid_arrays_and_stable_structure() -> None:
    state = validate_camera_batch(batch())
    validate_camera_batch(batch(timestamp=20), prior_state=state)

    with pytest.raises(StructuralExtractionError, match="exact format 'h265'"):
        validate_camera_batch(RawCameraBatch(**{**batch().__dict__, "format": "H265"}))
    with pytest.raises(StructuralExtractionError, match="identity.*changed"):
        validate_camera_batch(batch(timestamp=20, name="other"), prior_state=state)
    changed = batch(timestamp=20)
    changed_calibration = calibration()
    changed_calibration = replace(
        changed_calibration,
        intrinsic=replace(changed_calibration.intrinsic, focal_length_x=2),
    )
    changed_frame = RawCameraFrame(0, "cam/front", 25, HEVC_AU, changed_calibration)
    with pytest.raises(StructuralExtractionError, match="calibration.*changed"):
        validate_camera_batch(
            RawCameraBatch(**{**changed.__dict__, "frames": (changed_frame,)}), prior_state=state
        )


@pytest.mark.parametrize(
    "payload",
    [b"", b"not-annex-b", b"\x00\x00\x00\x01\x40\x01\xaa"],
)
def test_annex_b_requires_start_code_and_vcl(payload: bytes) -> None:
    with pytest.raises(StructuralExtractionError, match="Annex-B|VCL"):
        validate_annex_b_hevc_access_unit(payload)


class FakeDecoder:
    def __init__(self, outputs: list[list[Image.Image]]) -> None:
        self.outputs = outputs
        self.closed = False

    def decode(self, payload: bytes) -> list[Image.Image]:
        assert payload == HEVC_AU
        return self.outputs.pop(0)

    def close(self) -> None:
        self.closed = True


def test_decoder_is_persistent_and_cleanup_occurs_on_failure() -> None:
    created: list[FakeDecoder] = []

    def factory() -> FakeDecoder:
        decoder = FakeDecoder([[Image.new("RGB", (4, 3))], []])
        created.append(decoder)
        return decoder

    decoders = CameraDecoderSet(1, factory)
    assert decoders.decode(0, HEVC_AU).size == (4, 3)
    with pytest.raises(StructuralExtractionError, match="exactly one decoded frame"):
        decoders.decode(0, HEVC_AU)
    assert len(created) == 1
    assert created[0].closed


def test_stage_jpeg_uses_quality_95_atomic_replace_and_reopens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = Image.new("RGB", (4, 3), (12, 34, 56))
    save_calls: list[dict[str, Any]] = []
    original_save = Image.Image.save

    def save_spy(
        target: Image.Image, file: Any, format: str | None = None, **params: Any
    ) -> None:
        save_calls.append({"format": format, **params})
        original_save(target, file, format=format, **params)

    monkeypatch.setattr(Image.Image, "save", save_spy)
    staged = stage_jpeg(tmp_path, "recording-1", 0, "cam/front", 123, image, (4, 3))

    assert save_calls == [{"format": "JPEG", "quality": 95}]
    assert staged.path.name == "000-cam_front-123.jpg"
    assert staged.timestamp_ns == 123
    assert staged.path.read_bytes().startswith(b"\xff\xd8")
    assert not list(staged.path.parent.glob("*.tmp"))
    with Image.open(staged.path) as reopened:
        reopened.load()
        assert reopened.format == "JPEG"
        assert reopened.mode == "RGB"
        assert reopened.size == (4, 3)


def test_stage_jpeg_rejects_escape_symlink_and_hardlink_targets(tmp_path: Path) -> None:
    image = Image.new("RGB", (4, 3))
    with pytest.raises(StructuralExtractionError, match="recording identifier"):
        stage_jpeg(tmp_path, "../escape", 0, "cam", 1, image, (4, 3))

    recording = tmp_path / "recording"
    recording.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    target = recording / "000-cam-1.jpg"
    target.symlink_to(outside)
    with pytest.raises(StructuralExtractionError, match="unsafe existing staging target"):
        stage_jpeg(tmp_path, "recording", 0, "cam", 1, image, (4, 3))
    target.unlink()
    os.link(outside, target)
    with pytest.raises(StructuralExtractionError, match="unsafe existing staging target"):
        stage_jpeg(tmp_path, "recording", 0, "cam", 1, image, (4, 3))
