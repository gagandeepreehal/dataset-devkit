from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image

from dataset_devkit.extraction import staging as staging_module
from dataset_devkit.extraction.camera import (
    CameraDecoderSet,
    DecoderOutput,
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
from dataset_devkit.extraction.staging import stage_jpeg, verify_staged_image_identity

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
        frames=(RawCameraFrame(0, name, timestamp + 5, calibration()),),
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
    changed_frame = RawCameraFrame(0, "cam/front", 25, changed_calibration)
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


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"junk\x00\x00\x01\x26\x01\xaa", "before first"),
        (b"\x00\x00\x01\x26\x01\x00\x00\x00\x01", "empty NAL"),
        (b"\x00\x00\x01\xa6\x01\xaa", "forbidden_zero_bit"),
        (b"\x00\x00\x01\x26\x00\xaa", "temporal_id_plus1"),
        (b"\x00\x00\x01\x26", "two-byte"),
    ],
)
def test_annex_b_rejects_junk_empty_or_invalid_hevc_headers(payload: bytes, message: str) -> None:
    with pytest.raises(StructuralExtractionError, match=message):
        validate_annex_b_hevc_access_unit(payload)


def test_annex_b_accepts_mixed_three_and_four_byte_boundaries() -> None:
    validate_annex_b_hevc_access_unit(b"\x00\x00\x00\x01\x40\x01\xaa\x00\x00\x01\x26\x01\xbb")


class FakeDecoder:
    def __init__(self, outputs: list[list[Image.Image]]) -> None:
        self.outputs = outputs
        self.closed = False

    def decode(self, payload: bytes, pts: int, time_base: Fraction) -> list[DecoderOutput]:
        assert payload == HEVC_AU
        return [DecoderOutput(pts, image) for image in self.outputs.pop(0)]

    def flush(self) -> list[DecoderOutput]:
        return []

    def close(self) -> None:
        self.closed = True


def output_frame(pts: int | None, color: int) -> DecoderOutput:
    return DecoderOutput(pts, Image.new("RGB", (4, 3), (color, 0, 0)))


class DelayedDecoder:
    def __init__(self, outputs: list[list[DecoderOutput]], flush: list[DecoderOutput]) -> None:
        self.outputs = outputs
        self.flush_outputs = flush
        self.submitted_pts: list[int] = []
        self.closed = False

    def decode(self, payload: bytes, pts: int, time_base: Fraction) -> list[DecoderOutput]:
        assert payload == HEVC_AU
        assert time_base == Fraction(1, 1_000_000_000)
        self.submitted_pts.append(pts)
        return self.outputs.pop(0)

    def flush(self) -> list[DecoderOutput]:
        return self.flush_outputs

    def close(self) -> None:
        self.closed = True


def test_decoder_associates_zero_now_multiple_later_and_flush_outputs_by_pts() -> None:
    decoder = DelayedDecoder(
        outputs=[[], [], [output_frame(1, 20), output_frame(0, 10)]],
        flush=[output_frame(2, 30)],
    )
    decoders = CameraDecoderSet(1, lambda: decoder)

    assert decoders.submit(0, HEVC_AU, "first") == ()
    assert decoders.submit(0, HEVC_AU, "second") == ()
    emitted = decoders.submit(0, HEVC_AU, "third")
    flushed = decoders.finish()

    assert [
        (item.metadata, cast(tuple[int, ...], item.image.getpixel((0, 0)))[0]) for item in emitted
    ] == [
        ("second", 20),
        ("first", 10),
    ]
    assert [
        (item.metadata, cast(tuple[int, ...], item.image.getpixel((0, 0)))[0]) for item in flushed
    ] == [("third", 30)]
    assert decoder.submitted_pts == [0, 1, 2]
    assert decoder.closed


@pytest.mark.parametrize(
    ("outputs", "flush", "message"),
    [
        ([[], [output_frame(99, 1)]], [], "unknown"),
        ([[output_frame(0, 1)], [output_frame(0, 2)]], [], "duplicate"),
        ([[], []], [output_frame(0, 1)], "missing"),
        ([[]], [output_frame(None, 1), output_frame(None, 2)], "missing PTS|ambiguous"),
    ],
)
def test_decoder_rejects_unknown_duplicate_missing_or_ambiguous_pts_and_closes(
    outputs: list[list[DecoderOutput]], flush: list[DecoderOutput], message: str
) -> None:
    decoder = DelayedDecoder(outputs, flush)
    decoders = CameraDecoderSet(1, lambda: decoder)
    with pytest.raises(StructuralExtractionError, match=message):
        for index in range(len(outputs)):
            decoders.submit(0, HEVC_AU, f"frame-{index}")
        decoders.finish()
    assert decoder.closed


def test_decoder_is_persistent_and_cleanup_occurs_on_failure() -> None:
    created: list[FakeDecoder] = []

    def factory() -> FakeDecoder:
        decoder = FakeDecoder([[Image.new("RGB", (4, 3))], []])
        created.append(decoder)
        return decoder

    decoders = CameraDecoderSet(1, factory)
    assert decoders.submit(0, HEVC_AU, "first")[0].image.size == (4, 3)
    assert decoders.submit(0, HEVC_AU, "second") == ()
    with pytest.raises(StructuralExtractionError, match="missing"):
        decoders.finish()
    assert len(created) == 1
    assert created[0].closed


def test_decoder_construction_failure_closes_every_created_context_once() -> None:
    close_counts: list[int] = []

    class ConstructionDecoder:
        def __init__(self) -> None:
            close_counts.append(0)
            self.index = len(close_counts) - 1

        def decode(self, payload: bytes, pts: int, time_base: Fraction) -> list[DecoderOutput]:
            return [DecoderOutput(pts, Image.new("RGB", (4, 3)))]

        def flush(self) -> list[DecoderOutput]:
            return []

        def close(self) -> None:
            close_counts[self.index] += 1

    calls = 0

    def factory() -> ConstructionDecoder:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("construction failed")
        return ConstructionDecoder()

    with pytest.raises(RuntimeError, match="construction failed"):
        CameraDecoderSet(3, factory)
    assert close_counts == [1, 1]


def test_stage_jpeg_uses_quality_95_atomic_replace_and_reopens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = Image.new("RGB", (4, 3), (12, 34, 56))
    save_calls: list[dict[str, Any]] = []
    original_save = Image.Image.save

    def save_spy(target: Image.Image, file: Any, format: str | None = None, **params: Any) -> None:
        save_calls.append({"format": format, **params})
        original_save(target, file, format=format, **params)

    monkeypatch.setattr(Image.Image, "save", save_spy)
    staged = stage_jpeg(tmp_path, "recording-1", 0, "CAM_FRONT", 123, image, (4, 3))

    assert save_calls == [{"format": "JPEG", "quality": 95}]
    assert staged.path.name == "000-CAM_FRONT-123.jpg"
    assert staged.timestamp_ns == 123
    assert staged.path.read_bytes().startswith(b"\xff\xd8")
    assert not list(staged.path.parent.glob("*.tmp"))
    with Image.open(staged.path) as reopened:
        reopened.load()
        assert reopened.format == "JPEG"
        assert reopened.mode == "RGB"
        assert reopened.size == (4, 3)
    original = staged.path.read_bytes()
    with pytest.raises(StructuralExtractionError, match="refusing to clobber"):
        stage_jpeg(
            tmp_path,
            "recording-1",
            0,
            "CAM_FRONT",
            123,
            Image.new("RGB", (4, 3), (200, 1, 2)),
            (4, 3),
        )
    assert staged.path.read_bytes() == original


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


@pytest.mark.parametrize("attack", ["mutation", "replacement", "symlink", "hardlink"])
def test_staged_image_identity_rejects_content_and_name_attacks(
    tmp_path: Path, attack: str
) -> None:
    staged = stage_jpeg(tmp_path, "recording", 0, "CAM_FRONT", 1, Image.new("RGB", (4, 3)), (4, 3))
    verify_staged_image_identity(staged)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(staged.path.read_bytes())
    if attack == "mutation":
        staged.path.write_bytes(staged.path.read_bytes() + b"changed")
    elif attack == "replacement":
        staged.path.unlink()
        staged.path.write_bytes(outside.read_bytes())
    elif attack == "symlink":
        staged.path.unlink()
        staged.path.symlink_to(outside)
    else:
        os.link(staged.path, outside.with_name("second-link.jpg"))

    with pytest.raises(StructuralExtractionError, match="staged image"):
        verify_staged_image_identity(staged)


def test_staged_image_identity_rejects_forged_path_and_ancestor_swap(
    tmp_path: Path,
) -> None:
    staged = stage_jpeg(tmp_path, "recording", 0, "CAM_FRONT", 1, Image.new("RGB", (4, 3)), (4, 3))
    unrelated = tmp_path / "recording" / "unrelated.jpg"
    unrelated.write_bytes(staged.path.read_bytes())
    forged = replace(staged, path=unrelated, root_relative_path=unrelated.name)
    with pytest.raises(StructuralExtractionError, match="identity|ownership"):
        verify_staged_image_identity(forged)

    original_directory = tmp_path / "recording-original"
    staged.path.parent.rename(original_directory)
    (tmp_path / "recording").mkdir()
    with pytest.raises(StructuralExtractionError, match="ancestor"):
        verify_staged_image_identity(staged)


def test_stage_jpeg_rejects_symlink_in_staging_ancestor(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_ancestor = tmp_path / "linked"
    linked_ancestor.symlink_to(outside, target_is_directory=True)

    with pytest.raises(StructuralExtractionError, match="staging.*ancestor|directory"):
        stage_jpeg(
            linked_ancestor / "staging",
            "recording",
            0,
            "cam",
            1,
            Image.new("RGB", (4, 3)),
            (4, 3),
        )
    assert not (outside / "staging").exists()


def test_stage_jpeg_rejects_same_dimension_content_swap_and_removes_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alternate_stream = BytesIO()
    Image.new("RGB", (4, 3), (200, 1, 2)).save(alternate_stream, format="JPEG", quality=95)
    alternate = alternate_stream.getvalue()
    original_link = os.link

    def link_then_substitute(
        source: str,
        destination: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        original_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )
        target_fd = os.open(destination, os.O_WRONLY | os.O_TRUNC, dir_fd=dst_dir_fd)
        try:
            os.write(target_fd, alternate)
            os.fsync(target_fd)
        finally:
            os.close(target_fd)

    monkeypatch.setattr(os, "link", link_then_substitute)
    with pytest.raises(StructuralExtractionError, match="content.*changed|identity"):
        stage_jpeg(
            tmp_path,
            "recording",
            0,
            "cam",
            1,
            Image.new("RGB", (4, 3), (1, 2, 3)),
            (4, 3),
        )
    assert not (tmp_path / "recording" / "000-cam-1.jpg").exists()


def test_stage_jpeg_detects_ancestor_swap_without_touching_outside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging_root = tmp_path / "staging"
    recording_dir = staging_root / "recording"
    moved_dir = staging_root / "recording-original"
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_target = outside / "000-cam-1.jpg"
    Image.new("RGB", (4, 3), (220, 5, 6)).save(outside_target, format="JPEG", quality=95)
    outside_bytes = outside_target.read_bytes()
    original_link = os.link

    def link_then_swap_ancestor(
        source: str,
        destination: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        original_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )
        recording_dir.rename(moved_dir)
        recording_dir.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(os, "link", link_then_swap_ancestor)
    with pytest.raises(StructuralExtractionError, match="ancestor.*changed|directory"):
        stage_jpeg(
            staging_root,
            "recording",
            0,
            "cam",
            1,
            Image.new("RGB", (4, 3), (1, 2, 3)),
            (4, 3),
        )
    assert outside_target.read_bytes() == outside_bytes
    assert not list(moved_dir.glob("*.jpg"))


def test_failed_stage_does_not_unlink_replacement_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replacement = b"replacement-owned-by-someone-else"
    original_link = os.link

    def link_then_replace_inode(
        source: str,
        destination: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        original_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )
        os.unlink(destination, dir_fd=dst_dir_fd)
        target_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=dst_dir_fd,
        )
        try:
            os.write(target_fd, replacement)
        finally:
            os.close(target_fd)

    monkeypatch.setattr(os, "link", link_then_replace_inode)
    with pytest.raises(StructuralExtractionError, match="identity|content|verification"):
        stage_jpeg(
            tmp_path,
            "recording",
            0,
            "cam",
            1,
            Image.new("RGB", (4, 3)),
            (4, 3),
        )
    target = tmp_path / "recording" / "000-cam-1.jpg"
    assert target.read_bytes() == replacement


def test_staging_invocations_isolate_reruns_concurrency_and_sanitized_collisions(
    tmp_path: Path,
) -> None:
    create = getattr(staging_module, "create_staging_invocation", None)
    assert callable(create), "exclusive staging invocation API is required"

    def stage(recording_id: str) -> tuple[object, Path]:
        invocation = create(tmp_path / "staging", recording_id)
        staged = stage_jpeg(
            tmp_path / "staging",
            invocation.directory_name,
            0,
            "CAM_FRONT",
            123,
            Image.new("RGB", (4, 3), (1, 2, 3)),
            (4, 3),
            batch_ordinal=0,
            invocation=invocation,
        )
        return invocation, staged.path

    with ThreadPoolExecutor(max_workers=4) as executor:
        pairs = list(executor.map(stage, ["same", "same", "a/b", "a?b"]))

    roots = [pair[1].parent for pair in pairs]
    assert len(set(roots)) == 4
    assert all(path.is_file() for _, path in pairs)
    assert roots[2].name.startswith("a_b-")
    assert roots[3].name.startswith("a_b-")


def test_duplicate_timestamps_are_disambiguated_and_rollback_is_inode_owned(
    tmp_path: Path,
) -> None:
    create = getattr(staging_module, "create_staging_invocation", None)
    rollback = getattr(staging_module, "rollback_staging_invocation", None)
    assert callable(create) and callable(rollback)
    staging_root = tmp_path / "staging"
    prior = create(staging_root, "recording")
    first = stage_jpeg(
        staging_root,
        prior.directory_name,
        0,
        "cam",
        123,
        Image.new("RGB", (4, 3)),
        (4, 3),
        batch_ordinal=0,
        invocation=prior,
    )
    second = stage_jpeg(
        staging_root,
        prior.directory_name,
        0,
        "cam",
        123,
        Image.new("RGB", (4, 3)),
        (4, 3),
        batch_ordinal=1,
        invocation=prior,
    )
    failed = create(staging_root, "recording")
    failed_image = stage_jpeg(
        staging_root,
        failed.directory_name,
        0,
        "cam",
        123,
        Image.new("RGB", (4, 3)),
        (4, 3),
        batch_ordinal=0,
        invocation=failed,
    )

    rollback(failed)

    assert first.path != second.path
    assert first.path.is_file() and second.path.is_file()
    assert not failed_image.path.exists()
    assert not failed.path.exists()
