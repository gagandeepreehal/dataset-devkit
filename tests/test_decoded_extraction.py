from __future__ import annotations

import shutil
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataset_devkit.config import DEFAULT_DECODED_CAMERAS, CameraName
from dataset_devkit.decoded_acquisition import (
    AcquiredDecodedRecording,
    DecodedHfAcquirer,
)
from dataset_devkit.extraction import decoded as decoded_module
from dataset_devkit.extraction.decoded import DecodedRecordingExtractor
from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.staging import verify_staged_image_identity
from dataset_devkit.publication import OwnedDirectoryCleanupError
from decoded_v2_fixture import (
    DecodedFixture,
    decoded_source_config,
    rewrite_as_anonymizer_layout,
    write_decoded_fixture,
)


def _download(root: Path) -> Callable[..., str]:
    def copy(**kwargs: object) -> str:
        filename = str(kwargs["filename"])
        target = Path(str(kwargs["local_dir"])) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / filename, target)
        return str(target)

    return copy


def acquire_fixture_bundle(
    fixture: DecodedFixture,
    cache: Path,
    *,
    cameras: list[CameraName] | None = None,
) -> AcquiredDecodedRecording:
    acquirer = DecodedHfAcquirer(
        source=decoded_source_config(
            fixture.revision, [fixture.recording_id], cameras=cameras
        ),
        cache_dir=cache,
        download_file=_download(fixture.root),
    )
    return acquirer.acquire(acquirer.load_plan().recordings[0])


def make_decoded_extractor(tmp_path: Path) -> DecodedRecordingExtractor:
    return DecodedRecordingExtractor(
        target_fps=Fraction(1, 1),
        tolerance_ns=1,
        staging_root=tmp_path / "work",
    )


def test_decoded_extractor_stages_six_cameras_without_mcap(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path / "remote")
    bundle = acquire_fixture_bundle(fixture, tmp_path / "cache")

    owned = make_decoded_extractor(tmp_path).extract_owned(bundle)

    assert {sample.camera_name for sample in owned.result.samples} == set(
        DEFAULT_DECODED_CAMERAS
    )
    assert len(owned.result.samples) == 12
    assert all(
        sample.ego_pose.interpolation.pose_frame == "recording_local_enu_v1"
        for sample in owned.result.samples
    )
    assert all(
        sample.ego_pose.translation_xyz_m
        == sample.ego_pose.interpolation.translation_xyz_m
        for sample in owned.result.samples
    )
    for sample in owned.result.samples:
        verify_staged_image_identity(sample.staged_image)
    assert not any(path.suffix == ".mcap" for path in tmp_path.rglob("*"))


def test_decoded_extractor_reads_production_shards_and_batch_timing(
    tmp_path: Path,
) -> None:
    fixture = write_decoded_fixture(tmp_path / "remote")
    rewrite_as_anonymizer_layout(fixture)
    bundle = acquire_fixture_bundle(fixture, tmp_path / "cache")

    owned = make_decoded_extractor(tmp_path).extract_owned(bundle)

    assert len(owned.result.samples) == 12
    first_batch = [sample for sample in owned.result.samples if sample.batch_timestamp_ns == 0]
    assert len(first_batch) == len(DEFAULT_DECODED_CAMERAS)
    assert {sample.camera_timestamp_ns for sample in first_batch} == {
        index * 10 for index in range(len(DEFAULT_DECODED_CAMERAS))
    }


def test_decoded_extractor_supports_opt_in_tilted_camera(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path / "remote", include_tilted=True)
    cameras: list[CameraName] = [*DEFAULT_DECODED_CAMERAS, "cam_front_tilted"]
    bundle = acquire_fixture_bundle(
        fixture, tmp_path / "cache", cameras=cameras
    )

    owned = make_decoded_extractor(tmp_path).extract_owned(bundle)

    assert {sample.camera_name for sample in owned.result.samples} == set(cameras)
    assert len(owned.result.samples) == 14


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("duplicate_frame", "indices"),
        ("missing_frame", "indices"),
        ("out_of_order", "out of order"),
        ("unknown_camera", "unknown camera"),
        ("dimension", "dimensions"),
    ],
)
def test_decoded_extractor_rejects_invalid_frame_joins(
    tmp_path: Path, case: str, match: str
) -> None:
    fixture = write_decoded_fixture(tmp_path / "remote")
    bundle = acquire_fixture_bundle(fixture, tmp_path / "cache")
    frames = bundle.path(bundle.entry.artifact("camera_frames").path)
    rows = pq.ParquetFile(frames).read().to_pylist()
    if case == "duplicate_frame":
        rows[1]["frame_index"] = 0
    elif case == "missing_frame":
        rows[1]["frame_index"] = 2
    elif case == "out_of_order":
        rows[1]["recording_offset_ns"] = 0
    elif case == "unknown_camera":
        rows[0]["camera_name"] = "cam_unknown"
    else:
        rows[0]["width"] = 5
    pq.write_table(pa.Table.from_pylist(rows), frames)

    with pytest.raises(StructuralExtractionError, match=match):
        make_decoded_extractor(tmp_path).extract_owned(bundle)


def test_decoded_extractor_rejects_frame_without_gnss_bracket(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path / "remote")
    bundle = acquire_fixture_bundle(fixture, tmp_path / "cache")
    gnss = bundle.path(bundle.entry.artifact("gnss").path)
    rows = pq.ParquetFile(gnss).read().to_pylist()
    for row in rows:
        row["recording_offset_ns"] += 1
    pq.write_table(pa.Table.from_pylist(rows), gnss)

    with pytest.raises(StructuralExtractionError, match="GNSS interpolation bracket"):
        make_decoded_extractor(tmp_path).extract_owned(bundle)
    assert not tuple((tmp_path / "work").glob(f"{fixture.recording_id}-*"))


def test_decoded_extractor_reports_rollback_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = write_decoded_fixture(tmp_path / "remote")
    bundle = acquire_fixture_bundle(fixture, tmp_path / "cache")
    bundle.path(bundle.entry.videos[0].path).write_bytes(b"not-video")

    def fail_rollback(*_args: object, **_kwargs: object) -> None:
        raise OSError("fixture rollback failure")

    monkeypatch.setattr(
        decoded_module, "rollback_staging_invocation", fail_rollback
    )
    with pytest.raises(OwnedDirectoryCleanupError):
        make_decoded_extractor(tmp_path).extract_owned(bundle)


def test_decoded_extractor_rejects_frame_count_mismatch(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path / "remote")
    catalog = fixture.root / "data/recordings.parquet"
    rows = pq.ParquetFile(catalog).read().to_pylist()
    rows[0]["frame_counts"]["cam_front"] = 3
    pq.write_table(pa.Table.from_pylist(rows), catalog)
    bundle = acquire_fixture_bundle(fixture, tmp_path / "cache")

    with pytest.raises(StructuralExtractionError, match="frame count"):
        make_decoded_extractor(tmp_path).extract_owned(bundle)


def test_decoded_extractor_rejects_missing_calibration(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path / "remote")
    bundle = acquire_fixture_bundle(fixture, tmp_path / "cache")
    calibration = bundle.path(bundle.entry.artifact("calibration").path)
    rows = pq.ParquetFile(calibration).read().to_pylist()[1:]
    pq.write_table(pa.Table.from_pylist(rows), calibration)

    with pytest.raises(StructuralExtractionError, match="calibration"):
        make_decoded_extractor(tmp_path).extract_owned(bundle)


def test_decoded_extractor_rejects_corrupt_video_and_rolls_back_staging(
    tmp_path: Path,
) -> None:
    fixture = write_decoded_fixture(tmp_path / "remote")
    bundle = acquire_fixture_bundle(fixture, tmp_path / "cache")
    bundle.path(bundle.entry.videos[0].path).write_bytes(b"not-video")

    with pytest.raises(StructuralExtractionError, match="video"):
        make_decoded_extractor(tmp_path).extract_owned(bundle)
    assert not tuple((tmp_path / "work").glob(f"{fixture.recording_id}-*"))
