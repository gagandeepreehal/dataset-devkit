from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataset_devkit.config import DenseAssetSelection, GnssAnnotationSourceConfig
from dataset_devkit.huggingface_acquisition import AcquisitionError, IntegrityError
from dataset_devkit.indus_ad_acquisition import IndusAdHfAcquirer
from dataset_devkit.indus_ad_annotations import gnss_scenario_schema
from indus_ad_fixture import _repo_tree, write_executable_indus_ad_fixture, write_indus_ad_fixture


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("lfs", [False, True])
def test_video_must_match_manifest_even_when_tree_accepts_same_size_bytes(
    tmp_path: Path, cached: bool, lfs: bool
) -> None:
    fixture = write_indus_ad_fixture(tmp_path / "source")
    video = next(item for item in fixture.tree if item.path.endswith(".mp4"))
    path = fixture.root / video.path
    original = path.read_bytes()
    changed = bytes([original[0] ^ 1]) + original[1:]
    path.write_bytes(changed)
    tree = _repo_tree(fixture.root)
    identity = next(item for item in tree if item.path == video.path)
    if not lfs:
        identity = replace(
            identity,
            blob_id=hashlib.sha1(
                f"blob {len(changed)}\0".encode() + changed, usedforsecurity=False
            ).hexdigest(),
            sha256=None,
        )
        tree = tuple(identity if item.path == video.path else item for item in tree)
    calls: list[str] = []
    acquirer = IndusAdHfAcquirer(
        source=fixture.source,
        cache_dir=tmp_path / "cache",
        list_tree=lambda repo_id, revision: tree,
        download_file=_downloader(fixture.root, calls),
    )
    entry = acquirer.load_plan().recordings[0]
    if cached:
        acquirer._acquire_one(identity, "indus-ad-payload")
        calls.clear()
    with pytest.raises(IntegrityError, match="artifact integrity manifest"):
        acquirer.acquire(entry)
    if cached:
        assert video.path not in calls


def _downloader(source_root: Path, calls: list[str]) -> Callable[..., str]:
    def download_file(**kwargs: object) -> str:
        filename = str(kwargs["filename"])
        local_dir = Path(str(kwargs["local_dir"]))
        calls.append(filename)
        source = source_root / filename
        destination = local_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return str(destination)

    return download_file


def test_acquirer_loads_metadata_then_selected_payload_only(tmp_path: Path) -> None:
    fixture = write_indus_ad_fixture(tmp_path / "source")
    calls: list[str] = []
    acquirer = IndusAdHfAcquirer(
        source=fixture.source,
        cache_dir=tmp_path / "cache",
        list_tree=lambda repo_id, revision: fixture.tree,
        download_file=_downloader(fixture.root, calls),
    )

    plan = acquirer.load_plan()

    assert plan.recordings[0].recording_id == fixture.recording_id
    assert not any(path.endswith((".mp4", ".tar")) for path in calls)
    bundle = acquirer.acquire(plan.recordings[0])
    assert bundle.annotations.camera
    assert bundle.materialize_dense_assets(frozenset()) == ()
    assert not any(path.endswith(".tar") for path in calls)
    assert all(video.frame_count == 1 for video in bundle.entry.videos)


def test_acquirer_materializes_only_explicit_selected_frames(tmp_path: Path) -> None:
    fixture = write_indus_ad_fixture(tmp_path / "source")
    source = fixture.source.model_copy(
        update={
            "dense_assets": DenseAssetSelection(semantic="selected_frames", depth="selected_frames")
        }
    )
    calls: list[str] = []
    acquirer = IndusAdHfAcquirer(
        source=source,
        cache_dir=tmp_path / "cache",
        list_tree=lambda repo_id, revision: fixture.tree,
        download_file=_downloader(fixture.root, calls),
    )
    entry = acquirer.load_plan().recordings[0]
    bundle = acquirer.acquire(entry)

    materialized = bundle.materialize_dense_assets(frozenset({"frame-fixture"}))

    assert {item.ref.kind for item in materialized} == {"semantic", "depth"}
    assert sum(path.endswith(".tar") for path in calls) == 2
    assert all(item.path.is_file() for item in materialized)


def test_acquirer_rejects_blob_identity_drift(tmp_path: Path) -> None:
    fixture = write_indus_ad_fixture(tmp_path / "source")
    target = next(item for item in fixture.tree if item.path == "data/metadata.jsonl")
    bad = replace(target, blob_id="f" * 40, sha256=None)
    tree = tuple(bad if item.path == target.path else item for item in fixture.tree)
    acquirer = IndusAdHfAcquirer(
        source=fixture.source,
        cache_dir=tmp_path / "cache",
        list_tree=lambda repo_id, revision: tree,
        download_file=_downloader(fixture.root, []),
    )

    with pytest.raises(IntegrityError, match="blob ID"):
        acquirer.load_plan()


def test_acquirer_rejects_download_path_escape(tmp_path: Path) -> None:
    fixture = write_indus_ad_fixture(tmp_path / "source")

    def escaped(**kwargs: object) -> str:
        del kwargs
        path = tmp_path / "outside"
        path.write_bytes(b"outside")
        return str(path)

    acquirer = IndusAdHfAcquirer(
        source=fixture.source,
        cache_dir=tmp_path / "cache",
        list_tree=lambda repo_id, revision: fixture.tree,
        download_file=escaped,
    )

    with pytest.raises(AcquisitionError, match="outside"):
        acquirer.load_plan()


def test_acquirer_verifies_pinned_gnss_companion(tmp_path: Path) -> None:
    fixture = write_executable_indus_ad_fixture(tmp_path / "source")
    companion_root = tmp_path / "companion"
    (companion_root / "data").mkdir(parents=True)
    path = companion_root / "data/annotations.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "schema_version": "indus_ad_gnss_annotations_v1",
                    "source_revision": fixture.source.revision,
                    "recording_id": fixture.recording_id,
                    "window_index": 0,
                    "window_start_offset_ns": 0,
                    "window_end_offset_ns": 1_000_000_000,
                    "motion_label": "moving",
                    "validity_tags": ["valid"],
                    "quality_tags": ["good"],
                    "anomaly_candidate": False,
                    "anomaly_reasons": [],
                    "confidence": 0.9,
                    "run_id": "test-run",
                }
            ],
            schema=gnss_scenario_schema(),
        ),
        path,
    )
    companion = GnssAnnotationSourceConfig(
        repo_id="owner/gnss",
        revision="7" * 40,
        contract="indus_ad_gnss_annotations_v1",
        expected_source_revision=fixture.source.revision,
    )
    source = fixture.source.model_copy(
        update={
            "gnss_annotation_source": companion,
            "annotation_layers": [*fixture.source.annotation_layers, "gnss_scenarios"],
        }
    )
    calls: list[str] = []

    def download(**kwargs: object) -> str:
        root = companion_root if kwargs["repo_id"] == companion.repo_id else fixture.root
        return _downloader(root, calls)(**kwargs)

    acquirer = IndusAdHfAcquirer(
        source=source,
        cache_dir=tmp_path / "cache",
        list_tree=lambda repo_id, revision: fixture.tree,
        companion_list_tree=lambda repo_id, revision: _repo_tree(companion_root),
        download_file=download,
    )
    bundle = acquirer.acquire(acquirer.load_plan().recordings[0])
    assert bundle.annotations.gnss[0].window_index == 0
    assert bundle.annotations.availability["gnss"].state == "available"
