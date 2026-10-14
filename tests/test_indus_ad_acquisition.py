from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from dataset_devkit.config import DenseAssetSelection
from dataset_devkit.huggingface_acquisition import AcquisitionError, IntegrityError
from dataset_devkit.indus_ad_acquisition import IndusAdHfAcquirer
from indus_ad_fixture import write_indus_ad_fixture


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
            "dense_assets": DenseAssetSelection(
                semantic="selected_frames", depth="selected_frames"
            )
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
