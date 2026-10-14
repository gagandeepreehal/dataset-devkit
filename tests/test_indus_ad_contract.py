from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataset_devkit.indus_ad_contract import (
    VIDEO_PRIVACY_PATH,
    VIDEO_VIEWER_PATH,
    IndusAdContractError,
    parse_indus_ad_contract,
)
from dataset_devkit.indus_ad_models import RepositoryFile
from indus_ad_fixture import refreshed_tree, write_indus_ad_fixture


def test_contract_requires_repaired_privacy_controls_to_agree(tmp_path: Path) -> None:
    fixture = write_indus_ad_fixture(tmp_path)

    plan = parse_indus_ad_contract(fixture.root, fixture.source, fixture.tree)

    assert plan.viewer_privacy_counts == {"face_plate_blurred": 6, "missing": 1}
    assert plan.recordings[0].recording_id == fixture.recording_id
    assert all(video.kind == "video" for video in plan.recordings[0].videos)
    assert {video.camera for video in plan.recordings[0].videos} == set(
        fixture.source.cameras
    )


def test_contract_rejects_one_nonnull_viewer_cell_not_marked_blurred(
    tmp_path: Path,
) -> None:
    fixture = write_indus_ad_fixture(tmp_path)
    path = fixture.root / VIDEO_VIEWER_PATH
    table = pq.ParquetFile(path).read()
    rows = table.to_pylist()
    rows[0]["cam_front_video_privacy_status"] = "not_face_plate_blurred"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    privacy_path = fixture.root / VIDEO_PRIVACY_PATH
    privacy = json.loads(privacy_path.read_text(encoding="utf-8"))
    binding = next(
        item for item in privacy["files"] if item["path"] == VIDEO_VIEWER_PATH
    )
    binding["bytes"] = path.stat().st_size
    binding["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    privacy_path.write_text(json.dumps(privacy, sort_keys=True), encoding="utf-8")
    fixture = refreshed_tree(fixture)

    with pytest.raises(IndusAdContractError, match="privacy status"):
        parse_indus_ad_contract(fixture.root, fixture.source, fixture.tree)


def test_contract_rejects_privacy_manifest_digest_drift(tmp_path: Path) -> None:
    fixture = write_indus_ad_fixture(tmp_path)
    path = fixture.root / VIDEO_PRIVACY_PATH
    value = json.loads(path.read_text(encoding="utf-8"))
    value["files"][0]["sha256"] = "f" * 64
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    fixture = refreshed_tree(fixture)

    with pytest.raises(IndusAdContractError, match="digest differs"):
        parse_indus_ad_contract(fixture.root, fixture.source, fixture.tree)


def test_contract_rejects_tree_size_drift(tmp_path: Path) -> None:
    fixture = write_indus_ad_fixture(tmp_path)
    target = fixture.tree[0]
    bad = replace(target, size=target.size + 1)
    tree = tuple(bad if item.path == target.path else item for item in fixture.tree)

    with pytest.raises(IndusAdContractError, match="tree size"):
        parse_indus_ad_contract(fixture.root, fixture.source, tree)


def test_contract_rejects_duplicate_and_unknown_tree_paths(tmp_path: Path) -> None:
    fixture = write_indus_ad_fixture(tmp_path)
    with pytest.raises(IndusAdContractError, match="duplicate path"):
        parse_indus_ad_contract(
            fixture.root,
            fixture.source,
            (*fixture.tree, fixture.tree[0]),
        )

    unknown = RepositoryFile("data/unknown/control.json", 1, "a" * 40)
    with pytest.raises(IndusAdContractError, match="unknown data path"):
        parse_indus_ad_contract(
            fixture.root,
            fixture.source,
            (*fixture.tree, unknown),
        )


def test_contract_rejects_unmatched_recording_selection(tmp_path: Path) -> None:
    fixture = write_indus_ad_fixture(tmp_path)
    source = fixture.source.model_copy(update={"recording_ids": ["missing-recording"]})

    with pytest.raises(IndusAdContractError, match="matched no recordings"):
        parse_indus_ad_contract(fixture.root, source, fixture.tree)
