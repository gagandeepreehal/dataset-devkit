from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from conftest import FeatureFactory
from dataset_devkit.config import GlobalConfig, SplitConfig
from dataset_devkit.export import ExportEvidence, export_dataset
from dataset_devkit.provenance import SourceFingerprint
from dataset_devkit.scenes import build_recording_scenes
from dataset_devkit.split import split_selected_scenes
from dataset_devkit.validation import validate_dataset
from test_scenes import _config, _report
from test_split import _scenarios, _selection


def _export_stratified_dataset(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> Path:
    base_config = config_factory()
    scene_config = _config(base_config, max_duration_s=0.001)
    source = SourceFingerprint(
        "https://example.blob.core.windows.net",
        "recordings",
        "mcap-h265/strata.mcap",
        '"strata"',
        8,
    )
    timestamps = tuple(
        timestamp
        for index in range(8)
        for timestamp in (index * 2_000_000_000, index * 2_000_000_000 + 1_000_000)
    )
    graph = build_recording_scenes(
        _report(tmp_path / "input", timestamps), source, scene_config
    )
    assert len(graph.scenes) == 8
    selection = _selection(
        (graph,),
        feature_factory,
        ("alpha", "alpha", "alpha", "alpha", "beta", "beta", "beta", "beta"),
    )
    scenarios = _scenarios(selection)
    split_config = SplitConfig(test_fraction=0.5, seed=31, stratify=True)
    split = split_selected_scenes(
        selection,
        selection.selected_scenes,
        scenarios,
        (graph,),
        split_config,
    )
    resolved = scene_config.model_copy(
        update={"scenarios": scenarios, "split": split_config}
    )
    evidence = ExportEvidence(
        selection.selected_scenes,
        scenarios,
        selection,
        (graph,),
        split_config,
        split,
        resolved,
        {"schema_version": 1},
    )
    root = tmp_path / "dataset"
    export_dataset(root, evidence)
    return root


def _rewrite_leakage_for_assignments(root: Path, split: dict[str, object]) -> None:
    pipeline = json.loads((root / "mz_extensions/pipeline_audit.json").read_text())
    assignments = split["assignments"]
    assert isinstance(assignments, list)
    split_by_scene = {item["scene_token"]: item["split"] for item in assignments}
    by_source: dict[str, list[dict[str, object]]] = {}
    for row in pipeline["graph_scene_sequence"]:
        by_source.setdefault(row["source_digest"], []).append(row)
    pairs: list[dict[str, object]] = []
    for source in sorted(by_source):
        rows = by_source[source]
        for earlier, later in zip(rows, rows[1:], strict=False):
            earlier_split = split_by_scene[earlier["scene_token"]]
            later_split = split_by_scene[later["scene_token"]]
            if earlier_split != later_split:
                pairs.append(
                    {
                        "source_digest": source,
                        "source_blob_path": earlier["source_blob_path"],
                        "earlier_scene_token": earlier["scene_token"],
                        "later_scene_token": later["scene_token"],
                        "earlier_last_timestamp_ns": earlier["last_timestamp_ns"],
                        "later_first_timestamp_ns": later["first_timestamp_ns"],
                        "earlier_split": earlier_split,
                        "later_split": later_split,
                    }
                )
    leakage = split["adjacent_scene_leakage"]
    assert isinstance(leakage, dict)
    leakage["cross_split_pairs"] = pairs


def _write_split(root: Path, split: dict[str, object]) -> None:
    (root / "mz_extensions/split.json").write_text(
        json.dumps(split, sort_keys=True, separators=(",", ":")) + "\n"
    )


def test_validator_rejects_same_stratum_membership_swap_with_unchanged_count(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    root = _export_stratified_dataset(tmp_path, config_factory, feature_factory)
    path = root / "mz_extensions/split.json"
    split = json.loads(path.read_text())
    alpha = [
        item for item in split["assignments"] if item["primary_scenario"] == "alpha"
    ]
    test = next(item for item in alpha if item["split"] == "test")
    train = next(item for item in alpha if item["split"] == "train")
    test["split"], train["split"] = train["split"], test["split"]
    _rewrite_leakage_for_assignments(root, split)
    _write_split(root, split)

    report = validate_dataset(root, official_smoke=False, verify_manifest=False)

    assert report.succeeded is False
    assert any(item.code == "split_integrity" for item in report.findings)


def test_validator_rejects_cross_stratum_redistribution_with_updated_audits(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    root = _export_stratified_dataset(tmp_path, config_factory, feature_factory)
    path = root / "mz_extensions/split.json"
    split = json.loads(path.read_text())
    assignments = split["assignments"]
    alpha_test = next(
        item
        for item in assignments
        if item["primary_scenario"] == "alpha" and item["split"] == "test"
    )
    beta_train = next(
        item
        for item in assignments
        if item["primary_scenario"] == "beta" and item["split"] == "train"
    )
    alpha_test["split"] = "train"
    beta_train["split"] = "test"
    for audit in split["strata"]:
        tests = sum(
            item["split"] == "test"
            for item in assignments
            if item["primary_scenario"] == audit["primary_scenario"]
        )
        audit["target_test_count"] = tests
        audit["actual_test_count"] = tests
        audit["actual_train_count"] = audit["population_count"] - tests
        audit["stratification_applied"] = True
        audit["fallback_reason"] = None
    _rewrite_leakage_for_assignments(root, split)
    _write_split(root, split)

    report = validate_dataset(root, official_smoke=False, verify_manifest=False)

    assert report.succeeded is False
    assert any(item.code == "split_integrity" for item in report.findings)
