from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from conftest import FeatureFactory
from dataset_devkit.config import (
    GlobalConfig,
    ScenarioRuleConfig,
    ScenariosConfig,
    SplitConfig,
)
from dataset_devkit.export import ExportEvidence, export_dataset
from dataset_devkit.provenance import SourceFingerprint
from dataset_devkit.scenario_selection import select_scenarios
from dataset_devkit.scenes import build_recording_scenes
from dataset_devkit.split import split_selected_scenes
from dataset_devkit.validation import validate_dataset
from test_export_dataset import _evidence
from test_scenes import _config, _report


def _nonadjacent_selection_evidence(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> ExportEvidence:
    source = SourceFingerprint(
        "https://example.blob.core.windows.net",
        "recordings",
        "mcap-h265/nonadjacent.mcap",
        '"nonadjacent"',
        8,
    )
    timestamps = tuple(
        timestamp
        for index in range(4)
        for timestamp in (index * 2_000_000_000, index * 2_000_000_000 + 1_000_000)
    )
    scene_config = _config(config_factory(), max_duration_s=0.001)
    validity_report = _report(tmp_path / "input", timestamps)
    graph = build_recording_scenes(validity_report, source, scene_config)
    assert len(graph.scenes) == 4
    population = tuple(
        replace(
            feature_factory(
                scene_token=scene.token,
                scene_name=scene.name,
                source=source,
                source_blob_path=source.blob_path,
            ),
            computed_tags=("selected",) if index in {0, 2} else ("other",),
        )
        for index, scene in enumerate(graph.scenes)
    )
    scenarios = ScenariosConfig(
        seed=5,
        strict_quotas=True,
        rules=[
            ScenarioRuleConfig(
                name="selected",
                quota=2,
                required_all_tags=["selected"],
            )
        ],
    )
    selection = select_scenarios(population, scenarios)
    split_config = SplitConfig(test_fraction=0.5, seed=1, stratify=False)
    split = split_selected_scenes(
        selection, population, scenarios, (graph,), split_config
    )
    assert {item.split for item in split.assignments} == {"train", "test"}
    assert split.adjacent_scene_leakage.cross_split_pairs == ()
    resolved = scene_config.model_copy(
        update={"scenarios": scenarios, "split": split_config}
    )
    return ExportEvidence(
        population,
        scenarios,
        selection,
        (graph,),
        split_config,
        split,
        resolved,
        {"schema_version": 1},
        validity_reports=((source, validity_report),),
    )


def test_validator_recomputes_adjacency_from_complete_task5_graph_sequence(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    root = tmp_path / "dataset"
    evidence = _nonadjacent_selection_evidence(
        tmp_path, config_factory, feature_factory
    )
    export_dataset(root, evidence)

    report = validate_dataset(root, official_smoke=False, verify_manifest=False)

    assert report.succeeded is True
    audit = json.loads((root / "mz_extensions/pipeline_audit.json").read_text())
    sequence = audit["graph_scene_sequence"]
    assert [item["scene_token"] for item in sequence] == [
        scene.token for scene in evidence.graphs[0].scenes
    ]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda audit: audit["selection"]["assignments"].pop(),
        lambda audit: audit["filter"]["accepted"].pop(),
        lambda audit: audit["selection"]["unselected"].append(
            dict(audit["selection"]["assignments"][0])
        ),
        lambda audit: audit["graph_scene_sequence"].pop(),
        lambda audit: audit["graph_scene_sequence"][0].pop("scene_token"),
    ],
)
def test_validator_rejects_pipeline_membership_or_graph_sequence_mutation(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
    mutate: Callable[[dict[str, object]], object],
) -> None:
    root = tmp_path / "dataset"
    export_dataset(root, _evidence(tmp_path, config_factory, feature_factory))
    path = root / "mz_extensions/pipeline_audit.json"
    audit = json.loads(path.read_text())
    mutate(audit)
    path.write_text(json.dumps(audit), encoding="utf-8")

    report = validate_dataset(root, official_smoke=False, verify_manifest=False)

    assert report.succeeded is False
    assert any(item.code == "pipeline_audit" for item in report.findings)
