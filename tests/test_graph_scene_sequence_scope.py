from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

from conftest import FeatureFactory
from dataset_devkit.config import (
    GlobalConfig,
    ScenarioRuleConfig,
    ScenariosConfig,
    SplitConfig,
)
from dataset_devkit.export import (
    ExportEvidence,
    export_dataset,
    pipeline_graph_scene_sequence,
)
from dataset_devkit.provenance import SourceFingerprint
from dataset_devkit.scenario_selection import select_scenarios
from dataset_devkit.scene_models import RecordingSceneResult
from dataset_devkit.scenes import build_recording_scenes
from dataset_devkit.split import split_selected_scenes
from dataset_devkit.validation import validate_dataset
from dataset_devkit.validity import ValidityReport
from test_scenes import _config, _report


def _graph(
    tmp_path: Path,
    config: GlobalConfig,
    name: str,
    scene_count: int,
) -> tuple[RecordingSceneResult, ValidityReport]:
    timestamps = tuple(
        timestamp
        for index in range(scene_count)
        for timestamp in (index * 2_000_000_000, index * 2_000_000_000 + 1_000_000)
    )
    source = SourceFingerprint(
        "https://example.blob.core.windows.net",
        "recordings",
        f"mcap-h265/{name}.mcap",
        f'"{name}"',
        8,
    )
    report = _report(tmp_path / name, timestamps)
    return (
        build_recording_scenes(
            report,
            source,
            _config(config, max_duration_s=0.001),
        ),
        report,
    )


def _evidence(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> tuple[ExportEvidence, tuple[RecordingSceneResult, ...]]:
    config = config_factory()
    selected_graph, selected_validity = _graph(tmp_path, config, "selected", 3)
    unselected_graph, _ = _graph(tmp_path, config, "all-unselected", 1)
    population = tuple(
        replace(
            feature_factory(
                scene_token=scene.token,
                scene_name=scene.name,
                source=graph.source,
                source_blob_path=graph.source.blob_path,
            ),
            computed_tags=("selected",)
            if graph is selected_graph and index in {0, 2}
            else ("other",),
        )
        for graph in (selected_graph, unselected_graph)
        for index, scene in enumerate(graph.scenes)
    )
    scenarios = ScenariosConfig(
        seed=5,
        strict_quotas=True,
        rules=[
            ScenarioRuleConfig(
                name="selected", quota=2, required_all_tags=["selected"]
            )
        ],
    )
    selection = select_scenarios(population, scenarios)
    split_config = SplitConfig(test_fraction=0.5, seed=1, stratify=False)
    split = split_selected_scenes(
        selection,
        population,
        scenarios,
        (selected_graph,),
        split_config,
    )
    resolved = _config(config, max_duration_s=0.001).model_copy(
        update={"scenarios": scenarios, "split": split_config}
    )
    audit = {
        "schema_version": 1,
        "filter": {
            "accepted": [
                {
                    "scene_token": item.scene_token,
                    "source_digest": item.source.digest,
                }
                for item in population
            ],
            "rejected": [],
        },
        "selection": {
            "candidate_fingerprint": selection.candidate_fingerprint,
            "config_fingerprint": selection.config_fingerprint,
            "rules_fingerprint": selection.rules_fingerprint,
            "assignments": [asdict(item) for item in selection.assignments],
            "rule_audits": [asdict(item) for item in selection.rule_audits],
            "unselected": [asdict(item) for item in selection.unselected],
        },
        "graph_scene_sequence": pipeline_graph_scene_sequence(
            (selected_graph, unselected_graph), selection
        ),
    }
    evidence = ExportEvidence(
        population,
        scenarios,
        selection,
        (selected_graph,),
        split_config,
        split,
        resolved,
        {"schema_version": 1},
        audit,
        ((selected_graph.source, selected_validity),),
    )
    return evidence, (selected_graph, unselected_graph)


def test_graph_sequence_keeps_intervening_scenes_only_for_selected_sources(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    evidence, graphs = _evidence(tmp_path, config_factory, feature_factory)

    sequence = pipeline_graph_scene_sequence(graphs, evidence.selection)

    assert [item["scene_token"] for item in sequence] == [
        scene.token for scene in evidence.graphs[0].scenes
    ]


def test_validator_accepts_all_unselected_source_outside_graph_sequence(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    evidence, _ = _evidence(tmp_path, config_factory, feature_factory)
    root = tmp_path / "dataset"
    export_dataset(root, evidence)

    report = validate_dataset(root, official_smoke=False, verify_manifest=False)

    assert report.succeeded is True
