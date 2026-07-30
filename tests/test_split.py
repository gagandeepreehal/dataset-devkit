from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

import dataset_devkit.split as split_module
from conftest import FeatureFactory
from dataset_devkit.config import (
    GlobalConfig,
    ScenarioRuleConfig,
    ScenariosConfig,
    SplitConfig,
)
from dataset_devkit.features import SceneFeatures
from dataset_devkit.provenance import SourceFingerprint
from dataset_devkit.scenario_selection import ScenarioSelectionResult, select_scenarios
from dataset_devkit.scene_models import RecordingSceneResult
from dataset_devkit.scenes import build_recording_scenes
from dataset_devkit.split import (
    SceneSplitResult,
    split_extension_payload,
    split_selected_scenes,
    validate_scene_split,
    write_split_extension,
)
from test_scenes import _config, _report


def _graph(
    tmp_path: Path, config: GlobalConfig, name: str, count: int = 4
) -> RecordingSceneResult:
    source = SourceFingerprint(
        "https://example.blob.core.windows.net",
        "recordings",
        f"mcap-h265/{name}.mcap",
        f'"{name}"',
        count,
    )
    timestamps = tuple(
        timestamp
        for index in range(count)
        for timestamp in (index * 2_000_000_000, index * 2_000_000_000 + 1)
    )
    graph = build_recording_scenes(
        _report(tmp_path / name, timestamps),
        source,
        _config(config, max_duration_s=0.000000001),
    )
    assert len(graph.scenes) == count
    return graph


def _selection(
    graphs: Sequence[RecordingSceneResult],
    factory: FeatureFactory,
    scenarios: tuple[str, ...] | None = None,
) -> ScenarioSelectionResult:
    pairs = [(graph, scene) for graph in graphs for scene in graph.scenes]
    names = scenarios or tuple("road" for _ in pairs)
    features = tuple(
        factory(
            scene_token=scene.token,
            scene_name=scene.name,
            source=graph.source,
            source_blob_path=graph.source.blob_path,
        )
        for graph, scene in pairs
    )
    features = tuple(
        replace(feature, computed_tags=(scenario,))
        for feature, scenario in zip(features, names, strict=True)
    )
    config = ScenariosConfig(
        seed=5,
        rules=[
            ScenarioRuleConfig(
                name=scenario,
                quota=names.count(scenario),
                required_all_tags=[scenario],
            )
            for scenario in dict.fromkeys(names)
        ],
    )
    return select_scenarios(features, config)


def _scenarios(selection: ScenarioSelectionResult) -> ScenariosConfig:
    names = tuple(dict.fromkeys(item.primary_scenario for item in selection.assignments))
    return ScenariosConfig(
        seed=selection.seed,
        rules=[
            ScenarioRuleConfig(
                name=name,
                quota=sum(item.primary_scenario == name for item in selection.assignments),
                required_all_tags=[name],
            )
            for name in names
        ],
    )


def _split(
    selection: ScenarioSelectionResult,
    graphs: Sequence[RecordingSceneResult],
    config: SplitConfig,
    *,
    features_population: Sequence[SceneFeatures] | None = None,
    scenarios_config: ScenariosConfig | None = None,
) -> SceneSplitResult:
    return split_selected_scenes(
        selection,
        selection.selected_scenes if features_population is None else features_population,
        scenarios_config or _scenarios(selection),
        graphs,
        config,
    )


def test_half_up_target_is_exact_and_input_order_independent(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    graph = _graph(tmp_path, config_factory(), "a", 5)
    selection = _selection((graph,), feature_factory)
    config = SplitConfig(test_fraction=0.5, seed=17, stratify=False)

    first = _split(selection, (graph,), config)
    scenarios = _scenarios(selection)
    reordered_selection = select_scenarios(
        tuple(reversed(selection.selected_scenes)), scenarios
    )
    second = _split(reordered_selection, (graph,), config)

    assert first.test_count == 3
    assert first.train_count == 2
    assert {(item.scene_token, item.split) for item in first.assignments} == {
        (item.scene_token, item.split) for item in second.assignments
    }


@pytest.mark.parametrize(
    ("count", "fraction", "expected"), [(1, 0.01, 0), (1, 0.99, 1), (3, 0.01, 0), (3, 0.99, 3)]
)
def test_fraction_boundary_and_single_scene_semantics(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
    count: int,
    fraction: float,
    expected: int,
) -> None:
    graph = _graph(tmp_path, config_factory(), f"edge-{count}-{fraction}", count)
    result = _split(
        _selection((graph,), feature_factory),
        (graph,),
        SplitConfig(test_fraction=fraction, seed=1, stratify=True),
    )
    assert result.test_count == expected
    assert len(result.assignments) == count


def test_stratification_and_small_stratum_fallback_are_audited(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    graph = _graph(tmp_path, config_factory(), "strata", 5)
    selection = _selection(
        (graph,), feature_factory, ("common", "common", "common", "common", "singleton")
    )
    result = _split(
        selection, (graph,), SplitConfig(test_fraction=0.4, seed=9, stratify=True)
    )

    audits = {item.primary_scenario: item for item in result.strata}
    assert result.test_count == 2
    assert 0 < audits["common"].actual_test_count < 4
    assert audits["common"].fallback_reason is None
    assert audits["singleton"].fallback_reason == "stratum_too_small"


def test_global_target_constraint_on_strata_is_explicit(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    graph = _graph(tmp_path, config_factory(), "limited", 4)
    selection = _selection((graph,), feature_factory, ("a", "a", "b", "b"))
    result = _split(
        selection, (graph,), SplitConfig(test_fraction=0.01, seed=3, stratify=True)
    )
    assert result.test_count == 0
    assert {audit.fallback_reason for audit in result.strata} == {
        "global_target_prevents_both_sides"
    }


def test_full_disjoint_assignment_chain_ownership_and_recomputed_validation(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    graph = _graph(tmp_path, config_factory(), "chains", 4)
    selection = _selection((graph,), feature_factory)
    config = SplitConfig(test_fraction=0.5, seed=4, stratify=False)
    result = _split(selection, (graph,), config)

    validate_scene_split(
        result,
        selection,
        selection.selected_scenes,
        _scenarios(selection),
        (graph,),
        config,
    )
    by_scene = {item.scene_token: item.split for item in result.assignments}
    assert set(by_scene) == {item.token for item in graph.scenes}
    assert all(sample.scene_token in by_scene for sample in graph.samples)
    assert all(item.scene_token in by_scene for item in graph.sample_data)
    with pytest.raises(ValueError):
        validate_scene_split(
            replace(result, test_count=result.test_count + 1),
            selection,
            selection.selected_scenes,
            _scenarios(selection),
            (graph,),
            config,
        )


@pytest.mark.parametrize("mutation", ["missing", "foreign", "duplicate", "feature", "graph"])
def test_rejects_missing_foreign_duplicate_and_mutated_evidence(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
    mutation: str,
) -> None:
    first = _graph(tmp_path, config_factory(), "first", 2)
    second = _graph(tmp_path, config_factory(), "second", 2)
    selection = _selection((first,), feature_factory)
    graphs: tuple[RecordingSceneResult, ...] = (first,)
    if mutation == "missing":
        graphs = ()
    elif mutation == "foreign":
        graphs = (first, second)
    elif mutation == "duplicate":
        graphs = (first, first)
    elif mutation == "feature":
        selection = replace(
            selection,
            selected_scenes=(
                replace(selection.selected_scenes[0], source=second.source),
                *selection.selected_scenes[1:],
            ),
        )
    elif mutation == "graph":
        first = replace(first, source=replace(first.source, etag='"changed"'))
        graphs = (first,)
    with pytest.raises(ValueError):
        _split(selection, graphs, SplitConfig(test_fraction=0.5, seed=1, stratify=False))


def test_validation_rejects_config_selection_and_graph_replay_drift(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    first = _graph(tmp_path, config_factory(), "replay-a", 2)
    second = _graph(tmp_path, config_factory(), "replay-b", 2)
    selection = _selection((first,), feature_factory)
    config = SplitConfig(test_fraction=0.5, seed=11, stratify=True)
    result = _split(selection, (first,), config)

    with pytest.raises(ValueError, match="recomputed"):
        validate_scene_split(
            result,
            selection,
            selection.selected_scenes,
            _scenarios(selection),
            (first,),
            SplitConfig(test_fraction=0.5, seed=12, stratify=True),
        )
    changed_selection = replace(
        selection,
        assignments=(
            replace(selection.assignments[0], primary_scenario="changed"),
            *selection.assignments[1:],
        ),
    )
    with pytest.raises(ValueError):
        validate_scene_split(
            result,
            changed_selection,
            selection.selected_scenes,
            _scenarios(selection),
            (first,),
            config,
        )
    with pytest.raises(ValueError):
        validate_scene_split(
            result,
            selection,
            selection.selected_scenes,
            _scenarios(selection),
            (second,),
            config,
        )


def test_empty_selected_population_is_truthfully_complete() -> None:
    scenarios = ScenariosConfig(seed=1, rules=[])
    selection = select_scenarios((), scenarios)
    config = SplitConfig(test_fraction=0.5, seed=1, stratify=True)
    result = _split(selection, (), config)
    assert result.assignments == ()
    assert result.strata == ()
    assert (result.population_count, result.train_count, result.test_count) == (0, 0, 0)
    assert result.adjacent_scene_leakage.cross_split_pairs == ()


@pytest.mark.parametrize(
    "mutation",
    ["primary_scenario", "rule_index", "rank", "rule_audit", "config", "rules", "candidate"],
)
def test_rejects_any_mutated_task6_selection_evidence(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
    mutation: str,
) -> None:
    graph = _graph(tmp_path, config_factory(), "upstream", 3)
    selection = _selection((graph,), feature_factory)
    assignment = selection.assignments[0]
    if mutation == "primary_scenario":
        changed = replace(assignment, primary_scenario="other")
        selection = replace(selection, assignments=(changed, *selection.assignments[1:]))
    elif mutation == "rule_index":
        changed = replace(assignment, rule_index=9)
        selection = replace(selection, assignments=(changed, *selection.assignments[1:]))
    elif mutation == "rank":
        changed = replace(assignment, rank="0" * 64)
        selection = replace(selection, assignments=(changed, *selection.assignments[1:]))
    elif mutation == "rule_audit":
        audit = replace(selection.rule_audits[0], quota=99)
        selection = replace(selection, rule_audits=(audit, *selection.rule_audits[1:]))
    elif mutation == "config":
        selection = replace(selection, config_fingerprint="0" * 64)
    elif mutation == "rules":
        selection = replace(selection, rules_fingerprint="0" * 64)
    elif mutation == "candidate":
        selection = replace(selection, candidate_fingerprint="0" * 64)

    with pytest.raises(ValueError):
        _split(selection, (graph,), SplitConfig(test_fraction=0.5, seed=3, stratify=True))


def test_split_fingerprint_reuses_validated_task6_compact_fingerprints(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph(tmp_path, config_factory(), "compact", 3)
    selection = _selection((graph,), feature_factory)
    calls = 0
    original = split_module._jsonable

    def counted_jsonable(value: object) -> object:
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(split_module, "_jsonable", counted_jsonable)
    result = _split(
        selection, (graph,), SplitConfig(test_fraction=0.5, seed=3, stratify=True)
    )

    assert calls == 0
    assert result.upstream_fingerprint
    assert result.candidate_fingerprint


def test_adjacent_cross_split_leakage_and_no_leakage_are_truthful(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    graph = _graph(tmp_path, config_factory(), "adjacent", 4)
    selection = _selection((graph,), feature_factory)
    leaking = _split(
        selection, (graph,), SplitConfig(test_fraction=0.5, seed=1, stratify=False)
    )
    none = _split(
        selection, (graph,), SplitConfig(test_fraction=0.01, seed=1, stratify=False)
    )
    assert leaking.adjacent_scene_leakage.checked is True
    assert leaking.adjacent_scene_leakage.warning
    assert leaking.adjacent_scene_leakage.cross_split_pairs
    assert none.adjacent_scene_leakage.checked is True
    assert none.adjacent_scene_leakage.cross_split_pairs == ()


def test_canonical_extension_payload_and_writer_are_byte_deterministic(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    graph = _graph(tmp_path, config_factory(), "writer", 3)
    selection = _selection((graph,), feature_factory)
    result = _split(
        selection,
        (graph,),
        SplitConfig(test_fraction=0.5, seed=7, stratify=True),
    )
    split_config = SplitConfig(test_fraction=0.5, seed=7, stratify=True)
    scenarios = _scenarios(selection)
    path = tmp_path / "mz_extensions" / "split.json"
    write_split_extension(
        path,
        result,
        selection,
        selection.selected_scenes,
        scenarios,
        (graph,),
        split_config,
    )
    first = path.read_bytes()
    write_split_extension(
        path,
        result,
        selection,
        selection.selected_scenes,
        scenarios,
        (graph,),
        split_config,
    )
    assert path.read_bytes() == first
    assert json.loads(first) == split_extension_payload(result)
    assert first.endswith(b"\n")
    assert b"created_at" not in first
    with pytest.raises(ValueError, match="recomputed"):
        write_split_extension(
            path,
            replace(result, train_count=result.train_count + 1),
            selection,
            selection.selected_scenes,
            scenarios,
            (graph,),
            split_config,
        )
    assert path.read_bytes() == first
