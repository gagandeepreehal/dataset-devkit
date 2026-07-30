"""Deterministic, auditable individual-scene train/test splitting."""

from __future__ import annotations

import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Literal
from uuid import UUID

from dataset_devkit.config import ScenariosConfig, SplitConfig
from dataset_devkit.features import SceneFeatures
from dataset_devkit.provenance import canonical_hash, canonical_json
from dataset_devkit.scenario_selection import (
    ScenarioAssignment,
    ScenarioSelectionResult,
    validate_scenario_selection,
)
from dataset_devkit.scene_models import RecordingSceneResult, SceneRecord
from dataset_devkit.scenes import validate_scene_graph

SplitName = Literal["train", "test"]
FallbackReason = Literal[
    "stratification_disabled",
    "stratum_too_small",
    "global_target_prevents_both_sides",
]

_LEAKAGE_WARNING = (
    "Scene-level splitting can leak neighboring temporal context when chronologically "
    "adjacent scenes from one recording are assigned to different splits."
)


@dataclass(frozen=True)
class SceneSplitAssignment:
    scene_token: str
    source_digest: str
    primary_scenario: str
    split: SplitName
    rank: str


@dataclass(frozen=True)
class StratumSplitAudit:
    primary_scenario: str
    population_count: int
    expected_test_count: str
    target_test_count: int
    actual_test_count: int
    actual_train_count: int
    stratification_applied: bool
    fallback_reason: FallbackReason | None


@dataclass(frozen=True)
class AdjacentSceneLeakagePair:
    source_digest: str
    source_blob_path: str
    earlier_scene_token: str
    later_scene_token: str
    earlier_last_timestamp_ns: int
    later_first_timestamp_ns: int
    earlier_split: SplitName
    later_split: SplitName


@dataclass(frozen=True)
class AdjacentSceneLeakageAudit:
    checked: bool
    warning: str
    cross_split_pairs: tuple[AdjacentSceneLeakagePair, ...]


@dataclass(frozen=True)
class SceneSplitResult:
    assignments: tuple[SceneSplitAssignment, ...]
    strata: tuple[StratumSplitAudit, ...]
    adjacent_scene_leakage: AdjacentSceneLeakageAudit
    seed: int
    test_fraction: float
    stratify: bool
    population_count: int
    train_count: int
    test_count: int
    rounding_rule: Literal["floor(n * test_fraction + 0.5)"]
    config_fingerprint: str
    upstream_fingerprint: str
    candidate_fingerprint: str
    graph_fingerprint: str


@dataclass(frozen=True)
class _Candidate:
    identity: tuple[str, str]
    scenario: str
    feature: SceneFeatures
    assignment: ScenarioAssignment
    graph: RecordingSceneResult
    scene: SceneRecord


def _jsonable(value: object) -> object:
    if isinstance(value, (Path, UUID)):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _rank(seed: int, scenario: str, identity: tuple[str, str]) -> str:
    return canonical_hash(
        {
            "seed": seed,
            "primary_scenario": scenario,
            "scene_token": identity[0],
            "source_digest": identity[1],
        }
    )


def _half_up_target(count: int, fraction: Decimal) -> int:
    return int((Decimal(count) * fraction + Decimal("0.5")).to_integral_value(rounding=ROUND_FLOOR))


def _validate_inputs(
    selection: ScenarioSelectionResult,
    features_population: Sequence[SceneFeatures],
    scenarios_config: ScenariosConfig,
    graphs: Sequence[RecordingSceneResult],
) -> tuple[_Candidate, ...]:
    validate_scenario_selection(selection, list(features_population), scenarios_config)
    features: dict[tuple[str, str], SceneFeatures] = {}
    for feature in selection.selected_scenes:
        identity = (feature.scene_token, feature.source.digest)
        if identity in features:
            raise ValueError("selected features contain duplicate scene/source identities")
        features[identity] = feature
    assignments: dict[tuple[str, str], ScenarioAssignment] = {}
    for assignment in selection.assignments:
        identity = (assignment.scene_token, assignment.source_digest)
        if identity in assignments:
            raise ValueError("scenario assignments contain duplicate scene/source identities")
        assignments[identity] = assignment
    if set(features) != set(assignments):
        raise ValueError("selected feature and scenario assignment identities differ")

    graph_by_source: dict[str, RecordingSceneResult] = {}
    scene_by_identity: dict[tuple[str, str], SceneRecord] = {}
    for graph in graphs:
        validate_scene_graph(graph)
        digest = graph.source.digest
        if digest in graph_by_source:
            raise ValueError("recording graphs contain duplicate source identities")
        graph_by_source[digest] = graph
        for scene in graph.scenes:
            identity = (scene.token, digest)
            if identity in scene_by_identity:
                raise ValueError("recording graphs contain duplicate scene/source identities")
            scene_by_identity[identity] = scene
    selected_sources = {identity[1] for identity in features}
    if set(graph_by_source) != selected_sources:
        raise ValueError("recording graph sources are missing or foreign to selected scenes")

    candidates: list[_Candidate] = []
    for identity in sorted(features):
        feature = features[identity]
        assignment = assignments[identity]
        graph = graph_by_source[identity[1]]
        if feature.source != graph.source or feature.source_blob_path != graph.source.blob_path:
            raise ValueError("selected feature source evidence differs from its recording graph")
        selected_scene = scene_by_identity.get(identity)
        if selected_scene is None:
            raise ValueError("selected scene is missing from its recording graph")
        if feature.scene_name != selected_scene.name:
            raise ValueError("selected feature scene identity differs from its recording graph")
        candidates.append(
            _Candidate(
                identity,
                assignment.primary_scenario,
                feature,
                assignment,
                graph,
                selected_scene,
            )
        )
    return tuple(candidates)


def _upstream_fingerprint(
    selection: ScenarioSelectionResult,
    features_population: Sequence[SceneFeatures],
    scenarios_config: ScenariosConfig,
) -> str:
    """Bind validated Task 6 evidence without serializing its full records again.

    ``validate_scenario_selection`` has already recomputed the candidate, rules, and
    configuration fingerprints and all derived assignments/audits. Reusing those
    cryptographic commitments keeps Task 7 memory bounded instead of deep-copying the
    feature population and selection audit into another complete JSON document.
    """
    return canonical_hash(
        {
            "scenarios_config_fingerprint": canonical_hash(
                scenarios_config.model_dump(mode="json")
            ),
            "task6_config_fingerprint": selection.config_fingerprint,
            "task6_rules_fingerprint": selection.rules_fingerprint,
            "task6_candidate_fingerprint": selection.candidate_fingerprint,
            "task6_seed": selection.seed,
            "task6_strict_quotas": selection.strict_quotas,
            "feature_population_count": len(features_population),
            "selected_count": len(selection.assignments),
            "unselected_count": len(selection.unselected),
            "rule_count": len(selection.rule_audits),
        }
    )


def _graph_fingerprint(graphs: Sequence[RecordingSceneResult]) -> str:
    volatile_staging_keys = frozenset(
        {
            "path",
            "device",
            "inode",
            "invocation_root",
            "root_relative_path",
            "directory_device",
            "directory_inode",
            "directory_chain_identities",
        }
    )

    def stable_value(value: object, *, staged_image: bool = False) -> object:
        if isinstance(value, dict):
            return {
                key: stable_value(item, staged_image=key == "staged_image")
                for key, item in value.items()
                if not staged_image or key not in volatile_staging_keys
            }
        if isinstance(value, list):
            return [stable_value(item, staged_image=staged_image) for item in value]
        return value

    return canonical_hash(
        [
            stable_value(graph.to_dict())
            for graph in sorted(graphs, key=lambda item: item.source.digest)
        ]
    )


def _apportion_strata(
    groups: Mapping[str, tuple[_Candidate, ...]], fraction: Decimal, target: int, seed: int
) -> dict[str, int]:
    quotas: dict[str, int] = {}
    remainders: list[tuple[Decimal, str, str]] = []
    for scenario in sorted(groups):
        ideal = Decimal(len(groups[scenario])) * fraction
        floor = int(ideal.to_integral_value(rounding=ROUND_FLOOR))
        quotas[scenario] = floor
        tie = canonical_hash({"seed": seed, "primary_scenario": scenario, "purpose": "apportion"})
        remainders.append((ideal - Decimal(floor), tie, scenario))
    for _, _, scenario in sorted(remainders, key=lambda item: (-item[0], item[1], item[2]))[
        : target - sum(quotas.values())
    ]:
        quotas[scenario] += 1

    # Preserve both sides for as many non-singleton strata as the exact global target permits.
    for scenario in sorted(groups):
        count = len(groups[scenario])
        if count >= 2 and quotas[scenario] == 0:
            donors = [
                name
                for name in groups
                if quotas[name] > (1 if len(groups[name]) >= 2 else 0)
            ]
            if donors:
                donor = min(
                    donors,
                    key=lambda name: canonical_hash(
                        {"seed": seed, "from": name, "to": scenario, "purpose": "rebalance"}
                    ),
                )
                quotas[donor] -= 1
                quotas[scenario] += 1
    for scenario in sorted(groups):
        count = len(groups[scenario])
        if count >= 2 and quotas[scenario] == count:
            recipients = [
                name
                for name in groups
                if quotas[name] < (len(groups[name]) - 1 if len(groups[name]) >= 2 else 1)
            ]
            if recipients:
                recipient = min(
                    recipients,
                    key=lambda name: canonical_hash(
                        {"seed": seed, "from": scenario, "to": name, "purpose": "rebalance"}
                    ),
                )
                quotas[scenario] -= 1
                quotas[recipient] += 1
    return quotas


def _compute_split(
    selection: ScenarioSelectionResult,
    features_population: Sequence[SceneFeatures],
    scenarios_config: ScenariosConfig,
    graphs: Sequence[RecordingSceneResult],
    config: SplitConfig,
) -> SceneSplitResult:
    candidates = _validate_inputs(
        selection, features_population, scenarios_config, graphs
    )
    fraction = Decimal(str(config.test_fraction))
    target = _half_up_target(len(candidates), fraction)
    groups_lists: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        groups_lists[candidate.scenario].append(candidate)
    groups = {
        scenario: tuple(sorted(items, key=lambda item: item.identity))
        for scenario, items in groups_lists.items()
    }

    test_identities: set[tuple[str, str]] = set()
    quotas: dict[str, int] = {}
    if config.stratify:
        quotas = _apportion_strata(groups, fraction, target, config.seed)
        for scenario, items in groups.items():
            ranked = sorted(
                items,
                key=lambda item: (
                    _rank(config.seed, scenario, item.identity),
                    item.identity,
                ),
            )
            test_identities.update(item.identity for item in ranked[: quotas[scenario]])
    else:
        ranked = sorted(
            candidates,
            key=lambda item: (_rank(config.seed, "__all__", item.identity), item.identity),
        )
        test_identities.update(item.identity for item in ranked[:target])

    split_assignments = tuple(
        SceneSplitAssignment(
            candidate.identity[0],
            candidate.identity[1],
            candidate.scenario,
            "test" if candidate.identity in test_identities else "train",
            _rank(
                config.seed,
                candidate.scenario if config.stratify else "__all__",
                candidate.identity,
            ),
        )
        for candidate in candidates
    )
    audits: list[StratumSplitAudit] = []
    for scenario in sorted(groups):
        count = len(groups[scenario])
        actual = sum(item.identity in test_identities for item in groups[scenario])
        fallback: FallbackReason | None
        applied = config.stratify and count >= 2 and 0 < actual < count
        if not config.stratify:
            fallback = "stratification_disabled"
        elif count < 2:
            fallback = "stratum_too_small"
        elif not applied:
            fallback = "global_target_prevents_both_sides"
        else:
            fallback = None
        audits.append(
            StratumSplitAudit(
                scenario,
                count,
                str(Decimal(count) * fraction),
                quotas.get(scenario, actual),
                actual,
                count - actual,
                applied,
                fallback,
            )
        )

    split_by_identity = {
        (item.scene_token, item.source_digest): item.split for item in split_assignments
    }
    pairs: list[AdjacentSceneLeakagePair] = []
    selected = {item.identity for item in candidates}
    for graph in sorted(graphs, key=lambda item: item.source.digest):
        ordered = sorted(
            graph.scenes,
            key=lambda scene: (scene.first_timestamp_ns, scene.last_timestamp_ns, scene.ordinal),
        )
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            if (
                (earlier.token, graph.source.digest) in selected
                and (later.token, graph.source.digest) in selected
                and split_by_identity[(earlier.token, graph.source.digest)]
                != split_by_identity[(later.token, graph.source.digest)]
            ):
                pairs.append(
                    AdjacentSceneLeakagePair(
                        graph.source.digest,
                        graph.source.blob_path,
                        earlier.token,
                        later.token,
                        earlier.last_timestamp_ns,
                        later.first_timestamp_ns,
                        split_by_identity[(earlier.token, graph.source.digest)],
                        split_by_identity[(later.token, graph.source.digest)],
                    )
                )

    return SceneSplitResult(
        split_assignments,
        tuple(audits),
        AdjacentSceneLeakageAudit(True, _LEAKAGE_WARNING, tuple(pairs)),
        config.seed,
        config.test_fraction,
        config.stratify,
        len(candidates),
        len(candidates) - target,
        target,
        "floor(n * test_fraction + 0.5)",
        canonical_hash(config.model_dump(mode="json")),
        _upstream_fingerprint(selection, features_population, scenarios_config),
        selection.candidate_fingerprint,
        _graph_fingerprint(graphs),
    )


def split_selected_scenes(
    selection: ScenarioSelectionResult,
    features_population: Sequence[SceneFeatures],
    scenarios_config: ScenariosConfig,
    graphs: Sequence[RecordingSceneResult],
    config: SplitConfig,
) -> SceneSplitResult:
    """Split each selected scene atomically using deterministic SHA-256 ranks.

    The exact global test count uses ``floor(n * test_fraction + 0.5)``. With
    stratification enabled, largest-remainder apportionment assigns that exact count
    across primary-scenario strata and then preserves both sides where the global
    target permits. Singleton and globally constrained strata are explicitly audited.
    """
    result = _compute_split(
        selection, features_population, scenarios_config, graphs, config
    )
    validate_scene_split(
        result, selection, features_population, scenarios_config, graphs, config
    )
    return result


def validate_scene_split(
    result: SceneSplitResult,
    selection: ScenarioSelectionResult,
    features_population: Sequence[SceneFeatures],
    scenarios_config: ScenariosConfig,
    graphs: Sequence[RecordingSceneResult],
    config: SplitConfig,
) -> None:
    """Recompute the complete split and reject stale, replayed, or mutated evidence."""
    expected = _compute_split(
        selection, features_population, scenarios_config, graphs, config
    )
    if result != expected:
        raise ValueError("scene split differs from recomputed deterministic result")
    identities = [(item.scene_token, item.source_digest) for item in result.assignments]
    if len(identities) != len(set(identities)) or len(identities) != result.population_count:
        raise ValueError("scene split is not a complete unique assignment")
    train = {
        identity
        for identity, item in zip(identities, result.assignments, strict=True)
        if item.split == "train"
    }
    test = {
        identity
        for identity, item in zip(identities, result.assignments, strict=True)
        if item.split == "test"
    }
    if train & test or len(train | test) != result.population_count:
        raise ValueError("train/test assignments are not disjoint and complete")


def split_extension_payload(result: SceneSplitResult) -> dict[str, object]:
    """Return the canonical, wall-clock-free ``mz_extensions/split.json`` value."""
    value = _jsonable(result)
    assert isinstance(value, dict)
    return {"schema_version": 1, **value}


def write_split_extension(
    path: Path,
    result: SceneSplitResult,
    selection: ScenarioSelectionResult,
    features_population: Sequence[SceneFeatures],
    scenarios_config: ScenariosConfig,
    graphs: Sequence[RecordingSceneResult],
    config: SplitConfig,
) -> None:
    """Atomically write deterministic canonical split extension bytes."""
    validate_scene_split(
        result, selection, features_population, scenarios_config, graphs, config
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(split_extension_payload(result)))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
