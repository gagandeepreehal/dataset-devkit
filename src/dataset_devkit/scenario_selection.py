"""Ordered exact-quota deterministic scenario selection."""

from __future__ import annotations

from dataclasses import dataclass

from dataset_devkit.config import ScenarioRuleConfig, ScenariosConfig
from dataset_devkit.features import SceneFeatures
from dataset_devkit.filtering import filter_scenes
from dataset_devkit.provenance import canonical_hash


@dataclass(frozen=True)
class RankedCandidate:
    scene_token: str
    source_digest: str
    rank: str
    excluded_by_prior_rule: bool
    selected: bool


@dataclass(frozen=True)
class RuleAudit:
    rule_name: str
    rule_index: int
    quota: int
    matched_candidates: int
    eligible_unassigned: int
    selected: int
    deficit: int
    candidates: tuple[RankedCandidate, ...]


@dataclass(frozen=True)
class ScenarioAssignment:
    scene_token: str
    source_digest: str
    primary_scenario: str
    rule_index: int
    rank: str


@dataclass(frozen=True)
class UnselectedSceneAudit:
    scene_token: str
    source_digest: str
    reason: str
    matching_rules: tuple[str, ...]


@dataclass(frozen=True)
class ScenarioSelectionResult:
    assignments: tuple[ScenarioAssignment, ...]
    selected_scenes: tuple[SceneFeatures, ...]
    rule_audits: tuple[RuleAudit, ...]
    unselected: tuple[UnselectedSceneAudit, ...]
    seed: int
    config_fingerprint: str


class ScenarioQuotaError(ValueError):
    """Raised when strict exact quota cannot be filled without duplicate assignment."""

    def __init__(
        self,
        rule_name: str,
        eligible: int,
        selected: int,
        deficit: int,
        partial_rule_audits: tuple[RuleAudit, ...],
    ) -> None:
        super().__init__(
            f"scenario rule {rule_name!r} quota deficit: eligible={eligible}, "
            f"selected={selected}, deficit={deficit}"
        )
        self.rule_name = rule_name
        self.eligible = eligible
        self.selected = selected
        self.deficit = deficit
        self.partial_rule_audits = partial_rule_audits


def _matches(feature: SceneFeatures, rule: ScenarioRuleConfig) -> bool:
    tags = set(feature.computed_tags)
    labels = set(feature.human_labels)
    for kind, present in (("tags", tags), ("labels", labels)):
        required_any = set(getattr(rule, f"required_any_{kind}"))
        required_all = set(getattr(rule, f"required_all_{kind}"))
        excluded = set(getattr(rule, f"excluded_{kind}"))
        if required_any and not present & required_any:
            return False
        if not required_all <= present or excluded & present:
            return False
    return rule.filters is None or bool(filter_scenes((feature,), rule.filters).accepted)


def _identity(feature: SceneFeatures) -> tuple[str, str]:
    return feature.scene_token, feature.source.digest


def _rank(
    seed: int, rule_index: int, rule_name: str, feature: SceneFeatures
) -> str:
    return canonical_hash(
        {
            "seed": seed,
            "rule_index": rule_index,
            "rule_name": rule_name,
            "scene_token": feature.scene_token,
            "source": feature.source.to_dict(),
        }
    )


def select_scenarios(
    features: tuple[SceneFeatures, ...] | list[SceneFeatures], config: ScenariosConfig
) -> ScenarioSelectionResult:
    """Select exact quotas by SHA-256 rank, independent of input order and random state."""
    by_identity = {_identity(feature): feature for feature in features}
    if len(by_identity) != len(features):
        raise ValueError("scenario input contains duplicate scene/source identities")
    ordered_features = tuple(by_identity[key] for key in sorted(by_identity))
    assigned: set[tuple[str, str]] = set()
    assignments: list[ScenarioAssignment] = []
    selected_features: list[SceneFeatures] = []
    audits: list[RuleAudit] = []
    matching_names: dict[tuple[str, str], list[str]] = {key: [] for key in by_identity}
    for rule_index, rule in enumerate(config.rules):
        matched = tuple(feature for feature in ordered_features if _matches(feature, rule))
        for feature in matched:
            matching_names[_identity(feature)].append(rule.name)
        all_ranked = tuple(
            sorted(
                (
                    (_rank(config.seed, rule_index, rule.name, feature), feature)
                    for feature in matched
                ),
                key=lambda item: (item[0], *_identity(item[1])),
            )
        )
        ranked = tuple(item for item in all_ranked if _identity(item[1]) not in assigned)
        chosen = ranked[: rule.quota]
        chosen_keys = {_identity(feature) for _, feature in chosen}
        deficit = max(0, rule.quota - len(chosen))
        rank_by_key = {_identity(feature): rank for rank, feature in all_ranked}
        candidates = tuple(
            RankedCandidate(
                feature.scene_token,
                feature.source.digest,
                rank_by_key[_identity(feature)],
                _identity(feature) in assigned,
                _identity(feature) in chosen_keys,
            )
            for _, feature in all_ranked
        )
        audit = RuleAudit(
            rule.name,
            rule_index,
            rule.quota,
            len(matched),
            len(ranked),
            len(chosen),
            deficit,
            candidates,
        )
        audits.append(audit)
        if deficit and config.strict_quotas:
            raise ScenarioQuotaError(
                rule.name, len(ranked), len(chosen), deficit, tuple(audits)
            )
        for rank, feature in chosen:
            key = _identity(feature)
            assigned.add(key)
            assignments.append(
                ScenarioAssignment(
                    feature.scene_token,
                    feature.source.digest,
                    rule.name,
                    rule_index,
                    rank,
                )
            )
            selected_features.append(feature)
    unselected = tuple(
        UnselectedSceneAudit(
            feature.scene_token,
            feature.source.digest,
            "no_rule_match" if not matching_names[_identity(feature)] else "quota_filled",
            tuple(matching_names[_identity(feature)]),
        )
        for feature in ordered_features
        if _identity(feature) not in assigned
    )
    return ScenarioSelectionResult(
        tuple(assignments),
        tuple(selected_features),
        tuple(audits),
        unselected,
        config.seed,
        canonical_hash(config.model_dump(mode="json")),
    )
