from __future__ import annotations

from dataclasses import asdict, replace

import pytest

import dataset_devkit.filtering as filtering_module
import dataset_devkit.scenario_selection as selection_module
from conftest import FeatureFactory
from dataset_devkit.config import FiltersConfig, ScenarioRuleConfig, ScenariosConfig
from dataset_devkit.features import ChannelCoverage
from dataset_devkit.filtering import filter_scenes
from dataset_devkit.provenance import canonical_hash
from dataset_devkit.scenario_selection import (
    ScenarioQuotaError,
    UnselectedSceneAudit,
    select_scenarios,
    validate_scenario_selection,
)


def test_filter_evaluates_all_criteria_without_short_circuiting(
    feature_factory: FeatureFactory,
) -> None:
    feature = feature_factory(
        scene_token="scene-a",
        duration_s=1.0,
        total_distance_m=0.0,
        scene_valid_ratio=0.5,
        computed_tags=("stationary",),
    )
    config = FiltersConfig(
        min_duration_s=2.0,
        min_distance_m=1.0,
        min_scene_valid_ratio=1.0,
        required_all_tags=["moving"],
    )

    result = filter_scenes((feature,), config)

    assert result.accepted == ()
    assert [reason.code for reason in result.rejected[0].reasons] == [
        "duration_below_minimum",
        "scene_valid_ratio_below_minimum",
        "distance_below_minimum",
        "required_all_tags_missing",
    ]


def test_empty_filter_accepts_all_in_input_order(feature_factory: FeatureFactory) -> None:
    first = feature_factory(scene_token="one")
    second = feature_factory(scene_token="two")
    assert filter_scenes((first, second), FiltersConfig()).accepted == (first, second)


def test_filter_predicate_sets_compile_once_for_many_features(
    feature_factory: FeatureFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    features = tuple(feature_factory(scene_token=f"scene-{index}") for index in range(250))
    calls = 0
    original = filtering_module._compile_filter

    def counted_compile(config: FiltersConfig):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return original(config)

    monkeypatch.setattr(filtering_module, "_compile_filter", counted_compile)
    result = filter_scenes(features, FiltersConfig(required_all_tags=["moving"]))

    assert result.accepted == features
    assert calls == 1


def test_selection_is_input_order_independent_and_first_rule_claims_overlap(
    feature_factory: FeatureFactory,
) -> None:
    a = feature_factory(scene_token="a", computed_tags=("moving", "straight"))
    b = feature_factory(scene_token="b", computed_tags=("moving", "straight"))
    config = ScenariosConfig(
        seed=99,
        rules=[
            ScenarioRuleConfig(name="Straight", quota=1, required_all_tags=["straight"]),
            ScenarioRuleConfig(name="Moving", quota=1, required_all_tags=["moving"]),
        ],
    )

    first = select_scenarios((a, b), config)
    second = select_scenarios((b, a), config)

    assert first.assignments == second.assignments
    assert len({item.scene_token for item in first.assignments}) == 2
    assert [item.primary_scenario for item in first.assignments] == ["Straight", "Moving"]


def test_candidate_fingerprint_streams_each_many_candidate_record_once(
    feature_factory: FeatureFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    features = tuple(feature_factory(scene_token=f"scene-{index:05d}") for index in range(500))
    visits = 0
    original = selection_module._update_hash_with_feature

    def counted_update(hasher: object, feature: object) -> None:
        nonlocal visits
        visits += 1
        original(hasher, feature)  # type: ignore[arg-type]

    monkeypatch.setattr(selection_module, "_update_hash_with_feature", counted_update)
    first = selection_module._candidate_fingerprint(features)
    second = selection_module._candidate_fingerprint(tuple(reversed(features)))

    assert first == second
    assert first == canonical_hash([asdict(feature) for feature in features])
    assert visits == 2 * len(features)


def test_optional_rule_filter_compiles_once_for_entire_selection_call(
    feature_factory: FeatureFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    features = tuple(
        feature_factory(scene_token=f"scene-{index:03d}", computed_tags=("moving",))
        for index in range(100)
    )
    config = ScenariosConfig(
        seed=8,
        rules=[
            ScenarioRuleConfig(
                name="Moving Filtered",
                quota=1,
                required_all_tags=["moving"],
                filters=FiltersConfig(min_distance_m=0.0),
            )
        ],
    )
    calls = 0
    original = filtering_module._compile_filter

    def counted_compile(filters: FiltersConfig):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return original(filters)

    monkeypatch.setattr(filtering_module, "_compile_filter", counted_compile)
    result = select_scenarios(features, config)

    assert len(result.selected_scenes) == 1
    assert calls == 1


def test_strict_exact_quota_deficit_has_deterministic_partial_audit(
    feature_factory: FeatureFactory,
) -> None:
    feature = feature_factory(scene_token="a", computed_tags=("straight",))
    config = ScenariosConfig(
        seed=1,
        rules=[ScenarioRuleConfig(name="Straight", quota=2, required_all_tags=["straight"])],
    )

    with pytest.raises(ScenarioQuotaError) as caught:
        select_scenarios((feature,), config)

    assert caught.value.rule_name == "Straight"
    assert caught.value.eligible == 1
    assert caught.value.selected == 1
    assert caught.value.deficit == 1


def test_non_strict_deficit_and_quota_zero_are_audited(
    feature_factory: FeatureFactory,
) -> None:
    feature = feature_factory(scene_token="a", computed_tags=("straight",))
    config = ScenariosConfig(
        seed=1,
        strict_quotas=False,
        rules=[
            ScenarioRuleConfig(name="Zero", quota=0, required_all_tags=["straight"]),
            ScenarioRuleConfig(name="Short", quota=2, required_all_tags=["straight"]),
        ],
    )

    result = select_scenarios((feature,), config)

    assert result.rule_audits[0].selected == 0
    assert result.rule_audits[0].deficit == 0
    assert result.rule_audits[1].selected == 1
    assert result.rule_audits[1].deficit == 1


def test_selection_validator_rejects_mutation_and_fingerprint_changes(
    feature_factory: FeatureFactory,
) -> None:
    feature = feature_factory(scene_token="a", computed_tags=("straight",))
    config = ScenariosConfig(
        seed=1,
        rules=[ScenarioRuleConfig(name="Straight", quota=1, required_all_tags=["straight"])],
    )
    result = select_scenarios((feature,), config)

    with pytest.raises(ValueError, match="partition|duplicate|order"):
        validate_scenario_selection(
            replace(result, assignments=(result.assignments[0], result.assignments[0])),
            (feature,),
            config,
        )
    changed = config.model_copy(update={"seed": 2})
    with pytest.raises(ValueError, match="fingerprint|seed"):
        validate_scenario_selection(result, (feature,), changed)
    changed_feature = replace(feature, total_distance_m=feature.total_distance_m + 1.0)
    with pytest.raises(ValueError, match="candidate|fingerprint"):
        validate_scenario_selection(result, (changed_feature,), config)


def test_unselected_no_match_and_quota_filled_reasons(
    feature_factory: FeatureFactory,
) -> None:
    matching = feature_factory(scene_token="matching", computed_tags=("straight",))
    no_match = feature_factory(scene_token="none", computed_tags=("stationary",))
    config = ScenariosConfig(
        seed=4,
        rules=[ScenarioRuleConfig(name="Straight", quota=0, required_all_tags=["straight"])],
    )

    result = select_scenarios((matching, no_match), config)

    assert {item.scene_token: item.reason for item in result.unselected} == {
        "matching": "quota_filled",
        "none": "no_rule_match",
    }


@pytest.mark.parametrize(
    ("field", "config_field", "value", "rejected_value"),
    [
        ("duration_s", "min_duration_s", 2.0, 1.999),
        ("duration_s", "max_duration_s", 2.0, 2.001),
        ("scene_valid_ratio", "min_scene_valid_ratio", 0.8, 0.799),
        ("scene_valid_ratio", "max_scene_valid_ratio", 0.8, 0.801),
        ("source_gnss_valid_ratio", "min_source_gnss_valid_ratio", 0.8, 0.799),
        ("source_gnss_valid_ratio", "max_source_gnss_valid_ratio", 0.8, 0.801),
        ("camera_coverage_ratio", "min_camera_coverage_ratio", 0.8, 0.799),
        ("camera_coverage_ratio", "max_camera_coverage_ratio", 0.8, 0.801),
        ("max_abs_sync_error_ms", "max_sync_error_ms", 5.0, 5.001),
        ("total_distance_m", "min_distance_m", 3.0, 2.999),
        ("total_distance_m", "max_distance_m", 3.0, 3.001),
    ],
)
def test_filter_numeric_boundaries_are_inclusive_and_exceeding_rejects(
    feature_factory: FeatureFactory,
    field: str,
    config_field: str,
    value: float,
    rejected_value: float,
) -> None:
    boundary = feature_factory(**{field: value})
    config = FiltersConfig.model_validate({config_field: value})
    rejected = feature_factory(scene_token="rejected", **{field: rejected_value})
    assert filter_scenes((boundary,), config).accepted == (boundary,)
    assert filter_scenes((rejected,), config).rejected


def test_per_channel_coverage_boundaries(feature_factory: FeatureFactory) -> None:
    feature = feature_factory(
        camera_coverage_by_channel=(ChannelCoverage("front", 8, 10, 0.8),)
    )
    boundary = FiltersConfig(
        min_camera_coverage_by_channel={"front": 0.8},
        max_camera_coverage_by_channel={"front": 0.8},
    )
    assert filter_scenes((feature,), boundary).accepted == (feature,)
    assert filter_scenes(
        (feature,), FiltersConfig(min_camera_coverage_by_channel={"rear": 0.1})
    ).rejected[0].reasons[0].code == "channel_coverage_below_minimum"


def test_tag_and_human_label_predicates_remain_separate(
    feature_factory: FeatureFactory,
) -> None:
    feature = feature_factory(
        computed_tags=("moving", "straight"),
        human_labels=("weather/rain", "quality/good"),
    )
    accepted = FiltersConfig(
        required_any_tags=["turn", "moving"],
        required_all_tags=["straight"],
        excluded_tags=["stationary"],
        required_any_labels=["weather/rain", "weather/snow"],
        required_all_labels=["quality/good"],
        excluded_labels=["quality/bad"],
    )
    assert filter_scenes((feature,), accepted).accepted == (feature,)
    rejected = FiltersConfig(
        required_all_tags=["weather/rain"],
        required_all_labels=["moving"],
    )
    codes = {item.code for item in filter_scenes((feature,), rejected).rejected[0].reasons}
    assert codes == {"required_all_tags_missing", "required_all_labels_missing"}


def test_exact_blacklists_do_not_use_prefix_matching(feature_factory: FeatureFactory) -> None:
    feature = feature_factory(scene_token="scene-a")
    near = FiltersConfig(
        blacklisted_scene_tokens=["scene"],
        blacklisted_source_digests=[feature.source.digest[:-1]],
        blacklisted_blob_paths=[feature.source_blob_path + ".other"],
    )
    assert filter_scenes((feature,), near).accepted == (feature,)
    exact = FiltersConfig(
        blacklisted_scene_tokens=[feature.scene_token],
        blacklisted_source_digests=[feature.source.digest],
        blacklisted_blob_paths=[feature.source_blob_path],
    )
    assert {item.code for item in filter_scenes((feature,), exact).rejected[0].reasons} == {
        "scene_token_blacklisted",
        "source_digest_blacklisted",
        "blob_path_blacklisted",
    }


def test_selection_repeat_shuffle_prior_rule_audit_and_duplicate_prevention(
    feature_factory: FeatureFactory,
) -> None:
    features = tuple(
        feature_factory(scene_token=token, computed_tags=("moving", "straight"))
        for token in ("a", "b", "c")
    )
    config = ScenariosConfig(
        seed=15,
        rules=[
            ScenarioRuleConfig(name="First", quota=1, required_all_tags=["straight"]),
            ScenarioRuleConfig(name="Second", quota=1, required_all_tags=["moving"]),
        ],
    )
    first = select_scenarios(features, config)
    assert select_scenarios(tuple(reversed(features)), config) == first
    assert select_scenarios((features[1], features[2], features[0]), config) == first
    assert any(item.excluded_by_prior_rule for item in first.rule_audits[1].candidates)
    assert "claimed_by_prior_rule" in {
        item.reason for item in first.rule_audits[1].candidates
    }
    with pytest.raises(ValueError, match="duplicate"):
        select_scenarios((features[0], features[0]), config)


def test_validator_rejects_coherent_lower_rank_selection_forgery(
    feature_factory: FeatureFactory,
) -> None:
    features = tuple(
        feature_factory(scene_token=token, computed_tags=("straight",))
        for token in ("a", "b", "c")
    )
    config = ScenariosConfig(
        seed=18,
        rules=[ScenarioRuleConfig(name="Rule", quota=1, required_all_tags=["straight"])],
    )
    result = select_scenarios(features, config)
    audit = result.rule_audits[0]
    best = next(item for item in audit.candidates if item.selected)
    lower = next(item for item in audit.candidates if not item.selected)
    candidates = tuple(
        replace(item, selected=False, reason="quota_filled")
        if item.scene_token == best.scene_token
        else replace(item, selected=True, reason="selected")
        if item.scene_token == lower.scene_token
        else item
        for item in audit.candidates
    )
    lower_feature = next(item for item in features if item.scene_token == lower.scene_token)
    forged_assignment = replace(
        result.assignments[0],
        scene_token=lower.scene_token,
        source_digest=lower.source_digest,
        rank=lower.rank,
    )
    retained_unselected = tuple(
        item for item in result.unselected if item.scene_token != lower.scene_token
    )
    forged = replace(
        result,
        assignments=(forged_assignment,),
        selected_scenes=(lower_feature,),
        rule_audits=(replace(audit, candidates=candidates),),
        unselected=(
            *retained_unselected,
            UnselectedSceneAudit(best.scene_token, best.source_digest, "quota_filled", ("Rule",)),
        ),
    )
    with pytest.raises(ValueError, match="rank|order|audit"):
        validate_scenario_selection(forged, features, config)


def test_validator_rejects_candidate_reordering_rank_and_prior_claim_forgery(
    feature_factory: FeatureFactory,
) -> None:
    features = tuple(
        feature_factory(scene_token=token, computed_tags=("moving", "straight"))
        for token in ("a", "b", "c")
    )
    config = ScenariosConfig(
        seed=6,
        rules=[
            ScenarioRuleConfig(name="First", quota=1, required_all_tags=["straight"]),
            ScenarioRuleConfig(name="Second", quota=1, required_all_tags=["moving"]),
        ],
    )
    result = select_scenarios(features, config)
    first = result.rule_audits[0]
    reordered = replace(first, candidates=tuple(reversed(first.candidates)))
    with pytest.raises(ValueError):
        validate_scenario_selection(
            replace(result, rule_audits=(reordered, result.rule_audits[1])),
            features,
            config,
        )
    altered = replace(first.candidates[0], rank="0" * 64)
    altered_audit = replace(first, candidates=(altered, *first.candidates[1:]))
    with pytest.raises(ValueError, match="rank"):
        validate_scenario_selection(
            replace(result, rule_audits=(altered_audit, result.rule_audits[1])),
            features,
            config,
        )
    second = result.rule_audits[1]
    claimed_index = next(
        index
        for index, item in enumerate(second.candidates)
        if item.excluded_by_prior_rule
    )
    changed_candidates = list(second.candidates)
    changed_candidates[claimed_index] = replace(
        changed_candidates[claimed_index],
        excluded_by_prior_rule=False,
        reason="quota_filled",
    )
    changed_second = replace(
        second,
        candidates=tuple(changed_candidates),
        eligible_unassigned=second.eligible_unassigned + 1,
    )
    with pytest.raises(ValueError, match="prior|audit"):
        validate_scenario_selection(
            replace(result, rule_audits=(first, changed_second)),
            features,
            config,
        )


def test_validator_rejects_forged_strict_quota_deficit(
    feature_factory: FeatureFactory,
) -> None:
    feature = feature_factory(scene_token="a", computed_tags=("straight",))
    loose = ScenariosConfig(
        seed=5,
        strict_quotas=False,
        rules=[ScenarioRuleConfig(name="Rule", quota=2, required_all_tags=["straight"])],
    )
    result = select_scenarios((feature,), loose)
    strict = loose.model_copy(update={"strict_quotas": True})
    forged = replace(
        result,
        config_fingerprint=canonical_hash(strict.model_dump(mode="json")),
        strict_quotas=True,
    )
    with pytest.raises(ValueError, match="strict|deficit"):
        validate_scenario_selection(forged, (feature,), strict)
