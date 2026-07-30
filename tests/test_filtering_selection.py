from __future__ import annotations

import pytest

from conftest import FeatureFactory
from dataset_devkit.config import FiltersConfig, ScenarioRuleConfig, ScenariosConfig
from dataset_devkit.filtering import filter_scenes
from dataset_devkit.scenario_selection import ScenarioQuotaError, select_scenarios


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
