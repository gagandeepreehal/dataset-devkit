import json
from pathlib import Path

from conftest import FeatureFactory
from dataset_devkit.config import ScenariosConfig
from dataset_devkit.scenario_selection import select_scenarios
from dataset_devkit.scenario_templates import canonical_scenario_rules


def test_canonical_templates_validate_with_exact_public_names() -> None:
    rules = canonical_scenario_rules(quota=0, annotation_label="weather/rain")

    assert [rule.name for rule in rules] == [
        "Straight",
        "Stopping",
        "Left Turn",
        "Right Turn",
        "Left Curvature",
        "Right Curvature",
        "Annotation Category",
    ]
    assert rules[-1].required_all_labels == ["weather/rain"]


def test_json_templates_validate() -> None:
    path = Path(__file__).parents[1] / "examples" / "scenario_templates.json"
    config = ScenariosConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
    assert len(config.rules) == 7


def test_templates_match_computed_tags_and_annotation_labels_separately(
    feature_factory: FeatureFactory,
) -> None:
    features = (
        feature_factory(scene_token="straight", computed_tags=("moving", "straight")),
        feature_factory(scene_token="stopping", computed_tags=("stopping",)),
        feature_factory(scene_token="left-turn", computed_tags=("left_turn",)),
        feature_factory(scene_token="right-turn", computed_tags=("right_turn",)),
        feature_factory(scene_token="left-curve", computed_tags=("left_curvature",)),
        feature_factory(scene_token="right-curve", computed_tags=("right_curvature",)),
        feature_factory(
            scene_token="label",
            computed_tags=(),
            human_labels=("weather/rain",),
        ),
    )
    config = ScenariosConfig(
        seed=1,
        rules=list(canonical_scenario_rules(quota=1, annotation_label="weather/rain")),
    )

    result = select_scenarios(features, config)

    assert tuple(item.primary_scenario for item in result.assignments) == tuple(
        rule.name for rule in config.rules
    )
