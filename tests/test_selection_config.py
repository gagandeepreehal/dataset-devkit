from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from dataset_devkit.config import GlobalConfig, load_config
from test_config import minimal_config, write_config


def _task6_config() -> dict[str, object]:
    data = deepcopy(minimal_config())
    data["tags"] = {
        "reference_camera_channel": "front",
        "reference_camera_policy": "require",
        "stationary_speed_mps": 0.2,
        "minimum_movement_m": 0.1,
        "straight_max_heading_change_deg": 5.0,
        "curvature_min_heading_change_deg": 10.0,
        "turn_min_heading_change_deg": 45.0,
    }
    data["filters"] = {}
    data["scenarios"] = {"seed": 42, "strict_quotas": True, "rules": []}
    return data


def test_task6_configuration_is_strict_and_typed(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, _task6_config()))

    assert config.tags.reference_camera_channel == "front"
    assert config.scenarios.strict_quotas is True
    assert config.filters.required_all_tags == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("straight_max_heading_change_deg", 10.0),
        ("curvature_min_heading_change_deg", 5.0),
        ("turn_min_heading_change_deg", 10.0),
    ],
)
def test_tag_heading_thresholds_are_strictly_ordered(field: str, value: float) -> None:
    data = _task6_config()
    data["tags"][field] = value  # type: ignore[index]

    with pytest.raises(ValidationError, match="straight.*curvature.*turn"):
        GlobalConfig.model_validate_json(json.dumps(data))


def test_filter_ranges_and_contradictory_predicates_are_rejected() -> None:
    data = _task6_config()
    data["filters"] = {
        "min_duration_s": 2.0,
        "max_duration_s": 1.0,
        "required_all_tags": ["moving"],
        "excluded_tags": ["moving"],
    }

    with pytest.raises(ValidationError):
        GlobalConfig.model_validate_json(json.dumps(data))


def test_scenario_rules_have_exact_quota_and_no_contradictions() -> None:
    data = _task6_config()
    data["scenarios"] = {
        "seed": 7,
        "strict_quotas": True,
        "rules": [
            {
                "name": "Straight",
                "quota": 4,
                "required_all_tags": ["straight"],
                "excluded_tags": ["straight"],
            }
        ],
    }

    with pytest.raises(ValidationError, match="overlap"):
        GlobalConfig.model_validate_json(json.dumps(data))
