from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from dataset_devkit.config import GlobalConfig, TagsConfig, load_config
from dataset_devkit.schema import validate_config_schema_and_runtime
from test_config import minimal_config

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schema" / "dataset_config.schema.json"


def test_checked_in_schema_is_current_and_deterministic() -> None:
    expected = json.dumps(GlobalConfig.model_json_schema(), indent=2, sort_keys=True) + "\n"

    assert SCHEMA_PATH.read_text(encoding="utf-8") == expected


def _required_camera_schema() -> dict[str, object]:
    schema = GlobalConfig.model_json_schema()
    return schema["$defs"]["FrameValidityConfig"]["properties"]["required_cameras"]  # type: ignore[no-any-return]


def _schema_accepts_required_cameras(values: list[str]) -> bool:
    schema = _required_camera_schema()
    item_schema = schema["items"]
    assert isinstance(item_schema, dict)
    pattern = item_schema["pattern"]
    minimum = item_schema["minLength"]
    assert isinstance(pattern, str)
    assert isinstance(minimum, int)
    return (
        (not schema.get("uniqueItems") or len(values) == len(set(values)))
        and all(len(value) >= minimum and re.fullmatch(pattern, value) for value in values)
    )


def test_required_camera_json_schema_directly_rejects_duplicates_and_unsafe_segments() -> None:
    assert _schema_accepts_required_cameras(["front", "rear"])
    assert not _schema_accepts_required_cameras(["front", "front"])
    for unsafe in (
        "", ".", "..", "../front", "front/rear", r"front\rear", " front",
        "rear.", "rear\t", "CON", "con.txt", "LPT1.camera",
    ):
        assert not _schema_accepts_required_cameras([unsafe])


def _schema_errors(data: dict[str, object]) -> list[object]:
    return list(Draft202012Validator(GlobalConfig.model_json_schema()).iter_errors(data))


def test_example_shape_is_valid_under_generated_json_schema() -> None:
    assert not _schema_errors(minimal_config())


def test_schema_rejects_unsafe_per_channel_coverage_key() -> None:
    data = minimal_config()
    data["filters"] = {"min_camera_coverage_by_channel": {"../front": 0.5}}

    assert _schema_errors(data)


def test_safe_segment_maps_close_additional_properties_in_generated_schema() -> None:
    definition = GlobalConfig.model_json_schema()["$defs"]["FiltersConfig"]
    properties = definition["properties"]
    for field in (
        "min_camera_coverage_by_channel",
        "max_camera_coverage_by_channel",
    ):
        assert properties[field]["additionalProperties"] is False


def test_schema_rejects_duplicate_task6_predicate_arrays() -> None:
    filter_fields = (
        "required_any_tags",
        "required_all_tags",
        "excluded_tags",
        "required_any_labels",
        "required_all_labels",
        "excluded_labels",
        "blacklisted_scene_tokens",
        "blacklisted_source_digests",
        "blacklisted_repo_paths",
    )
    rule_fields = filter_fields[:6]
    for field in filter_fields:
        data = minimal_config()
        data["filters"] = {field: ["duplicate", "duplicate"]}
        assert _schema_errors(data), field
    for field in rule_fields:
        data = minimal_config()
        data["scenarios"] = {
            "seed": 1,
            "rules": [{"name": "rule", "quota": 0, field: ["duplicate", "duplicate"]}],
        }
        assert _schema_errors(data), field


def test_schema_directly_enforces_reference_camera_require_coupling() -> None:
    required = minimal_config()
    required["tags"]["reference_camera_channel"] = None  # type: ignore[index]
    required["tags"]["reference_camera_policy"] = "require"  # type: ignore[index]
    fallback = deepcopy(required)
    fallback["tags"]["reference_camera_policy"] = "lexicographic_fallback"  # type: ignore[index]

    assert _schema_errors(required)
    with pytest.raises(ValidationError, match="reference_camera_channel"):
        TagsConfig.model_validate(required["tags"])
    assert not _schema_errors(fallback)
    TagsConfig.model_validate(fallback["tags"])


def test_task6_runtime_constraint_metadata_is_complete_and_exact() -> None:
    definitions = GlobalConfig.model_json_schema()["$defs"]
    expected = {
        "TagsConfig": {
            (
                "tags.heading_threshold_order",
                "straight_max_heading_change_deg < curvature_min_heading_change_deg < "
                "turn_min_heading_change_deg is required",
                (
                    "straight_max_heading_change_deg",
                    "curvature_min_heading_change_deg",
                    "turn_min_heading_change_deg",
                ),
            )
        },
        "FiltersConfig": {
            *{
                (
                    f"filters.{minimum}_lte_{maximum}",
                    f"{minimum} must be <= {maximum}",
                    (minimum, maximum),
                )
                for minimum, maximum in (
                    ("min_duration_s", "max_duration_s"),
                    ("min_scene_valid_ratio", "max_scene_valid_ratio"),
                    ("min_source_gnss_valid_ratio", "max_source_gnss_valid_ratio"),
                    ("min_camera_coverage_ratio", "max_camera_coverage_ratio"),
                    ("min_distance_m", "max_distance_m"),
                )
            },
            (
                "filters.channel_coverage_min_lte_max",
                "camera coverage minimum exceeds maximum for {channel}",
                ("min_camera_coverage_by_channel", "max_camera_coverage_by_channel"),
            ),
            (
                "filters.required_excluded_tags_disjoint",
                "required and excluded tags overlap",
                ("required_any_tags", "required_all_tags", "excluded_tags"),
            ),
            (
                "filters.required_excluded_labels_disjoint",
                "required and excluded labels overlap",
                ("required_any_labels", "required_all_labels", "excluded_labels"),
            ),
        },
        "ScenarioRuleConfig": {
            (
                "scenario_rule.required_excluded_tags_disjoint",
                "required and excluded tags overlap",
                ("required_any_tags", "required_all_tags", "excluded_tags"),
            ),
            (
                "scenario_rule.required_excluded_labels_disjoint",
                "required and excluded labels overlap",
                ("required_any_labels", "required_all_labels", "excluded_labels"),
            ),
        },
        "ScenariosConfig": {
            (
                "scenarios.unique_rule_names",
                "scenario rule names must be unique",
                ("rules[].name",),
            )
        },
    }
    for definition_name, expected_constraints in expected.items():
        actual = {
            (item["code"], item["message"], tuple(item["fields"]))
            for item in definitions[definition_name][
                "x-dataset-devkit-runtime-constraints"
            ]
        }
        assert actual == expected_constraints


def test_unrepresentable_task6_constraints_are_schema_visible_and_runtime_rejected() -> None:
    invalid_configs: list[dict[str, object]] = []
    heading = minimal_config()
    heading["tags"]["straight_max_heading_change_deg"] = 20.0  # type: ignore[index]
    invalid_configs.append(heading)
    for minimum, maximum in (
        ("min_duration_s", "max_duration_s"),
        ("min_scene_valid_ratio", "max_scene_valid_ratio"),
        ("min_source_gnss_valid_ratio", "max_source_gnss_valid_ratio"),
        ("min_camera_coverage_ratio", "max_camera_coverage_ratio"),
        ("min_distance_m", "max_distance_m"),
    ):
        data = minimal_config()
        data["filters"] = {minimum: 1.0, maximum: 0.0}
        invalid_configs.append(data)
    channel = minimal_config()
    channel["filters"] = {
        "min_camera_coverage_by_channel": {"front": 0.8},
        "max_camera_coverage_by_channel": {"front": 0.2},
    }
    invalid_configs.append(channel)
    for kind in ("tags", "labels"):
        filtered = minimal_config()
        filtered["filters"] = {
            f"required_all_{kind}": ["same"],
            f"excluded_{kind}": ["same"],
        }
        invalid_configs.append(filtered)
        ruled = minimal_config()
        ruled["scenarios"] = {
            "seed": 1,
            "rules": [
                {
                    "name": "rule",
                    "quota": 0,
                    f"required_all_{kind}": ["same"],
                    f"excluded_{kind}": ["same"],
                }
            ],
        }
        invalid_configs.append(ruled)
    duplicate_names = minimal_config()
    duplicate_names["scenarios"] = {
        "seed": 1,
        "rules": [{"name": "same", "quota": 0}, {"name": "same", "quota": 1}],
    }
    invalid_configs.append(duplicate_names)

    for data in invalid_configs:
        assert not _schema_errors(data)
        with pytest.raises(ValidationError):
            GlobalConfig.model_validate(data)


def test_combined_schema_and_runtime_validator_is_public_and_authoritative(
    tmp_path: Path,
) -> None:
    valid_path = tmp_path / "valid.json"
    valid_path.write_text(json.dumps(minimal_config()), encoding="utf-8")
    assert validate_config_schema_and_runtime(valid_path).schema_version == "1.0"

    invalid = minimal_config()
    invalid["filters"] = {"min_distance_m": 2.0, "max_distance_m": 1.0}
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValidationError, match="min_distance_m"):
        validate_config_schema_and_runtime(invalid_path)


@pytest.mark.parametrize("location", ["global", "rule"])
@pytest.mark.parametrize(
    "field", ["min_camera_coverage_by_channel", "max_camera_coverage_by_channel"]
)
@pytest.mark.parametrize("value", [-1e-12, 1.000000000001])
def test_per_channel_ratio_bounds_have_schema_and_runtime_parity(
    tmp_path: Path, location: str, field: str, value: float
) -> None:
    data = minimal_config()
    if location == "global":
        data["filters"] = {field: {"front": value}}
    else:
        data["scenarios"] = {
            "seed": 1,
            "rules": [
                {"name": "Internal Space", "quota": 0, "filters": {field: {"front": value}}}
            ],
        }

    assert _schema_errors(data)
    path = tmp_path / f"{location}-{field}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)


def test_task6_string_item_and_name_schema_constraints_cover_every_field() -> None:
    definitions = GlobalConfig.model_json_schema()["$defs"]
    filters = definitions["FiltersConfig"]["properties"]
    for field in (
        "required_any_tags",
        "required_all_tags",
        "excluded_tags",
        "required_any_labels",
        "required_all_labels",
        "excluded_labels",
        "blacklisted_scene_tokens",
        "blacklisted_source_digests",
        "blacklisted_repo_paths",
    ):
        assert filters[field]["items"]["minLength"] == 1
        assert filters[field]["items"]["pattern"]
        assert filters[field]["items"]["not"] == {"pattern": r"\s$"}
    rule = definitions["ScenarioRuleConfig"]["properties"]
    assert rule["name"]["minLength"] == 1
    assert rule["name"]["pattern"]
    assert rule["name"]["not"] == {"pattern": r"\s$"}
    for field in (
        "required_any_tags",
        "required_all_tags",
        "excluded_tags",
        "required_any_labels",
        "required_all_labels",
        "excluded_labels",
    ):
        assert rule[field]["items"]["minLength"] == 1
        assert rule[field]["items"]["pattern"]
        assert rule[field]["items"]["not"] == {"pattern": r"\s$"}


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        " leading",
        "trailing ",
        "\n",
        "value\n",
        "\r",
        "value\r\n",
        "value\u2028",
        "value\u2029",
        "value\t",
    ],
)
@pytest.mark.parametrize(
    ("location", "field"),
    [
        ("filters", "required_all_tags"),
        ("filters", "required_all_labels"),
        ("filters", "blacklisted_scene_tokens"),
        ("filters", "blacklisted_source_digests"),
        ("filters", "blacklisted_repo_paths"),
        ("rule", "required_all_tags"),
        ("rule", "required_all_labels"),
        ("rule", "name"),
    ],
)
def test_task6_trimmed_nonblank_strings_have_schema_and_load_parity(
    tmp_path: Path, location: str, field: str, value: str
) -> None:
    data = minimal_config()
    if location == "filters":
        data["filters"] = {field: [value]}
    else:
        rule: dict[str, object] = {"name": "Valid Name", "quota": 0}
        rule[field] = value if field == "name" else [value]
        data["scenarios"] = {"seed": 1, "rules": [rule]}

    assert _schema_errors(data)
    path = tmp_path / f"{location}-{field}-{len(value)}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)


@pytest.mark.parametrize(
    "name", ["Left Turn Urban", "Left\tTurn", "Left\nTurn", "Left\u2028Turn", "Left\u2029Turn"]
)
def test_internal_space_scenario_name_passes_schema_and_runtime(
    tmp_path: Path, name: str
) -> None:
    data = minimal_config()
    data["scenarios"] = {
        "seed": 1,
        "rules": [{"name": name, "quota": 0}],
    }
    assert not _schema_errors(data)
    path = tmp_path / "internal-space.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_config(path).scenarios.rules[0].name == name
