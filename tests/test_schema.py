from __future__ import annotations

import json
import re
from pathlib import Path

from jsonschema import Draft202012Validator

from dataset_devkit.config import GlobalConfig
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
        "blacklisted_blob_paths",
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
