from __future__ import annotations

import copy
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from dataset_devkit.config import GlobalConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


def _load_with_literal(tmp_path: Path, field: str, old: str, literal: str) -> GlobalConfig:
    text = (ROOT / "examples" / "dataset_config.json").read_text(encoding="utf-8")
    replaced = text.replace(f'"{field}": {old}', f'"{field}": {literal}')
    assert replaced != text
    config_path = tmp_path / f"{field}.json"
    config_path.write_text(replaced, encoding="utf-8")
    return load_config(config_path)


def test_scene_defaults_and_exact_integer_nanoseconds(
    config_factory: Callable[[], GlobalConfig],
) -> None:
    config = config_factory()

    assert config.scenes.mode == "hybrid"
    assert config.scenes.dataset_namespace == UUID("8d55f58b-4a7b-5a9a-a95a-a3989610795b")
    assert config.scenes.min_duration_ns == 10_000_000_000
    assert config.scenes.max_duration_ns == 40_000_000_000
    assert config.scenes.max_sample_gap_ns == 650_000_000
    assert config.scenes.skip_between_scenes_ns == 0
    assert config.annotations.match_tolerance_ns == 500_000_000
    assert config.annotations.before_ns == 20_000_000_000
    assert config.annotations.after_ns == 20_000_000_000
    assert all(
        isinstance(value, Decimal)
        for value in (
            config.scenes.min_duration_s,
            config.scenes.max_duration_s,
            config.scenes.max_sample_gap_ms,
            config.scenes.skip_between_scenes_s,
            config.annotations.match_tolerance_ms,
            config.annotations.before_s,
            config.annotations.after_s,
        )
    )


@pytest.mark.parametrize("mode", ["automatic", "annotation_only", "hybrid"])
def test_all_exact_scene_modes_are_accepted(config_factory: object, mode: str) -> None:
    data = config_factory().model_dump(mode="python")  # type: ignore[operator]
    data["scenes"]["mode"] = mode
    assert GlobalConfig.model_validate(data).scenes.mode == mode


@pytest.mark.parametrize("mode", ["fixed", "annotation", "AUTO", ""])
def test_old_or_unknown_scene_modes_are_rejected(config_factory: object, mode: str) -> None:
    data = config_factory().model_dump(mode="python")  # type: ignore[operator]
    data["scenes"]["mode"] = mode
    with pytest.raises(ValidationError, match="mode"):
        GlobalConfig.model_validate(data)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("scenes", "unknown", 1),
        ("annotations", "unknown", 1),
        ("scenes", "dataset_namespace", "not-a-uuid"),
        ("scenes", "min_duration_s", float("nan")),
        ("scenes", "max_duration_s", float("inf")),
        ("annotations", "before_s", float("nan")),
    ],
)
def test_scene_annotation_configuration_is_strict(
    config_factory: object, section: str, field: str, value: object
) -> None:
    data = copy.deepcopy(config_factory().model_dump(mode="python"))  # type: ignore[operator]
    data[section][field] = value
    with pytest.raises(ValidationError):
        GlobalConfig.model_validate(data)


@pytest.mark.parametrize(
    ("field", "old", "literal", "property_name", "expected"),
    [
        ("min_duration_s", "10.0", "0.000000001", "min_duration_ns", 1),
        ("min_duration_s", "10.0", "1e-9", "min_duration_ns", 1),
        ("max_sample_gap_ms", "650.0", "0.000001", "max_sample_gap_ns", 1),
        ("match_tolerance_ms", "500.0", "1e-6", "match_tolerance_ns", 1),
        ("before_s", "20.0", "1e-9", "before_ns", 1),
        ("after_s", "20.0", "0", "after_ns", 0),
    ],
)
def test_json_time_literals_preserve_exact_decimal_text(
    tmp_path: Path,
    field: str,
    old: str,
    literal: str,
    property_name: str,
    expected: int,
) -> None:
    config = _load_with_literal(tmp_path, field, old, literal)
    section = config.annotations if hasattr(config.annotations, property_name) else config.scenes
    assert getattr(section, property_name) == expected


@pytest.mark.parametrize(
    ("literal", "message"),
    [
        ("0.000000000999999999999999999999", "exact integer"),
        ("0.000000001000000000000000000001", "exact integer"),
        ("9.999999999999999999e-10", "exact integer"),
    ],
)
def test_json_sub_nanosecond_precision_is_rejected_without_float_rounding(
    tmp_path: Path, literal: str, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        _load_with_literal(tmp_path, "min_duration_s", "10.0", literal)


def test_direct_model_api_requires_decimal_for_task5_times(
    config_factory: Callable[[], GlobalConfig],
) -> None:
    data = config_factory().model_dump(mode="python")
    data["scenes"]["min_duration_s"] = 0.29
    with pytest.raises(ValidationError, match="min_duration_s"):
        GlobalConfig.model_validate(data)

    data["scenes"]["min_duration_s"] = Decimal("0.29")
    config = GlobalConfig.model_validate(data)
    assert config.scenes.min_duration_ns == 290_000_000


def test_task5_time_schema_accepts_json_numbers_not_decimal_strings() -> None:
    schema = GlobalConfig.model_json_schema()
    for model_name, fields in {
        "ScenesConfig": (
            "min_duration_s",
            "max_duration_s",
            "max_sample_gap_ms",
            "skip_between_scenes_s",
        ),
        "AnnotationsConfig": ("match_tolerance_ms", "before_s", "after_s"),
    }.items():
        properties = schema["$defs"][model_name]["properties"]
        for field in fields:
            assert properties[field]["type"] == "number"
            assert "anyOf" not in properties[field]
