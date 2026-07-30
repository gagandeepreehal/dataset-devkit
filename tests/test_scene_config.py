from __future__ import annotations

import copy
from uuid import UUID

import pytest
from pydantic import ValidationError

from dataset_devkit.config import GlobalConfig


def test_scene_defaults_and_exact_integer_nanoseconds(config_factory: object) -> None:
    config = config_factory()  # type: ignore[operator]

    assert config.scenes.mode == "hybrid"
    assert config.scenes.dataset_namespace == UUID("8d55f58b-4a7b-5a9a-a95a-a3989610795b")
    assert config.scenes.min_duration_ns == 10_000_000_000
    assert config.scenes.max_duration_ns == 40_000_000_000
    assert config.scenes.max_sample_gap_ns == 650_000_000
    assert config.scenes.skip_between_scenes_ns == 0
    assert config.annotations.match_tolerance_ns == 500_000_000
    assert config.annotations.before_ns == 20_000_000_000
    assert config.annotations.after_ns == 20_000_000_000


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


def test_decimal_duration_conversion_has_no_binary_float_truncation(
    config_factory: object,
) -> None:
    data = config_factory().model_dump(mode="python")  # type: ignore[operator]
    data["scenes"]["min_duration_s"] = 0.29
    data["scenes"]["max_duration_s"] = 0.58
    data["scenes"]["max_sample_gap_ms"] = 0.29
    config = GlobalConfig.model_validate(data)
    assert config.scenes.min_duration_ns == 290_000_000
    assert config.scenes.max_duration_ns == 580_000_000
    assert config.scenes.max_sample_gap_ns == 290_000
