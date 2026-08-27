from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from dataset_devkit.config import GlobalConfig


def test_validity_configuration_has_explicit_strict_defaults(
    config_factory: object,
) -> None:
    config = config_factory()  # type: ignore[operator]

    assert config.frame_validity.invalid_sample_policy == "retain_for_audit"
    assert config.frame_validity.required_cameras == ["front", "rear"]
    assert config.frame_validity.camera_timestamp_gap_max_ms == 1000.0
    assert set(config.frame_validity.invalidate_on.model_dump()) == {
        "gnss_source_invalid",
        "position_sigma_exceeded",
        "orientation_variance_exceeded",
        "gnss_sync_gap_exceeded",
        "camera_timestamp_non_monotonic",
        "camera_timestamp_gap_exceeded",
        "missing_required_camera",
        "grid_miss",
    }
    assert set(config.sanity_checks.model_dump()) == {
        "empty_selected_grid",
        "empty_final_candidates",
        "all_gnss_sources_invalid",
        "zero_required_camera_coverage",
    }


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("frame_validity", "unknown", True),
        ("invalidate_on", "unknown_reason", True),
        ("sanity_checks", "unknown_check", "warn"),
    ],
)
def test_unknown_validity_and_sanity_policy_keys_are_rejected(
    config_factory: object, section: str, key: str, value: object
) -> None:
    data = copy.deepcopy(config_factory().model_dump(mode="python"))  # type: ignore[operator]
    if section == "invalidate_on":
        data["frame_validity"]["invalidate_on"][key] = value
    else:
        data[section][key] = value

    with pytest.raises(ValidationError, match=key):
        GlobalConfig.model_validate(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("invalid_sample_policy", "retain"),
        ("camera_timestamp_gap_max_ms", 0.0),
        ("camera_timestamp_gap_max_ms", -1.0),
        ("required_cameras", ["front", "front"]),
        ("required_cameras", [""]),
        ("required_cameras", ["../front"]),
    ],
)
def test_frame_validity_contract_rejects_invalid_values(
    config_factory: object, field: str, value: object
) -> None:
    data = copy.deepcopy(config_factory().model_dump(mode="python"))  # type: ignore[operator]
    data["frame_validity"][field] = value

    with pytest.raises(ValidationError, match=field):
        GlobalConfig.model_validate(data)


@pytest.mark.parametrize("policy", ["quarantine", "ignore", True])
def test_sanity_policy_is_exact_error_warn_or_off(
    config_factory: object, policy: object
) -> None:
    data = copy.deepcopy(config_factory().model_dump(mode="python"))  # type: ignore[operator]
    data["sanity_checks"]["empty_selected_grid"] = policy

    with pytest.raises(ValidationError, match="empty_selected_grid"):
        GlobalConfig.model_validate(data)
