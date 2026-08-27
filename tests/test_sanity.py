from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest

from dataset_devkit.config import GlobalConfig, SanityChecksConfig
from dataset_devkit.extraction.errors import NonstructuralSanityError
from dataset_devkit.extraction.grid import GridSelection
from dataset_devkit.sanity import SANITY_CHECK_CODES, SanityCheckCode, evaluate_sanity
from dataset_devkit.validity import evaluate_validity
from test_validity import _result


def _config(
    base: GlobalConfig,
    code: SanityCheckCode,
    policy: Literal["error", "warn", "off"],
) -> GlobalConfig:
    policies = {name: "off" for name in SANITY_CHECK_CODES}
    policies[code] = policy
    sanity = SanityChecksConfig.model_validate(policies)
    return base.model_copy(update={"sanity_checks": sanity})


def _trigger(
    tmp_path: Path, base: GlobalConfig, code: SanityCheckCode
) -> tuple[object, object, GlobalConfig]:
    result = _result(tmp_path)
    if code == "empty_selected_grid":
        result = replace(
            result,
            selected_grid=GridSelection((), (), tuple(
                batch.rec_timestamp_ns for batch in result.camera_batches
            )),
            samples=(),
            ego_poses_by_timestamp={},
        )
    elif code == "all_gnss_sources_invalid":
        result = replace(
            result,
            gnss_samples=tuple(replace(sample, is_valid=False) for sample in result.gnss_samples),
        )
    elif code == "zero_required_camera_coverage":
        base = base.model_copy(
            update={
                "frame_validity": base.frame_validity.model_copy(
                    update={"required_cameras": ["never-present"]}
                )
            }
        )
    validity = evaluate_validity(result, base)
    return result, validity, base


@pytest.mark.parametrize("code", SANITY_CHECK_CODES)
@pytest.mark.parametrize("policy", ["error", "warn", "off"])
def test_each_sanity_check_honors_error_warn_and_off(
    tmp_path: Path,
    config_factory: object,
    code: SanityCheckCode,
    policy: Literal["error", "warn", "off"],
) -> None:
    result, validity, base = _trigger(tmp_path / f"{code}-{policy}", config_factory(), code)  # type: ignore[operator]
    config = _config(base, code, policy)

    if policy == "error":
        with pytest.raises(NonstructuralSanityError) as caught:
            evaluate_sanity(result, validity, config)  # type: ignore[arg-type]
        assert [item.code for item in caught.value.observations] == [code]
        assert caught.value.observations[0].policy == "error"
    else:
        report = evaluate_sanity(result, validity, config)  # type: ignore[arg-type]
        if policy == "warn":
            assert [item.code for item in report.observations] == [code]
            assert report.warnings == report.observations
            assert report.observations[0].message
            assert report.observations[0].scope in {"recording", "sample"}
        else:
            assert not report.observations
            assert not report.warnings


@pytest.mark.parametrize("sample_policy", ["retain_for_audit", "drop"])
def test_required_camera_coverage_uses_extracted_presence_before_policy_filtering(
    tmp_path: Path,
    config_factory: object,
    sample_policy: Literal["retain_for_audit", "drop"],
) -> None:
    base = config_factory()  # type: ignore[operator]
    config = base.model_copy(
        update={
            "frame_validity": base.frame_validity.model_copy(
                update={
                    "invalid_sample_policy": sample_policy,
                    "required_cameras": ["front", "rear"],
                }
            ),
            "sanity_checks": SanityChecksConfig(
                empty_selected_grid="off",
                empty_final_candidates="off",
                all_gnss_sources_invalid="off",
                zero_required_camera_coverage="warn",
            ),
        }
    )
    result = _result(tmp_path / sample_policy)
    validity = evaluate_validity(result, config)

    sanity = evaluate_sanity(result, validity, config)

    assert not sanity.observations
