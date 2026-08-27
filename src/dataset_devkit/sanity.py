"""Typed nonstructural checks evaluated after validity classification."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, cast

from dataset_devkit.config import GlobalConfig
from dataset_devkit.extraction.errors import NonstructuralSanityError
from dataset_devkit.extraction.models import RecordingExtractionResult
from dataset_devkit.validity import ValidityReport

type SanityCheckCode = Literal[
    "empty_selected_grid",
    "empty_final_candidates",
    "all_gnss_sources_invalid",
    "zero_required_camera_coverage",
]
type SanityPolicy = Literal["error", "warn", "off"]
type SanityScope = Literal["recording", "sample"]

SANITY_CHECK_CODES: tuple[SanityCheckCode, ...] = (
    "empty_selected_grid",
    "empty_final_candidates",
    "all_gnss_sources_invalid",
    "zero_required_camera_coverage",
)


@dataclass(frozen=True)
class SanityObservation:
    code: SanityCheckCode
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)
    scope: SanityScope = "recording"
    policy: Literal["error", "warn"] = "warn"

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


@dataclass(frozen=True)
class SanityReport:
    observations: tuple[SanityObservation, ...]
    warnings: tuple[SanityObservation, ...]


@dataclass(frozen=True)
class _CheckResult:
    triggered: bool
    message: str
    details: Mapping[str, Any]
    scope: SanityScope = "recording"


def _empty_selected(
    result: RecordingExtractionResult, validity: ValidityReport, config: GlobalConfig
) -> _CheckResult:
    del validity, config
    count = len(result.selected_grid.entries)
    return _CheckResult(count == 0, "recording selected no camera grid samples", {"count": count})


def _empty_final(
    result: RecordingExtractionResult, validity: ValidityReport, config: GlobalConfig
) -> _CheckResult:
    del result, config
    count = len(validity.final_candidates)
    return _CheckResult(
        count == 0,
        "recording has no valid final sample candidates",
        {"count": count},
    )


def _all_gnss_invalid(
    result: RecordingExtractionResult, validity: ValidityReport, config: GlobalConfig
) -> _CheckResult:
    del validity, config
    total = len(result.gnss_samples)
    invalid = sum(not sample.is_valid for sample in result.gnss_samples)
    return _CheckResult(
        total > 0 and invalid == total,
        "every GNSS source sample is marked invalid",
        {"total_gnss_samples": total, "invalid_gnss_samples": invalid},
    )


def _zero_required_coverage(
    result: RecordingExtractionResult, validity: ValidityReport, config: GlobalConfig
) -> _CheckResult:
    del validity
    required = tuple(config.frame_validity.required_cameras)
    covered = {
        sample.camera_name
        for sample in result.samples
        if sample.camera_name in required
    }
    return _CheckResult(
        bool(required) and not covered,
        "recording has zero coverage for configured required cameras",
        {"required_cameras": required, "covered_required_cameras": tuple(sorted(covered))},
        "sample",
    )


_CHECKS: Mapping[
    SanityCheckCode,
    Callable[[RecordingExtractionResult, ValidityReport, GlobalConfig], _CheckResult],
] = {
    "empty_selected_grid": _empty_selected,
    "empty_final_candidates": _empty_final,
    "all_gnss_sources_invalid": _all_gnss_invalid,
    "zero_required_camera_coverage": _zero_required_coverage,
}


def evaluate_sanity(
    result: RecordingExtractionResult,
    validity: ValidityReport,
    config: GlobalConfig,
) -> SanityReport:
    """Emit warnings or one aggregate policy error; off checks are not evaluated."""
    observations: list[SanityObservation] = []
    errors: list[SanityObservation] = []
    warnings: list[SanityObservation] = []
    for code in SANITY_CHECK_CODES:
        policy = cast(SanityPolicy, getattr(config.sanity_checks, code))
        if policy == "off":
            continue
        check = _CHECKS[code](result, validity, config)
        if not check.triggered:
            continue
        observation = SanityObservation(
            code,
            check.message,
            check.details,
            check.scope,
            policy,
        )
        observations.append(observation)
        if policy == "error":
            errors.append(observation)
        else:
            warnings.append(observation)
    if errors:
        raise NonstructuralSanityError(
            tuple(observations), error_observations=tuple(errors)
        )
    return SanityReport(tuple(observations), tuple(warnings))
