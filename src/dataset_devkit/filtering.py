"""Declarative scene-feature filtering with complete rejection evidence."""

from __future__ import annotations

from dataclasses import dataclass

from dataset_devkit.config import FiltersConfig
from dataset_devkit.features import SceneFeatures


@dataclass(frozen=True)
class RejectionReason:
    code: str
    measured_value: object
    operator: str
    threshold: object
    scene_token: str
    source_digest: str
    source_blob_path: str


@dataclass(frozen=True)
class RejectedScene:
    feature: SceneFeatures
    reasons: tuple[RejectionReason, ...]


@dataclass(frozen=True)
class FilterResult:
    accepted: tuple[SceneFeatures, ...]
    rejected: tuple[RejectedScene, ...]


@dataclass(frozen=True)
class _CompiledFilter:
    min_channel_coverage: tuple[tuple[str, float], ...]
    max_channel_coverage: tuple[tuple[str, float], ...]
    required_any_tags: frozenset[str]
    required_all_tags: frozenset[str]
    excluded_tags: frozenset[str]
    required_any_labels: frozenset[str]
    required_all_labels: frozenset[str]
    excluded_labels: frozenset[str]
    blacklisted_scene_tokens: frozenset[str]
    blacklisted_source_digests: frozenset[str]
    blacklisted_blob_paths: frozenset[str]


def _compile_filter(config: FiltersConfig) -> _CompiledFilter:
    return _CompiledFilter(
        tuple(sorted(config.min_camera_coverage_by_channel.items())),
        tuple(sorted(config.max_camera_coverage_by_channel.items())),
        frozenset(config.required_any_tags),
        frozenset(config.required_all_tags),
        frozenset(config.excluded_tags),
        frozenset(config.required_any_labels),
        frozenset(config.required_all_labels),
        frozenset(config.excluded_labels),
        frozenset(config.blacklisted_scene_tokens),
        frozenset(config.blacklisted_source_digests),
        frozenset(config.blacklisted_blob_paths),
    )


def _evaluate(
    feature: SceneFeatures, config: FiltersConfig, compiled: _CompiledFilter
) -> tuple[RejectionReason, ...]:
    reasons: list[RejectionReason] = []

    def reject(code: str, measured: object, operator: str, threshold: object) -> None:
        reasons.append(
            RejectionReason(
                code,
                measured,
                operator,
                threshold,
                feature.scene_token,
                feature.source.digest,
                feature.source_blob_path,
            )
        )

    ranges = (
        ("duration", feature.duration_s, config.min_duration_s, config.max_duration_s),
        (
            "scene_valid_ratio",
            feature.scene_valid_ratio,
            config.min_scene_valid_ratio,
            config.max_scene_valid_ratio,
        ),
        (
            "camera_coverage_ratio",
            feature.camera_coverage_ratio,
            config.min_camera_coverage_ratio,
            config.max_camera_coverage_ratio,
        ),
        (
            "source_gnss_valid_ratio",
            feature.source_gnss_valid_ratio,
            config.min_source_gnss_valid_ratio,
            config.max_source_gnss_valid_ratio,
        ),
        ("distance", feature.total_distance_m, config.min_distance_m, config.max_distance_m),
    )
    for name, value, minimum, maximum in ranges:
        if minimum is not None and value < minimum:
            reject(f"{name}_below_minimum", value, ">=", minimum)
        if maximum is not None and value > maximum:
            reject(f"{name}_above_maximum", value, "<=", maximum)
    coverage = {item.channel: item.ratio for item in feature.camera_coverage_by_channel}
    for channel, minimum in compiled.min_channel_coverage:
        value = coverage.get(channel, 0.0)
        if value < minimum:
            reject("channel_coverage_below_minimum", value, ">=", (channel, minimum))
    for channel, maximum in compiled.max_channel_coverage:
        value = coverage.get(channel, 0.0)
        if value > maximum:
            reject("channel_coverage_above_maximum", value, "<=", (channel, maximum))
    if (
        config.max_sync_error_ms is not None
        and feature.max_abs_sync_error_ms > config.max_sync_error_ms
    ):
        reject(
            "sync_error_above_maximum",
            feature.max_abs_sync_error_ms,
            "<=",
            config.max_sync_error_ms,
        )
    for kind, present_values, required_any, required_all, excluded in (
        (
            "tags",
            set(feature.computed_tags),
            compiled.required_any_tags,
            compiled.required_all_tags,
            compiled.excluded_tags,
        ),
        (
            "labels",
            set(feature.human_labels),
            compiled.required_any_labels,
            compiled.required_all_labels,
            compiled.excluded_labels,
        ),
    ):
        if required_any and not present_values & required_any:
            reject(
                f"required_any_{kind}_missing",
                tuple(sorted(present_values)),
                "intersects",
                tuple(sorted(required_any)),
            )
        missing = required_all - present_values
        if missing:
            reject(
                f"required_all_{kind}_missing",
                tuple(sorted(present_values)),
                "contains",
                tuple(sorted(required_all)),
            )
        found = excluded & present_values
        if found:
            reject(
                f"excluded_{kind}_present",
                tuple(sorted(found)),
                "disjoint",
                tuple(sorted(excluded)),
            )
    if feature.scene_token in compiled.blacklisted_scene_tokens:
        reject(
            "scene_token_blacklisted",
            feature.scene_token,
            "not in",
            tuple(config.blacklisted_scene_tokens),
        )
    if feature.source.digest in compiled.blacklisted_source_digests:
        reject(
            "source_digest_blacklisted",
            feature.source.digest,
            "not in",
            tuple(config.blacklisted_source_digests),
        )
    if feature.source_blob_path in compiled.blacklisted_blob_paths:
        reject(
            "blob_path_blacklisted",
            feature.source_blob_path,
            "not in",
            tuple(config.blacklisted_blob_paths),
        )
    return tuple(reasons)


def filter_scenes(
    features: tuple[SceneFeatures, ...] | list[SceneFeatures], config: FiltersConfig
) -> FilterResult:
    """Evaluate every configured predicate and preserve deterministic input order."""
    accepted: list[SceneFeatures] = []
    rejected: list[RejectedScene] = []
    compiled = _compile_filter(config)
    for feature in features:
        reasons = _evaluate(feature, config, compiled)
        if reasons:
            rejected.append(RejectedScene(feature, reasons))
        else:
            accepted.append(feature)
    return FilterResult(tuple(accepted), tuple(rejected))
