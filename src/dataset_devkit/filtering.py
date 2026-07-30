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


def _evaluate(feature: SceneFeatures, config: FiltersConfig) -> tuple[RejectionReason, ...]:
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
    for channel, minimum in sorted(config.min_camera_coverage_by_channel.items()):
        value = coverage.get(channel, 0.0)
        if value < minimum:
            reject("channel_coverage_below_minimum", value, ">=", (channel, minimum))
    for channel, maximum in sorted(config.max_camera_coverage_by_channel.items()):
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
    for kind, present_values in (
        ("tags", set(feature.computed_tags)),
        ("labels", set(feature.human_labels)),
    ):
        required_any = set(getattr(config, f"required_any_{kind}"))
        required_all = set(getattr(config, f"required_all_{kind}"))
        excluded = set(getattr(config, f"excluded_{kind}"))
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
    if feature.scene_token in config.blacklisted_scene_tokens:
        reject(
            "scene_token_blacklisted",
            feature.scene_token,
            "not in",
            tuple(config.blacklisted_scene_tokens),
        )
    if feature.source.digest in config.blacklisted_source_digests:
        reject(
            "source_digest_blacklisted",
            feature.source.digest,
            "not in",
            tuple(config.blacklisted_source_digests),
        )
    if feature.source_blob_path in config.blacklisted_blob_paths:
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
    for feature in features:
        reasons = _evaluate(feature, config)
        if reasons:
            rejected.append(RejectedScene(feature, reasons))
        else:
            accepted.append(feature)
    return FilterResult(tuple(accepted), tuple(rejected))
