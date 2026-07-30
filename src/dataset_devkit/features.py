"""Timestamp-aware deterministic CPU trajectory features over Task 5 scene graphs."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass

from dataset_devkit.config import TagsConfig
from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.provenance import SourceFingerprint
from dataset_devkit.scene_models import (
    RecordingSceneResult,
    SampleDataRecord,
    SampleRecord,
    SceneRecord,
    SourceSampleRecord,
)
from dataset_devkit.scenes import validate_scene_graph


@dataclass(frozen=True)
class ChannelCoverage:
    channel: str
    present: int
    expected: int
    ratio: float


@dataclass(frozen=True)
class SceneFeatures:
    scene_token: str
    scene_name: str
    source: SourceFingerprint
    source_blob_path: str
    human_labels: tuple[str, ...]
    computed_tags: tuple[str, ...]
    reference_camera_requested: str | None
    reference_channels_used: tuple[str, ...]
    reference_fallback_count: int
    observation_timestamps_ns: tuple[int, ...]
    segment_dt_s: tuple[float, ...]
    segment_distances_m: tuple[float, ...]
    cumulative_distances_m: tuple[float, ...]
    segment_speeds_mps: tuple[float, ...]
    observation_stationary: tuple[bool, ...]
    segment_stationary: tuple[bool, ...]
    duration_s: float
    total_distance_m: float
    mean_speed_mps: float
    median_speed_mps: float
    max_speed_mps: float
    start_speed_mps: float
    end_speed_mps: float
    moving_to_stationary_count: int
    stationary_to_moving_count: int
    stationary_duration_s: float
    headings_rad: tuple[float, ...]
    signed_heading_changes_rad: tuple[float, ...]
    absolute_heading_change_rad: float
    net_heading_change_rad: float
    total_heading_change_rad: float
    signed_curvatures_rad_per_m: tuple[float, ...]
    source_gnss_valid_ratio: float
    camera_coverage_ratio: float
    camera_coverage_by_channel: tuple[ChannelCoverage, ...]
    max_abs_sync_error_ms: float
    mean_abs_sync_error_ms: float
    scene_valid_ratio: float = 1.0
    time_weighted_speed_mps: float = 0.0


@dataclass(frozen=True)
class RecordingFeatureResult:
    source: SourceFingerprint
    scenes: tuple[SceneFeatures, ...]


@dataclass(frozen=True)
class _FeatureRecordIndex:
    samples_by_scene: dict[str, tuple[SampleRecord, ...]]
    data_by_sample: dict[str, tuple[SampleDataRecord, ...]]
    data_by_scene: dict[str, tuple[SampleDataRecord, ...]]
    source_by_timestamp: dict[int, SourceSampleRecord]


def _index_feature_records(result: RecordingSceneResult) -> _FeatureRecordIndex:
    """Index each Task 5 feature evidence record exactly once in canonical order."""
    samples_by_scene_lists: dict[str, list[SampleRecord]] = defaultdict(list)
    data_by_sample_lists: dict[str, list[SampleDataRecord]] = defaultdict(list)
    data_by_scene_lists: dict[str, list[SampleDataRecord]] = defaultdict(list)
    for sample in result.samples:
        samples_by_scene_lists[sample.scene_token].append(sample)
    for item in result.sample_data:
        data_by_sample_lists[item.sample_token].append(item)
        data_by_scene_lists[item.scene_token].append(item)
    return _FeatureRecordIndex(
        {key: tuple(value) for key, value in samples_by_scene_lists.items()},
        {key: tuple(value) for key, value in data_by_sample_lists.items()},
        {key: tuple(value) for key, value in data_by_scene_lists.items()},
        {item.timestamp_ns: item for item in result.source_samples},
    )


def _yaw_wxyz(quaternion: tuple[float, float, float, float]) -> float:
    if not all(math.isfinite(value) for value in quaternion):
        raise StructuralExtractionError("trajectory quaternion contains non-finite values")
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 0 or not math.isfinite(norm):
        raise StructuralExtractionError("trajectory quaternion has zero or invalid norm")
    w, x, y, z = (value / norm for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _shortest_angle(value: float) -> float:
    wrapped = (value + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if wrapped == -math.pi and value > 0 else wrapped


def _select_observations(
    scene: SceneRecord,
    samples: tuple[SampleRecord, ...],
    data_by_sample: dict[str, tuple[SampleDataRecord, ...]],
    config: TagsConfig,
) -> tuple[tuple[SampleDataRecord, ...], int]:
    selected: list[SampleDataRecord] = []
    fallback_count = 0
    for sample in samples:
        token = sample.token
        candidates = data_by_sample[token]
        requested = config.reference_camera_channel
        matches = tuple(item for item in candidates if item.channel == requested)
        if requested is not None and len(matches) == 1:
            selected.append(matches[0])
        elif config.reference_camera_policy == "lexicographic_fallback" and candidates:
            selected.append(min(candidates, key=lambda item: (item.channel, item.camera_index)))
            fallback_count += 1
        else:
            raise StructuralExtractionError(
                f"scene {scene.token} sample {token} does not contain exactly one configured "
                "trajectory reference camera"
            )
    return tuple(selected), fallback_count


def derive_computed_tags(
    config: TagsConfig,
    distance: float,
    segment_stationary: tuple[bool, ...],
    heading_change: float,
    stopping: int,
    starting: int,
    gnss_ratio: float,
    coverage_ratio: float,
) -> tuple[str, ...]:
    tags: set[str] = set()
    moving = distance >= config.minimum_movement_m and any(
        not item for item in segment_stationary
    )
    tags.add("moving" if moving else "stationary")
    if stopping:
        tags.add("stopping")
    if starting:
        tags.add("starting")
    degrees = math.degrees(abs(heading_change))
    direction = "left" if heading_change > 0 else "right"
    if degrees <= config.straight_max_heading_change_deg:
        tags.add("straight")
    elif degrees >= config.turn_min_heading_change_deg:
        tags.add(f"{direction}_turn")
    elif degrees >= config.curvature_min_heading_change_deg:
        tags.add(f"{direction}_curvature")
    if gnss_ratio == 1.0:
        tags.add("gnss_valid")
    elif gnss_ratio == 0.0:
        tags.add("gnss_invalid")
    else:
        tags.add("gnss_partial")
    if coverage_ratio == 1.0:
        tags.add("camera_coverage_complete")
    else:
        tags.add("camera_coverage_partial")
    return tuple(sorted(tags))


def _compute_scene(
    result: RecordingSceneResult,
    scene: SceneRecord,
    config: TagsConfig,
    index: _FeatureRecordIndex,
) -> SceneFeatures:
    samples = index.samples_by_scene[scene.token]
    data_by_sample = index.data_by_sample
    observations, fallback_count = _select_observations(
        scene, samples, data_by_sample, config
    )
    scene_data = index.data_by_scene[scene.token]
    if any(len(item.gnss_source_validity) != 2 for item in scene_data):
        raise StructuralExtractionError(
            "trajectory source GNSS endpoint-validity evidence is missing"
        )
    timestamps = tuple(item.timestamp_ns for item in observations)
    if any(
        current <= previous
        for previous, current in zip(timestamps, timestamps[1:], strict=False)
    ):
        raise StructuralExtractionError("trajectory reference timestamps must increase strictly")
    positions: list[tuple[float, float]] = []
    headings: list[float] = []
    for item in observations:
        pose = item.ego_pose
        if not pose.available or pose.translation_xyz_m is None or pose.rotation_wxyz is None:
            raise StructuralExtractionError("trajectory reference pose is unavailable")
        if not all(math.isfinite(value) for value in pose.translation_xyz_m):
            raise StructuralExtractionError("trajectory reference pose contains non-finite values")
        positions.append((pose.translation_xyz_m[0], pose.translation_xyz_m[1]))
        headings.append(_yaw_wxyz(pose.rotation_wxyz))
    dt = tuple(
        (current - previous) / 1_000_000_000
        for previous, current in zip(timestamps, timestamps[1:], strict=False)
    )
    distances = tuple(
        math.hypot(current[0] - previous[0], current[1] - previous[1])
        for previous, current in zip(positions, positions[1:], strict=False)
    )
    speeds = tuple(distance / seconds for distance, seconds in zip(distances, dt, strict=True))
    heading_changes = tuple(
        _shortest_angle(current - previous)
        for previous, current in zip(headings, headings[1:], strict=False)
    )
    curvatures = tuple(
        change / distance if distance > 0 else 0.0
        for change, distance in zip(heading_changes, distances, strict=True)
    )
    if not all(
        math.isfinite(value)
        for values in (dt, distances, speeds, heading_changes, curvatures)
        for value in values
    ):
        raise StructuralExtractionError("trajectory computation produced a non-finite metric")
    segment_stationary = tuple(speed <= config.stationary_speed_mps for speed in speeds)
    observation_stationary = (
        (True,)
        if not speeds
        else (segment_stationary[0], *segment_stationary)
    )
    stopping = sum(
        not previous and current
        for previous, current in zip(segment_stationary, segment_stationary[1:], strict=False)
    )
    starting = sum(
        previous and not current
        for previous, current in zip(segment_stationary, segment_stationary[1:], strict=False)
    )
    cumulative = [0.0]
    for distance in distances:
        cumulative.append(cumulative[-1] + distance)
    duration = sum(dt)
    distance = sum(distances)
    source_records = tuple(index.source_by_timestamp[item.timestamp_ns] for item in samples)
    gnss_ratio = sum(item.source_gnss_valid for item in source_records) / len(source_records)
    expected_slots = sum(len(item.expected_channels) for item in source_records)
    present_slots = sum(len(item.present_channels) for item in source_records)
    coverage_ratio = present_slots / expected_slots
    channels = sorted({channel for item in source_records for channel in item.expected_channels})
    channel_coverage = tuple(
        ChannelCoverage(
            channel,
            sum(channel in item.present_channels for item in source_records),
            sum(channel in item.expected_channels for item in source_records),
            sum(channel in item.present_channels for item in source_records)
            / sum(channel in item.expected_channels for item in source_records),
        )
        for channel in channels
    )
    sync_errors_ns = (
        *(abs(item.grid_signed_sync_error_ns) for item in source_records),
        *(abs(item.camera_signed_sync_error_ns) for item in scene_data),
    )
    net_heading = sum(heading_changes)
    return SceneFeatures(
        scene.token,
        scene.name,
        result.source,
        scene.source_blob_path,
        tuple(scene.labels),
        derive_computed_tags(
            config,
            distance,
            segment_stationary,
            net_heading,
            stopping,
            starting,
            gnss_ratio,
            coverage_ratio,
        ),
        config.reference_camera_channel,
        tuple(item.channel for item in observations),
        fallback_count,
        timestamps,
        dt,
        distances,
        tuple(cumulative),
        speeds,
        tuple(observation_stationary),
        segment_stationary,
        duration,
        distance,
        statistics.fmean(speeds) if speeds else 0.0,
        statistics.median(speeds) if speeds else 0.0,
        max(speeds, default=0.0),
        speeds[0] if speeds else 0.0,
        speeds[-1] if speeds else 0.0,
        stopping,
        starting,
        sum(
            seconds
            for seconds, stationary in zip(dt, segment_stationary, strict=True)
            if stationary
        ),
        tuple(headings),
        heading_changes,
        abs(net_heading),
        net_heading,
        sum(abs(value) for value in heading_changes),
        curvatures,
        gnss_ratio,
        coverage_ratio,
        channel_coverage,
        max(sync_errors_ns, default=0) / 1_000_000,
        statistics.fmean(sync_errors_ns) / 1_000_000 if sync_errors_ns else 0.0,
        time_weighted_speed_mps=distance / duration if duration else 0.0,
    )


def compute_recording_features(
    result: RecordingSceneResult, config: TagsConfig
) -> RecordingFeatureResult:
    """Validate Task 5 input and compute one immutable feature record per scene."""
    validate_scene_graph(result)
    index = _index_feature_records(result)
    return RecordingFeatureResult(
        result.source,
        tuple(_compute_scene(result, scene, config, index) for scene in result.scenes),
    )
