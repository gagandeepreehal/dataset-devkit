"""Deterministic Task 8 nuScenes export into an empty staging dataroot."""

from __future__ import annotations

import hashlib
import math
import os
import stat
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid5

from PIL import Image

from dataset_devkit.config import GlobalConfig, ScenariosConfig, SplitConfig
from dataset_devkit.extraction.models import CameraCalibration, EgoPose, StagedImage
from dataset_devkit.features import SceneFeatures
from dataset_devkit.identifiers import validate_safe_segment
from dataset_devkit.provenance import canonical_hash, canonical_json
from dataset_devkit.scenario_selection import ScenarioSelectionResult, validate_scenario_selection
from dataset_devkit.scene_models import (
    RecordingSceneResult,
    SampleDataRecord,
    SampleRecord,
    SceneRecord,
)
from dataset_devkit.scenes import validate_scene_graph
from dataset_devkit.split import SceneSplitResult, split_extension_payload, validate_scene_split

NUSCENES_VERSION = "v1.0-trainval"
OFFICIAL_TABLES = (
    "category",
    "attribute",
    "visibility",
    "instance",
    "sensor",
    "calibrated_sensor",
    "ego_pose",
    "log",
    "scene",
    "sample",
    "sample_data",
    "sample_annotation",
    "map",
)
_EMPTY_TABLES = frozenset(
    {"category", "attribute", "visibility", "instance", "sample_annotation"}
)


@dataclass(frozen=True)
class ExportEvidence:
    """Validated upstream evidence required by the standalone exporter."""

    features_population: Sequence[SceneFeatures]
    scenarios_config: ScenariosConfig
    selection: ScenarioSelectionResult
    graphs: Sequence[RecordingSceneResult]
    split_config: SplitConfig
    split: SceneSplitResult
    resolved_config: GlobalConfig
    content_manifest: object


@dataclass(frozen=True)
class ExportResult:
    dataroot: Path
    version: str
    scene_count: int
    sample_count: int
    sample_data_count: int
    image_count: int
    content_fingerprint: str


def _jsonable(value: object) -> object:
    if isinstance(value, (Path, UUID)):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _token(namespace: UUID, kind: str, identity: object) -> str:
    return uuid5(namespace, f"dataset-devkit:{kind}:{canonical_json(identity)}").hex


def _microseconds(timestamp_ns: int) -> int:
    if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int) or timestamp_ns < 0:
        raise ValueError("timestamps must be nonnegative integer nanoseconds")
    return timestamp_ns // 1_000


def _normalized_channel(channel: str) -> str:
    validate_safe_segment(channel)
    if not channel[0].isalnum() or any(not (ch.isalnum() or ch in "_-") for ch in channel):
        raise ValueError(f"unsafe camera channel {channel!r}")
    body = channel.upper().replace("-", "_")
    normalized = body if body.startswith("CAM_") else f"CAM_{body}"
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    validate_safe_segment(normalized)
    return normalized


def _finite_vector(value: Sequence[float] | None, length: int, label: str) -> list[float]:
    if value is None or len(value) != length or not all(math.isfinite(item) for item in value):
        raise ValueError(f"{label} must contain {length} finite values")
    return [float(item) for item in value]


def _normalized_quaternion(value: Sequence[float] | None, label: str) -> list[float]:
    numbers = _finite_vector(value, 4, label)
    norm = math.sqrt(sum(item * item for item in numbers))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError(f"{label} has zero or invalid norm")
    return [item / norm for item in numbers]


def _rotation_vector_quaternion(value: Sequence[float]) -> list[float]:
    vector = _finite_vector(value, 3, "camera rotation vector")
    angle = math.sqrt(sum(item * item for item in vector))
    if angle == 0:
        return [1.0, 0.0, 0.0, 0.0]
    half = angle / 2.0
    scale = math.sin(half) / angle
    return _normalized_quaternion(
        [math.cos(half), vector[0] * scale, vector[1] * scale, vector[2] * scale],
        "camera rotation quaternion",
    )


def _calibration_value(
    calibration: CameraCalibration, width: int, height: int
) -> dict[str, object]:
    intrinsic = calibration.intrinsic
    values = (
        intrinsic.focal_length_x,
        intrinsic.focal_length_y,
        intrinsic.optical_center_x,
        intrinsic.optical_center_y,
        intrinsic.skew,
        intrinsic.width,
        intrinsic.height,
    )
    if not all(math.isfinite(item) for item in values):
        raise ValueError("camera intrinsic contains non-finite values")
    if intrinsic.width != width or intrinsic.height != height or width <= 0 or height <= 0:
        raise ValueError("camera calibration and image dimensions differ")
    translation = _finite_vector(calibration.extrinsic.translation_vector, 3, "camera translation")
    return {
        "translation": translation,
        "rotation": _rotation_vector_quaternion(calibration.extrinsic.rotation_vector),
        "camera_intrinsic": [
            [
                float(intrinsic.focal_length_x),
                float(intrinsic.skew),
                float(intrinsic.optical_center_x),
            ],
            [0.0, float(intrinsic.focal_length_y), float(intrinsic.optical_center_y)],
            [0.0, 0.0, 1.0],
        ],
    }


def _pose_value(pose: EgoPose) -> tuple[list[float], list[float]]:
    if not pose.available:
        raise ValueError("camera ego pose is unavailable")
    return (
        _finite_vector(pose.translation_xyz_m, 3, "ego pose translation"),
        _normalized_quaternion(pose.rotation_wxyz, "ego pose rotation"),
    )


def _read_verified_jpeg(staged: StagedImage) -> tuple[bytes, str]:
    path = staged.path
    try:
        before_lstat = path.lstat()
    except OSError as error:
        raise ValueError(f"cannot inspect staged image {path}") from error
    if stat.S_ISLNK(before_lstat.st_mode) or not stat.S_ISREG(before_lstat.st_mode):
        raise ValueError("staged image must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("staged image cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("staged image must be a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise ValueError("staged image changed while it was read")
    if staged.device is not None and staged.device != before.st_dev:
        raise ValueError("staged image device evidence differs")
    if staged.inode is not None and staged.inode != before.st_ino:
        raise ValueError("staged image inode evidence differs")
    data = b"".join(chunks)
    digest = hashlib.sha256(data).hexdigest()
    if staged.size is not None and staged.size != len(data):
        raise ValueError("staged image size evidence differs")
    if staged.sha256 is not None and staged.sha256 != digest:
        raise ValueError("staged image hash evidence differs")
    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            if image.format != "JPEG" or image.size != (staged.width, staged.height):
                raise ValueError("staged image JPEG dimensions differ")
    except (OSError, ValueError) as error:
        raise ValueError("staged image is not the expected JPEG") from error
    return data, digest


def _write_json(path: Path, value: object) -> bytes:
    content = (canonical_json(value) + "\n").encode("utf-8")
    _write_bytes(path, content)
    return content


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error
    try:
        offset = 0
        while offset < len(content):
            offset += os.write(descriptor, content[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _compatibility_mask() -> bytes:
    stream = BytesIO()
    Image.new("L", (1, 1), 0).save(stream, format="PNG", optimize=False)
    return stream.getvalue()


def _copy_image(root: Path, relative: str, staged: StagedImage) -> str:
    parts = relative.split("/")
    if len(parts) != 3 or parts[0] != "samples":
        raise ValueError("unsafe exported image filename")
    for part in parts:
        validate_safe_segment(part)
    content, digest = _read_verified_jpeg(staged)
    destination = root.joinpath(*parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FileExistsError(f"refusing image destination collision at {relative}") from error
    try:
        offset = 0
        while offset < len(content):
            offset += os.write(descriptor, content[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    copied = destination.read_bytes()
    if len(copied) != len(content) or hashlib.sha256(copied).hexdigest() != digest:
        raise ValueError("copied image verification failed")
    with Image.open(BytesIO(copied)) as image:
        image.load()
        if image.format != "JPEG" or image.size != (staged.width, staged.height):
            raise ValueError("copied image dimensions differ")
    return digest


def _validate_boundary(evidence: ExportEvidence) -> None:
    if evidence.resolved_config.scenarios != evidence.scenarios_config:
        raise ValueError("resolved global scenario config differs from export evidence")
    if evidence.resolved_config.split != evidence.split_config:
        raise ValueError("resolved global split config differs from export evidence")
    configured_paths = (
        evidence.resolved_config.azure.blob_list,
        evidence.resolved_config.paths.work_dir,
        evidence.resolved_config.paths.cache_dir,
        evidence.resolved_config.paths.output_dir,
        evidence.resolved_config.annotations.path,
        evidence.resolved_config.quarantine.directory,
    )
    if any(not path.is_absolute() for path in configured_paths):
        raise ValueError("global config paths must be fully resolved before export")
    validate_scenario_selection(
        evidence.selection, list(evidence.features_population), evidence.scenarios_config
    )
    validate_scene_split(
        evidence.split,
        evidence.selection,
        evidence.features_population,
        evidence.scenarios_config,
        evidence.graphs,
        evidence.split_config,
    )
    for graph in evidence.graphs:
        validate_scene_graph(graph)
    selected = {
        (item.scene_token, item.source_digest) for item in evidence.split.assignments
    }
    feature_selected = {
        (item.scene_token, item.source.digest) for item in evidence.selection.selected_scenes
    }
    if selected != feature_selected:
        raise ValueError("selected feature and split scene population differs")
    config_namespace = evidence.resolved_config.scenes.dataset_namespace
    if any(graph.dataset_namespace != config_namespace for graph in evidence.graphs):
        raise ValueError("scene graph namespace differs from resolved config")
    # Ensure the supplied content manifest is canonicalizable and finite.
    canonical_json(evidence.content_manifest)


def _selected_records(
    evidence: ExportEvidence,
) -> tuple[
    tuple[tuple[RecordingSceneResult, SceneRecord], ...],
    dict[str, SampleRecord],
    dict[str, tuple[SampleDataRecord, ...]],
]:
    selected = {
        (item.scene_token, item.source_digest) for item in evidence.split.assignments
    }
    scenes: list[tuple[RecordingSceneResult, SceneRecord]] = []
    samples: dict[str, SampleRecord] = {}
    data_lists: dict[str, list[SampleDataRecord]] = defaultdict(list)
    for graph in sorted(evidence.graphs, key=lambda item: item.source.digest):
        for scene in graph.scenes:
            if (scene.token, graph.source.digest) in selected:
                scenes.append((graph, scene))
                for sample in graph.samples:
                    if sample.scene_token == scene.token:
                        if sample.token in samples:
                            raise ValueError("selected samples contain duplicate tokens")
                        samples[sample.token] = sample
                for item in graph.sample_data:
                    if item.scene_token == scene.token:
                        data_lists[item.sample_token].append(item)
    if len(scenes) != len(selected):
        raise ValueError("selected scene graph coverage is missing or duplicated")
    return (
        tuple(sorted(scenes, key=lambda item: (item[0].source.digest, item[1].ordinal))),
        samples,
        {
            key: tuple(
                sorted(value, key=lambda item: (item.channel, item.timestamp_ns, item.token))
            )
            for key, value in data_lists.items()
        },
    )


def _assert_converted_chain(timestamps_ns: Sequence[int], label: str) -> None:
    converted = tuple(_microseconds(value) for value in timestamps_ns)
    if any(
        current <= previous
        for previous, current in zip(converted, converted[1:], strict=False)
    ):
        raise ValueError(f"{label} has a microsecond conversion collision or order hazard")


def export_dataset(staging_dataroot: str | Path, evidence: ExportEvidence) -> ExportResult:
    """Export selected scenes into a new or existing empty staging dataroot.

    Integer nanoseconds are converted to official nuScenes integer microseconds with
    floor division. Exact nanoseconds remain in extensions, and any chain whose order
    would collapse or reverse at microsecond precision is rejected.
    """
    _validate_boundary(evidence)
    root = Path(staging_dataroot)
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise ValueError("staging dataroot must be a real directory")
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError("staging dataroot must be empty; refusing overwrite")

    namespace = evidence.resolved_config.scenes.dataset_namespace
    selected_scenes, samples_by_token, data_by_sample = _selected_records(evidence)
    feature_by_identity = {
        (item.scene_token, item.source.digest): item
        for item in evidence.selection.selected_scenes
    }

    original_channels = sorted(
        {
            item.channel
            for graph, scene in selected_scenes
            for item in graph.sample_data
            if item.scene_token == scene.token
        }
    )
    channel_map = {channel: _normalized_channel(channel) for channel in original_channels}
    reverse: dict[str, list[str]] = defaultdict(list)
    for original, normalized in channel_map.items():
        reverse[normalized].append(original)
    collisions = {key: value for key, value in reverse.items() if len(value) > 1}
    if collisions:
        raise ValueError(f"camera channel normalization collision: {collisions}")

    sensors = [
        {
            "token": _token(namespace, "sensor", normalized),
            "channel": normalized,
            "modality": "camera",
        }
        for normalized in sorted(reverse)
    ]
    sensor_by_channel = {item["channel"]: item["token"] for item in sensors}
    logs: list[dict[str, object]] = []
    log_by_source: dict[str, str] = {}
    selected_graphs = sorted(
        {item[0].source.digest: item[0] for item in selected_scenes}.values(),
        key=lambda item: item.source.digest,
    )
    for graph in selected_graphs:
        token = _token(namespace, "log", graph.source.to_dict())
        log_by_source[graph.source.digest] = token
        logs.append(
            {
                "token": token,
                "logfile": graph.source.blob_path,
                "vehicle": "",
                "date_captured": "",
                "location": "",
            }
        )

    calibrations: list[dict[str, object]] = []
    calibration_tokens: dict[tuple[str, str, str], str] = {}
    official_scenes: list[dict[str, object]] = []
    official_samples: list[dict[str, object]] = []
    official_data: list[dict[str, object]] = []
    poses: list[dict[str, object]] = []
    gnss_extension: list[dict[str, object]] = []
    validity_extension: list[dict[str, object]] = []
    tags_extension: list[dict[str, object]] = []
    annotation_scene_refs: list[dict[str, object]] = []
    image_count = 0

    for graph, scene_value in selected_scenes:
        scene = scene_value
        feature = feature_by_identity[(scene.token, graph.source.digest)]
        scene_samples = sorted(
            (item for item in samples_by_token.values() if item.scene_token == scene.token),
            key=lambda item: (item.timestamp_ns, item.token),
        )
        if not scene_samples or len(scene_samples) != scene.nbr_samples:
            raise ValueError("selected scene sample count differs")
        _assert_converted_chain([item.timestamp_ns for item in scene_samples], "sample chain")
        official_scenes.append(
            {
                "token": scene.token,
                "name": scene.name,
                "description": "",
                "log_token": log_by_source[graph.source.digest],
                "nbr_samples": len(scene_samples),
                "first_sample_token": scene_samples[0].token,
                "last_sample_token": scene_samples[-1].token,
            }
        )
        for index, sample in enumerate(scene_samples):
            official_samples.append(
                {
                    "token": sample.token,
                    "timestamp": _microseconds(sample.timestamp_ns),
                    "prev": scene_samples[index - 1].token if index else "",
                    "next": (
                        scene_samples[index + 1].token
                        if index + 1 < len(scene_samples)
                        else ""
                    ),
                    "scene_token": scene.token,
                }
            )

        scene_data = [item for sample in scene_samples for item in data_by_sample[sample.token]]
        by_channel: dict[str, list[SampleDataRecord]] = defaultdict(list)
        for item in scene_data:
            by_channel[item.channel].append(item)
        for channel, chain in by_channel.items():
            chain.sort(key=lambda item: (item.timestamp_ns, item.token))
            _assert_converted_chain(
                [item.timestamp_ns for item in chain], f"camera {channel} chain"
            )
            for index, item in enumerate(chain):
                if item.calibration is None:
                    raise ValueError("camera calibration is absent")
                if item.timestamp_ns != item.ego_pose.timestamp_ns:
                    raise ValueError("ego pose timestamp differs from real camera timestamp")
                calibration_value = _calibration_value(
                    item.calibration, item.staged_image.width, item.staged_image.height
                )
                calibration_identity = canonical_hash(calibration_value)
                key = (graph.source.digest, channel, calibration_identity)
                calibration_token = calibration_tokens.get(key)
                if calibration_token is None:
                    calibration_token = _token(namespace, "calibrated-sensor", key)
                    calibration_tokens[key] = calibration_token
                    calibrations.append(
                        {
                            "token": calibration_token,
                            "sensor_token": sensor_by_channel[channel_map[channel]],
                            **calibration_value,
                        }
                    )
                pose_translation, pose_rotation = _pose_value(item.ego_pose)
                pose_token = _token(namespace, "ego-pose", [item.token, item.timestamp_ns])
                poses.append(
                    {
                        "token": pose_token,
                        "timestamp": _microseconds(item.timestamp_ns),
                        "rotation": pose_rotation,
                        "translation": pose_translation,
                    }
                )
                relative = f"samples/{channel_map[channel]}/{item.token}.jpg"
                image_sha = _copy_image(root, relative, item.staged_image)
                image_count += 1
                official_data.append(
                    {
                        "token": item.token,
                        "sample_token": item.sample_token,
                        "ego_pose_token": pose_token,
                        "calibrated_sensor_token": calibration_token,
                        "timestamp": _microseconds(item.timestamp_ns),
                        "fileformat": "jpg",
                        "is_key_frame": True,
                        "height": item.staged_image.height,
                        "width": item.staged_image.width,
                        "filename": relative,
                        "prev": chain[index - 1].token if index else "",
                        "next": chain[index + 1].token if index + 1 < len(chain) else "",
                    }
                )
                interpolation = item.ego_pose.interpolation
                gnss_extension.append(
                    {
                        "sample_data_token": item.token,
                        "scene_token": scene.token,
                        "source_digest": graph.source.digest,
                        "original_channel": channel,
                        "normalized_channel": channel_map[channel],
                        "timestamp_ns": item.timestamp_ns,
                        "image_sha256": image_sha,
                        "available": interpolation.available,
                        "latitude_deg": interpolation.latitude_deg,
                        "longitude_deg": interpolation.longitude_deg,
                        "height_m": interpolation.height_m,
                        "quaternion_wxyz": _jsonable(interpolation.quaternion_wxyz),
                        "fraction": interpolation.fraction,
                        "sync_gap_before_ns": interpolation.sync_gap_before_ns,
                        "sync_gap_after_ns": interpolation.sync_gap_after_ns,
                        "source_validity": _jsonable(interpolation.source_validity),
                        "position_uncertainty": _jsonable(interpolation.position_uncertainty),
                        "orientation_uncertainty": _jsonable(interpolation.orientation_uncertainty),
                        "before": _jsonable(interpolation.before),
                        "after": _jsonable(interpolation.after),
                    }
                )
        validity_extension.append(
            {
                "scene_token": scene.token,
                "source_digest": graph.source.digest,
                "scene_valid_ratio": feature.scene_valid_ratio,
                "source_gnss_valid_ratio": feature.source_gnss_valid_ratio,
                "camera_coverage_ratio": feature.camera_coverage_ratio,
                "camera_coverage_by_channel": _jsonable(feature.camera_coverage_by_channel),
                "max_abs_sync_error_ms": feature.max_abs_sync_error_ms,
                "mean_abs_sync_error_ms": feature.mean_abs_sync_error_ms,
                "sample_data": [
                    {
                        "sample_data_token": item.token,
                        "timestamp_ns": item.timestamp_ns,
                        "grid_signed_sync_error_ns": item.grid_signed_sync_error_ns,
                        "camera_signed_sync_error_ns": item.camera_signed_sync_error_ns,
                        "gnss_source_validity": list(item.gnss_source_validity),
                    }
                    for item in sorted(scene_data, key=lambda value: value.token)
                ],
                "samples": [
                    {
                        "sample_token": item.token,
                        "timestamp_ns": item.timestamp_ns,
                        "grid_timestamp_ns": item.grid_timestamp_ns,
                        "batch_timestamp_ns": item.batch_timestamp_ns,
                    }
                    for item in scene_samples
                ],
            }
        )
        tags_extension.append(
            {
                "scene_token": scene.token,
                "source_digest": graph.source.digest,
                "computed_tags": list(feature.computed_tags),
            }
        )
        annotation_scene_refs.append(
            {
                "scene_token": scene.token,
                "source_digest": graph.source.digest,
                "human_labels": list(feature.human_labels),
                "annotation_refs": list(scene.annotation_refs),
                "annotation_window_ref": scene.annotation_window_ref,
            }
        )

    official_scenes.sort(key=lambda item: str(item["token"]))
    official_samples.sort(key=lambda item: str(item["token"]))
    official_data.sort(key=lambda item: str(item["token"]))
    poses.sort(key=lambda item: str(item["token"]))
    calibrations.sort(key=lambda item: str(item["token"]))
    gnss_extension.sort(key=lambda item: str(item["sample_data_token"]))
    validity_extension.sort(key=lambda item: str(item["scene_token"]))
    tags_extension.sort(key=lambda item: str(item["scene_token"]))
    annotation_scene_refs.sort(key=lambda item: str(item["scene_token"]))

    tables: dict[str, object] = {name: [] for name in _EMPTY_TABLES}
    compatibility_mask_filename = "maps/dataset-devkit-loader-compatibility.png"
    compatibility_map = {
        "token": _token(
            namespace,
            "loader-compatibility-map",
            sorted(log_by_source.values()),
        ),
        "log_tokens": sorted(log_by_source.values()),
        "category": "compatibility_scaffold",
        "filename": compatibility_mask_filename,
    }
    tables.update(
        sensor=sensors,
        calibrated_sensor=calibrations,
        ego_pose=poses,
        log=logs,
        scene=official_scenes,
        sample=official_samples,
        sample_data=official_data,
        map=[compatibility_map],
    )
    _write_bytes(root / compatibility_mask_filename, _compatibility_mask())
    version_dir = root / NUSCENES_VERSION
    for name in OFFICIAL_TABLES:
        _write_json(version_dir / f"{name}.json", tables[name])

    selected_tokens_by_source: dict[str, set[str]] = defaultdict(set)
    for selected_graph, selected_scene in selected_scenes:
        selected_tokens_by_source[selected_graph.source.digest].add(selected_scene.token)
    recordings = [
        {
            "source": graph.source.to_dict(),
            "source_digest": graph.source.digest,
            "log_token": log_by_source[graph.source.digest],
            "channels": [
                {"original": channel, "normalized": channel_map[channel]}
                for channel in sorted(
                    {
                        item.channel
                        for item in graph.sample_data
                        if item.scene_token in selected_tokens_by_source[graph.source.digest]
                    }
                )
            ],
        }
        for graph in selected_graphs
    ]
    selected_annotation_tokens = {
        token for _, scene in selected_scenes for token in scene.annotation_refs
    }
    selected_window_tokens = {
        scene.annotation_window_ref
        for _, scene in selected_scenes
        if scene.annotation_window_ref
    }
    annotation_payload = {
        "scenes": annotation_scene_refs,
        "records": [
            {"source_digest": graph.source.digest, **asdict(item)}
            for graph in sorted(evidence.graphs, key=lambda value: value.source.digest)
            for item in graph.annotations
            if item.token in selected_annotation_tokens
        ],
        "matches": [
            {"source_digest": graph.source.digest, **asdict(item)}
            for graph in sorted(evidence.graphs, key=lambda value: value.source.digest)
            for item in graph.annotation_matches
            if item.annotation_token in selected_annotation_tokens
        ],
        "windows": [
            {"source_digest": graph.source.digest, **asdict(item)}
            for graph in sorted(evidence.graphs, key=lambda value: value.source.digest)
            for item in graph.annotation_windows
            if item.token in selected_window_tokens
        ],
    }
    extensions = root / "mz_extensions"
    _write_json(extensions / "recordings.json", recordings)
    _write_json(extensions / "gnss.json", gnss_extension)
    _write_json(extensions / "validity.json", validity_extension)
    _write_json(
        extensions / "validation.json",
        {"schema_version": 1, "state": "not_run", "succeeded": False, "report": None},
    )
    _write_json(extensions / "tags.json", tags_extension)
    _write_json(extensions / "annotations.json", annotation_payload)
    _write_json(extensions / "split.json", split_extension_payload(evidence.split))
    _write_json(
        extensions / "config.json", evidence.resolved_config.model_dump(mode="json")
    )
    manifest_bytes = _write_json(extensions / "content_manifest.json", evidence.content_manifest)
    return ExportResult(
        root.resolve(),
        NUSCENES_VERSION,
        len(official_scenes),
        len(official_samples),
        len(official_data),
        image_count,
        hashlib.sha256(manifest_bytes).hexdigest(),
    )
