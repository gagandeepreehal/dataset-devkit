"""Typed, immutable records exchanged by native extraction stages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from dataset_devkit.extraction.grid import GridSelection
from dataset_devkit.extraction.uncertainty import bounded_freeze


@dataclass(frozen=True)
class CameraIntrinsic:
    focal_length_x: float
    focal_length_y: float
    optical_center_x: float
    optical_center_y: float
    rmse: float
    skew: float
    distortion_coeffs: tuple[float, ...]
    width: float
    height: float


@dataclass(frozen=True)
class CameraExtrinsic:
    rotation_vector: tuple[float, ...]
    translation_vector: tuple[float, ...]


@dataclass(frozen=True)
class CameraCalibration:
    intrinsic: CameraIntrinsic
    extrinsic: CameraExtrinsic


@dataclass(frozen=True)
class RawCameraFrame:
    camera_index: int
    camera_name: str
    camera_timestamp_ns: int
    calibration: CameraCalibration
    camera_timestamp_source: str = "camera_timestamp"


@dataclass(frozen=True)
class RawCameraBatch:
    rec_timestamp_ns: int
    recorded_timestamp_ns: int
    frame_id: int
    rec_frame_id: int
    format: str
    width: int
    height: int
    frames: tuple[RawCameraFrame, ...]


@dataclass(frozen=True)
class CameraStructure:
    names: tuple[str, ...]
    width: int
    height: int
    calibrations: tuple[CameraCalibration, ...]


@dataclass(frozen=True)
class TimestampObservation:
    stream: str
    previous_timestamp_ns: int
    current_timestamp_ns: int
    delta_ns: int


@dataclass(frozen=True)
class SourceIdentity:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class RawRecording:
    source_identity: SourceIdentity
    camera_batches: tuple[RawCameraBatch, ...]
    gnss_samples: Sequence[GnssSample]
    timestamp_observations: tuple[TimestampObservation, ...]


@dataclass(frozen=True)
class CameraAccessUnit:
    batch_ordinal: int
    batch: RawCameraBatch
    frame: RawCameraFrame
    payload: bytes


@dataclass(frozen=True)
class StagedImage:
    camera_index: int
    camera_name: str
    timestamp_ns: int
    path: Path
    width: int
    height: int
    device: int | None = None
    inode: int | None = None
    size: int | None = None
    sha256: str | None = None
    invocation_root: Path | None = None
    root_relative_path: str | None = None
    directory_device: int | None = None
    directory_inode: int | None = None
    directory_chain_identities: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class GnssSample:
    timestamp_ns: int
    rec_timestamp_ns: int
    is_valid: bool
    latitude_deg: float
    longitude_deg: float
    height_m: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    position_uncertainty: Mapping[str, Any]
    orientation_uncertainty: Mapping[str, Any]
    raw_identifiers: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_uncertainty",
            bounded_freeze(
                self.position_uncertainty, root_path="position_uncertainty"
            ),
        )
        object.__setattr__(
            self,
            "orientation_uncertainty",
            bounded_freeze(
                self.orientation_uncertainty, root_path="orientation_uncertainty"
            ),
        )
        object.__setattr__(
            self,
            "raw_identifiers",
            bounded_freeze(self.raw_identifiers, root_path="raw_identifiers"),
        )


type PoseFrame = Literal["web_mercator_v1", "recording_local_enu_v1"]


@dataclass(frozen=True)
class PrivacyGnssSample:
    recording_offset_ns: int
    is_valid: bool
    local_east_m: float | None
    local_north_m: float | None
    local_up_m: float | None
    roll_rad: float | None
    pitch_rad: float | None
    yaw_rad: float | None
    published_east_m: float | None
    published_north_m: float | None
    latitude_deg: float | None
    longitude_deg: float | None
    height_m: float | None
    position_uncertainty: Mapping[str, Any]
    orientation_uncertainty: Mapping[str, Any]
    quality: Mapping[str, Any]
    local_enu_origin_offset_ns: Literal[0]
    published_horizontal_resolution_m: float | None

    @property
    def timestamp_ns(self) -> int:
        """Compatibility view of the authoritative recording-relative offset."""
        return self.recording_offset_ns

    @property
    def rec_timestamp_ns(self) -> int:
        return self.recording_offset_ns

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_uncertainty",
            bounded_freeze(self.position_uncertainty, root_path="position_uncertainty"),
        )
        object.__setattr__(
            self,
            "orientation_uncertainty",
            bounded_freeze(
                self.orientation_uncertainty, root_path="orientation_uncertainty"
            ),
        )
        object.__setattr__(
            self,
            "quality",
            bounded_freeze(self.quality, root_path="quality"),
        )


@dataclass(frozen=True)
class GnssInterpolation:
    timestamp_ns: int
    available: bool
    before: GnssSample | PrivacyGnssSample | None
    after: GnssSample | PrivacyGnssSample | None
    fraction: float | None
    sync_gap_before_ns: int | None
    sync_gap_after_ns: int | None
    latitude_deg: float | None = None
    longitude_deg: float | None = None
    height_m: float | None = None
    quaternion_wxyz: tuple[float, float, float, float] | None = None
    projected_x_m: float | None = None
    projected_y_m: float | None = None
    position_uncertainty: Mapping[str, Any] = field(default_factory=dict)
    orientation_uncertainty: Mapping[str, Any] = field(default_factory=dict)
    source_validity: tuple[bool, bool] | None = None
    position_uncertainty_uninterpolated_paths: tuple[str, ...] = ()
    orientation_uncertainty_uninterpolated_paths: tuple[str, ...] = ()
    translation_xyz_m: tuple[float, float, float] | None = None
    pose_frame: PoseFrame | None = None
    published_east_m: float | None = None
    published_north_m: float | None = None
    published_horizontal_resolution_m: float | None = None
    quality: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_uncertainty",
            bounded_freeze(
                self.position_uncertainty, root_path="position_uncertainty"
            ),
        )
        object.__setattr__(
            self,
            "orientation_uncertainty",
            bounded_freeze(
                self.orientation_uncertainty, root_path="orientation_uncertainty"
            ),
        )
        object.__setattr__(
            self,
            "quality",
            bounded_freeze(self.quality, root_path="quality"),
        )


@dataclass(frozen=True)
class EgoPose:
    timestamp_ns: int
    available: bool
    translation_xyz_m: tuple[float, float, float] | None
    rotation_wxyz: tuple[float, float, float, float] | None
    interpolation: GnssInterpolation


@dataclass(frozen=True)
class ExtractedCameraSample:
    grid_target_timestamp_ns: int
    batch_timestamp_ns: int
    camera_timestamp_ns: int
    camera_index: int
    camera_name: str
    staged_image: StagedImage
    ego_pose: EgoPose
    calibration: CameraCalibration | None = None


@dataclass(frozen=True)
class RecordingExtractionResult:
    source_path: Path
    staging_root: Path
    camera_batches: tuple[RawCameraBatch, ...]
    gnss_samples: tuple[GnssSample | PrivacyGnssSample, ...]
    selected_grid: GridSelection
    samples: tuple[ExtractedCameraSample, ...]
    ego_poses_by_timestamp: Mapping[int, EgoPose]
    timestamp_observations: tuple[TimestampObservation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ego_poses_by_timestamp",
            MappingProxyType(dict(self.ego_poses_by_timestamp)),
        )
