"""One-recording extraction orchestration, stopping before validity policy or export."""

from __future__ import annotations

import re
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

from dataset_devkit.extraction.camera import CameraDecoderSet, HevcDecoder, PyAvHevcDecoder
from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.gnss import interpolate_gnss
from dataset_devkit.extraction.grid import select_camera_grid
from dataset_devkit.extraction.mcap_source import read_recording
from dataset_devkit.extraction.models import (
    EgoPose,
    ExtractedCameraSample,
    RecordingExtractionResult,
)
from dataset_devkit.extraction.staging import stage_jpeg


def _recording_id(path: Path) -> str:
    identifier = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-")
    return identifier or "recording"


class RecordingExtractor:
    """Read one local MCAP, decode its stream in order, and stage selected JPEGs."""

    def __init__(
        self,
        *,
        camera_topic: str,
        gnss_topic: str,
        target_fps: Fraction | float | str,
        tolerance_ns: int,
        staging_root: Path,
        decoder_factory: Callable[[], HevcDecoder] = PyAvHevcDecoder,
    ) -> None:
        self.camera_topic = camera_topic
        self.gnss_topic = gnss_topic
        self.target_fps = (
            target_fps if isinstance(target_fps, Fraction) else Fraction(str(target_fps))
        )
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if tolerance_ns < 0:
            raise ValueError("tolerance_ns must be nonnegative")
        self.tolerance_ns = tolerance_ns
        self.staging_root = staging_root
        self.decoder_factory = decoder_factory

    def extract(self, path: Path) -> RecordingExtractionResult:
        recording = read_recording(path, self.camera_topic, self.gnss_topic)
        selection = select_camera_grid(
            [batch.rec_timestamp_ns for batch in recording.camera_batches],
            self.target_fps,
            self.tolerance_ns,
        )
        target_by_batch = {
            selected.batch_timestamp_ns: selected.target_timestamp_ns
            for selected in selection.entries
        }
        first_batch = recording.camera_batches[0]
        poses: dict[int, EgoPose] = {}
        samples: list[ExtractedCameraSample] = []

        def pose_for(timestamp_ns: int) -> EgoPose:
            existing = poses.get(timestamp_ns)
            if existing is not None:
                return existing
            interpolation = interpolate_gnss(recording.gnss_samples, timestamp_ns)
            if interpolation.available:
                assert interpolation.projected_x_m is not None
                assert interpolation.projected_y_m is not None
                assert interpolation.height_m is not None
                assert interpolation.quaternion_wxyz is not None
                pose = EgoPose(
                    timestamp_ns,
                    True,
                    (
                        interpolation.projected_x_m,
                        interpolation.projected_y_m,
                        interpolation.height_m,
                    ),
                    interpolation.quaternion_wxyz,
                    interpolation,
                )
            else:
                pose = EgoPose(timestamp_ns, False, None, None, interpolation)
            poses[timestamp_ns] = pose
            return pose

        with CameraDecoderSet(len(first_batch.frames), self.decoder_factory) as decoders:
            for batch in recording.camera_batches:
                selected_target = target_by_batch.get(batch.rec_timestamp_ns)
                for frame in batch.frames:
                    image = decoders.decode(frame.camera_index, frame.payload)
                    if image.size != (batch.width, batch.height):
                        raise StructuralExtractionError(
                            f"decoded camera[{frame.camera_index}] dimensions differ from schema"
                        )
                    if selected_target is None:
                        continue
                    staged = stage_jpeg(
                        self.staging_root,
                        _recording_id(path),
                        frame.camera_index,
                        frame.camera_name,
                        frame.camera_timestamp_ns,
                        image,
                        (batch.width, batch.height),
                    )
                    samples.append(
                        ExtractedCameraSample(
                            selected_target,
                            batch.rec_timestamp_ns,
                            frame.camera_timestamp_ns,
                            frame.camera_index,
                            frame.camera_name,
                            staged,
                            pose_for(frame.camera_timestamp_ns),
                        )
                    )
        return RecordingExtractionResult(
            path.resolve(),
            recording.camera_batches,
            recording.gnss_samples,
            selection,
            tuple(samples),
            poses,
            recording.timestamp_observations,
        )
