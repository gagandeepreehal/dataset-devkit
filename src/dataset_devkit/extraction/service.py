"""One-recording extraction orchestration, stopping before validity policy or export."""

from __future__ import annotations

import re
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from dataset_devkit.extraction.camera import (
    AssociatedDecodedFrame,
    CameraDecoderSet,
    HevcDecoder,
    PyAvHevcDecoder,
)
from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.gnss import interpolate_gnss
from dataset_devkit.extraction.grid import select_camera_grid
from dataset_devkit.extraction.mcap_source import iter_camera_access_units, read_recording
from dataset_devkit.extraction.models import (
    CameraAccessUnit,
    EgoPose,
    ExtractedCameraSample,
    RecordingExtractionResult,
)
from dataset_devkit.extraction.staging import (
    create_staging_invocation,
    rollback_staging_invocation,
    stage_jpeg,
)


def _recording_id(path: Path) -> str:
    identifier = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-")
    return identifier or "recording"


@dataclass(frozen=True)
class _Submission:
    access_unit: CameraAccessUnit
    selected_target_ns: int | None


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
        samples: list[tuple[int, ExtractedCameraSample]] = []
        recording_id = _recording_id(path)
        invocation = create_staging_invocation(self.staging_root, recording_id)

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

        def consume(outputs: tuple[AssociatedDecodedFrame, ...]) -> None:
            for output in outputs:
                metadata = output.metadata
                if not isinstance(metadata, _Submission):
                    raise StructuralExtractionError("decoder returned invalid submission metadata")
                access = metadata.access_unit
                if output.image.size != (access.batch.width, access.batch.height):
                    raise StructuralExtractionError(
                        f"decoded camera[{access.frame.camera_index}] dimensions "
                        "differ from schema"
                    )
                if metadata.selected_target_ns is None:
                    continue
                staged = stage_jpeg(
                    self.staging_root,
                    invocation.directory_name,
                    access.frame.camera_index,
                    access.frame.camera_name,
                    access.frame.camera_timestamp_ns,
                    output.image,
                    (access.batch.width, access.batch.height),
                    batch_ordinal=access.batch_ordinal,
                    invocation=invocation,
                )
                samples.append(
                    (
                        output.submission_index,
                        ExtractedCameraSample(
                            metadata.selected_target_ns,
                            access.batch.rec_timestamp_ns,
                            access.frame.camera_timestamp_ns,
                            access.frame.camera_index,
                            access.frame.camera_name,
                            staged,
                            pose_for(access.frame.camera_timestamp_ns),
                        ),
                    )
                )

        try:
            with CameraDecoderSet(len(first_batch.frames), self.decoder_factory) as decoders:
                for access in iter_camera_access_units(path, self.camera_topic, recording):
                    consume(
                        decoders.submit(
                            access.frame.camera_index,
                            access.payload,
                            _Submission(
                                access,
                                target_by_batch.get(access.batch.rec_timestamp_ns),
                            ),
                        )
                    )
                consume(decoders.finish())
        except Exception:
            with suppress(StructuralExtractionError):
                rollback_staging_invocation(invocation)
            raise
        ordered_samples = tuple(sample for _, sample in sorted(samples, key=lambda item: item[0]))
        return RecordingExtractionResult(
            path.resolve(),
            invocation.path,
            recording.camera_batches,
            tuple(recording.gnss_samples),
            selection,
            ordered_samples,
            poses,
            recording.timestamp_observations,
        )
