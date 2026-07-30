"""Camera-batch validation and persistent HEVC decoding."""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import suppress
from typing import Any, Protocol

from PIL import Image

from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.models import CameraStructure, RawCameraBatch


class HevcDecoder(Protocol):
    def decode(self, payload: bytes) -> list[Image.Image]: ...

    def close(self) -> None: ...


def _annex_b_nal_units(payload: bytes) -> list[bytes]:
    starts: list[tuple[int, int]] = []
    index = 0
    while index <= len(payload) - 3:
        if payload[index : index + 4] == b"\x00\x00\x00\x01":
            starts.append((index, 4))
            index += 4
        elif payload[index : index + 3] == b"\x00\x00\x01":
            starts.append((index, 3))
            index += 3
        else:
            index += 1
    units: list[bytes] = []
    for unit_index, (start, prefix_length) in enumerate(starts):
        end = starts[unit_index + 1][0] if unit_index + 1 < len(starts) else len(payload)
        unit = payload[start + prefix_length : end]
        if unit:
            units.append(unit)
    return units


def validate_annex_b_hevc_access_unit(payload: bytes) -> None:
    """Require Annex-B framing and at least one HEVC VCL NAL unit."""
    units = _annex_b_nal_units(payload)
    if not units:
        raise StructuralExtractionError("camera payload is not a nonempty Annex-B access unit")
    if not any(len(unit) >= 2 and ((unit[0] >> 1) & 0x3F) <= 31 for unit in units):
        raise StructuralExtractionError("Annex-B access unit contains no VCL NAL unit")


def _finite(values: tuple[float, ...]) -> bool:
    return all(math.isfinite(value) for value in values)


def validate_camera_batch(
    batch: RawCameraBatch, prior_state: CameraStructure | None = None
) -> CameraStructure:
    """Validate structural camera invariants and recording-level stability."""
    if batch.format != "h265":
        raise StructuralExtractionError("camera message must use exact format 'h265'")
    if batch.width <= 0 or batch.height <= 0:
        raise StructuralExtractionError("camera batch dimensions must be positive")
    if not batch.frames:
        raise StructuralExtractionError("number_of_cameras must be greater than zero")
    names = tuple(frame.camera_name for frame in batch.frames)
    if any(not name.strip() for name in names) or len(names) != len(set(names)):
        raise StructuralExtractionError("camera names must be unique and nonblank")
    calibrations = tuple(frame.calibration for frame in batch.frames)
    for expected_index, frame in enumerate(batch.frames):
        if frame.camera_index != expected_index:
            raise StructuralExtractionError("camera arrays are not index aligned")
        intrinsic = frame.calibration.intrinsic
        extrinsic = frame.calibration.extrinsic
        if intrinsic.width != batch.width or intrinsic.height != batch.height:
            raise StructuralExtractionError("camera intrinsic dimensions differ from batch")
        intrinsic_values = (
            intrinsic.focal_length_x,
            intrinsic.focal_length_y,
            intrinsic.optical_center_x,
            intrinsic.optical_center_y,
            intrinsic.rmse,
            intrinsic.skew,
            *intrinsic.distortion_coeffs,
        )
        if not _finite(intrinsic_values):
            raise StructuralExtractionError("camera intrinsic values must be finite")
        if len(extrinsic.rotation_vector) != 3 or len(extrinsic.translation_vector) != 3:
            raise StructuralExtractionError("camera extrinsic vectors must each have length three")
        if not _finite(extrinsic.rotation_vector + extrinsic.translation_vector):
            raise StructuralExtractionError("camera extrinsic values must be finite")
        validate_annex_b_hevc_access_unit(frame.payload)
    current = CameraStructure(names, batch.width, batch.height, calibrations)
    if prior_state is not None:
        if names != prior_state.names:
            raise StructuralExtractionError("camera identity changed during recording")
        if (batch.width, batch.height) != (prior_state.width, prior_state.height):
            raise StructuralExtractionError("camera dimensions changed during recording")
        if calibrations != prior_state.calibrations:
            raise StructuralExtractionError("camera calibration changed during recording")
    return current


class CameraDecoderSet:
    """Own exactly one persistent decoder context for every camera index."""

    def __init__(self, camera_count: int, factory: Callable[[], HevcDecoder]) -> None:
        if camera_count <= 0:
            raise ValueError("camera_count must be positive")
        self._decoders: list[HevcDecoder] = []
        try:
            for _ in range(camera_count):
                self._decoders.append(factory())
        except Exception:
            for decoder in self._decoders:
                with suppress(Exception):
                    decoder.close()
            raise
        self._closed = False

    def decode(self, camera_index: int, payload: bytes) -> Image.Image:
        if self._closed:
            raise StructuralExtractionError("camera decoder set is closed")
        validate_annex_b_hevc_access_unit(payload)
        try:
            frames = self._decoders[camera_index].decode(payload)
            if len(frames) != 1:
                raise StructuralExtractionError(
                    "each HEVC access unit must yield exactly one decoded frame"
                )
            return frames[0].convert("RGB")
        except StructuralExtractionError:
            self.close()
            raise
        except Exception as error:
            self.close()
            raise StructuralExtractionError(
                f"HEVC decoder failed for camera index {camera_index}: {error}"
            ) from error

    def close(self) -> None:
        if not self._closed:
            for decoder in self._decoders:
                with suppress(Exception):
                    decoder.close()
            self._closed = True

    def __enter__(self) -> CameraDecoderSet:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class PyAvHevcDecoder:
    """Persistent software HEVC decoder backed by PyAV."""

    def __init__(self) -> None:
        import av

        self._context: Any = av.CodecContext.create("hevc", "r")

    def decode(self, payload: bytes) -> list[Image.Image]:
        import av

        return [
            frame.to_image().convert("RGB")
            for frame in self._context.decode(av.Packet(payload))
        ]

    def close(self) -> None:
        self._context = None
