"""Camera-batch validation and persistent HEVC decoding."""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Protocol

from PIL import Image

from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.models import CameraStructure, RawCameraBatch


class HevcDecoder(Protocol):
    def decode(
        self, payload: bytes, pts: int, time_base: Fraction
    ) -> list[DecoderOutput]: ...

    def flush(self) -> list[DecoderOutput]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class DecoderOutput:
    pts: int | None
    image: Image.Image


@dataclass(frozen=True)
class AssociatedDecodedFrame:
    submission_index: int
    camera_index: int
    metadata: Any
    image: Image.Image


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
    if not starts or starts[0][0] != 0:
        raise StructuralExtractionError(
            "Annex-B access unit has bytes before first start code"
        )
    units: list[bytes] = []
    for unit_index, (start, prefix_length) in enumerate(starts):
        end = starts[unit_index + 1][0] if unit_index + 1 < len(starts) else len(payload)
        unit = payload[start + prefix_length : end]
        if not unit:
            raise StructuralExtractionError("Annex-B access unit contains an empty NAL unit")
        units.append(unit)
    return units


def validate_annex_b_hevc_access_unit(payload: bytes) -> None:
    """Require Annex-B framing and at least one HEVC VCL NAL unit."""
    units = _annex_b_nal_units(payload)
    if not units:
        raise StructuralExtractionError("camera payload is not a nonempty Annex-B access unit")
    for unit in units:
        if len(unit) < 2:
            raise StructuralExtractionError("HEVC NAL unit lacks its two-byte header")
        if unit[0] & 0x80:
            raise StructuralExtractionError("HEVC NAL forbidden_zero_bit must be zero")
        if unit[1] & 0x07 == 0:
            raise StructuralExtractionError("HEVC NAL nuh_temporal_id_plus1 must be nonzero")
    if not any(((unit[0] >> 1) & 0x3F) <= 31 for unit in units):
        raise StructuralExtractionError("Annex-B access unit contains no VCL NAL unit")


def _finite(values: tuple[float, ...]) -> bool:
    return all(math.isfinite(value) for value in values)


def validate_camera_batch(
    batch: RawCameraBatch,
    prior_state: CameraStructure | None = None,
    *,
    allow_native_calibration_resolution: bool = False,
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
        intrinsic_values = (
            intrinsic.focal_length_x,
            intrinsic.focal_length_y,
            intrinsic.optical_center_x,
            intrinsic.optical_center_y,
            intrinsic.rmse,
            intrinsic.skew,
            *intrinsic.distortion_coeffs,
            intrinsic.width,
            intrinsic.height,
        )
        if not _finite(intrinsic_values):
            raise StructuralExtractionError("camera intrinsic values must be finite")
        if intrinsic.width <= 0 or intrinsic.height <= 0:
            raise StructuralExtractionError("camera intrinsic dimensions must be positive")
        if intrinsic.width != batch.width or intrinsic.height != batch.height:
            if not allow_native_calibration_resolution:
                raise StructuralExtractionError(
                    "camera intrinsic dimensions differ from batch"
                )
            if not math.isclose(
                intrinsic.width * batch.height,
                intrinsic.height * batch.width,
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                raise StructuralExtractionError(
                    "camera intrinsic and batch dimensions have different aspect ratios"
                )
        if len(extrinsic.rotation_vector) != 3 or len(extrinsic.translation_vector) != 3:
            raise StructuralExtractionError("camera extrinsic vectors must each have length three")
        if not _finite(extrinsic.rotation_vector + extrinsic.translation_vector):
            raise StructuralExtractionError("camera extrinsic values must be finite")
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
        self._finished = False
        self._next_pts = 0
        self._pending: list[dict[int, tuple[int, Any]]] = [
            {} for _ in range(camera_count)
        ]
        self._emitted: list[set[int]] = [set() for _ in range(camera_count)]
        self._submitted_counts = [0 for _ in range(camera_count)]
        self._decoded_counts = [0 for _ in range(camera_count)]

    def _cardinality_message(self, camera_index: int, detail: str) -> str:
        return (
            "decoder must produce exactly one decoded frame per HEVC access unit "
            f"for camera index {camera_index}: {detail}; "
            f"submitted={self._submitted_counts[camera_index]}, "
            f"decoded={self._decoded_counts[camera_index]}"
        )

    def _associate(
        self, camera_index: int, outputs: list[DecoderOutput]
    ) -> tuple[AssociatedDecodedFrame, ...]:
        pending = self._pending[camera_index]
        associated: list[AssociatedDecodedFrame] = []
        for output in outputs:
            pts = output.pts
            if pts is None:
                if len(outputs) == 1 and len(pending) == 1:
                    pts = next(iter(pending))
                else:
                    raise StructuralExtractionError(
                        "decoded HEVC frame has missing PTS and association is ambiguous"
                    )
            if pts in self._emitted[camera_index]:
                raise StructuralExtractionError(
                    self._cardinality_message(
                        camera_index, f"decoder emitted duplicate PTS {pts}"
                    )
                )
            submitted = pending.pop(pts, None)
            if submitted is None:
                raise StructuralExtractionError(
                    self._cardinality_message(
                        camera_index, f"decoder emitted unknown PTS {pts}"
                    )
                )
            submission_index, metadata = submitted
            self._emitted[camera_index].add(pts)
            self._decoded_counts[camera_index] += 1
            associated.append(
                AssociatedDecodedFrame(
                    submission_index,
                    camera_index,
                    metadata,
                    output.image.convert("RGB"),
                )
            )
        return tuple(associated)

    def submit(
        self, camera_index: int, payload: bytes, metadata: Any
    ) -> tuple[AssociatedDecodedFrame, ...]:
        if self._closed or self._finished:
            raise StructuralExtractionError("camera decoder set is closed")
        validate_annex_b_hevc_access_unit(payload)
        try:
            decoder = self._decoders[camera_index]
            pts = self._next_pts
            self._next_pts += 1
            self._pending[camera_index][pts] = (pts, metadata)
            self._submitted_counts[camera_index] += 1
            return self._associate(
                camera_index,
                decoder.decode(payload, pts, Fraction(1, 1_000_000_000)),
            )
        except StructuralExtractionError:
            self.close()
            raise
        except Exception as error:
            self.close()
            raise StructuralExtractionError(
                f"HEVC decoder failed for camera index {camera_index}: {error}"
            ) from error

    def finish(self) -> tuple[AssociatedDecodedFrame, ...]:
        if self._closed or self._finished:
            raise StructuralExtractionError("camera decoder set is closed")
        emitted: list[AssociatedDecodedFrame] = []
        try:
            for camera_index, decoder in enumerate(self._decoders):
                emitted.extend(self._associate(camera_index, decoder.flush()))
            for camera_index, pending in enumerate(self._pending):
                if pending or (
                    self._decoded_counts[camera_index]
                    != self._submitted_counts[camera_index]
                ):
                    raise StructuralExtractionError(
                        self._cardinality_message(
                            camera_index,
                            f"decoder is missing {len(pending)} frame output(s) after EOF flush",
                        )
                    )
            self._finished = True
            self.close()
            return tuple(emitted)
        except StructuralExtractionError:
            self.close()
            raise
        except Exception as error:
            self.close()
            raise StructuralExtractionError(f"HEVC decoder EOF flush failed: {error}") from error

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

    def decode(
        self, payload: bytes, pts: int, time_base: Fraction
    ) -> list[DecoderOutput]:
        import av

        packet = av.Packet(payload)
        packet.pts = pts
        packet.dts = pts
        packet.time_base = time_base
        return [
            DecoderOutput(frame.pts, frame.to_image().convert("RGB"))
            for frame in self._context.decode(packet)
        ]

    def flush(self) -> list[DecoderOutput]:
        return [
            DecoderOutput(frame.pts, frame.to_image().convert("RGB"))
            for frame in self._context.decode(None)
        ]

    def close(self) -> None:
        self._context = None
