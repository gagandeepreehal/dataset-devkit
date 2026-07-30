"""GNSS indexing, geodetic interpolation, attitude SLERP, and projection."""

from __future__ import annotations

import bisect
import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import cast, overload

from pyproj import Transformer

from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.models import GnssInterpolation, GnssSample
from dataset_devkit.extraction.uncertainty import bounded_leaf_items

Quaternion = tuple[float, float, float, float]
_WEB_MERCATOR = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


@dataclass(frozen=True)
class GnssIndex(Sequence[GnssSample]):
    """One validated timestamp-sorted GNSS index reusable across all camera queries."""

    samples: tuple[GnssSample, ...]
    timestamps: tuple[int, ...]

    def __iter__(self) -> Iterator[GnssSample]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    @overload
    def __getitem__(self, index: int) -> GnssSample: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[GnssSample, ...]: ...

    def __getitem__(self, index: int | slice) -> GnssSample | tuple[GnssSample, ...]:
        return self.samples[index]


def _normalize_quaternion(quaternion: Quaternion) -> Quaternion:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if not math.isfinite(norm) or norm == 0:
        raise StructuralExtractionError("orientation produced an invalid quaternion")
    normalized = tuple(value / norm for value in quaternion)
    return normalized  # type: ignore[return-value]


def euler_to_quaternion_wxyz(roll: float, pitch: float, yaw: float) -> Quaternion:
    """Return active right-handed fixed-axis XYZ qz*qy*qx, in (w,x,y,z) order."""
    if not all(math.isfinite(value) for value in (roll, pitch, yaw)):
        raise StructuralExtractionError("GNSS orientation values must be finite")
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return _normalize_quaternion(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


def quaternion_slerp_shortest(first: Quaternion, second: Quaternion, fraction: float) -> Quaternion:
    """SLERP normalized quaternions along the shortest path, including antipodal inputs."""
    if not 0 <= fraction <= 1 or not math.isfinite(fraction):
        raise ValueError("fraction must be finite and between zero and one")
    left = _normalize_quaternion(first)
    right = _normalize_quaternion(second)
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    if dot < 0:
        right = tuple(-value for value in right)  # type: ignore[assignment]
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return _normalize_quaternion(
            tuple(a + fraction * (b - a) for a, b in zip(left, right, strict=True))  # type: ignore[arg-type]
        )
    angle = math.acos(dot)
    denominator = math.sin(angle)
    left_weight = math.sin((1 - fraction) * angle) / denominator
    right_weight = math.sin(fraction * angle) / denominator
    return _normalize_quaternion(
        tuple(
            left_weight * a + right_weight * b
            for a, b in zip(left, right, strict=True)
        )  # type: ignore[arg-type]
    )


def index_gnss_samples(samples: Iterable[GnssSample]) -> GnssIndex:
    """Deterministically order by protobuf timestamp and reject duplicate ambiguity."""
    ordered = tuple(sorted(samples, key=lambda sample: sample.timestamp_ns))
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if previous.timestamp_ns == current.timestamp_ns:
            raise StructuralExtractionError("duplicate GNSS timestamp is structurally ambiguous")
    return GnssIndex(ordered, tuple(sample.timestamp_ns for sample in ordered))


def _linear(first: float, second: float, fraction: float) -> float:
    return first + (second - first) * fraction


_CANONICAL_PROTOBUF_INTEGER = re.compile(r"(?:0|-?[1-9][0-9]*)")
_MIN_PROTOBUF_INTEGER = -(2**63)
_MAX_PROTOBUF_INTEGER = 2**64 - 1
_MISSING = object()


def parse_numeric_uncertainty_leaf(value: object) -> float | None:
    """Parse numeric scalars and canonical protobuf-JSON int64/uint64 strings."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise StructuralExtractionError("uncertainty numeric value must be finite")
        return number
    if isinstance(value, str) and _CANONICAL_PROTOBUF_INTEGER.fullmatch(value):
        integer = int(value)
        if _MIN_PROTOBUF_INTEGER <= integer <= _MAX_PROTOBUF_INTEGER:
            number = float(integer)
            if math.isfinite(number):
                return number
    return None


def _numeric_paths(value: object, path: str) -> tuple[str, ...]:
    return tuple(
        leaf_path
        for leaf_path, leaf in bounded_leaf_items(value, root_path=path)
        if parse_numeric_uncertainty_leaf(leaf) is not None
    )


def _interpolate_uncertainty_value(
    first: object,
    second: object,
    fraction: float,
    path: str,
    uninterpolated: set[str],
) -> object:
    left_number = parse_numeric_uncertainty_leaf(first)
    right_number = parse_numeric_uncertainty_leaf(second)
    if left_number is not None and right_number is not None:
        return _linear(left_number, right_number, fraction)
    if left_number is not None or right_number is not None:
        uninterpolated.update(_numeric_paths(first, path))
        uninterpolated.update(_numeric_paths(second, path))
        return _MISSING
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        result: dict[str, object] = {}
        first_keys = {str(key): key for key in first}
        second_keys = {str(key): key for key in second}
        for key in sorted(first_keys.keys() | second_keys.keys()):
            child_path = f"{path}.{key}" if path else key
            if key not in first_keys:
                uninterpolated.update(
                    _numeric_paths(second[second_keys[key]], child_path)
                )
                continue
            if key not in second_keys:
                uninterpolated.update(
                    _numeric_paths(first[first_keys[key]], child_path)
                )
                continue
            interpolated = _interpolate_uncertainty_value(
                first[first_keys[key]],
                second[second_keys[key]],
                fraction,
                child_path,
                uninterpolated,
            )
            if interpolated is not _MISSING:
                result[key] = interpolated
        return result if result else _MISSING
    first_sequence = (
        isinstance(first, Sequence) and not isinstance(first, (str, bytes, bytearray))
    )
    second_sequence = (
        isinstance(second, Sequence) and not isinstance(second, (str, bytes, bytearray))
    )
    if first_sequence and second_sequence:
        left_items: tuple[object, ...] = tuple(cast(Sequence[object], first))
        right_items: tuple[object, ...] = tuple(cast(Sequence[object], second))
        if len(left_items) != len(right_items):
            uninterpolated.update(_numeric_paths(left_items, path))
            uninterpolated.update(_numeric_paths(right_items, path))
            return _MISSING
        result_items: list[object] = []
        for index, (left, right) in enumerate(
            zip(left_items, right_items, strict=True)
        ):
            interpolated = _interpolate_uncertainty_value(
                left,
                right,
                fraction,
                f"{path}[{index}]",
                uninterpolated,
            )
            if interpolated is _MISSING:
                uninterpolated.update(_numeric_paths(left_items, path))
                uninterpolated.update(_numeric_paths(right_items, path))
                return _MISSING
            result_items.append(interpolated)
        return tuple(result_items)
    uninterpolated.update(_numeric_paths(first, path))
    uninterpolated.update(_numeric_paths(second, path))
    return _MISSING


def _numeric_interpolation(
    first: Mapping[str, object], second: Mapping[str, object], fraction: float
) -> tuple[dict[str, object], tuple[str, ...]]:
    bounded_leaf_items(first, root_path="")
    bounded_leaf_items(second, root_path="")
    uninterpolated: set[str] = set()
    result = _interpolate_uncertainty_value(
        first, second, fraction, "", uninterpolated
    )
    mapping = (
        {}
        if result is _MISSING
        else dict(cast(Mapping[str, object], result))
    )
    return mapping, tuple(sorted(uninterpolated))


def _unavailable(
    timestamp_ns: int, before: GnssSample | None, after: GnssSample | None
) -> GnssInterpolation:
    return GnssInterpolation(
        timestamp_ns=timestamp_ns,
        available=False,
        before=before,
        after=after,
        fraction=None,
        sync_gap_before_ns=None if before is None else timestamp_ns - before.timestamp_ns,
        sync_gap_after_ns=None if after is None else after.timestamp_ns - timestamp_ns,
    )


def interpolate_gnss(
    samples: GnssIndex | Iterable[GnssSample], timestamp_ns: int
) -> GnssInterpolation:
    """Bracket without extrapolation and interpolate geodetic, attitude, and numeric uncertainty."""
    indexed = samples if isinstance(samples, GnssIndex) else index_gnss_samples(samples)
    if not indexed:
        return _unavailable(timestamp_ns, None, None)
    position = bisect.bisect_left(indexed.timestamps, timestamp_ns)
    if position == len(indexed):
        return _unavailable(timestamp_ns, indexed[-1], None)
    if indexed[position].timestamp_ns == timestamp_ns:
        before = after = indexed[position]
        fraction = 0.0
    elif position == 0:
        return _unavailable(timestamp_ns, None, indexed[0])
    else:
        before, after = indexed[position - 1], indexed[position]
        fraction = (timestamp_ns - before.timestamp_ns) / (
            after.timestamp_ns - before.timestamp_ns
        )

    latitude = _linear(before.latitude_deg, after.latitude_deg, fraction)
    longitude = _linear(before.longitude_deg, after.longitude_deg, fraction)
    height = _linear(before.height_m, after.height_m, fraction)
    first_quaternion = euler_to_quaternion_wxyz(
        before.roll_rad, before.pitch_rad, before.yaw_rad
    )
    second_quaternion = euler_to_quaternion_wxyz(
        after.roll_rad, after.pitch_rad, after.yaw_rad
    )
    quaternion = quaternion_slerp_shortest(first_quaternion, second_quaternion, fraction)
    projected_x, projected_y = _WEB_MERCATOR.transform(longitude, latitude)
    if not all(
        math.isfinite(value)
        for value in (latitude, longitude, height, projected_x, projected_y, *quaternion)
    ):
        raise StructuralExtractionError(
            "GNSS interpolation or projection produced non-finite values"
        )
    position_uncertainty, position_uninterpolated = _numeric_interpolation(
        before.position_uncertainty, after.position_uncertainty, fraction
    )
    orientation_uncertainty, orientation_uninterpolated = _numeric_interpolation(
        before.orientation_uncertainty, after.orientation_uncertainty, fraction
    )
    return GnssInterpolation(
        timestamp_ns=timestamp_ns,
        available=True,
        before=before,
        after=after,
        fraction=fraction,
        sync_gap_before_ns=timestamp_ns - before.timestamp_ns,
        sync_gap_after_ns=after.timestamp_ns - timestamp_ns,
        latitude_deg=latitude,
        longitude_deg=longitude,
        height_m=height,
        quaternion_wxyz=quaternion,
        projected_x_m=projected_x,
        projected_y_m=projected_y,
        position_uncertainty=position_uncertainty,
        orientation_uncertainty=orientation_uncertainty,
        source_validity=(before.is_valid, after.is_valid),
        position_uncertainty_uninterpolated_paths=position_uninterpolated,
        orientation_uncertainty_uninterpolated_paths=orientation_uninterpolated,
    )
