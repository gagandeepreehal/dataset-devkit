"""Privacy-transformed GNSS parsing and recording-local ENU interpolation."""

from __future__ import annotations

import bisect
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq
from pyproj import Transformer

from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.gnss import (
    _numeric_interpolation,
    euler_to_quaternion_wxyz,
    parse_numeric_uncertainty_leaf,
    quaternion_slerp_shortest,
)
from dataset_devkit.extraction.models import GnssInterpolation, PrivacyGnssSample

_TO_WGS84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
_EXPECTED_COLUMNS = {
    "recording_id",
    "recording_offset_ns",
    "is_valid",
    "local_enu",
    "orientation",
    "enu",
    "lat_lon_ht",
    "position_uncertainty",
    "orientation_uncertainty",
    "quality",
    "local_enu_origin_offset_ns",
    "published_horizontal_resolution_m",
}
_ANONYMIZER_COLUMNS = {
    "recording_id",
    "log_time_of_day_ns",
    "log_time_offset_ns",
    "publish_time_of_day_ns",
    "publish_time_offset_ns",
    "channel",
    "schema_hash",
    "sequence",
    "fields_json",
}
_FORBIDDEN_TIME_FIELDS = {
    "timestamp_ns",
    "rec_timestamp_ns",
    "batch_timestamp_ns",
    "log_time_ns",
    "publish_time_ns",
    "date",
    "datetime",
}

__all__ = ["PrivacyGnssSample", "interpolate_privacy_gnss", "parse_privacy_gnss"]


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise StructuralExtractionError(f"privacy GNSS {label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _float(value: object, label: str) -> float:
    if not isinstance(value, float) or not math.isfinite(value):
        raise StructuralExtractionError(f"privacy GNSS {label} must be finite float64")
    return value


def _int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise StructuralExtractionError(f"privacy GNSS {label} must be an integer")
    return value


def _reject_absolute_fields(value: object, location: str = "row") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            name = str(key).casefold()
            if name in _FORBIDDEN_TIME_FIELDS or "absolute_date" in name:
                raise StructuralExtractionError(
                    f"privacy GNSS contains forbidden absolute time field at {location}.{key}"
                )
            _reject_absolute_fields(nested, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _reject_absolute_fields(nested, f"{location}[{index}]")


def _round_half_away(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def _quality_interpolation(
    first: Mapping[str, Any], second: Mapping[str, Any], fraction: float
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key in sorted(set(first) | set(second)):
        if key not in first or key not in second:
            continue
        left = first[key]
        right = second[key]
        left_number = parse_numeric_uncertainty_leaf(left)
        right_number = parse_numeric_uncertainty_leaf(right)
        if left_number is not None and right_number is not None:
            result[key] = left_number + (right_number - left_number) * fraction
        elif left == right:
            result[key] = left
        else:
            result[key] = left if fraction < 0.5 else right
    return result


def parse_privacy_gnss(path: Path) -> tuple[PrivacyGnssSample, ...]:
    """Parse a strict date-free GNSS shard without Hive partition inference."""
    try:
        parquet = pq.ParquetFile(path)
        columns = set(parquet.schema_arrow.names)
        if columns != _EXPECTED_COLUMNS and columns != _ANONYMIZER_COLUMNS:
            raise StructuralExtractionError(
                "privacy GNSS contains missing, extra, or forbidden fields"
            )
        rows = cast(list[dict[str, object]], parquet.read().to_pylist())
    except StructuralExtractionError:
        raise
    except Exception as error:
        raise StructuralExtractionError("privacy GNSS Parquet is invalid") from error
    if not rows:
        raise StructuralExtractionError("privacy GNSS contains no rows")
    if columns == _ANONYMIZER_COLUMNS:
        normalized: list[dict[str, object]] = []
        for row in rows:
            fields_json = row.get("fields_json")
            if not isinstance(fields_json, str):
                raise StructuralExtractionError("privacy GNSS fields JSON is invalid")
            try:
                fields = json.loads(fields_json)
            except json.JSONDecodeError as error:
                raise StructuralExtractionError(
                    "privacy GNSS fields JSON is invalid"
                ) from error
            if not isinstance(fields, dict):
                raise StructuralExtractionError("privacy GNSS fields JSON is invalid")
            if "recording_id" in fields or "recording_offset_ns" in fields:
                raise StructuralExtractionError(
                    "privacy GNSS fields JSON shadows row identity"
                )
            normalized.append(
                {
                    "recording_id": row.get("recording_id"),
                    "recording_offset_ns": row.get("log_time_offset_ns"),
                    **fields,
                }
            )
        rows = normalized

    samples: list[PrivacyGnssSample] = []
    recording_id: str | None = None
    for row in rows:
        _reject_absolute_fields(row)
        row_recording_id = row.get("recording_id")
        if not isinstance(row_recording_id, str) or not row_recording_id:
            raise StructuralExtractionError("privacy GNSS recording ID is invalid")
        if recording_id is None:
            recording_id = row_recording_id
        elif row_recording_id != recording_id:
            raise StructuralExtractionError("privacy GNSS mixes recording IDs")
        is_valid = row.get("is_valid")
        if not isinstance(is_valid, bool):
            raise StructuralExtractionError("privacy GNSS validity must be boolean")
        origin = _int(row.get("local_enu_origin_offset_ns"), "origin offset")
        if origin != 0:
            raise StructuralExtractionError("privacy GNSS local ENU origin offset must be zero")
        if not is_valid:
            if any(
                row.get(field) is not None
                for field in (
                    "local_enu",
                    "enu",
                    "lat_lon_ht",
                    "published_horizontal_resolution_m",
                )
            ):
                raise StructuralExtractionError(
                    "invalid privacy GNSS row contains published coordinates"
                )
            samples.append(
                PrivacyGnssSample(
                    recording_offset_ns=_int(
                        row.get("recording_offset_ns"), "recording offset"
                    ),
                    is_valid=False,
                    local_east_m=None,
                    local_north_m=None,
                    local_up_m=None,
                    roll_rad=None,
                    pitch_rad=None,
                    yaw_rad=None,
                    published_east_m=None,
                    published_north_m=None,
                    latitude_deg=None,
                    longitude_deg=None,
                    height_m=None,
                    position_uncertainty=_mapping(
                        row.get("position_uncertainty"), "position uncertainty"
                    ),
                    orientation_uncertainty=_mapping(
                        row.get("orientation_uncertainty"), "orientation uncertainty"
                    ),
                    quality=_mapping(row.get("quality"), "quality"),
                    local_enu_origin_offset_ns=0,
                    published_horizontal_resolution_m=None,
                )
            )
            continue
        local = _mapping(row.get("local_enu"), "local ENU")
        orientation = _mapping(row.get("orientation"), "orientation")
        published = _mapping(row.get("enu"), "published ENU")
        lat_lon_ht = _mapping(row.get("lat_lon_ht"), "latitude/longitude/height")
        resolution = _float(
            row.get("published_horizontal_resolution_m"), "published resolution"
        )
        if resolution != 1.0:
            raise StructuralExtractionError("privacy GNSS published resolution must be 1 metre")
        east = _float(published.get("east_m"), "published east")
        north = _float(published.get("north_m"), "published north")
        if east != float(_round_half_away(east)) or north != float(_round_half_away(north)):
            raise StructuralExtractionError(
                "privacy GNSS published horizontal coordinates are not integer metres"
            )
        longitude, latitude = _TO_WGS84.transform(east, north)
        published_latitude = _float(lat_lon_ht.get("latitude_deg"), "latitude")
        published_longitude = _float(lat_lon_ht.get("longitude_deg"), "longitude")
        if (
            abs(latitude - published_latitude) > 1e-9
            or abs(longitude - published_longitude) > 1e-9
        ):
            raise StructuralExtractionError(
                "privacy GNSS latitude/longitude differs from rounded global ENU"
            )
        samples.append(
            PrivacyGnssSample(
                recording_offset_ns=_int(row.get("recording_offset_ns"), "recording offset"),
                is_valid=is_valid,
                local_east_m=_float(local.get("local_east_m"), "local east"),
                local_north_m=_float(local.get("local_north_m"), "local north"),
                local_up_m=_float(local.get("local_up_m"), "local up"),
                roll_rad=_float(orientation.get("roll_rad"), "roll"),
                pitch_rad=_float(orientation.get("pitch_rad"), "pitch"),
                yaw_rad=_float(orientation.get("yaw_rad"), "yaw"),
                published_east_m=east,
                published_north_m=north,
                latitude_deg=published_latitude,
                longitude_deg=published_longitude,
                height_m=_float(lat_lon_ht.get("height_m"), "height"),
                position_uncertainty=_mapping(
                    row.get("position_uncertainty"), "position uncertainty"
                ),
                orientation_uncertainty=_mapping(
                    row.get("orientation_uncertainty"), "orientation uncertainty"
                ),
                quality=_mapping(row.get("quality"), "quality"),
                local_enu_origin_offset_ns=0,
                published_horizontal_resolution_m=resolution,
            )
        )
    offsets = tuple(sample.recording_offset_ns for sample in samples)
    if any(current <= previous for previous, current in zip(offsets, offsets[1:], strict=False)):
        raise StructuralExtractionError(
            "privacy GNSS recording offsets must be unique and strictly increasing"
        )
    return tuple(samples)


def _unavailable(
    offset_ns: int,
    before: PrivacyGnssSample | None,
    after: PrivacyGnssSample | None,
) -> GnssInterpolation:
    return GnssInterpolation(
        timestamp_ns=offset_ns,
        available=False,
        before=before,
        after=after,
        fraction=None,
        sync_gap_before_ns=(
            None if before is None else offset_ns - before.recording_offset_ns
        ),
        sync_gap_after_ns=(
            None if after is None else after.recording_offset_ns - offset_ns
        ),
        pose_frame="recording_local_enu_v1",
        published_horizontal_resolution_m=1.0,
    )


def interpolate_privacy_gnss(
    samples: Sequence[PrivacyGnssSample], recording_offset_ns: int
) -> GnssInterpolation:
    """Interpolate precise local ENU without leading or trailing extrapolation."""
    if not samples:
        return _unavailable(recording_offset_ns, None, None)
    offsets = tuple(sample.recording_offset_ns for sample in samples)
    if any(current <= previous for previous, current in zip(offsets, offsets[1:], strict=False)):
        raise StructuralExtractionError(
            "privacy GNSS recording offsets must be unique and strictly increasing"
        )
    position = bisect.bisect_left(offsets, recording_offset_ns)
    if position == len(samples):
        return _unavailable(recording_offset_ns, samples[-1], None)
    if offsets[position] == recording_offset_ns:
        before = after = samples[position]
        fraction = 0.0
    elif position == 0:
        return _unavailable(recording_offset_ns, None, samples[0])
    else:
        before, after = samples[position - 1], samples[position]
        fraction = (recording_offset_ns - before.recording_offset_ns) / (
            after.recording_offset_ns - before.recording_offset_ns
        )

    required = (
        before.local_east_m,
        before.local_north_m,
        before.local_up_m,
        before.roll_rad,
        before.pitch_rad,
        before.yaw_rad,
        before.published_east_m,
        before.published_north_m,
        before.height_m,
        after.local_east_m,
        after.local_north_m,
        after.local_up_m,
        after.roll_rad,
        after.pitch_rad,
        after.yaw_rad,
        after.published_east_m,
        after.published_north_m,
        after.height_m,
    )
    if not before.is_valid or not after.is_valid or any(value is None for value in required):
        return _unavailable(recording_offset_ns, before, after)
    (
        before_local_east,
        before_local_north,
        before_local_up,
        before_roll,
        before_pitch,
        before_yaw,
        before_published_east,
        before_published_north,
        before_height,
        after_local_east,
        after_local_north,
        after_local_up,
        after_roll,
        after_pitch,
        after_yaw,
        after_published_east,
        after_published_north,
        after_height,
    ) = cast(tuple[float, ...], required)

    def linear(left: float, right: float) -> float:
        return left + (right - left) * fraction

    local = (
        linear(before_local_east, after_local_east),
        linear(before_local_north, after_local_north),
        linear(before_local_up, after_local_up),
    )
    height = linear(before_height, after_height)
    published_east = float(
        _round_half_away(linear(before_published_east, after_published_east))
    )
    published_north = float(
        _round_half_away(linear(before_published_north, after_published_north))
    )
    longitude, latitude = _TO_WGS84.transform(published_east, published_north)
    quaternion = quaternion_slerp_shortest(
        euler_to_quaternion_wxyz(before_roll, before_pitch, before_yaw),
        euler_to_quaternion_wxyz(after_roll, after_pitch, after_yaw),
        fraction,
    )
    values = (*local, height, published_east, published_north, latitude, longitude, *quaternion)
    if not all(math.isfinite(value) for value in values):
        raise StructuralExtractionError("privacy GNSS interpolation produced non-finite values")
    position_uncertainty, position_uninterpolated = _numeric_interpolation(
        before.position_uncertainty, after.position_uncertainty, fraction
    )
    orientation_uncertainty, orientation_uninterpolated = _numeric_interpolation(
        before.orientation_uncertainty, after.orientation_uncertainty, fraction
    )
    return GnssInterpolation(
        timestamp_ns=recording_offset_ns,
        available=True,
        before=before,
        after=after,
        fraction=fraction,
        sync_gap_before_ns=recording_offset_ns - before.recording_offset_ns,
        sync_gap_after_ns=after.recording_offset_ns - recording_offset_ns,
        latitude_deg=latitude,
        longitude_deg=longitude,
        height_m=height,
        quaternion_wxyz=quaternion,
        projected_x_m=published_east,
        projected_y_m=published_north,
        position_uncertainty=position_uncertainty,
        orientation_uncertainty=orientation_uncertainty,
        source_validity=(before.is_valid, after.is_valid),
        position_uncertainty_uninterpolated_paths=position_uninterpolated,
        orientation_uncertainty_uninterpolated_paths=orientation_uninterpolated,
        translation_xyz_m=local,
        pose_frame="recording_local_enu_v1",
        published_east_m=published_east,
        published_north_m=published_north,
        published_horizontal_resolution_m=1.0,
        quality=_quality_interpolation(before.quality, after.quality, fraction),
    )
