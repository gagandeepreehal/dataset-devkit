from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyproj import Transformer

from dataset_devkit.extraction.decoded_gnss import (
    PrivacyGnssSample,
    interpolate_privacy_gnss,
    parse_privacy_gnss,
)
from dataset_devkit.extraction.errors import StructuralExtractionError
from decoded_v2_fixture import write_decoded_fixture

_TO_WGS84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


def privacy_sample(
    offset: int,
    *,
    local: tuple[float, float, float],
    global_xy: tuple[float, float],
) -> PrivacyGnssSample:
    longitude, latitude = _TO_WGS84.transform(*global_xy)
    return PrivacyGnssSample(
        recording_offset_ns=offset,
        is_valid=True,
        local_east_m=local[0],
        local_north_m=local[1],
        local_up_m=local[2],
        roll_rad=0.0,
        pitch_rad=0.0,
        yaw_rad=0.2,
        published_east_m=global_xy[0],
        published_north_m=global_xy[1],
        latitude_deg=latitude,
        longitude_deg=longitude,
        height_m=900.0 + local[2],
        position_uncertainty={"sigma_m": 0.1},
        orientation_uncertainty={"variance": 0.01},
        quality={"fix": "rtk", "satellites": 18, "hdop": 0.7},
        local_enu_origin_offset_ns=0,
        published_horizontal_resolution_m=1.0,
    )


def test_privacy_gnss_pose_uses_local_enu_not_global() -> None:
    samples = (
        privacy_sample(0, local=(0.0, 0.0, 0.0), global_xy=(8_000_000.0, 1_500_000.0)),
        privacy_sample(
            2_000_000_000,
            local=(10.0, 4.0, 0.5),
            global_xy=(8_000_010.0, 1_500_004.0),
        ),
    )

    result = interpolate_privacy_gnss(samples, 1_000_000_000)

    assert result.pose_frame == "recording_local_enu_v1"
    assert result.translation_xyz_m == pytest.approx((5.0, 2.0, 0.25))
    assert result.published_horizontal_resolution_m == 1.0
    assert result.translation_xyz_m != pytest.approx((8_000_005.0, 1_500_002.0, 900.25))


def test_privacy_global_rounding_is_half_away_from_zero() -> None:
    positive = interpolate_privacy_gnss(
        (
            privacy_sample(0, local=(0.0, 0.0, 0.0), global_xy=(0.0, 0.0)),
            privacy_sample(2, local=(2.0, 0.0, 0.0), global_xy=(1.0, 0.0)),
        ),
        1,
    )
    negative = interpolate_privacy_gnss(
        (
            privacy_sample(0, local=(0.0, 0.0, 0.0), global_xy=(-1.0, 0.0)),
            privacy_sample(2, local=(2.0, 0.0, 0.0), global_xy=(0.0, 0.0)),
        ),
        1,
    )

    assert positive.published_east_m == 1.0
    assert negative.published_east_m == -1.0


def test_parse_privacy_gnss_reads_precise_fixture_rows(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path)
    path = next((fixture.root / "data/tables/gnss").rglob("*.parquet"))

    samples = parse_privacy_gnss(path, expected_recording_id=fixture.recording_id)

    assert [sample.recording_offset_ns for sample in samples] == [0, 1_000_000_000]
    assert samples[1].local_east_m == 10.0
    assert samples[1].published_east_m == 8_000_010.0


def test_parse_privacy_gnss_binds_rows_to_expected_recording(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path)
    path = next((fixture.root / "data/tables/gnss").rglob("*.parquet"))

    with pytest.raises(StructuralExtractionError, match="selected recording"):
        parse_privacy_gnss(path, expected_recording_id="another-recording")


def test_parse_privacy_gnss_accepts_anonymizer_fields_json_schema(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path)
    source = next((fixture.root / "data/tables/gnss").rglob("*.parquet"))
    flat_rows = pq.ParquetFile(source).read().to_pylist()
    production_rows: list[dict[str, object]] = []
    for row in flat_rows:
        fields = {
            key: value
            for key, value in row.items()
            if key not in {"recording_id", "recording_offset_ns"}
        }
        production_rows.append(
            {
                "recording_id": row["recording_id"],
                "log_time_of_day_ns": 10_000_000_000 + row["recording_offset_ns"],
                "log_time_offset_ns": row["recording_offset_ns"],
                "publish_time_of_day_ns": 10_000_000_000 + row["recording_offset_ns"],
                "publish_time_offset_ns": row["recording_offset_ns"],
                "channel": "gnss",
                "schema_hash": "a" * 64,
                "sequence": len(production_rows),
                "fields_json": json.dumps(fields, sort_keys=True),
            }
        )
    pq.write_table(pa.Table.from_pylist(production_rows), source)

    samples = parse_privacy_gnss(source, expected_recording_id=fixture.recording_id)

    assert [item.recording_offset_ns for item in samples] == [0, 1_000_000_000]
    assert samples[1].local_east_m == 10.0


def test_anonymizer_invalid_gnss_row_is_an_interpolation_barrier(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path)
    source = next((fixture.root / "data/tables/gnss").rglob("*.parquet"))
    flat_rows = pq.ParquetFile(source).read().to_pylist()
    production_rows: list[dict[str, object]] = []
    for index, row in enumerate(flat_rows):
        fields = {
            key: value
            for key, value in row.items()
            if key not in {"recording_id", "recording_offset_ns"}
        }
        if index == 1:
            fields.update(
                {
                    "is_valid": False,
                    "local_enu": None,
                    "enu": None,
                    "lat_lon_ht": None,
                    "published_horizontal_resolution_m": None,
                }
            )
        production_rows.append(
            {
                "recording_id": row["recording_id"],
                "log_time_of_day_ns": 10_000_000_000 + row["recording_offset_ns"],
                "log_time_offset_ns": row["recording_offset_ns"],
                "publish_time_of_day_ns": 10_000_000_000 + row["recording_offset_ns"],
                "publish_time_offset_ns": row["recording_offset_ns"],
                "channel": "gnss",
                "schema_hash": "a" * 64,
                "sequence": index,
                "fields_json": json.dumps(fields, sort_keys=True),
            }
        )
    pq.write_table(pa.Table.from_pylist(production_rows), source)

    samples = parse_privacy_gnss(source, expected_recording_id=fixture.recording_id)

    assert not samples[1].is_valid
    assert not interpolate_privacy_gnss(samples, 500_000_000).available
    assert not interpolate_privacy_gnss(samples, 1_000_000_000).available


def test_privacy_gnss_rejects_nonzero_origin_offset(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path)
    path = next((fixture.root / "data/tables/gnss").rglob("*.parquet"))
    rows = pq.ParquetFile(path).read().to_pylist()
    rows[0]["local_enu_origin_offset_ns"] = 1
    pq.write_table(pa.Table.from_pylist(rows), path)

    with pytest.raises(StructuralExtractionError, match="origin offset"):
        parse_privacy_gnss(path, expected_recording_id=fixture.recording_id)


def test_privacy_gnss_rejects_duplicate_offsets_and_absolute_time_fields(
    tmp_path: Path,
) -> None:
    fixture = write_decoded_fixture(tmp_path)
    path = next((fixture.root / "data/tables/gnss").rglob("*.parquet"))
    rows = pq.ParquetFile(path).read().to_pylist()
    rows[1]["recording_offset_ns"] = rows[0]["recording_offset_ns"]
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(StructuralExtractionError, match="offset"):
        parse_privacy_gnss(path, expected_recording_id=fixture.recording_id)

    rows[1]["recording_offset_ns"] = 1_000_000_000
    rows[0]["timestamp_ns"] = 123
    rows[1]["timestamp_ns"] = 124
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(StructuralExtractionError, match="absolute.*time|forbidden"):
        parse_privacy_gnss(path, expected_recording_id=fixture.recording_id)


@pytest.mark.parametrize(
    "field_name",
    ["gps_week", "gps_day", "day_of_year", "utc_timestamp", "epoch_seconds"],
)
def test_privacy_gnss_rejects_nested_reversible_time_fields(
    tmp_path: Path, field_name: str
) -> None:
    fixture = write_decoded_fixture(tmp_path)
    path = next((fixture.root / "data/tables/gnss").rglob("*.parquet"))
    rows = pq.ParquetFile(path).read().to_pylist()
    fields = {
        key: value
        for key, value in rows[0].items()
        if key not in {"recording_id", "recording_offset_ns"}
    }
    quality = dict(fields["quality"])
    quality[field_name] = 123
    fields["quality"] = quality
    production_row = {
        "recording_id": fixture.recording_id,
        "log_time_of_day_ns": 10_000_000_000,
        "log_time_offset_ns": 0,
        "publish_time_of_day_ns": 10_000_000_000,
        "publish_time_offset_ns": 0,
        "channel": "gnss",
        "schema_hash": "a" * 64,
        "sequence": 0,
        "fields_json": json.dumps(fields, sort_keys=True),
    }
    pq.write_table(pa.Table.from_pylist([production_row]), path)

    with pytest.raises(StructuralExtractionError, match="forbidden absolute time"):
        parse_privacy_gnss(path, expected_recording_id=fixture.recording_id)


@pytest.mark.parametrize("value", ["2026-09-14", "2026-09-14T12:34:56Z"])
def test_privacy_gnss_rejects_nested_iso_date_values(
    tmp_path: Path, value: str
) -> None:
    fixture = write_decoded_fixture(tmp_path)
    path = next((fixture.root / "data/tables/gnss").rglob("*.parquet"))
    rows = pq.ParquetFile(path).read().to_pylist()
    quality = dict(rows[0]["quality"])
    quality["note"] = value
    rows[0]["quality"] = quality
    pq.write_table(pa.Table.from_pylist(rows), path)

    with pytest.raises(StructuralExtractionError, match="forbidden absolute time"):
        parse_privacy_gnss(path, expected_recording_id=fixture.recording_id)


def test_privacy_gnss_never_extrapolates() -> None:
    samples = (
        privacy_sample(10, local=(0.0, 0.0, 0.0), global_xy=(0.0, 0.0)),
        privacy_sample(20, local=(1.0, 0.0, 0.0), global_xy=(1.0, 0.0)),
    )

    assert not interpolate_privacy_gnss(samples, 9).available
    assert not interpolate_privacy_gnss(samples, 21).available
    assert interpolate_privacy_gnss(samples, 10).available


def test_local_translation_is_independent_of_global_context() -> None:
    base = (
        privacy_sample(0, local=(0.0, 0.0, 0.0), global_xy=(0.0, 0.0)),
        privacy_sample(2, local=(2.0, 4.0, 1.0), global_xy=(2.0, 4.0)),
    )
    shifted = tuple(
        replace(
            sample,
            published_east_m=cast(float, sample.published_east_m) + 1_000_000.0,
            published_north_m=cast(float, sample.published_north_m) - 500_000.0,
        )
        for sample in base
    )

    assert interpolate_privacy_gnss(base, 1).translation_xyz_m == (
        interpolate_privacy_gnss(shifted, 1).translation_xyz_m
    )
