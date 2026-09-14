from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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

    samples = parse_privacy_gnss(path)

    assert [sample.recording_offset_ns for sample in samples] == [0, 1_000_000_000]
    assert samples[1].local_east_m == 10.0
    assert samples[1].published_east_m == 8_000_010.0


def test_privacy_gnss_rejects_nonzero_origin_offset(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path)
    path = next((fixture.root / "data/tables/gnss").rglob("*.parquet"))
    rows = pq.ParquetFile(path).read().to_pylist()
    rows[0]["local_enu_origin_offset_ns"] = 1
    pq.write_table(pa.Table.from_pylist(rows), path)

    with pytest.raises(StructuralExtractionError, match="origin offset"):
        parse_privacy_gnss(path)


def test_privacy_gnss_rejects_duplicate_offsets_and_absolute_time_fields(
    tmp_path: Path,
) -> None:
    fixture = write_decoded_fixture(tmp_path)
    path = next((fixture.root / "data/tables/gnss").rglob("*.parquet"))
    rows = pq.ParquetFile(path).read().to_pylist()
    rows[1]["recording_offset_ns"] = rows[0]["recording_offset_ns"]
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(StructuralExtractionError, match="offset"):
        parse_privacy_gnss(path)

    rows[1]["recording_offset_ns"] = 1_000_000_000
    rows[0]["timestamp_ns"] = 123
    rows[1]["timestamp_ns"] = 124
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(StructuralExtractionError, match="absolute.*time|forbidden"):
        parse_privacy_gnss(path)


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
            published_east_m=sample.published_east_m + 1_000_000.0,
            published_north_m=sample.published_north_m - 500_000.0,
        )
        for sample in base
    )

    assert interpolate_privacy_gnss(base, 1).translation_xyz_m == (
        interpolate_privacy_gnss(shifted, 1).translation_xyz_m
    )
