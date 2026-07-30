from __future__ import annotations

import math
from typing import cast

import pytest

from dataset_devkit.extraction import gnss as gnss_module
from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.gnss import (
    Quaternion,
    euler_to_quaternion_wxyz,
    index_gnss_samples,
    interpolate_gnss,
    quaternion_slerp_shortest,
)
from dataset_devkit.extraction.models import GnssSample


def sample(timestamp: int, *, lon: float, yaw: float, valid: bool = True) -> GnssSample:
    return GnssSample(
        timestamp_ns=timestamp,
        rec_timestamp_ns=timestamp + 1,
        is_valid=valid,
        latitude_deg=0.0,
        longitude_deg=lon,
        height_m=10.0 + lon,
        roll_rad=0.0,
        pitch_rad=0.0,
        yaw_rad=yaw,
        position_uncertainty={"east_sigma_m": 1.0 + lon, "hdop": 2.0 + lon},
        orientation_uncertainty={"yaw_variance": 0.1 + lon, "label": "raw"},
        raw_identifiers={"fix_id": f"fix-{timestamp}"},
    )


def test_gnss_index_sorts_and_rejects_duplicate_recorded_timestamps() -> None:
    indexed = index_gnss_samples(
        [sample(10, lon=1, yaw=0), sample(0, lon=0, yaw=0)]
    )
    assert [item.timestamp_ns for item in indexed] == [0, 10]
    assert getattr(indexed, "timestamps", None) == (0, 10)
    with pytest.raises(StructuralExtractionError, match="duplicate GNSS timestamp"):
        index_gnss_samples([sample(0, lon=0, yaw=0), sample(0, lon=1, yaw=0)])


def test_interpolation_retains_raw_endpoints_uncertainty_and_height() -> None:
    before = sample(0, lon=0, yaw=math.radians(179), valid=True)
    after = sample(10, lon=2, yaw=math.radians(-179), valid=False)
    result = interpolate_gnss((before, after), 5)

    assert result.available
    assert result.before is before and result.after is after
    assert result.fraction == 0.5
    assert result.longitude_deg == 1.0
    assert result.height_m == 11.0
    assert result.position_uncertainty == {"east_sigma_m": 2.0, "hdop": 3.0}
    assert result.orientation_uncertainty["yaw_variance"] == pytest.approx(1.1)
    assert result.source_validity == (True, False)
    assert result.sync_gap_before_ns == result.sync_gap_after_ns == 5
    assert result.projected_x_m == pytest.approx(111_319.490793, rel=1e-8)
    assert result.projected_y_m == pytest.approx(0.0, abs=1e-8)
    assert result.quaternion_wxyz is not None
    assert math.sqrt(sum(value * value for value in result.quaternion_wxyz)) == pytest.approx(1)


def test_interpolation_exact_endpoint_and_outside_range_are_explicit() -> None:
    samples = (sample(0, lon=0, yaw=0), sample(10, lon=2, yaw=1))
    endpoint = interpolate_gnss(samples, 0)
    assert endpoint.available and endpoint.fraction == 0
    assert endpoint.before is samples[0] and endpoint.after is samples[0]

    early = interpolate_gnss(samples, -1)
    late = interpolate_gnss(samples, 11)
    assert not early.available and early.before is None and early.after is samples[0]
    assert not late.available and late.before is samples[1] and late.after is None


def test_web_mercator_known_point_and_quaternion_shortest_path() -> None:
    projected = interpolate_gnss(
        (sample(0, lon=180, yaw=0), sample(10, lon=180, yaw=0)), 0
    )
    assert projected.projected_x_m == pytest.approx(20_037_508.342789, rel=1e-8)

    first = euler_to_quaternion_wxyz(0, 0, math.radians(179))
    second = euler_to_quaternion_wxyz(0, 0, math.radians(-179))
    midpoint = quaternion_slerp_shortest(first, second, 0.5)
    assert abs(midpoint[0]) < 1e-6
    assert abs(midpoint[3]) == pytest.approx(1.0)
    antipodal = cast(Quaternion, tuple(-value for value in first))
    assert quaternion_slerp_shortest(first, antipodal, 0.5) == pytest.approx(first)


def test_gnss_index_is_built_once_and_reused_for_many_interpolations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = index_gnss_samples(
        sample(timestamp, lon=float(timestamp) / 1_000, yaw=0)
        for timestamp in range(10_000, -1, -1)
    )
    rebuilds = 0
    real_index = gnss_module.index_gnss_samples

    def counted_index(samples: object) -> object:
        nonlocal rebuilds
        rebuilds += 1
        return real_index(samples)  # type: ignore[arg-type]

    monkeypatch.setattr(gnss_module, "index_gnss_samples", counted_index)
    for timestamp in range(50, 9_950, 50):
        assert interpolate_gnss(index, timestamp).available
    assert rebuilds == 0


def test_result_uncertainty_mappings_are_defensively_immutable() -> None:
    mutable_position = {"east_sigma_m": 1.0}
    mutable_orientation: dict[str, object] = {"nested": {"yaw": 0.1}}
    item = GnssSample(
        timestamp_ns=0,
        rec_timestamp_ns=1,
        is_valid=True,
        latitude_deg=0,
        longitude_deg=0,
        height_m=0,
        roll_rad=0,
        pitch_rad=0,
        yaw_rad=0,
        position_uncertainty=mutable_position,
        orientation_uncertainty=mutable_orientation,
    )
    mutable_position["east_sigma_m"] = 99
    cast(dict[str, object], mutable_orientation["nested"])["yaw"] = 99

    assert item.position_uncertainty["east_sigma_m"] == 1
    assert cast(object, item.orientation_uncertainty["nested"])["yaw"] == 0.1  # type: ignore[index]
    with pytest.raises(TypeError):
        item.position_uncertainty["east_sigma_m"] = 2  # type: ignore[index]


def test_mixed_axis_active_xyz_euler_quaternion_golden() -> None:
    quaternion = euler_to_quaternion_wxyz(
        math.radians(30), math.radians(-20), math.radians(40)
    )
    assert quaternion == pytest.approx(
        (0.8785122060499201, 0.296882904556291, -0.0704393377846027, 0.36758011983238364)
    )
