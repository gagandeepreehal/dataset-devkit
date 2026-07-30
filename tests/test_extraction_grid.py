from __future__ import annotations

from fractions import Fraction
from typing import Any, cast

import pytest

from dataset_devkit.extraction import grid as grid_module
from dataset_devkit.extraction.grid import select_camera_grid


def test_grid_uses_rational_period_nearest_unused_and_earlier_tie() -> None:
    result = select_camera_grid(
        batch_timestamps_ns=[0, 400_000_000, 600_000_000, 1_000_000_000],
        target_fps=Fraction(2, 1),
        tolerance_ns=150_000_000,
    )

    assert [(entry.target_timestamp_ns, entry.batch_timestamp_ns) for entry in result.entries] == [
        (0, 0),
        (500_000_000, 400_000_000),
        (1_000_000_000, 1_000_000_000),
    ]
    assert result.entries[1].signed_sync_error_ns == -100_000_000
    assert result.unused_batch_timestamps_ns == (600_000_000,)


def test_grid_records_dropped_and_out_of_tolerance_targets() -> None:
    result = select_camera_grid(
        batch_timestamps_ns=[0, 1_020_000_000],
        target_fps=Fraction(2, 1),
        tolerance_ns=30_000_000,
    )

    assert [miss.target_timestamp_ns for miss in result.misses] == [500_000_000]
    assert result.entries[-1].absolute_sync_error_ns == 20_000_000


def test_grid_keeps_final_target_when_last_batch_is_early_jitter() -> None:
    result = select_camera_grid([0, 490_000_000], Fraction(2, 1), 20_000_000)

    assert [(entry.target_timestamp_ns, entry.batch_timestamp_ns) for entry in result.entries] == [
        (0, 0),
        (500_000_000, 490_000_000),
    ]


def test_grid_is_deterministic_for_unsorted_input_and_rejects_duplicates() -> None:
    unsorted = select_camera_grid([1_000, 0, 1_510], Fraction(2_000_000, 1), 20)
    chosen = [
        (entry.target_timestamp_ns, entry.batch_timestamp_ns)
        for entry in unsorted.entries
    ]
    assert chosen == [
        (1_000, 1_000),
        (1_500, 1_510),
    ]
    assert unsorted.unused_batch_timestamps_ns == (0,)

    with pytest.raises(ValueError, match="duplicate camera batch timestamp"):
        select_camera_grid([0, 0], Fraction(1, 1), 0)


def test_fractional_fps_has_no_cumulative_float_drift() -> None:
    result = select_camera_grid(
        [0, 33_366_667, 66_733_333, 100_100_000],
        Fraction(30_000, 1_001),
        0,
    )
    assert [entry.target_timestamp_ns for entry in result.entries] == [
        0,
        33_366_667,
        66_733_333,
        100_100_000,
    ]


def test_grid_rejects_sub_nanosecond_period_and_excessive_target_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="at least 1 ns|1e9"):
        select_camera_grid([0, 1], Fraction(1_000_000_001, 1), 0)

    monkeypatch.setattr(grid_module, "MAX_GRID_TARGETS", 5, raising=False)
    with pytest.raises(ValueError, match="target count.*5|safety limit.*5"):
        select_camera_grid([0, 10], Fraction(1_000_000_000, 1), 0)


def test_grid_sorts_candidates_once_and_scales_to_large_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_sorted = sorted
    sort_calls = 0

    def counted_sorted(*args: Any, **kwargs: Any) -> list[int]:
        nonlocal sort_calls
        sort_calls += 1
        return cast(list[int], real_sorted(*args, **kwargs))

    monkeypatch.setattr(grid_module, "sorted", counted_sorted, raising=False)
    timestamps = list(range(0, 100_000, 2))
    result = select_camera_grid(timestamps, Fraction(500_000_000, 1), 0)

    assert len(result.entries) == len(timestamps)
    assert sort_calls == 1
    targets = [entry.target_timestamp_ns for entry in result.entries]
    assert all(left < right for left, right in zip(targets, targets[1:], strict=False))
