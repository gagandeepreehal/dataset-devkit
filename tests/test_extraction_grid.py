from __future__ import annotations

from fractions import Fraction

import pytest

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
