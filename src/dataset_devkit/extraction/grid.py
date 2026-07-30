"""Deterministic target-grid selection without floating-point accumulation."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class SelectedGridEntry:
    target_timestamp_ns: int
    batch_timestamp_ns: int
    signed_sync_error_ns: int
    absolute_sync_error_ns: int


@dataclass(frozen=True)
class GridMiss:
    target_timestamp_ns: int


@dataclass(frozen=True)
class GridSelection:
    entries: tuple[SelectedGridEntry, ...]
    misses: tuple[GridMiss, ...]
    unused_batch_timestamps_ns: tuple[int, ...]


def _rounded_fraction(value: Fraction) -> int:
    quotient, remainder = divmod(value.numerator, value.denominator)
    if remainder * 2 < value.denominator:
        return quotient
    return quotient + 1


def select_camera_grid(
    batch_timestamps_ns: list[int] | tuple[int, ...],
    target_fps: Fraction,
    tolerance_ns: int,
) -> GridSelection:
    """Select nearest unused batches for an anchored rational target grid."""
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    if tolerance_ns < 0:
        raise ValueError("tolerance_ns must be nonnegative")
    ordered = sorted(batch_timestamps_ns)
    if len(ordered) != len(set(ordered)):
        raise ValueError("duplicate camera batch timestamp is ambiguous")
    if not ordered:
        return GridSelection((), (), ())

    anchor = ordered[0]
    end = ordered[-1]
    period_ns = Fraction(1_000_000_000, 1) / target_fps
    available = set(ordered)
    entries: list[SelectedGridEntry] = []
    misses: list[GridMiss] = []
    grid_index = 0
    while True:
        target = anchor + _rounded_fraction(grid_index * period_ns)
        if target > end + tolerance_ns:
            break
        candidates = sorted(available, key=lambda timestamp: (abs(timestamp - target), timestamp))
        if candidates and abs(candidates[0] - target) <= tolerance_ns:
            chosen = candidates[0]
            available.remove(chosen)
            signed_error = chosen - target
            entries.append(
                SelectedGridEntry(target, chosen, signed_error, abs(signed_error))
            )
        else:
            misses.append(GridMiss(target))
        grid_index += 1
    return GridSelection(tuple(entries), tuple(misses), tuple(sorted(available)))
