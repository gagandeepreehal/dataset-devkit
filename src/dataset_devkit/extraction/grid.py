"""Deterministic target-grid selection without floating-point accumulation."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from fractions import Fraction

MAX_GRID_TARGETS = 10_000_000

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

    anchor = batch_timestamps_ns[0]
    end = ordered[-1]
    period_ns = Fraction(1_000_000_000, 1) / target_fps
    if period_ns < 1:
        raise ValueError("target period must be at least 1 ns (target_fps must be <= 1e9)")
    limit = end + tolerance_ns
    if (
        anchor + _rounded_fraction(MAX_GRID_TARGETS * period_ns) <= limit
    ):
        raise ValueError(
            f"target count exceeds safety limit {MAX_GRID_TARGETS}; "
            "lower target_fps or split the recording"
        )

    candidate_count = len(ordered)
    available = [True] * candidate_count
    predecessor = list(range(candidate_count + 1))
    successor = list(range(candidate_count + 1))

    def find_predecessor(encoded_index: int) -> int:
        root = encoded_index
        while predecessor[root] != root:
            root = predecessor[root]
        while predecessor[encoded_index] != encoded_index:
            parent = predecessor[encoded_index]
            predecessor[encoded_index] = root
            encoded_index = parent
        return root

    def find_successor(index: int) -> int:
        root = index
        while successor[root] != root:
            root = successor[root]
        while successor[index] != index:
            parent = successor[index]
            successor[index] = root
            index = parent
        return root

    def remove(index: int) -> None:
        available[index] = False
        predecessor[index + 1] = find_predecessor(index)
        successor[index] = find_successor(index + 1)

    entries: list[SelectedGridEntry] = []
    misses: list[GridMiss] = []
    grid_index = 0
    previous_target: int | None = None
    while True:
        target = anchor + _rounded_fraction(grid_index * period_ns)
        if target > limit:
            break
        if previous_target is not None and target <= previous_target:
            raise ValueError("rounded target timestamps must be strictly increasing")
        previous_target = target

        insertion = bisect.bisect_left(ordered, target)
        left_encoded = find_predecessor(insertion)
        left = left_encoded - 1
        right = find_successor(insertion)
        candidate_indices = [
            index for index in (left, right) if 0 <= index < candidate_count
        ]
        if candidate_indices:
            chosen_index = min(
                candidate_indices,
                key=lambda index: (abs(ordered[index] - target), ordered[index]),
            )
            chosen = ordered[chosen_index]
        else:
            chosen_index = -1
            chosen = 0
        if chosen_index >= 0 and abs(chosen - target) <= tolerance_ns:
            remove(chosen_index)
            signed_error = chosen - target
            entries.append(
                SelectedGridEntry(target, chosen, signed_error, abs(signed_error))
            )
        else:
            misses.append(GridMiss(target))
        grid_index += 1
    unused = tuple(timestamp for index, timestamp in enumerate(ordered) if available[index])
    return GridSelection(tuple(entries), tuple(misses), unused)
