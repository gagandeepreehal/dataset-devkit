"""Deterministic bounded traversal for untrusted nested value trees."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from dataset_devkit.extraction.errors import StructuralExtractionError

UNCERTAINTY_MAX_DEPTH = 32
UNCERTAINTY_MAX_NODES = 10_000
UNCERTAINTY_MAX_LEAVES = 8_000
UNCERTAINTY_MAX_WORK = 200_000


@dataclass
class TraversalBudget:
    """One deterministic traversal budget with stable structural errors."""

    nodes: int = 0
    leaves: int = 0
    work: int = 0

    def visit(self, path: str, *, leaf: bool, work: int = 0) -> None:
        self.nodes += 1
        self.work += work
        if leaf:
            self.leaves += 1
        if self.nodes > UNCERTAINTY_MAX_NODES:
            self._raise(path, f"maximum nodes {UNCERTAINTY_MAX_NODES} exceeded")
        if self.leaves > UNCERTAINTY_MAX_LEAVES:
            self._raise(path, f"maximum leaves {UNCERTAINTY_MAX_LEAVES} exceeded")
        if self.work > UNCERTAINTY_MAX_WORK:
            self._raise(path, f"maximum work {UNCERTAINTY_MAX_WORK} exceeded")

    @staticmethod
    def check_depth(path: str, depth: int) -> None:
        if depth > UNCERTAINTY_MAX_DEPTH:
            TraversalBudget._raise(
                path, f"maximum depth {UNCERTAINTY_MAX_DEPTH} exceeded"
            )

    @staticmethod
    def _raise(path: str, reason: str) -> None:
        raise StructuralExtractionError(
            f"bounded value traversal failed at {path!r}: {reason}"
        )


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _child_path(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _freeze(
    value: Any,
    *,
    path: str,
    depth: int,
    ancestors: frozenset[int],
    budget: TraversalBudget,
    work: int,
) -> Any:
    is_mapping = isinstance(value, Mapping)
    is_sequence = _is_sequence(value)
    is_set = isinstance(value, (set, frozenset))
    is_container = is_mapping or is_sequence or is_set
    if is_container:
        budget.check_depth(path, depth)
    budget.visit(path, leaf=not is_container, work=work)
    if not is_container:
        return value
    identity = id(value)
    if identity in ancestors:
        budget._raise(path, "container cycle detected")
    nested_ancestors = ancestors | {identity}
    if is_mapping:
        mapping = value
        keys = sorted(mapping, key=str)
        string_keys = [str(key) for key in keys]
        if len(string_keys) != len(set(string_keys)):
            budget._raise(path, "mapping keys collide after string conversion")
        return MappingProxyType(
            {
                str(key): _freeze(
                    mapping[key],
                    path=_child_path(path, str(key)),
                    depth=depth + 1,
                    ancestors=nested_ancestors,
                    budget=budget,
                    work=len(str(key)),
                )
                for key in keys
            }
        )
    if is_sequence:
        sequence = value
        return tuple(
            _freeze(
                item,
                path=f"{path}[{index}]",
                depth=depth + 1,
                ancestors=nested_ancestors,
                budget=budget,
                work=len(str(index)) + 2,
            )
            for index, item in enumerate(sequence)
        )
    return frozenset(
        _freeze(
            item,
            path=f"{path}{{{index}}}",
            depth=depth + 1,
            ancestors=nested_ancestors,
            budget=budget,
            work=len(str(index)) + 2,
        )
        for index, item in enumerate(sorted(value, key=repr))
    )


def bounded_freeze(value: Any, *, root_path: str) -> Any:
    """Deep-copy/freeze a bounded acyclic mapping/sequence value tree."""
    return _freeze(
        value,
        path=root_path,
        depth=0,
        ancestors=frozenset(),
        budget=TraversalBudget(),
        work=0,
    )


def bounded_leaf_items(value: Any, *, root_path: str = "") -> tuple[tuple[str, Any], ...]:
    """Return deterministic flattened leaves after enforcing the shared limits."""
    frozen = bounded_freeze(value, root_path=root_path)
    leaves: list[tuple[str, Any]] = []
    stack: list[tuple[str, Any]] = [(root_path, frozen)]
    while stack:
        path, item = stack.pop()
        if isinstance(item, Mapping):
            for key in reversed(tuple(item)):
                stack.append((_child_path(path, str(key)), item[key]))
        elif _is_sequence(item):
            for index in range(len(item) - 1, -1, -1):
                stack.append((f"{path}[{index}]", item[index]))
        elif isinstance(item, (set, frozenset)):
            for index, nested in reversed(tuple(enumerate(sorted(item, key=repr)))):
                stack.append((f"{path}{{{index}}}", nested))
        else:
            leaves.append((path, item))
    return tuple(leaves)
