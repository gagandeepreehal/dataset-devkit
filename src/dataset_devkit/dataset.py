"""Stable dataset access boundary.

The indexing and query implementation is intentionally deferred to the dataset API task.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Dataset:
    """Identify a published dataset version without eagerly loading it."""

    dataroot: Path
    version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataroot", self.dataroot.resolve())
