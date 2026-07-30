"""Service boundaries for pipeline stages implemented in later tasks."""

from __future__ import annotations

from pathlib import Path

from dataset_devkit.config import GlobalConfig


class ServiceNotImplementedError(NotImplementedError):
    """Raised when a stable boundary has no pipeline implementation yet."""


def build_dataset(config: GlobalConfig) -> None:
    """Build a dataset from validated configuration."""
    raise ServiceNotImplementedError("the build pipeline is not implemented yet")


def validate_dataset(dataroot: Path, version: str) -> None:
    """Validate an exported dataset version."""
    raise ServiceNotImplementedError("dataset validation is not implemented yet")


def inspect_dataset(dataroot: Path, version: str) -> None:
    """Inspect an exported dataset version."""
    raise ServiceNotImplementedError("dataset inspection is not implemented yet")
