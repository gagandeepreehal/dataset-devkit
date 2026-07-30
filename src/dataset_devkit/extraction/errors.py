"""Extraction failures that make a recording structurally unusable."""


class StructuralExtractionError(ValueError):
    """Raised when an MCAP cannot satisfy the version-one extraction contract."""
