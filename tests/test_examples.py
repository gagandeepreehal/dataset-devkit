from __future__ import annotations

from pathlib import Path

from dataset_devkit.blob_list import parse_blob_list
from dataset_devkit.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_example_configuration_is_valid() -> None:
    config = load_config(ROOT / "examples" / "dataset_config.json")

    assert config.schema_version == "1.0"
    assert config.azure.blob_list.name == "mcap_blobs.txt"


def test_shipped_blob_list_has_only_relative_mcap_blob_names() -> None:
    lines = parse_blob_list(ROOT / "examples" / "mcap_blobs.txt")

    assert lines
    assert all(line.startswith("mcap-h265/") for line in lines)
