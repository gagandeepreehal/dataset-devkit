from __future__ import annotations

from pathlib import Path

from dataset_devkit.annotations import parse_annotations
from dataset_devkit.config import load_config, privacy_classification
from dataset_devkit.schema import validate_config_schema_and_runtime

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_example_configuration_is_valid() -> None:
    config = load_config(ROOT / "examples" / "dataset_config.json")

    assert config.schema_version == "1.0"
    assert config.huggingface.repo_id == (
        "gagandeepreehal/minuszero-indian-autonomous-driving-monocam"
    )
    assert config.huggingface.revision == "b13c3bd3a049c73b560910ef5dbc60cbd28c441b"
    assert config.huggingface.manifest_path == "manifest.jsonl"


def test_shipped_annotation_jsonl_is_strict_and_parseable() -> None:
    records = parse_annotations(ROOT / "examples" / "annotations.jsonl")

    assert records and records[0].repo_path.startswith("data/")


def test_decoded_example_is_schema_and_runtime_valid() -> None:
    path = ROOT / "examples" / "decoded_v2_config.json"

    config = validate_config_schema_and_runtime(path)

    assert config.schema_version == "2.0"
    assert privacy_classification(config) == "privacy_transformed"


def test_readme_distinguishes_raw_and_privacy_transformed_sources() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "restricted raw" in readme.casefold()
    assert "recording-local ENU" in readme
    assert "does not independently rerun the anonymizer" in readme
