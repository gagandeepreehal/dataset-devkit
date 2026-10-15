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
        "owner/restricted-raw-mcap"
    )
    assert config.huggingface.revision == "0" * 40
    assert config.huggingface.manifest_path == "manifest.jsonl"
    assert all(config.mcap_compatibility.model_dump().values())


def test_shipped_annotation_jsonl_is_strict_and_parseable() -> None:
    records = parse_annotations(ROOT / "examples" / "annotations.jsonl")

    assert records and records[0].repo_path.startswith("data/")


def test_decoded_example_is_schema_and_runtime_valid() -> None:
    path = ROOT / "examples" / "decoded_v2_config.json"

    config = validate_config_schema_and_runtime(path)

    assert config.schema_version == "2.0"
    assert privacy_classification(config) == "privacy_transformed"


def test_indus_ad_example_is_schema_and_runtime_valid() -> None:
    path = ROOT / "examples" / "indus_ad_v21_config.json"

    config = validate_config_schema_and_runtime(path)

    assert config.schema_version == "2.1"
    assert config.source.type == "indus_ad_hf"
    assert privacy_classification(config) == "privacy_transformed"


def test_readme_distinguishes_raw_and_privacy_transformed_sources() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "restricted raw" in readme.casefold()
    assert "recording-local ENU" in readme
    assert "does not independently rerun the anonymizer" in readme
