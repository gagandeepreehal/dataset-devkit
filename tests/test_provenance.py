from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from dataset_devkit.config import GlobalConfig
from dataset_devkit.provenance import (
    AcquisitionManifest,
    ArtifactIdentity,
    DecodedSourceFingerprint,
    ExtractionManifest,
    SourceFingerprint,
    canonical_json,
    extraction_config_hash,
    fingerprint_from_dict,
    fingerprint_to_dict,
    load_manifest,
    write_manifest,
)


def test_decoded_fingerprint_binds_recording_and_artifact_set() -> None:
    fingerprint = DecodedSourceFingerprint(
        repo_id="owner/dataset",
        revision="0123456789abcdef0123456789abcdef01234567",
        recording_id="recording-code",
        catalog_row_sha256="a" * 64,
        artifact_set_sha256="b" * 64,
        total_size=123,
    )

    assert fingerprint.source_type == "decoded_hf"
    assert fingerprint.privacy_classification == "privacy_transformed"
    assert fingerprint_from_dict(fingerprint_to_dict(fingerprint)) == fingerprint


def test_decoded_digest_changes_with_artifact_set() -> None:
    first = DecodedSourceFingerprint(
        "owner/dataset",
        "0123456789abcdef0123456789abcdef01234567",
        "recording-code",
        "a" * 64,
        "b" * 64,
        123,
    )
    second = replace(first, artifact_set_sha256="c" * 64)

    assert first.digest != second.digest


@pytest.mark.parametrize(
    "mutation",
    [
        {"source_type": "mcap_hf"},
        {"privacy_classification": "restricted_raw"},
        {"revision": "main"},
        {"catalog_row_sha256": "a" * 63},
        {"artifact_set_sha256": "g" * 64},
        {"total_size": True},
        {"unknown": "value"},
    ],
)
def test_decoded_fingerprint_parser_rejects_hybrid_or_invalid_shapes(
    mutation: dict[str, object],
) -> None:
    fingerprint = DecodedSourceFingerprint(
        "owner/dataset",
        "0123456789abcdef0123456789abcdef01234567",
        "recording-code",
        "a" * 64,
        "b" * 64,
        123,
    )
    payload = fingerprint_to_dict(fingerprint)
    payload.update(mutation)

    with pytest.raises(ValueError, match="fingerprint"):
        fingerprint_from_dict(payload)


def test_canonical_json_and_source_fingerprint_are_deterministic() -> None:
    assert canonical_json({"z": 1, "a": [True, None]}) == '{"a":[true,null],"z":1}'
    first = SourceFingerprint(
        repo_id="owner/dataset",
        revision="a" * 40,
        repo_path="data/fleet/a.mcap",
        sha256="b" * 64,
        size=12,
    )
    second = SourceFingerprint.from_dict(json.loads(canonical_json(first.to_dict())))

    assert first == second
    assert first.digest == second.digest
    assert first.cache_key != SourceFingerprint(
        repo_id=first.repo_id,
        revision=first.revision,
        repo_path="data/fleet/b.mcap",
        sha256=first.sha256,
        size=first.size,
    ).cache_key


def test_canonical_json_rejects_nonfinite_numbers() -> None:
    with pytest.raises(ValueError, match="JSON compliant"):
        canonical_json({"invalid": math.nan})


def test_extraction_config_hash_is_deterministic_and_scoped(
    config_factory: Callable[[], GlobalConfig],
) -> None:
    config = config_factory()
    same = config.model_copy(deep=True)
    execution_changed = config.model_copy(
        update={"execution": config.execution.model_copy(update={"workers": 99})}
    )
    extraction_changed = config.model_copy(
        update={"image": config.image.model_copy(update={"jpeg_quality": 80})}
    )
    compatibility_changed = config.model_copy(
        update={
            "mcap_compatibility": config.mcap_compatibility.model_copy(
                update={"compatible_gnss_numeric_types": False}
            )
        }
    )

    assert extraction_config_hash(config) == extraction_config_hash(same)
    assert extraction_config_hash(config) == extraction_config_hash(execution_changed)
    assert extraction_config_hash(config) != extraction_config_hash(extraction_changed)
    assert extraction_config_hash(config) != extraction_config_hash(compatibility_changed)


def _manifest(tmp_path: Path) -> AcquisitionManifest:
    return AcquisitionManifest(
        source=SourceFingerprint(
            repo_id="owner/dataset",
            revision="a" * 40,
            repo_path="data/a.mcap",
            sha256="b" * 64,
            size=3,
        ),
        status="downloaded",
        artifact=ArtifactIdentity(
            cache_relative_path="objects/item.mcap", size=3, sha256="b" * 64
        ),
        requested_extraction_config_hash="b" * 64,
    )


def test_manifest_round_trip_and_malformed_manifest_is_a_cache_miss(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    expected = _manifest(tmp_path)

    write_manifest(path, expected)

    assert load_manifest(path) == expected
    path.write_text("{broken", encoding="utf-8")
    assert load_manifest(path) is None
    invalid = expected.to_dict()
    invalid["manifest_version"] = 2
    path.write_text(canonical_json(invalid) + "\n", encoding="utf-8")
    assert load_manifest(path) is None
    assert load_manifest(tmp_path / "missing.json") is None


def test_extraction_manifest_round_trip_requires_exact_fields(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    expected = ExtractionManifest(
        source=manifest.source,
        extraction_config_hash=manifest.requested_extraction_config_hash,
    )

    assert ExtractionManifest.from_dict(expected.to_dict()) == expected
    with pytest.raises(ValueError, match="extraction manifest"):
        ExtractionManifest.from_dict({"manifest_version": 1})


def test_manifest_hard_link_is_a_cache_miss(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    outside = tmp_path / "outside-manifest"
    write_manifest(outside, manifest)
    outside_content = outside.read_bytes()
    link = tmp_path / "manifest.json"
    os.link(outside, link)

    assert load_manifest(link) is None
    assert outside.read_bytes() == outside_content
