from __future__ import annotations

import base64
import json
import math
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from dataset_devkit.config import GlobalConfig
from dataset_devkit.provenance import (
    AcquisitionManifest,
    ArtifactIdentity,
    ExtractionManifest,
    IntegrityVerification,
    SourceFingerprint,
    canonical_json,
    extraction_config_hash,
    load_manifest,
    write_manifest,
)


def test_canonical_json_and_source_fingerprint_are_deterministic() -> None:
    assert canonical_json({"z": 1, "a": [True, None]}) == '{"a":[true,null],"z":1}'
    first = SourceFingerprint(
        account_url="https://data.blob.core.windows.net",
        container="recordings",
        blob_path="mcap-h265/fleet/a.mcap",
        etag='"abc"',
        size=12,
    )
    second = SourceFingerprint.from_dict(json.loads(canonical_json(first.to_dict())))

    assert first == second
    assert first.digest == second.digest
    assert first.cache_key != SourceFingerprint(
        account_url=first.account_url,
        container=first.container,
        blob_path="mcap-h265/fleet/b.mcap",
        etag=first.etag,
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

    assert extraction_config_hash(config) == extraction_config_hash(same)
    assert extraction_config_hash(config) == extraction_config_hash(execution_changed)
    assert extraction_config_hash(config) != extraction_config_hash(extraction_changed)


def _manifest(tmp_path: Path) -> AcquisitionManifest:
    return AcquisitionManifest(
        source=SourceFingerprint(
            account_url="https://data.blob.core.windows.net",
            container="recordings",
            blob_path="mcap-h265/a.mcap",
            etag='"abc"',
            size=3,
        ),
        status="downloaded",
        artifact=ArtifactIdentity(
            cache_relative_path="objects/item.mcap", size=3, sha256="a" * 64
        ),
        integrity=IntegrityVerification(method="size_etag", verified=True, content_md5=None),
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
    invalid["integrity"] = {
        "method": "size_etag",
        "verified": False,
        "content_md5": None,
    }
    path.write_text(canonical_json(invalid) + "\n", encoding="utf-8")
    assert load_manifest(path) is None
    assert load_manifest(tmp_path / "missing.json") is None


@pytest.mark.parametrize(
    "integrity",
    [
        {"method": "size_etag", "verified": False, "content_md5": None},
        {"method": "size_etag", "verified": True, "content_md5": "claimed"},
        {"method": "content_md5", "verified": False, "content_md5": "claimed"},
        {"method": "content_md5", "verified": True, "content_md5": None},
        {"method": "content_md5", "verified": True, "content_md5": "not-base64"},
        {
            "method": "content_md5",
            "verified": True,
            "content_md5": base64.b64encode(b"too-short").decode("ascii"),
        },
    ],
)
def test_integrity_verification_rejects_inconsistent_or_malformed_values(
    integrity: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="integrity"):
        IntegrityVerification.from_dict(integrity)


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
