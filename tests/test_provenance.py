from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import pytest

from dataset_devkit.config import GlobalConfig
from dataset_devkit.provenance import (
    AcquisitionManifest,
    ArtifactIdentity,
    IntegrityVerification,
    SourceFingerprint,
    canonical_json,
    extraction_cache_reusable,
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
        extraction_config_hash="b" * 64,
    )


def test_manifest_round_trip_and_malformed_manifest_is_a_cache_miss(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    expected = _manifest(tmp_path)

    write_manifest(path, expected)

    assert load_manifest(path) == expected
    path.write_text("{broken", encoding="utf-8")
    assert load_manifest(path) is None
    assert load_manifest(tmp_path / "missing.json") is None


def test_extraction_cache_requires_exact_source_and_config_hash(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = _manifest(tmp_path)
    write_manifest(path, manifest)

    assert extraction_cache_reusable(path, manifest.source, manifest.extraction_config_hash)
    assert not extraction_cache_reusable(
        path,
        SourceFingerprint(
            account_url=manifest.source.account_url,
            container=manifest.source.container,
            blob_path=manifest.source.blob_path,
            etag='"changed"',
            size=manifest.source.size,
        ),
        manifest.extraction_config_hash,
    )
    assert not extraction_cache_reusable(path, manifest.source, "c" * 64)
    path.write_text("[]", encoding="utf-8")
    assert not extraction_cache_reusable(path, manifest.source, manifest.extraction_config_hash)
