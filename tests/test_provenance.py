from __future__ import annotations

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
    extraction_cache_reusable,
    extraction_config_hash,
    load_extraction_manifest,
    load_manifest,
    record_extraction_complete,
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
    assert load_manifest(tmp_path / "missing.json") is None


def test_extraction_cache_requires_exact_source_and_config_hash(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = _manifest(tmp_path)
    record_extraction_complete(
        path, manifest.source, manifest.requested_extraction_config_hash
    )

    assert extraction_cache_reusable(
        path, manifest.source, manifest.requested_extraction_config_hash
    )
    assert not extraction_cache_reusable(
        path,
        SourceFingerprint(
            account_url=manifest.source.account_url,
            container=manifest.source.container,
            blob_path=manifest.source.blob_path,
            etag='"changed"',
            size=manifest.source.size,
        ),
        manifest.requested_extraction_config_hash,
    )
    assert not extraction_cache_reusable(path, manifest.source, "c" * 64)
    path.write_text("[]", encoding="utf-8")
    assert not extraction_cache_reusable(
        path, manifest.source, manifest.requested_extraction_config_hash
    )


def test_extraction_completion_manifest_round_trip(tmp_path: Path) -> None:
    acquisition = _manifest(tmp_path)
    path = tmp_path / "extraction.manifest.json"

    record_extraction_complete(
        path, acquisition.source, acquisition.requested_extraction_config_hash
    )

    assert load_extraction_manifest(path) == ExtractionManifest(
        source=acquisition.source,
        extraction_config_hash=acquisition.requested_extraction_config_hash,
    )


def test_extraction_manifest_symlink_is_a_miss_and_completion_replaces_link(
    tmp_path: Path,
) -> None:
    acquisition = _manifest(tmp_path)
    outside = tmp_path / "outside-extraction-manifest"
    record_extraction_complete(
        outside, acquisition.source, acquisition.requested_extraction_config_hash
    )
    outside_content = outside.read_text(encoding="utf-8")
    link = tmp_path / "extraction.manifest.json"
    link.symlink_to(outside)

    assert not extraction_cache_reusable(
        link, acquisition.source, acquisition.requested_extraction_config_hash
    )

    record_extraction_complete(
        link, acquisition.source, acquisition.requested_extraction_config_hash
    )

    assert outside.read_text(encoding="utf-8") == outside_content
    assert link.is_file()
    assert not link.is_symlink()
    assert extraction_cache_reusable(
        link, acquisition.source, acquisition.requested_extraction_config_hash
    )


def test_manifest_hard_link_is_a_cache_miss(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    outside = tmp_path / "outside-manifest"
    write_manifest(outside, manifest)
    outside_content = outside.read_bytes()
    link = tmp_path / "manifest.json"
    os.link(outside, link)

    assert load_manifest(link) is None
    assert outside.read_bytes() == outside_content
