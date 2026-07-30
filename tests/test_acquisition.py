from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import pytest
from azure.core import MatchConditions

from dataset_devkit.acquisition import (
    AcquisitionError,
    AzureBlobAcquirer,
    BlobChangedError,
    BlobClientProtocol,
    DownloadProtocol,
    IntegrityError,
)
from dataset_devkit.config import AzureConfig
from dataset_devkit.provenance import (
    SourceFingerprint,
    canonical_json,
    extraction_cache_reusable,
    load_manifest,
    record_extraction_complete,
)


@dataclass(frozen=True)
class FakeContentSettings:
    content_md5: bytes | None = None


@dataclass(frozen=True)
class FakeProperties:
    size: int
    etag: str
    content_settings: FakeContentSettings


class FakeDownload:
    def __init__(self, payload: bytes, offset: int, failure: Exception | None) -> None:
        self.payload = payload
        self.offset = offset
        self.failure = failure

    def readinto(self, stream: IO[bytes]) -> int:
        if self.failure is not None:
            stream.write(self.payload[self.offset : self.offset + 2])
            raise self.failure
        return stream.write(self.payload[self.offset :])


class FakeBlobClient:
    def __init__(self, payload: bytes, properties: list[FakeProperties]) -> None:
        self.payload = payload
        self.properties = properties
        self.property_calls = 0
        self.download_offsets: list[int] = []
        self.download_kwargs: list[dict[str, object]] = []
        self.download_failure: Exception | None = None
        self.properties_failure: Exception | None = None

    def get_blob_properties(self) -> FakeProperties:
        if self.properties_failure is not None:
            raise self.properties_failure
        value = self.properties[min(self.property_calls, len(self.properties) - 1)]
        self.property_calls += 1
        return value

    def download_blob(self, *, offset: int, **kwargs: object) -> DownloadProtocol:
        self.download_offsets.append(offset)
        self.download_kwargs.append(kwargs)
        return FakeDownload(self.payload, offset, self.download_failure)


class FakeBlobServiceClient:
    def __init__(self, client: FakeBlobClient) -> None:
        self.client = client
        self.requests: list[tuple[str, str]] = []

    def get_blob_client(self, *, container: str, blob: str) -> BlobClientProtocol:
        self.requests.append((container, blob))
        return self.client


def properties(payload: bytes, etag: str = '"etag-1"', *, md5: bool = True) -> FakeProperties:
    digest = hashlib.md5(payload, usedforsecurity=False).digest() if md5 else None
    return FakeProperties(len(payload), etag, FakeContentSettings(digest))


def acquirer(
    tmp_path: Path,
    client: FakeBlobClient,
    extraction_hash: str = "a" * 64,
) -> AzureBlobAcquirer:
    azure = AzureConfig(
        account_url="https://data.blob.core.windows.net",
        container="recordings",
        blob_list=tmp_path / "blobs.txt",
    )
    return AzureBlobAcquirer(
        azure=azure,
        cache_dir=tmp_path / "cache",
        extraction_config_hash=extraction_hash,
        service_client=FakeBlobServiceClient(client),
    )


def test_fresh_download_is_verified_and_atomically_finalized(tmp_path: Path) -> None:
    payload = b"complete-mcap"
    prop = properties(payload)
    client = FakeBlobClient(payload, [prop, prop])
    downloader = acquirer(tmp_path, client)

    result = downloader.acquire("mcap-h265/fleet/a.mcap")

    assert result.manifest.status == "downloaded"
    assert result.artifact_path.read_bytes() == payload
    assert result.manifest.artifact.sha256 == hashlib.sha256(payload).hexdigest()
    assert result.manifest.integrity.method == "content_md5"
    assert result.manifest.integrity.content_md5 == base64.b64encode(
        prop.content_settings.content_md5 or b""
    ).decode("ascii")
    assert load_manifest(result.manifest_path) == result.manifest
    paths = downloader.paths_for(result.manifest.source)
    assert not paths.partial.exists()
    assert not paths.partial_sidecar.exists()
    assert client.download_offsets == [0]
    assert client.download_kwargs == [
        {
            "etag": prop.etag,
            "match_condition": MatchConditions.IfNotModified,
            "validate_content": True,
        }
    ]


def test_exact_fingerprint_cache_hit_reverifies_without_downloading(tmp_path: Path) -> None:
    payload = b"cached-mcap"
    prop = properties(payload)
    client = FakeBlobClient(payload, [prop, prop, prop])
    downloader = acquirer(tmp_path, client)
    first = downloader.acquire("mcap-h265/fleet/a.mcap")

    second = downloader.acquire("mcap-h265/fleet/a.mcap")

    assert second.artifact_path == first.artifact_path
    assert second.manifest.status == "cache_hit"
    assert client.download_offsets == [0]


@pytest.mark.parametrize(
    ("new_payload", "new_etag"),
    [(b"new", '"etag-2"'), (b"longer-payload", '"etag-1"')],
)
def test_fingerprint_change_downloads_a_new_cache_object(
    tmp_path: Path, new_payload: bytes, new_etag: str
) -> None:
    old_payload = b"old"
    old_prop = properties(old_payload)
    new_prop = properties(new_payload, new_etag)
    client = FakeBlobClient(old_payload, [old_prop, old_prop])
    downloader = acquirer(tmp_path, client)
    first = downloader.acquire("mcap-h265/fleet/a.mcap")
    client.payload = new_payload
    client.properties = [new_prop, new_prop]
    client.property_calls = 0

    second = downloader.acquire("mcap-h265/fleet/a.mcap")

    assert second.manifest.status == "downloaded"
    assert second.artifact_path != first.artifact_path
    assert second.artifact_path.read_bytes() == new_payload
    assert client.download_offsets == [0, 0]


def write_partial(
    downloader: AzureBlobAcquirer, source: SourceFingerprint, content: bytes
) -> None:
    paths = downloader.paths_for(source)
    paths.directory.mkdir(parents=True, exist_ok=True)
    paths.partial.write_bytes(content)
    paths.partial_sidecar.write_text(canonical_json(source.to_dict()) + "\n", encoding="utf-8")


def test_compatible_partial_is_resumed_with_a_ranged_read(tmp_path: Path) -> None:
    payload = b"resume-this"
    prop = properties(payload)
    client = FakeBlobClient(payload, [prop, prop])
    downloader = acquirer(tmp_path, client)
    source = downloader.fingerprint_for("mcap-h265/a.mcap", prop)
    write_partial(downloader, source, payload[:6])

    result = downloader.acquire("mcap-h265/a.mcap")

    assert result.manifest.status == "resumed"
    assert result.artifact_path.read_bytes() == payload
    assert client.download_offsets == [6]


def test_incompatible_partial_is_discarded_and_restarted(tmp_path: Path) -> None:
    payload = b"current"
    prop = properties(payload, '"new"')
    client = FakeBlobClient(payload, [prop, prop])
    downloader = acquirer(tmp_path, client)
    current = downloader.fingerprint_for("mcap-h265/a.mcap", prop)
    stale = SourceFingerprint(
        current.account_url, current.container, current.blob_path, '"old"', current.size
    )
    write_partial(downloader, stale, b"stale")

    result = downloader.acquire("mcap-h265/a.mcap")

    assert result.manifest.status == "downloaded"
    assert result.artifact_path.read_bytes() == payload
    assert client.download_offsets == [0]


def test_pre_and_post_download_property_change_is_rejected(tmp_path: Path) -> None:
    payload = b"changing"
    before = properties(payload, '"before"')
    after = properties(payload, '"after"')
    client = FakeBlobClient(payload, [before, after])
    downloader = acquirer(tmp_path, client)

    with pytest.raises(BlobChangedError, match="changed"):
        downloader.acquire("mcap-h265/a.mcap")

    assert not downloader.paths_for(
        downloader.fingerprint_for("mcap-h265/a.mcap", before)
    ).final.exists()


def test_size_mismatch_never_finalizes(tmp_path: Path) -> None:
    payload = b"short"
    claimed = FakeProperties(len(payload) + 1, '"etag"', FakeContentSettings(None))
    client = FakeBlobClient(payload, [claimed, claimed])
    downloader = acquirer(tmp_path, client)

    with pytest.raises(IntegrityError, match="size"):
        downloader.acquire("mcap-h265/a.mcap")

    assert not downloader.paths_for(
        downloader.fingerprint_for("mcap-h265/a.mcap", claimed)
    ).final.exists()


def test_md5_mismatch_fails_closed_and_never_finalizes(tmp_path: Path) -> None:
    payload = b"payload"
    wrong = FakeProperties(len(payload), '"etag"', FakeContentSettings(b"x" * 16))
    client = FakeBlobClient(payload, [wrong, wrong])
    downloader = acquirer(tmp_path, client)

    with pytest.raises(IntegrityError, match="MD5"):
        downloader.acquire("mcap-h265/a.mcap")

    assert not downloader.paths_for(
        downloader.fingerprint_for("mcap-h265/a.mcap", wrong)
    ).final.exists()


def test_manifest_reports_when_azure_integrity_metadata_is_unavailable(tmp_path: Path) -> None:
    payload = b"payload"
    prop = properties(payload, md5=False)
    client = FakeBlobClient(payload, [prop, prop])

    result = acquirer(tmp_path, client).acquire("mcap-h265/a.mcap")

    assert result.manifest.integrity.method == "size_etag"
    assert result.manifest.integrity.verified
    assert result.manifest.integrity.content_md5 is None


def test_azure_download_failure_never_produces_final_artifact(tmp_path: Path) -> None:
    payload = b"payload"
    prop = properties(payload)
    client = FakeBlobClient(payload, [prop])
    client.download_failure = RuntimeError("transport failed")
    downloader = acquirer(tmp_path, client)
    source = downloader.fingerprint_for("mcap-h265/a.mcap", prop)

    with pytest.raises(AcquisitionError, match="download"):
        downloader.acquire("mcap-h265/a.mcap")

    paths = downloader.paths_for(source)
    assert not paths.final.exists()
    assert paths.partial.exists()
    assert paths.partial_sidecar.exists()


def test_azure_property_read_failure_never_produces_cache_files(tmp_path: Path) -> None:
    payload = b"payload"
    prop = properties(payload)
    client = FakeBlobClient(payload, [prop])
    client.properties_failure = RuntimeError("properties failed")
    downloader = acquirer(tmp_path, client)

    with pytest.raises(AcquisitionError, match="properties"):
        downloader.acquire("mcap-h265/a.mcap")

    assert not (tmp_path / "cache").exists()


def test_corrupt_cached_artifact_is_removed_and_redownloaded(tmp_path: Path) -> None:
    payload = b"payload"
    prop = properties(payload)
    client = FakeBlobClient(payload, [prop, prop, prop, prop])
    downloader = acquirer(tmp_path, client)
    first = downloader.acquire("mcap-h265/a.mcap")
    first.artifact_path.write_bytes(b"corrupt")

    second = downloader.acquire("mcap-h265/a.mcap")

    assert second.manifest.status == "downloaded"
    assert second.artifact_path.read_bytes() == payload
    assert client.download_offsets == [0, 0]


@pytest.mark.parametrize("unsafe", ["../x.mcap", "/mcap-h265/x.mcap", r"mcap-h265\x.mcap"])
def test_acquisition_rejects_unsafe_paths_before_azure_access(
    tmp_path: Path, unsafe: str
) -> None:
    payload = b"payload"
    prop = properties(payload)
    client = FakeBlobClient(payload, [prop])
    downloader = acquirer(tmp_path, client)

    with pytest.raises(ValueError, match="invalid"):
        downloader.acquire(unsafe)

    assert client.property_calls == 0


def test_cache_paths_are_hash_derived_and_cannot_escape_cache_root(tmp_path: Path) -> None:
    payload = b"payload"
    prop = properties(payload)
    downloader = acquirer(tmp_path, FakeBlobClient(payload, [prop]))
    source = SourceFingerprint(
        account_url="https://data.blob.core.windows.net",
        container="recordings",
        blob_path="mcap-h265/../../escape.mcap",
        etag='"etag"',
        size=len(payload),
    )

    paths = downloader.paths_for(source)

    assert paths.directory.is_relative_to((tmp_path / "cache").resolve())
    assert ".." not in paths.final.parts


def test_cache_layout_rejects_a_preexisting_directory_symlink_escape(tmp_path: Path) -> None:
    payload = b"payload"
    prop = properties(payload)
    downloader = acquirer(tmp_path, FakeBlobClient(payload, [prop]))
    source = downloader.fingerprint_for("mcap-h265/a.mcap", prop)
    paths = downloader.paths_for(source)
    paths.directory.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    paths.directory.symlink_to(outside, target_is_directory=True)

    with pytest.raises(AcquisitionError, match="escape"):
        downloader.paths_for(source)


def test_partial_symlink_is_unlinked_without_modifying_target_and_download_restarts(
    tmp_path: Path,
) -> None:
    payload = b"safe-payload"
    prop = properties(payload)
    client = FakeBlobClient(payload, [prop, prop])
    downloader = acquirer(tmp_path, client)
    source = downloader.fingerprint_for("mcap-h265/a.mcap", prop)
    paths = downloader.paths_for(source)
    paths.directory.mkdir(parents=True)
    outside = tmp_path / "outside-partial"
    outside.write_bytes(b"evil")
    paths.partial.symlink_to(outside)
    paths.partial_sidecar.write_text(
        canonical_json(source.to_dict()) + "\n", encoding="utf-8"
    )

    result = downloader.acquire(source.blob_path)

    assert outside.read_bytes() == b"evil"
    assert client.download_offsets == [0]
    assert result.artifact_path.read_bytes() == payload
    assert result.artifact_path.is_file()
    assert not result.artifact_path.is_symlink()


def test_partial_sidecar_symlink_is_not_read_and_download_restarts(tmp_path: Path) -> None:
    payload = b"safe-payload"
    prop = properties(payload)
    client = FakeBlobClient(payload, [prop, prop])
    downloader = acquirer(tmp_path, client)
    source = downloader.fingerprint_for("mcap-h265/a.mcap", prop)
    paths = downloader.paths_for(source)
    paths.directory.mkdir(parents=True)
    paths.partial.write_bytes(payload[:4])
    outside = tmp_path / "outside-sidecar"
    outside.write_text(canonical_json(source.to_dict()) + "\n", encoding="utf-8")
    paths.partial_sidecar.symlink_to(outside)

    result = downloader.acquire(source.blob_path)

    assert outside.read_text(encoding="utf-8") == canonical_json(source.to_dict()) + "\n"
    assert client.download_offsets == [0]
    assert result.artifact_path.read_bytes() == payload
    assert not result.artifact_path.is_symlink()


def test_existing_final_symlink_is_not_read_or_exposed_as_cache_hit(tmp_path: Path) -> None:
    payload = b"safe-payload"
    prop = properties(payload)
    client = FakeBlobClient(payload, [prop, prop, prop, prop])
    downloader = acquirer(tmp_path, client)
    first = downloader.acquire("mcap-h265/a.mcap")
    outside = tmp_path / "outside-final"
    outside.write_bytes(payload)
    first.artifact_path.unlink()
    first.artifact_path.symlink_to(outside)

    second = downloader.acquire("mcap-h265/a.mcap")

    assert outside.read_bytes() == payload
    assert client.download_offsets == [0, 0]
    assert second.manifest.status == "downloaded"
    assert second.artifact_path.read_bytes() == payload
    assert not second.artifact_path.is_symlink()


def test_acquisition_manifest_symlink_is_not_read_or_persisted(tmp_path: Path) -> None:
    payload = b"safe-payload"
    prop = properties(payload)
    client = FakeBlobClient(payload, [prop, prop, prop, prop])
    downloader = acquirer(tmp_path, client)
    first = downloader.acquire("mcap-h265/a.mcap")
    serialized_manifest = first.manifest_path.read_text(encoding="utf-8")
    outside = tmp_path / "outside-manifest"
    outside.write_text(serialized_manifest, encoding="utf-8")
    first.manifest_path.unlink()
    first.manifest_path.symlink_to(outside)

    second = downloader.acquire("mcap-h265/a.mcap")

    assert outside.read_text(encoding="utf-8") == serialized_manifest
    assert client.download_offsets == [0, 0]
    assert second.manifest_path.is_file()
    assert not second.manifest_path.is_symlink()
    assert not second.artifact_path.is_symlink()


def test_cache_hit_does_not_claim_extraction_complete_for_new_config(tmp_path: Path) -> None:
    payload = b"safe-payload"
    prop = properties(payload)
    client = FakeBlobClient(payload, [prop, prop, prop])
    hash_a = "a" * 64
    hash_b = "b" * 64
    first = acquirer(tmp_path, client, hash_a).acquire("mcap-h265/a.mcap")

    assert not extraction_cache_reusable(
        first.extraction_manifest_path, first.manifest.source, hash_a
    )
    record_extraction_complete(first.extraction_manifest_path, first.manifest.source, hash_a)
    assert extraction_cache_reusable(
        first.extraction_manifest_path, first.manifest.source, hash_a
    )

    second = acquirer(tmp_path, client, hash_b).acquire("mcap-h265/a.mcap")

    assert second.manifest.status == "cache_hit"
    assert second.manifest.requested_extraction_config_hash == hash_b
    assert not extraction_cache_reusable(
        second.extraction_manifest_path, second.manifest.source, hash_b
    )
    assert extraction_cache_reusable(
        second.extraction_manifest_path, second.manifest.source, hash_a
    )


def test_unsafe_leaf_swap_during_finalization_is_cleaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"safe-payload"
    prop = properties(payload)
    client = FakeBlobClient(payload, [prop, prop])
    downloader = acquirer(tmp_path, client)
    source = downloader.fingerprint_for("mcap-h265/a.mcap", prop)
    paths = downloader.paths_for(source)
    outside = tmp_path / "outside-finalization"
    outside.write_bytes(b"must-not-change")
    real_replace = os.replace

    def replace_with_symlink(
        source_path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination_path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    ) -> None:
        if Path(os.fsdecode(destination_path)) == paths.final:
            Path(os.fsdecode(source_path)).unlink()
            paths.final.symlink_to(outside)
            return
        real_replace(source_path, destination_path)

    monkeypatch.setattr(os, "replace", replace_with_symlink)

    with pytest.raises(AcquisitionError, match="final"):
        downloader.acquire(source.blob_path)

    assert outside.read_bytes() == b"must-not-change"
    assert not paths.final.exists()
    assert not paths.final.is_symlink()
