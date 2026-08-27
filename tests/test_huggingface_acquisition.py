from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from dataset_devkit.config import HuggingFaceConfig
from dataset_devkit.huggingface_acquisition import (
    AcquisitionError,
    HuggingFaceAcquirer,
    IntegrityError,
)
from dataset_devkit.huggingface_manifest import ManifestEntry
from dataset_devkit.provenance import canonical_hash


class FakeDownload:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        filename = str(kwargs["filename"])
        local_dir = Path(str(kwargs["local_dir"]))
        target = local_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[filename])
        return str(target)


def make_acquirer(tmp_path: Path, download: FakeDownload) -> HuggingFaceAcquirer:
    return HuggingFaceAcquirer(
        huggingface=HuggingFaceConfig(
            repo_id="owner/dataset",
            revision="a" * 40,
            manifest_path="manifest.jsonl",
        ),
        cache_dir=tmp_path / "cache",
        extraction_config_hash="c" * 64,
        download_file=download,
    )


def test_acquire_uses_pinned_dataset_download_and_verifies_sha256(tmp_path: Path) -> None:
    payload = b"mcap-content"
    download = FakeDownload({"data/a.mcap": payload})
    acquirer = make_acquirer(tmp_path, download)
    entry = ManifestEntry("data/a.mcap", len(payload), hashlib.sha256(payload).hexdigest())

    result = acquirer.acquire(entry)

    assert result.artifact_path.read_bytes() == payload
    assert result.manifest.source.repo_path == entry.repo_path
    assert result.manifest.artifact.sha256 == entry.sha256
    assert download.calls == [
        {
            "repo_id": "owner/dataset",
            "repo_type": "dataset",
            "revision": "a" * 40,
            "filename": "data/a.mcap",
            "local_dir": result.artifact_path.parent / "download",
            "force_download": True,
        }
    ]


def test_load_entries_downloads_the_configured_manifest_at_the_pinned_commit(
    tmp_path: Path,
) -> None:
    row = (
        b'{"repo_path":"data/a.mcap","source_size":1,'
        b'"sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}\n'
    )
    download = FakeDownload({"manifest.jsonl": row})
    acquirer = make_acquirer(tmp_path, download)
    identity = canonical_hash(
        {
            "repo_id": "owner/dataset",
            "revision": "a" * 40,
            "manifest_path": "manifest.jsonl",
        }
    )
    local_dir = tmp_path / "cache" / "huggingface-manifests" / identity

    assert acquirer.load_entries() == (ManifestEntry("data/a.mcap", 1, "b" * 64),)
    assert download.calls == [
        {
            "repo_id": "owner/dataset",
            "repo_type": "dataset",
            "revision": "a" * 40,
            "filename": "manifest.jsonl",
            "local_dir": local_dir,
            "force_download": True,
        }
    ]


def test_manifest_scratch_is_isolated_by_repository_revision(tmp_path: Path) -> None:
    row = (
        b'{"repo_path":"data/a.mcap","source_size":1,'
        b'"sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}\n'
    )
    first_download = FakeDownload({"manifest.jsonl": row})
    second_download = FakeDownload({"manifest.jsonl": row})
    first = make_acquirer(tmp_path, first_download)
    second = HuggingFaceAcquirer(
        huggingface=HuggingFaceConfig(
            repo_id="owner/dataset", revision="c" * 40, manifest_path="manifest.jsonl"
        ),
        cache_dir=tmp_path / "cache",
        extraction_config_hash="c" * 64,
        download_file=second_download,
    )

    first.load_entries()
    second.load_entries()

    assert first_download.calls[0]["local_dir"] != second_download.calls[0]["local_dir"]


def test_acquire_rejects_hash_mismatch(tmp_path: Path) -> None:
    download = FakeDownload({"data/a.mcap": b"wrong"})
    acquirer = make_acquirer(tmp_path, download)
    entry = ManifestEntry("data/a.mcap", 5, "0" * 64)

    with pytest.raises(IntegrityError, match="SHA-256"):
        acquirer.acquire(entry)


def test_acquire_rejects_download_path_outside_owned_scratch(tmp_path: Path) -> None:
    outside = tmp_path / "outside.mcap"
    outside.write_bytes(b"x")

    def escape(**_: object) -> str:
        return str(outside)

    acquirer = HuggingFaceAcquirer(
        huggingface=HuggingFaceConfig(
            repo_id="owner/dataset", revision="a" * 40, manifest_path="manifest.jsonl"
        ),
        cache_dir=tmp_path / "cache",
        extraction_config_hash="c" * 64,
        download_file=escape,
    )

    with pytest.raises(AcquisitionError, match="outside"):
        acquirer.acquire(ManifestEntry("data/a.mcap", 1, hashlib.sha256(b"x").hexdigest()))


def test_acquire_rejects_size_mismatch(tmp_path: Path) -> None:
    download = FakeDownload({"data/a.mcap": b"content"})
    acquirer = make_acquirer(tmp_path, download)

    with pytest.raises(IntegrityError, match="size"):
        acquirer.acquire(
            ManifestEntry("data/a.mcap", 1, hashlib.sha256(b"content").hexdigest())
        )


def test_verified_cache_hit_avoids_a_second_download(tmp_path: Path) -> None:
    payload = b"mcap-content"
    download = FakeDownload({"data/a.mcap": payload})
    acquirer = make_acquirer(tmp_path, download)
    entry = ManifestEntry("data/a.mcap", len(payload), hashlib.sha256(payload).hexdigest())

    first = acquirer.acquire(entry)
    second = acquirer.acquire(entry)

    assert first.artifact_path == second.artifact_path
    assert second.manifest.status == "cache_hit"
    assert len(download.calls) == 1


def test_corrupt_cache_is_replaced_from_repository(tmp_path: Path) -> None:
    payload = b"mcap-content"
    download = FakeDownload({"data/a.mcap": payload})
    acquirer = make_acquirer(tmp_path, download)
    entry = ManifestEntry("data/a.mcap", len(payload), hashlib.sha256(payload).hexdigest())
    first = acquirer.acquire(entry)
    first.artifact_path.write_bytes(b"corrupt")

    second = acquirer.acquire(entry)

    assert second.artifact_path.read_bytes() == payload
    assert len(download.calls) == 2


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_acquire_rejects_linked_download_output(tmp_path: Path, kind: str) -> None:
    outside = tmp_path / "outside.mcap"
    outside.write_bytes(b"x")

    def linked(**kwargs: object) -> str:
        target = Path(str(kwargs["local_dir"])) / str(kwargs["filename"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if kind == "symlink":
            target.symlink_to(outside)
        else:
            os.link(outside, target)
        return str(target)

    acquirer = HuggingFaceAcquirer(
        huggingface=HuggingFaceConfig(
            repo_id="owner/dataset", revision="a" * 40, manifest_path="manifest.jsonl"
        ),
        cache_dir=tmp_path / "cache",
        extraction_config_hash="c" * 64,
        download_file=linked,
    )

    with pytest.raises(AcquisitionError, match="owned regular file"):
        acquirer.acquire(ManifestEntry("data/a.mcap", 1, hashlib.sha256(b"x").hexdigest()))


def test_acquire_rejects_symlinked_cache_ancestor(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "huggingface").symlink_to(outside, target_is_directory=True)
    payload = b"mcap-content"
    acquirer = HuggingFaceAcquirer(
        huggingface=HuggingFaceConfig(
            repo_id="owner/dataset", revision="a" * 40, manifest_path="manifest.jsonl"
        ),
        cache_dir=cache,
        extraction_config_hash="c" * 64,
        download_file=FakeDownload({"data/a.mcap": payload}),
    )

    with pytest.raises(AcquisitionError, match="cache component"):
        acquirer.acquire(
            ManifestEntry("data/a.mcap", len(payload), hashlib.sha256(payload).hexdigest())
        )
    assert not tuple(outside.iterdir())


def test_extraction_completion_is_source_and_config_bound(tmp_path: Path) -> None:
    payload = b"mcap-content"
    acquirer = make_acquirer(tmp_path, FakeDownload({"data/a.mcap": payload}))
    entry = ManifestEntry("data/a.mcap", len(payload), hashlib.sha256(payload).hexdigest())
    source = acquirer.acquire(entry).manifest.source

    assert not acquirer.extraction_cache_reusable(source, "d" * 64)
    completion = acquirer.record_extraction_complete(source, "d" * 64)

    assert completion.is_file()
    assert acquirer.extraction_cache_reusable(source, "d" * 64)
    assert not acquirer.extraction_cache_reusable(source, "e" * 64)

    outside = tmp_path / "outside-completion.json"
    completion.replace(outside)
    os.link(outside, completion)
    assert not acquirer.extraction_cache_reusable(source, "d" * 64)
