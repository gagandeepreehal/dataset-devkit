"""Selective acquisition for privacy-transformed decoded V2 artifacts."""

from __future__ import annotations

import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from dataset_devkit.config import DecodedHfSourceConfig, GlobalConfigV2
from dataset_devkit.decoded_manifest import (
    DecodedRecordingEntry,
    DecodedSelectionPlan,
    parse_decoded_control_plane,
)
from dataset_devkit.huggingface_acquisition import (
    AcquisitionError,
    DownloadFile,
    VerifiedHfFileCache,
    _download_repository_file,
    _ensure_beneath,
    _recording_lock,
    _verify_file,
)
from dataset_devkit.provenance import canonical_hash

_CONTROL_PATHS = (
    "data/recordings.parquet",
    "data/manifests/output-files.parquet",
    "data/manifests/processing-config.json",
    "data/manifests/model-receipt.json",
    "data/manifests/schema-audit.parquet",
)


@dataclass(frozen=True, slots=True)
class AcquiredDecodedRecording:
    entry: DecodedRecordingEntry
    files: Mapping[str, Path]

    def path(self, repo_path: str) -> Path:
        try:
            return self.files[repo_path]
        except KeyError as error:
            raise AcquisitionError("decoded bundle lacks an advertised artifact") from error


class DecodedHfAcquirer:
    """Acquire a validated decoded control plane and only selected payloads."""

    def __init__(
        self,
        *,
        source: DecodedHfSourceConfig,
        cache_dir: Path,
        download_file: DownloadFile | None = None,
    ) -> None:
        self.source = source
        self.cache_dir = cache_dir.absolute()
        if download_file is None:
            from huggingface_hub import hf_hub_download

            download_file = hf_hub_download
        self.download_file = download_file

    @classmethod
    def from_config(cls, config: GlobalConfigV2) -> DecodedHfAcquirer:
        if not isinstance(config.source, DecodedHfSourceConfig):
            raise AcquisitionError("decoded acquirer requires a decoded_hf source")
        return cls(source=config.source, cache_dir=config.paths.cache_dir)

    def _acquire_control_files(self) -> Path:
        identity = canonical_hash(
            {"repo_id": self.source.repo_id, "revision": self.source.revision}
        )
        directory = _ensure_beneath(self.cache_dir, "decoded-control", identity)
        with _recording_lock(directory):
            snapshot = _ensure_beneath(directory, f"snapshot-{secrets.token_hex(12)}")
            for repo_path in _CONTROL_PATHS:
                returned = _download_repository_file(
                    self.download_file,
                    repo_id=self.source.repo_id,
                    revision=self.source.revision,
                    filename=repo_path,
                    local_dir=snapshot,
                )
                verified = _verify_file(returned, snapshot)
                if verified.size <= 0:
                    raise AcquisitionError("decoded control file is empty")
            return snapshot

    def load_plan(self) -> DecodedSelectionPlan:
        control_root = self._acquire_control_files()
        return parse_decoded_control_plane(control_root, self.source)

    def acquire(self, entry: DecodedRecordingEntry) -> AcquiredDecodedRecording:
        cache = VerifiedHfFileCache(self.cache_dir, self.download_file)
        ordered = (*entry.videos, *entry.artifacts)
        files = {
            artifact.path: cache.acquire(
                repo_id=self.source.repo_id,
                revision=self.source.revision,
                artifact=artifact,
            ).path
            for artifact in ordered
        }
        if len(files) != len(ordered):
            raise AcquisitionError("decoded bundle contains duplicate artifact paths")
        return AcquiredDecodedRecording(entry, MappingProxyType(files))
