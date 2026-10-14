"""Selective, revision-pinned acquisition for integrated INDUS-AD data."""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import cast

import pyarrow.parquet as pq

from dataset_devkit.config import DenseAssetSelection, IndusAdHfSourceConfig
from dataset_devkit.huggingface_acquisition import (
    AcquisitionError,
    DownloadFile,
    IntegrityError,
    _atomic_write,
    _download_repository_file,
    _ensure_beneath,
    _recording_lock,
    _verify_file,
)
from dataset_devkit.indus_ad_annotations import (
    parse_camera_annotations,
    parse_dense_indexes,
)
from dataset_devkit.indus_ad_contract import (
    CAMERA_ANNOTATION_VIEWER_PATH,
    METADATA_PATH,
    VIDEO_INTEGRITY_PATH,
    VIDEO_PRIVACY_PATH,
    VIDEO_VIEWER_PATH,
    parse_indus_ad_contract,
)
from dataset_devkit.indus_ad_dense import (
    MaterializedDenseAsset,
    materialize_selected_dense_assets,
)
from dataset_devkit.indus_ad_models import (
    AnnotationAvailability,
    IndusAdAnnotationBundle,
    IndusAdRecordingEntry,
    IndusAdSelectionPlan,
    RepositoryFile,
)
from dataset_devkit.provenance import canonical_hash

ListRepositoryTree = Callable[[str, str], tuple[RepositoryFile, ...]]


def _git_blob_id(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def _default_list_tree(repo_id: str, revision: str) -> tuple[RepositoryFile, ...]:
    from huggingface_hub import HfApi, RepoFile

    relevant_prefixes = (
        "data/annotations/camera/dense-index/",
        "data/annotations/camera/semantic/",
        "data/annotations/camera/depth/",
        "data/tables/camera-frames/",
        "data/tables/calibrations/",
        "data/tables/gnss/",
        "data/videos/",
        "data/viewer/",
        "data/manifests/",
    )
    exact = {METADATA_PATH}
    result: list[RepositoryFile] = []
    api = HfApi()
    for item in api.list_repo_tree(
        repo_id,
        recursive=True,
        expand=True,
        revision=revision,
        repo_type="dataset",
    ):
        if not isinstance(item, RepoFile):
            continue
        if item.path not in exact and not item.path.startswith(relevant_prefixes):
            continue
        if Path(item.path).suffix not in {".json", ".jsonl", ".parquet", ".mp4", ".tar"}:
            continue
        if item.size is None or item.blob_id is None:
            raise AcquisitionError("Hub tree file lacks a stable identity")
        lfs_sha256 = item.lfs.sha256 if item.lfs is not None else None
        result.append(RepositoryFile(item.path, item.size, item.blob_id, lfs_sha256))
    return tuple(result)


def _verify_tree_file(
    *,
    path: Path,
    local_root: Path,
    identity: RepositoryFile,
) -> str:
    verified = _verify_file(path, local_root)
    if verified.size != identity.size:
        raise IntegrityError("downloaded size differs from pinned repository tree")
    data = path.read_bytes()
    if identity.sha256 is not None:
        if verified.sha256 != identity.sha256:
            raise IntegrityError("downloaded SHA-256 differs from pinned repository tree")
    elif _git_blob_id(data) != identity.blob_id:
        raise IntegrityError("downloaded blob ID differs from pinned repository tree")
    return verified.sha256


@dataclass(frozen=True)
class AcquiredIndusAdRecording:
    entry: IndusAdRecordingEntry
    files: Mapping[str, Path]
    annotations: IndusAdAnnotationBundle
    dense_mode: DenseAssetSelection
    materialize_dense_assets: Callable[
        [frozenset[str]], tuple[MaterializedDenseAsset, ...]
    ]

    def path(self, repo_path: str) -> Path:
        try:
            return self.files[repo_path]
        except KeyError as error:
            raise AcquisitionError("integrated bundle lacks an advertised artifact") from error


class IndusAdHfAcquirer:
    """Acquire controls first, then only artifacts selected by the validated plan."""

    def __init__(
        self,
        *,
        source: IndusAdHfSourceConfig,
        cache_dir: Path,
        list_tree: ListRepositoryTree | None = None,
        download_file: DownloadFile | None = None,
    ) -> None:
        self.source = source
        self.cache_dir = cache_dir.absolute()
        self.list_tree = _default_list_tree if list_tree is None else list_tree
        if download_file is None:
            from huggingface_hub import hf_hub_download

            download_file = hf_hub_download
        self.download_file = download_file
        self._plan: IndusAdSelectionPlan | None = None
        self._tree: dict[str, RepositoryFile] = {}
        self._control_files: dict[str, Path] = {}
        self._annotations = IndusAdAnnotationBundle()

    def _acquire_one(self, identity: RepositoryFile, namespace: str) -> Path:
        cache_identity = canonical_hash(
            {
                "repo_id": self.source.repo_id,
                "revision": self.source.revision,
                "path": identity.path,
                "size": identity.size,
                "blob_id": identity.blob_id,
                "sha256": identity.sha256,
            }
        )
        directory = _ensure_beneath(self.cache_dir, namespace, cache_identity)
        final = directory / f"artifact{Path(identity.path).suffix}"
        receipt = directory / "receipt.json"
        with _recording_lock(directory):
            if final.exists():
                digest = _verify_tree_file(
                    path=final, local_root=directory, identity=identity
                )
                if not receipt.exists():
                    _atomic_write(
                        receipt,
                        {
                            "repo_id": self.source.repo_id,
                            "revision": self.source.revision,
                            "path": identity.path,
                            "size": identity.size,
                            "blob_id": identity.blob_id,
                            "sha256": digest,
                        },
                    )
                return final
            scratch = _ensure_beneath(directory, "download")
            returned = _download_repository_file(
                self.download_file,
                repo_id=self.source.repo_id,
                revision=self.source.revision,
                filename=identity.path,
                local_dir=scratch,
            )
            digest = _verify_tree_file(path=returned, local_root=scratch, identity=identity)
            os.replace(returned, final)
            promoted = _verify_tree_file(
                path=final, local_root=directory, identity=identity
            )
            if promoted != digest:
                raise IntegrityError("promoted INDUS-AD artifact changed identity")
            _atomic_write(
                receipt,
                {
                    "repo_id": self.source.repo_id,
                    "revision": self.source.revision,
                    "path": identity.path,
                    "size": identity.size,
                    "blob_id": identity.blob_id,
                    "sha256": digest,
                },
            )
            return final

    def load_plan(self) -> IndusAdSelectionPlan:
        if self._plan is not None:
            return self._plan
        tree = self.list_tree(self.source.repo_id, self.source.revision)
        self._tree = {item.path: item for item in tree}
        dense_indexes = sorted(
            path
            for path in self._tree
            if path.startswith("data/annotations/camera/dense-index/")
            and path.endswith(".parquet")
        )
        controls = [
            METADATA_PATH,
            VIDEO_VIEWER_PATH,
            CAMERA_ANNOTATION_VIEWER_PATH,
            VIDEO_INTEGRITY_PATH,
            VIDEO_PRIVACY_PATH,
            *dense_indexes,
        ]
        self._control_files = {
            path: self._acquire_one(self._tree[path], "indus-ad-control")
            for path in controls
            if path in self._tree
        }
        root = _ensure_beneath(
            self.cache_dir,
            "indus-ad-control-view",
            canonical_hash(
                {"repo_id": self.source.repo_id, "revision": self.source.revision}
            ),
        )
        for repo_path, cached in self._control_files.items():
            destination = root / repo_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copyfile(cached, destination)
        self._plan = parse_indus_ad_contract(root, self.source, tree)
        annotation_table = pq.ParquetFile(
            self._control_files[CAMERA_ANNOTATION_VIEWER_PATH]
        ).read(columns=["recording_id", "recording_offset_ns"])
        durations: dict[str, int] = {}
        for row in annotation_table.to_pylist():
            recording_id = cast(str, row["recording_id"])
            offset = cast(int, row["recording_offset_ns"])
            durations[recording_id] = max(durations.get(recording_id, 1), offset + 1)
        camera_bundle = parse_camera_annotations(
            self._control_files[CAMERA_ANNOTATION_VIEWER_PATH],
            expected_recordings=durations,
        )
        dense_refs = parse_dense_indexes(
            [self._control_files[path] for path in dense_indexes],
            camera_annotations=camera_bundle.camera,
        )
        self._annotations = replace(
            camera_bundle,
            dense_refs=dense_refs,
            availability={
                **camera_bundle.availability,
                "dense": AnnotationAvailability("available", "dense_indexes_loaded"),
            },
        )
        return self._plan

    def acquire(self, entry: IndusAdRecordingEntry) -> AcquiredIndusAdRecording:
        self.load_plan()
        selected_artifacts = [
            item
            for item in (*entry.videos, *entry.artifacts)
            if item.kind in {"video", "camera_frames", "calibration", "gnss"}
        ]
        files = {
            item.path: self._acquire_one(self._tree[item.path], "indus-ad-payload")
            for item in selected_artifacts
        }
        files.update(self._control_files)
        counts = {
            item.camera: pq.ParquetFile(files[item.path]).metadata.num_rows
            for item in entry.artifacts_of("camera_frames")
        }
        acquired_entry = replace(
            entry,
            videos=tuple(
                replace(video, frame_count=counts.get(video.camera))
                for video in entry.videos
            ),
        )
        annotations = IndusAdAnnotationBundle(
            camera=tuple(
                item
                for item in self._annotations.camera
                if item.recording_id == entry.recording_id
            ),
            gnss=tuple(
                item
                for item in self._annotations.gnss
                if item.recording_id == entry.recording_id
            ),
            dense_refs=tuple(
                item
                for item in self._annotations.dense_refs
                if item.recording_id == entry.recording_id
            ),
            availability=self._annotations.availability,
        )

        def materialize(frame_ids: frozenset[str]) -> tuple[MaterializedDenseAsset, ...]:
            enabled = {
                kind
                for kind, mode in (
                    ("semantic", self.source.dense_assets.semantic),
                    ("depth", self.source.dense_assets.depth),
                )
                if mode == "selected_frames"
            }
            selected_refs = tuple(
                ref
                for ref in annotations.dense_refs
                if ref.frame_id in frame_ids and ref.kind in enabled
            )
            if not selected_refs:
                return ()
            archive_paths = {
                shard_path: self._acquire_one(
                    self._tree[shard_path], "indus-ad-dense-archive"
                )
                for shard_path in sorted({ref.shard_path for ref in selected_refs})
            }
            return materialize_selected_dense_assets(
                selected_refs=selected_refs,
                all_refs=self._annotations.dense_refs,
                archive_paths=archive_paths,
                output_dir=_ensure_beneath(self.cache_dir, "indus-ad-dense-members"),
            )

        return AcquiredIndusAdRecording(
            acquired_entry,
            MappingProxyType(files),
            annotations,
            self.source.dense_assets,
            materialize,
        )
