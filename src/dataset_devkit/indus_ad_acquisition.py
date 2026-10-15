"""Selective, revision-pinned acquisition for integrated INDUS-AD data."""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import cast

import pyarrow.parquet as pq

from dataset_devkit.config import (
    DenseAssetSelection,
    GlobalConfigV21,
    IndusAdHfSourceConfig,
)
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
    parse_gnss_scenarios,
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


def _default_companion_tree(repo_id: str, revision: str) -> tuple[RepositoryFile, ...]:
    from huggingface_hub import HfApi, RepoFile

    return tuple(
        RepositoryFile(
            item.path,
            item.size,
            item.blob_id,
            None if item.lfs is None else item.lfs.sha256,
        )
        for item in HfApi().list_repo_tree(
            repo_id, recursive=True, revision=revision, repo_type="dataset"
        )
        if isinstance(item, RepoFile)
        and item.path.startswith("data/")
        and item.path.endswith(".parquet")
    )


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
        expand=False,
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
    expected_sha256: str | None = None,
) -> str:
    verified = _verify_file(path, local_root)
    if verified.size != identity.size:
        raise IntegrityError("downloaded size differs from pinned repository tree")
    if identity.sha256 is not None:
        if verified.sha256 != identity.sha256:
            raise IntegrityError("downloaded SHA-256 differs from pinned repository tree")
    elif _git_blob_id(path.read_bytes()) != identity.blob_id:
        raise IntegrityError("downloaded blob ID differs from pinned repository tree")
    if expected_sha256 is not None and verified.sha256 != expected_sha256:
        raise IntegrityError("downloaded SHA-256 differs from artifact integrity manifest")
    return verified.sha256


@dataclass(frozen=True)
class AcquiredIndusAdRecording:
    entry: IndusAdRecordingEntry
    files: Mapping[str, Path]
    annotations: IndusAdAnnotationBundle
    dense_mode: DenseAssetSelection
    materialize_dense_assets: Callable[[frozenset[str]], tuple[MaterializedDenseAsset, ...]]

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
        companion_list_tree: ListRepositoryTree | None = None,
    ) -> None:
        self.source = source
        self.cache_dir = cache_dir.resolve()
        self.list_tree = _default_list_tree if list_tree is None else list_tree
        self.companion_list_tree = (
            _default_companion_tree if companion_list_tree is None else companion_list_tree
        )
        if download_file is None:
            from huggingface_hub import hf_hub_download

            download_file = hf_hub_download
        self.download_file = download_file
        self._plan: IndusAdSelectionPlan | None = None
        self._tree: dict[str, RepositoryFile] = {}
        self._control_files: dict[str, Path] = {}
        self._annotations = IndusAdAnnotationBundle()

    @classmethod
    def from_config(cls, config: GlobalConfigV21) -> IndusAdHfAcquirer:
        if not isinstance(config.source, IndusAdHfSourceConfig):
            raise AcquisitionError("INDUS-AD acquirer requires an indus_ad_hf source")
        return cls(source=config.source, cache_dir=config.paths.cache_dir)

    def _acquire_one(
        self, identity: RepositoryFile, namespace: str, *, expected_sha256: str | None = None
    ) -> Path:
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
                    path=final,
                    local_root=directory,
                    identity=identity,
                    expected_sha256=expected_sha256,
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
            digest = _verify_tree_file(
                path=returned,
                local_root=scratch,
                identity=identity,
                expected_sha256=expected_sha256,
            )
            os.replace(returned, final)
            promoted = _verify_tree_file(
                path=final,
                local_root=directory,
                identity=identity,
                expected_sha256=expected_sha256,
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
            if path.startswith("data/annotations/camera/dense-index/") and path.endswith(".parquet")
        )
        controls = [
            METADATA_PATH,
            VIDEO_VIEWER_PATH,
            CAMERA_ANNOTATION_VIEWER_PATH,
            VIDEO_INTEGRITY_PATH,
            VIDEO_PRIVACY_PATH,
            *dense_indexes,
        ]
        available_controls = tuple(path for path in controls if path in self._tree)
        with ThreadPoolExecutor(max_workers=8) as executor:
            acquired_controls = tuple(
                executor.map(
                    lambda path: self._acquire_one(self._tree[path], "indus-ad-control"),
                    available_controls,
                )
            )
        self._control_files = dict(zip(available_controls, acquired_controls, strict=True))
        root = _ensure_beneath(
            self.cache_dir,
            "indus-ad-control-view",
            canonical_hash({"repo_id": self.source.repo_id, "revision": self.source.revision}),
        )
        for repo_path, cached in self._control_files.items():
            destination = root / repo_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copyfile(cached, destination)
        plan = parse_indus_ad_contract(root, self.source, tree)
        annotation_table = pq.ParquetFile(self._control_files[CAMERA_ANNOTATION_VIEWER_PATH]).read(
            columns=["recording_id", "recording_offset_ns"]
        )
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
        self._plan = plan
        return plan

    def acquire(self, entry: IndusAdRecordingEntry) -> AcquiredIndusAdRecording:
        self.load_plan()
        selected_artifacts = [
            item
            for item in (*entry.videos, *entry.artifacts)
            if item.kind in {"video", "camera_frames", "calibration", "gnss"}
        ]
        files = {
            item.path: self._acquire_one(
                self._tree[item.path], "indus-ad-payload", expected_sha256=item.sha256
            )
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
                replace(video, frame_count=counts.get(video.camera)) for video in entry.videos
            ),
        )
        frame_evidence: dict[tuple[str, int, int], tuple[int, int]] = {}
        for artifact in entry.artifacts_of("camera_frames"):
            for row in pq.ParquetFile(files[artifact.path]).read().to_pylist():
                camera_name = row.get("camera_name", row.get("source_camera_name"))
                frame_index = row.get("frame_index", row.get("packet_index"))
                offset = row.get("recording_offset_ns", row.get("camera_time_offset_ns"))
                if row.get("recording_id") != entry.recording_id:
                    raise AcquisitionError("camera frame table recording identity differs")
                key = (str(camera_name), int(frame_index), int(offset))
                if key in frame_evidence:
                    raise AcquisitionError("camera frame evidence contains duplicate identities")
                frame_evidence[key] = (int(row["width"]), int(row["height"]))
        selected_cameras = {video.camera for video in entry.videos}
        camera_annotations = tuple(
            item
            for item in self._annotations.camera
            if item.recording_id == entry.recording_id and item.camera_name in selected_cameras
        )
        for annotation in camera_annotations:
            dimensions = frame_evidence.get(
                (
                    annotation.camera_name,
                    annotation.frame_index,
                    annotation.recording_offset_ns,
                )
            )
            if dimensions != (annotation.decoded_width, annotation.decoded_height):
                raise AcquisitionError("camera annotation differs from source frame evidence")
        annotations = IndusAdAnnotationBundle(
            camera=camera_annotations,
            gnss=tuple(
                item for item in self._annotations.gnss if item.recording_id == entry.recording_id
            ),
            dense_refs=tuple(
                item
                for item in self._annotations.dense_refs
                if item.recording_id == entry.recording_id
            ),
            availability=self._annotations.availability,
        )
        companion = self.source.gnss_annotation_source
        if companion is not None:
            if companion.expected_source_revision != self.source.revision:
                raise AcquisitionError("GNSS companion source revision differs")
            companion_tree = self.companion_list_tree(companion.repo_id, companion.revision)
            if not companion_tree:
                raise AcquisitionError("GNSS companion contains no annotation Parquet files")
            companion_acquirer = IndusAdHfAcquirer(
                source=self.source.model_copy(
                    update={
                        "repo_id": companion.repo_id,
                        "revision": companion.revision,
                    }
                ),
                cache_dir=self.cache_dir,
                download_file=self.download_file,
            )
            gnss_table = pq.ParquetFile(files[entry.artifact("gnss").path]).read(
                columns=["recording_offset_ns"]
            )
            duration = max(gnss_table.column(0).to_pylist())
            gnss_annotations = tuple(
                annotation
                for identity in sorted(companion_tree, key=lambda item: item.path)
                for annotation in parse_gnss_scenarios(
                    companion_acquirer._acquire_one(identity, "indus-ad-gnss-companion"),
                    expected_recordings={entry.recording_id: duration},
                    expected_source_revision=self.source.revision,
                    selected_recording_ids=frozenset({entry.recording_id}),
                )
            )
            identities = [(item.recording_id, item.window_index) for item in gnss_annotations]
            if len(identities) != len(set(identities)):
                raise AcquisitionError("GNSS companion contains duplicate annotation windows")
            annotations = replace(
                annotations,
                gnss=gnss_annotations,
                availability={
                    **annotations.availability,
                    "gnss": AnnotationAvailability(
                        "available" if gnss_annotations else "missing",
                        "companion_verified" if gnss_annotations else "no_recording_windows",
                    ),
                },
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
                shard_path: self._acquire_one(self._tree[shard_path], "indus-ad-dense-archive")
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
