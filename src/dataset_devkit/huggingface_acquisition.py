"""Pinned Hugging Face dataset acquisition with verified local caching."""

from __future__ import annotations

import fcntl
import hashlib
import os
import secrets
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from pathlib import Path

from dataset_devkit.config import GlobalConfig, HuggingFaceConfig
from dataset_devkit.huggingface_manifest import ManifestEntry, parse_manifest
from dataset_devkit.provenance import (
    AcquisitionManifest,
    ArtifactIdentity,
    ExtractionManifest,
    SourceFingerprint,
    canonical_hash,
    canonical_json,
    extraction_config_hash,
    load_extraction_manifest,
    load_manifest,
)

DownloadFile = Callable[..., str]


class AcquisitionError(RuntimeError):
    """Raised when acquisition cannot safely produce a cache artifact."""


class IntegrityError(AcquisitionError):
    """Raised when downloaded bytes disagree with the repository manifest."""


@dataclass(frozen=True)
class CachePaths:
    directory: Path
    final: Path
    manifest: Path
    extraction_manifest: Path


@dataclass(frozen=True)
class AcquisitionResult:
    artifact_path: Path
    manifest_path: Path
    extraction_manifest_path: Path
    manifest: AcquisitionManifest


@dataclass(frozen=True)
class _VerifiedFile:
    device: int
    inode: int
    size: int
    sha256: str


def _ensure_cache_root(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    value = path.lstat()
    if not stat.S_ISDIR(value.st_mode):
        raise AcquisitionError(f"cache component is not a directory: {path}")


def _ensure_beneath(root: Path, *components: str) -> Path:
    _ensure_cache_root(root)
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(root, flags)
    try:
        for component in components:
            if not component or component in {".", ".."} or "/" in component:
                raise AcquisitionError("invalid cache directory component")
            try:
                os.mkdir(component, 0o700, dir_fd=descriptor)
                os.fsync(descriptor)
            except FileExistsError:
                pass
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise AcquisitionError(
                    f"cache component is not a trusted directory: {component}"
                ) from error
            os.close(descriptor)
            descriptor = child
        return root.joinpath(*components)
    finally:
        os.close(descriptor)


@contextmanager
def _open_parent_beneath(root: Path, path: Path) -> Iterator[tuple[int, str]]:
    try:
        relative = path.absolute().relative_to(root.absolute())
    except ValueError as error:
        raise AcquisitionError("download returned a path outside owned scratch") from error
    if not relative.parts:
        raise AcquisitionError("download did not return a file path")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    descriptors = [os.open(root, flags)]
    try:
        for component in relative.parts[:-1]:
            try:
                descriptors.append(os.open(component, flags, dir_fd=descriptors[-1]))
            except OSError as error:
                raise AcquisitionError(
                    f"download parent is not a trusted directory: {component}"
                ) from error
        yield descriptors[-1], relative.parts[-1]
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _verify_file(path: Path, root: Path) -> _VerifiedFile:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    with _open_parent_beneath(root, path) as (parent_fd, name):
        try:
            descriptor = os.open(name, flags, dir_fd=parent_fd)
        except OSError as error:
            raise AcquisitionError(
                f"download is not an owned regular file: {path.name}"
            ) from error
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            os.close(descriptor)
            raise AcquisitionError(f"download is not an owned regular file: {path.name}")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
        try:
            leaf = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as error:
            raise AcquisitionError("download identity changed during verification") from error
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_nlink)
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_nlink) or identity != (
            leaf.st_dev,
            leaf.st_ino,
            leaf.st_size,
            leaf.st_nlink,
        ):
            raise AcquisitionError("download identity changed during verification")
        return _VerifiedFile(before.st_dev, before.st_ino, before.st_size, digest.hexdigest())


def _atomic_write(path: Path, value: object) -> None:
    temporary = path.parent / f".{path.name}.{secrets.token_hex(12)}.tmp"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


@contextmanager
def _recording_lock(directory: Path) -> Iterator[None]:
    lock_path = directory / ".acquisition.lock"
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    value = os.fstat(descriptor)
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        os.close(descriptor)
        raise AcquisitionError("acquisition lock must be an owned regular file")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        current = lock_path.lstat()
        if (current.st_dev, current.st_ino) != (value.st_dev, value.st_ino):
            raise AcquisitionError("acquisition lock identity changed")
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.absolute().relative_to(root.absolute())
    except ValueError:
        return False
    return True


class HuggingFaceAcquirer:
    """Acquire commit-pinned dataset files into a SHA-256-verified cache."""

    def __init__(
        self,
        *,
        huggingface: HuggingFaceConfig,
        cache_dir: Path,
        extraction_config_hash: str,
        download_file: DownloadFile | None = None,
    ) -> None:
        self.huggingface = huggingface
        self.cache_dir = cache_dir.absolute()
        self.extraction_config_hash = extraction_config_hash
        if download_file is None:
            from huggingface_hub import hf_hub_download

            download_file = hf_hub_download
        self.download_file = download_file

    @classmethod
    def from_config(cls, config: GlobalConfig) -> HuggingFaceAcquirer:
        return cls(
            huggingface=config.huggingface,
            cache_dir=config.paths.cache_dir,
            extraction_config_hash=extraction_config_hash(config),
        )

    def _download(self, filename: str, local_dir: Path) -> Path:
        try:
            returned = Path(
                self.download_file(
                    repo_id=self.huggingface.repo_id,
                    repo_type="dataset",
                    revision=self.huggingface.revision,
                    filename=filename,
                    local_dir=local_dir,
                    force_download=True,
                )
            )
        except Exception as error:
            raise AcquisitionError(f"failed to download {filename!r}") from error
        if not _inside(returned, local_dir):
            raise AcquisitionError("download returned a path outside owned scratch")
        return returned

    def load_entries(self) -> tuple[ManifestEntry, ...]:
        identity = canonical_hash(
            {
                "repo_id": self.huggingface.repo_id,
                "revision": self.huggingface.revision,
                "manifest_path": self.huggingface.manifest_path,
            }
        )
        local_dir = _ensure_beneath(self.cache_dir, "huggingface-manifests", identity)
        with _recording_lock(local_dir):
            manifest = self._download(self.huggingface.manifest_path, local_dir)
            _verify_file(manifest, local_dir)
            return parse_manifest(manifest)

    def paths_for(self, source: SourceFingerprint) -> CachePaths:
        directory = self.cache_dir / "huggingface" / source.digest
        stem = f"artifact-{source.digest}"
        return CachePaths(
            directory=directory,
            final=directory / f"{stem}.mcap",
            manifest=directory / f"{stem}.manifest.json",
            extraction_manifest=directory / f"{stem}.extraction.manifest.json",
        )

    def acquire(self, entry: ManifestEntry) -> AcquisitionResult:
        source = SourceFingerprint(
            self.huggingface.repo_id,
            self.huggingface.revision,
            entry.repo_path,
            entry.sha256,
            entry.size,
        )
        paths = self.paths_for(source)
        _ensure_beneath(self.cache_dir, "huggingface", source.digest)
        with _recording_lock(paths.directory):
            cached = load_manifest(paths.manifest)
            if cached is not None and cached.source == source and paths.final.exists():
                verified = _verify_file(paths.final, paths.directory)
                if verified.size == source.size and verified.sha256 == source.sha256:
                    hit = replace(cached, status="cache_hit")
                    _atomic_write(paths.manifest, hit.to_dict())
                    return AcquisitionResult(
                        paths.final, paths.manifest, paths.extraction_manifest, hit
                    )

            scratch = _ensure_beneath(
                self.cache_dir, "huggingface", source.digest, "download"
            )
            returned = self._download(entry.repo_path, scratch)
            verified = _verify_file(returned, scratch)
            if verified.size != source.size:
                raise IntegrityError("downloaded size differs from repository manifest")
            if verified.sha256 != source.sha256:
                raise IntegrityError("downloaded SHA-256 differs from repository manifest")
            os.replace(returned, paths.final)
            final = _verify_file(paths.final, paths.directory)
            if final != verified:
                raise AcquisitionError("promoted cache artifact changed identity")
            artifact = ArtifactIdentity(
                paths.final.relative_to(self.cache_dir).as_posix(), source.size, final.sha256
            )
            manifest = AcquisitionManifest(
                source, "downloaded", artifact, self.extraction_config_hash
            )
            _atomic_write(paths.manifest, manifest.to_dict())
            return AcquisitionResult(
                paths.final, paths.manifest, paths.extraction_manifest, manifest
            )

    def extraction_cache_reusable(
        self, source: SourceFingerprint, expected_extraction_config_hash: str
    ) -> bool:
        paths = self.paths_for(source)
        if not paths.directory.exists():
            return False
        with _recording_lock(paths.directory):
            value = load_extraction_manifest(paths.extraction_manifest)
            if value is None:
                return False
            return (
                value.source == source
                and value.extraction_config_hash == expected_extraction_config_hash
            )

    def record_extraction_complete(
        self, source: SourceFingerprint, completed_extraction_config_hash: str
    ) -> Path:
        paths = self.paths_for(source)
        _ensure_beneath(self.cache_dir, "huggingface", source.digest)
        with _recording_lock(paths.directory):
            manifest = ExtractionManifest(source, completed_extraction_config_hash)
            _atomic_write(paths.extraction_manifest, manifest.to_dict())
        return paths.extraction_manifest
