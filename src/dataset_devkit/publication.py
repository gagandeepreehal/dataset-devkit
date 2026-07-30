"""Identity-checked, no-overwrite atomic dataset publication."""

from __future__ import annotations

import os
import stat
import sys
import uuid
from contextlib import suppress
from ctypes import CDLL, c_char_p, c_int, get_errno
from dataclasses import dataclass
from pathlib import Path

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | _NOFOLLOW


def _identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _stable_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_nlink,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_directory_chain(path: Path, *, create: bool) -> tuple[int, tuple[tuple[int, int], ...]]:
    absolute = path.absolute()
    if not absolute.is_absolute() or ".." in absolute.parts:
        raise ValueError("output path must be absolute and traversal-safe")
    current = os.open("/", _DIRECTORY_FLAGS)
    identities = [_identity(os.fstat(current))]
    try:
        for component in absolute.parts[1:]:
            if create:
                with suppress(FileExistsError):
                    os.mkdir(component, 0o700, dir_fd=current)
            child = os.open(component, _DIRECTORY_FLAGS, dir_fd=current)
            listed = os.stat(component, dir_fd=current, follow_symlinks=False)
            opened = os.fstat(child)
            if not stat.S_ISDIR(listed.st_mode) or _identity(listed) != _identity(opened):
                os.close(child)
                raise ValueError("output directory identity changed")
            identities.append(_identity(opened))
            os.close(current)
            current = child
        return current, tuple(identities)
    except Exception:
        os.close(current)
        raise


@dataclass
class StagingLease:
    """Pinned authority for one invocation-owned output staging directory."""

    root: Path
    parent: Path
    name: str
    _parent_fd: int
    _root_fd: int
    _parent_chain: tuple[tuple[int, int], ...]
    parent_identity: tuple[int, int]
    root_identity: tuple[int, int]
    _closed: bool = False

    @classmethod
    def create(cls, output: str | Path, prefix: str) -> StagingLease:
        parent = Path(output).absolute()
        parent_fd, chain = _open_directory_chain(parent, create=True)
        try:
            for _ in range(128):
                name = f"{prefix}{uuid.uuid4().hex}"
                try:
                    os.mkdir(name, 0o700, dir_fd=parent_fd)
                except FileExistsError:
                    continue
                root_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
                listed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                opened = os.fstat(root_fd)
                if not stat.S_ISDIR(listed.st_mode) or _identity(listed) != _identity(opened):
                    os.close(root_fd)
                    raise ValueError("staging directory identity changed during creation")
                return cls(
                    parent / name,
                    parent,
                    name,
                    parent_fd,
                    root_fd,
                    chain,
                    _identity(os.fstat(parent_fd)),
                    _identity(opened),
                )
            raise FileExistsError("unable to allocate a unique staging directory")
        except Exception:
            os.close(parent_fd)
            raise

    def assert_bound(self) -> None:
        if self._closed:
            raise ValueError("staging lease is closed")
        if _identity(os.fstat(self._parent_fd)) != self.parent_identity:
            raise ValueError("output parent identity changed")
        if _identity(os.fstat(self._root_fd)) != self.root_identity:
            raise ValueError("staging root identity changed")
        listed = os.stat(self.name, dir_fd=self._parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(listed.st_mode) or _identity(listed) != self.root_identity:
            raise ValueError("staging directory entry no longer names the leased root")
        check_fd, chain = _open_directory_chain(self.parent, create=False)
        try:
            if chain != self._parent_chain or _identity(os.fstat(check_fd)) != self.parent_identity:
                raise ValueError("configured output parent binding changed")
        finally:
            os.close(check_fd)

    def duplicate_root_fd(self) -> int:
        self.assert_bound()
        return os.dup(self._root_fd)

    def cleanup(self) -> bool:
        """Boundedly clean this invocation through retained authoritative descriptors."""
        try:
            return cleanup_pinned_directory(
                self._parent_fd,
                self.name,
                self._root_fd,
                self.root_identity,
            )
        except (OSError, ValueError):
            return False

    def close(self) -> None:
        if not self._closed:
            os.close(self._root_fd)
            os.close(self._parent_fd)
            self._closed = True


def _open_tree_snapshot(
    directory_fd: int,
    *,
    prefix: str,
    files: list[tuple[str, str, int, int, os.stat_result, os.stat_result]],
    directories: list[tuple[str, str, int, int, os.stat_result, os.stat_result]],
) -> None:
    for name in sorted(os.listdir(directory_fd)):
        relative = f"{prefix}/{name}" if prefix else name
        listed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(listed.st_mode):
            child = os.open(name, _DIRECTORY_FLAGS, dir_fd=directory_fd)
            opened = os.fstat(child)
            if _identity(opened) != _identity(listed):
                os.close(child)
                raise ValueError(f"directory changed while walking: {relative}")
            directories.append((relative, name, directory_fd, child, listed, opened))
            _open_tree_snapshot(
                child,
                prefix=relative,
                files=files,
                directories=directories,
            )
        elif stat.S_ISREG(listed.st_mode) and listed.st_nlink == 1:
            descriptor = os.open(name, os.O_RDONLY | _NOFOLLOW, dir_fd=directory_fd)
            opened = os.fstat(descriptor)
            if _stable_identity(opened) != _stable_identity(listed):
                os.close(descriptor)
                raise ValueError(f"path changed while hashing: {relative}")
            files.append((relative, name, directory_fd, descriptor, listed, opened))
        else:
            raise ValueError(f"symlink or unsafe hardlink: {relative}")


def hash_regular_files_fd(
    root_fd: int, *, excluded: frozenset[str] = frozenset()
) -> dict[str, tuple[int, str]]:
    """Hash one whole pinned-tree snapshot, retaining every descriptor until recheck."""
    import hashlib

    root_before = os.fstat(root_fd)
    files: list[tuple[str, str, int, int, os.stat_result, os.stat_result]] = []
    directories: list[tuple[str, str, int, int, os.stat_result, os.stat_result]] = []
    entries: dict[str, tuple[int, str]] = {}
    try:
        _open_tree_snapshot(
            root_fd,
            prefix="",
            files=files,
            directories=directories,
        )
        for relative, _, _, descriptor, _, opened in files:
            digest = hashlib.sha256()
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
            if relative not in excluded:
                entries[relative] = (opened.st_size, digest.hexdigest())

        for relative, name, parent_fd, descriptor, listed, opened in files:
            after = os.fstat(descriptor)
            relisted = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if not (
                _stable_identity(listed)
                == _stable_identity(opened)
                == _stable_identity(after)
                == _stable_identity(relisted)
            ):
                raise ValueError(f"path changed while hashing: {relative}")
        for relative, name, parent_fd, descriptor, listed, opened in reversed(directories):
            after = os.fstat(descriptor)
            relisted = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if not (
                _stable_identity(listed)
                == _stable_identity(opened)
                == _stable_identity(after)
                == _stable_identity(relisted)
            ):
                raise ValueError(f"directory changed while walking: {relative}")
        if _stable_identity(root_before) != _stable_identity(os.fstat(root_fd)):
            raise ValueError("directory changed while walking: .")
        return entries
    finally:
        for _, _, _, descriptor, _, _ in files:
            os.close(descriptor)
        for _, _, _, descriptor, _, _ in reversed(directories):
            os.close(descriptor)


def _cleanup_contents_fd(directory_fd: int) -> None:
    """Recursively remove entries below one already-open authoritative directory."""
    for name in sorted(os.listdir(directory_fd)):
        listed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        identity = _identity(listed)
        if stat.S_ISDIR(listed.st_mode):
            child = os.open(name, _DIRECTORY_FLAGS, dir_fd=directory_fd)
            try:
                if _identity(os.fstat(child)) != identity:
                    raise ValueError("cleanup child identity changed")
                _cleanup_contents_fd(child)
                current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if stat.S_ISDIR(current.st_mode) and _identity(current) == identity:
                    os.rmdir(name, dir_fd=directory_fd)
            finally:
                os.close(child)
        elif stat.S_ISREG(listed.st_mode):
            descriptor = os.open(name, os.O_RDONLY | _NOFOLLOW, dir_fd=directory_fd)
            try:
                if _identity(os.fstat(descriptor)) != identity:
                    raise ValueError("cleanup file identity changed")
                current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if stat.S_ISREG(current.st_mode) and _identity(current) == identity:
                    os.unlink(name, dir_fd=directory_fd)
            finally:
                os.close(descriptor)
        else:
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if _identity(current) == identity:
                os.unlink(name, dir_fd=directory_fd)
    os.fsync(directory_fd)


def cleanup_pinned_directory(
    parent_fd: int,
    name: str,
    directory_fd: int,
    expected_identity: tuple[int, int],
) -> bool:
    """Empty one pinned directory and remove its name only while identity-bound."""
    if _identity(os.fstat(directory_fd)) != expected_identity:
        raise ValueError("cleanup authority differs from expected directory identity")
    _cleanup_contents_fd(directory_fd)
    try:
        listed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(listed.st_mode) or _identity(listed) != expected_identity:
        return False
    os.rmdir(name, dir_fd=parent_fd)
    os.fsync(parent_fd)
    return True


def _rename_exclusive(parent_fd: int, source: str, destination: str) -> None:
    """Use the platform's atomic no-replace rename primitive, failing closed."""
    library = CDLL(None, use_errno=True)
    encoded_source = os.fsencode(source)
    encoded_destination = os.fsencode(destination)
    if sys.platform == "darwin":
        rename = library.renameatx_np
        rename.argtypes = (c_int, c_char_p, c_int, c_char_p, c_int)
        result = rename(parent_fd, encoded_source, parent_fd, encoded_destination, 0x00000004)
    elif sys.platform.startswith("linux"):
        rename = library.renameat2
        rename.argtypes = (c_int, c_char_p, c_int, c_char_p, c_int)
        result = rename(parent_fd, encoded_source, parent_fd, encoded_destination, 0x1)
    else:
        raise OSError("atomic no-overwrite rename is unavailable on this platform")
    if result != 0:
        error_number = get_errno()
        if error_number in {17, 39}:  # EEXIST / ENOTEMPTY
            raise FileExistsError(
                f"refusing to overwrite existing final dataset: {destination}"
            )
        raise OSError(error_number, os.strerror(error_number))


def _quarantine_published_identity(
    parent_fd: int, final_name: str, expected_identity: tuple[int, int]
) -> str:
    """Atomically remove an owned failed publication from its final name."""
    listed = os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(listed.st_mode) or _identity(listed) != expected_identity:
        raise ValueError("refusing to roll back a replaced final dataset")
    for _ in range(128):
        quarantine_name = f".{final_name}.rejected-{uuid.uuid4().hex}"
        try:
            _rename_exclusive(parent_fd, final_name, quarantine_name)
        except FileExistsError:
            continue
        quarantined = os.stat(
            quarantine_name, dir_fd=parent_fd, follow_symlinks=False
        )
        if not stat.S_ISDIR(quarantined.st_mode) or _identity(quarantined) != expected_identity:
            with suppress(Exception):
                _rename_exclusive(parent_fd, quarantine_name, final_name)
            raise ValueError("publication rollback identity changed")
        return quarantine_name
    raise FileExistsError("unable to allocate publication rollback quarantine")


def _directory_identity(path: Path) -> tuple[int, int]:
    value = path.lstat()
    if not stat.S_ISDIR(value.st_mode):
        raise ValueError(f"not a regular directory: {path}")
    return value.st_dev, value.st_ino


def _fsync_tree_fd(directory_fd: int) -> None:
    directory_before = os.fstat(directory_fd)
    for name in sorted(os.listdir(directory_fd)):
        listed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(listed.st_mode):
            child = os.open(name, _DIRECTORY_FLAGS, dir_fd=directory_fd)
            try:
                if _identity(os.fstat(child)) != _identity(listed):
                    raise ValueError("staging directory changed before fsync")
                _fsync_tree_fd(child)
                relisted = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if _stable_identity(os.fstat(child)) != _stable_identity(relisted):
                    raise ValueError("staging directory changed during fsync")
            finally:
                os.close(child)
        elif stat.S_ISREG(listed.st_mode) and listed.st_nlink == 1:
            descriptor = os.open(name, os.O_RDONLY | _NOFOLLOW, dir_fd=directory_fd)
            try:
                opened = os.fstat(descriptor)
                if _stable_identity(opened) != _stable_identity(listed):
                    raise ValueError("staging file changed before fsync")
                os.fsync(descriptor)
                after = os.fstat(descriptor)
                relisted = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if (
                    _stable_identity(opened) != _stable_identity(after)
                    or _stable_identity(after) != _stable_identity(relisted)
                ):
                    raise ValueError("staging file changed during fsync")
            finally:
                os.close(descriptor)
        else:
            raise ValueError("staging contains a symlink or unsafe hardlink")
    os.fsync(directory_fd)
    directory_after = os.fstat(directory_fd)
    if _stable_identity(directory_before) != _stable_identity(directory_after):
        raise ValueError("staging directory changed during fsync")


def fsync_tree(root: Path, *, expected_identity: tuple[int, int] | None = None) -> None:
    """Flush one no-follow tree, optionally requiring caller-supplied authority."""
    descriptor, _ = _open_directory_chain(root, create=False)
    try:
        if expected_identity is not None and _identity(os.fstat(descriptor)) != expected_identity:
            raise ValueError("staging directory identity differs from caller authority")
        _fsync_tree_fd(descriptor)
    finally:
        os.close(descriptor)


def publish_staging(
    staging: str | Path | StagingLease,
    final: str | Path,
    *,
    expected_identity: tuple[int, int] | None = None,
    expected_content_hash: str | None = None,
) -> Path:
    """Atomically rename one sibling staging directory, refusing any overwrite."""
    if isinstance(staging, StagingLease):
        lease: StagingLease | None = staging
        source = staging.root
    else:
        lease = None
        source = Path(staging).absolute()
    destination = Path(final).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing final dataset: {destination}")
    if source.parent != destination.parent:
        raise ValueError("staging and final dataroot must be siblings")
    if lease is not None:
        lease.assert_bound()
        source_identity = lease.root_identity
    elif expected_identity is not None:
        source_identity = expected_identity
    else:
        raise ValueError("publication requires the original staging identity")
    parent = destination.parent
    if lease is None:
        parent.mkdir(parents=True, exist_ok=True)
    parent_fd = os.dup(lease._parent_fd) if lease is not None else os.open(
        parent, _DIRECTORY_FLAGS
    )
    try:
        listed = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
        if (listed.st_dev, listed.st_ino) != source_identity:
            raise ValueError("staging directory identity changed")
        try:
            os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"refusing to overwrite existing final dataset: {destination}")
        if lease is not None:
            from dataset_devkit.validation import verify_publication_manifest

            if expected_content_hash is None:
                raise ValueError("leased publication requires an expected content hash")
            verify_publication_manifest(lease, expected_content_hash)
            _fsync_tree_fd(lease._root_fd)
            verify_publication_manifest(lease, expected_content_hash)
            lease.assert_bound()
        else:
            fsync_tree(source, expected_identity=source_identity)
        listed = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
        if (listed.st_dev, listed.st_ino) != source_identity:
            raise ValueError("staging directory identity changed before publication")
        published = False
        try:
            _rename_exclusive(parent_fd, source.name, destination.name)
            published = True
            if lease is not None:
                from dataset_devkit.validation import verify_publication_manifest_fd

                if expected_content_hash is None:
                    raise ValueError("leased publication requires an expected content hash")
                root_fd = os.dup(lease._root_fd)
                try:
                    if _identity(os.fstat(root_fd)) != source_identity:
                        raise ValueError("leased root identity changed after publication")
                    verify_publication_manifest_fd(root_fd, expected_content_hash)
                finally:
                    os.close(root_fd)
            os.fsync(parent_fd)
        except Exception:
            if published:
                _quarantine_published_identity(
                    parent_fd, destination.name, source_identity
                )
                os.fsync(parent_fd)
            raise
    finally:
        os.close(parent_fd)
    return destination
