"""Identity-checked, no-overwrite atomic dataset publication."""

from __future__ import annotations

import os
import stat
import sys
from ctypes import CDLL, c_char_p, c_int, get_errno
from pathlib import Path

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


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


def _directory_identity(path: Path) -> tuple[int, int]:
    value = path.lstat()
    if not stat.S_ISDIR(value.st_mode):
        raise ValueError(f"not a regular directory: {path}")
    return value.st_dev, value.st_ino


def fsync_tree(root: Path) -> None:
    """Flush every regular file and directory without following links."""
    for directory, names, filenames in os.walk(root, topdown=False, followlinks=False):
        base = Path(directory)
        for name in names:
            if not stat.S_ISDIR((base / name).lstat().st_mode):
                raise ValueError("staging contains a symlink or unsafe directory")
        for name in filenames:
            path = base / name
            current = path.lstat()
            if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
                raise ValueError("staging contains a symlink or unsafe hardlink")
            descriptor = os.open(path, os.O_RDONLY | _NOFOLLOW)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        descriptor = os.open(base, os.O_RDONLY | os.O_DIRECTORY | _NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def publish_staging(staging: str | Path, final: str | Path) -> Path:
    """Atomically rename one sibling staging directory, refusing any overwrite."""
    source = Path(staging).absolute()
    destination = Path(final).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing final dataset: {destination}")
    if source.parent != destination.parent:
        raise ValueError("staging and final dataroot must be siblings")
    source_identity = _directory_identity(source)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | _NOFOLLOW)
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
        fsync_tree(source)
        listed = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
        if (listed.st_dev, listed.st_ino) != source_identity:
            raise ValueError("staging directory identity changed before publication")
        _rename_exclusive(parent_fd, source.name, destination.name)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return destination
