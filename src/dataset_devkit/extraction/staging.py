"""Narrow POSIX-safe atomic JPEG staging."""

from __future__ import annotations

import os
import re
import stat
import uuid
from contextlib import suppress
from pathlib import Path

from PIL import Image

from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.models import StagedImage

_SAFE_RECORDING = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _camera_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return slug or "camera"


def stage_jpeg(
    staging_root: Path,
    recording_id: str,
    camera_index: int,
    camera_name: str,
    timestamp_ns: int,
    image: Image.Image,
    expected_dimensions: tuple[int, int],
) -> StagedImage:
    """Atomically write, then independently decode and verify a quality-95 JPEG."""
    if _SAFE_RECORDING.fullmatch(recording_id) is None:
        raise StructuralExtractionError("unsafe recording identifier for staging")
    staging_root.mkdir(parents=True, exist_ok=True)
    if staging_root.is_symlink():
        raise StructuralExtractionError("staging root may not be a symlink")
    recording_dir = staging_root / recording_id
    with suppress(FileExistsError):
        recording_dir.mkdir(mode=0o700)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(recording_dir, directory_flags)
    except OSError as error:
        raise StructuralExtractionError("unsafe recording staging directory") from error

    filename = f"{camera_index:03d}-{_camera_slug(camera_name)}-{timestamp_ns}.jpg"
    temporary_name = f".{filename}.{uuid.uuid4().hex}.tmp"
    try:
        try:
            existing = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and (not stat.S_ISREG(existing.st_mode) or existing.st_nlink != 1):
            raise StructuralExtractionError("unsafe existing staging target")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        temporary_fd = os.open(temporary_name, flags, 0o600, dir_fd=directory_fd)
        try:
            with os.fdopen(temporary_fd, "wb", closefd=True) as stream:
                image.convert("RGB").save(stream, format="JPEG", quality=95)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(
                temporary_name,
                filename,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            os.fsync(directory_fd)
        except Exception:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=directory_fd)
            raise
    finally:
        os.close(directory_fd)

    path = recording_dir / filename
    try:
        with Image.open(path) as reopened:
            reopened.load()
            if reopened.format != "JPEG" or reopened.mode != "RGB":
                raise StructuralExtractionError("staged image did not reopen as RGB JPEG")
            if reopened.size != expected_dimensions:
                raise StructuralExtractionError(
                    "staged JPEG dimensions differ from camera dimensions"
                )
    except StructuralExtractionError:
        raise
    except Exception as error:
        raise StructuralExtractionError(
            "staged JPEG failed independent decode verification"
        ) from error
    return StagedImage(
        camera_index,
        camera_name,
        timestamp_ns,
        path,
        expected_dimensions[0],
        expected_dimensions[1],
    )
