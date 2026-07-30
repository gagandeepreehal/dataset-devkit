"""Safe JSON extraction-result cache with verified copied JPEG materialization."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, cast

from pydantic import TypeAdapter, ValidationError

from dataset_devkit.extraction.models import RecordingExtractionResult, StagedImage
from dataset_devkit.extraction.staging import staged_directory_metadata
from dataset_devkit.provenance import SourceFingerprint, canonical_json
from dataset_devkit.publication import publish_staging

_ADAPTER = TypeAdapter(RecordingExtractionResult)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _read_owned(path: Path) -> tuple[bytes, os.stat_result]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("cache artifact must be a single-link regular file")
    descriptor = os.open(path, os.O_RDONLY | _NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    def identity(item: os.stat_result) -> tuple[int, int, int, int]:
        return item.st_dev, item.st_ino, item.st_size, item.st_nlink
    if identity(before) != identity(opened) or identity(opened) != identity(after):
        raise ValueError("cache artifact identity changed")
    return b"".join(chunks), after


def _verified_image(image: StagedImage) -> bytes:
    content, current = _read_owned(image.path)
    if image.device is not None and image.device != current.st_dev:
        raise ValueError("staged image device differs")
    if image.inode is not None and image.inode != current.st_ino:
        raise ValueError("staged image inode differs")
    if image.size is not None and image.size != len(content):
        raise ValueError("staged image size differs")
    if image.sha256 is not None and image.sha256 != hashlib.sha256(content).hexdigest():
        raise ValueError("staged image digest differs")
    return content


class ExtractionResultCache:
    """Persist and reconstruct trusted extraction evidence without executable formats."""

    def __init__(self, cache_dir: Path) -> None:
        self.root = cache_dir.resolve() / "extraction-results"

    def path_for(self, source: SourceFingerprint, config_hash: str) -> Path:
        if len(config_hash) != 64 or any(
            character not in "0123456789abcdef" for character in config_hash
        ):
            raise ValueError("extraction config hash must be lowercase SHA-256")
        return self.root / source.digest / config_hash

    def load(
        self,
        source: SourceFingerprint,
        config_hash: str,
        source_path: Path,
    ) -> RecordingExtractionResult | None:
        root = self.path_for(source, config_hash)
        try:
            if not stat.S_ISDIR(root.lstat().st_mode):
                return None
            names = sorted(
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file()
            )
            manifest_bytes, _ = _read_owned(root / "manifest.json")
            result_bytes, _ = _read_owned(root / "result.json")
            manifest = json.loads(manifest_bytes)
            if not isinstance(manifest, dict):
                return None
            images = manifest.get("images")
            if (
                manifest.get("schema_version") != 1
                or manifest.get("source") != source.to_dict()
                or manifest.get("extraction_config_hash") != config_hash
                or manifest.get("result_sha256") != hashlib.sha256(result_bytes).hexdigest()
                or not isinstance(images, list)
            ):
                return None
            expected_names = ["manifest.json", "result.json"]
            for item in images:
                if not isinstance(item, dict):
                    return None
                relative = item.get("path")
                if not isinstance(relative, str) or not relative.startswith("images/"):
                    return None
                content, current = _read_owned(root / relative)
                if (
                    item.get("size") != len(content)
                    or item.get("sha256") != hashlib.sha256(content).hexdigest()
                    or current.st_nlink != 1
                ):
                    return None
                expected_names.append(relative)
            if names != sorted(expected_names):
                return None
            raw = json.loads(result_bytes)
            result = _ADAPTER.validate_python(raw)
            image_root = root / "images"
            directory_device, directory_inode, directory_identities = (
                staged_directory_metadata(image_root)
            )
            samples = []
            for index, sample in enumerate(result.samples):
                path = image_root / f"{index:08d}.jpg"
                _, current = _read_owned(path)
                staged = replace(
                    sample.staged_image,
                    path=path,
                    device=current.st_dev,
                    inode=current.st_ino,
                    size=current.st_size,
                    sha256=cast(dict[str, Any], images[index])["sha256"],
                    invocation_root=image_root,
                    root_relative_path=f"{index:08d}.jpg",
                    directory_device=directory_device,
                    directory_inode=directory_inode,
                    directory_chain_identities=directory_identities,
                )
                samples.append(replace(sample, staged_image=staged))
            return replace(
                result,
                source_path=source_path.resolve(),
                staging_root=image_root,
                samples=tuple(samples),
            )
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ValidationError,
            ValueError,
            KeyError,
            IndexError,
        ):
            return None

    def store(
        self,
        source: SourceFingerprint,
        config_hash: str,
        result: RecordingExtractionResult,
    ) -> RecordingExtractionResult:
        final = self.path_for(source, config_hash)
        cached = self.load(source, config_hash, result.source_path)
        if cached is not None:
            return cached
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{config_hash}.staging-", dir=final.parent))
        try:
            image_directory = temporary / "images"
            image_directory.mkdir()
            image_rows: list[dict[str, object]] = []
            cached_samples = []
            for index, sample in enumerate(result.samples):
                content = _verified_image(sample.staged_image)
                relative = f"images/{index:08d}.jpg"
                path = temporary / relative
                path.write_bytes(content)
                current = path.stat()
                digest = hashlib.sha256(content).hexdigest()
                image_rows.append({"path": relative, "size": len(content), "sha256": digest})
                cached_samples.append(
                    replace(
                        sample,
                        staged_image=replace(
                            sample.staged_image,
                            path=final / relative,
                            device=current.st_dev,
                            inode=current.st_ino,
                            size=len(content),
                            sha256=digest,
                            invocation_root=final,
                            root_relative_path=relative,
                            directory_chain_identities=(),
                        ),
                    )
                )
            cached_result = replace(
                result,
                staging_root=final,
                samples=tuple(cached_samples),
            )
            result_bytes = (canonical_json(_jsonable(cached_result)) + "\n").encode()
            (temporary / "result.json").write_bytes(result_bytes)
            manifest = {
                "schema_version": 1,
                "source": source.to_dict(),
                "extraction_config_hash": config_hash,
                "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
                "images": image_rows,
            }
            (temporary / "manifest.json").write_text(canonical_json(manifest) + "\n")
            try:
                publish_staging(temporary, final)
            except FileExistsError:
                shutil.rmtree(temporary)
            loaded = self.load(source, config_hash, result.source_path)
            if loaded is None:
                raise ValueError("stored extraction result did not revalidate")
            return loaded
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
