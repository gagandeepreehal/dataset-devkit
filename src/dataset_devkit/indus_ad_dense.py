"""Safe selected-member materialization for INDUS-AD dense label archives."""

from __future__ import annotations

import hashlib
import os
import secrets
import tarfile
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from dataset_devkit.indus_ad_models import DenseAssetRef

MAX_DENSE_MEMBER_BYTES = 256 * 1024 * 1024
MAX_DENSE_MEMBERS_PER_REQUEST = 250_000
MAX_DENSE_TOTAL_BYTES_PER_REQUEST = 64 * 1024 * 1024 * 1024


class DenseAssetError(ValueError):
    """Raised when a dense archive or requested member is unsafe or inconsistent."""


@dataclass(frozen=True, slots=True)
class MaterializedDenseAsset:
    ref: DenseAssetRef
    path: Path
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if self.size != self.ref.payload_bytes or self.sha256 != self.ref.payload_sha256:
            raise ValueError("materialized dense identity differs from its source reference")


@contextmanager
def _open_archive(path: Path) -> Iterator[tarfile.TarFile]:
    try:
        with tarfile.open(path, mode="r:*") as archive:
            yield archive
    except (OSError, tarfile.TarError) as error:
        raise DenseAssetError("dense archive is invalid") from error


def _safe_member_name(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        bool(value)
        and not path.is_absolute()
        and path.as_posix() == value
        and "\\" not in value
        and all(part not in {"", ".", ".."} for part in value.split("/"))
    )


def _validate_decoded_payload(path: Path, ref: DenseAssetRef) -> None:
    try:
        if ref.kind == "semantic":
            with Image.open(path) as image:
                image.load()
                if image.size != (ref.width, ref.height):
                    raise DenseAssetError("semantic payload dimensions differ")
                if ref.dtype == "uint8" and image.mode not in {"L", "P"}:
                    raise DenseAssetError("semantic payload dtype differs")
        else:
            array = np.load(path, allow_pickle=False)
            if array.shape != (ref.height, ref.width) or str(array.dtype) != ref.dtype:
                raise DenseAssetError("depth payload shape or dtype differs")
    except DenseAssetError:
        raise
    except Exception as error:
        raise DenseAssetError("dense payload cannot be decoded safely") from error


def materialize_selected_dense_assets(
    *,
    selected_refs: Sequence[DenseAssetRef],
    all_refs: Sequence[DenseAssetRef],
    archive_paths: Mapping[str, Path],
    output_dir: Path,
) -> tuple[MaterializedDenseAsset, ...]:
    """Verify archive structure and atomically publish only requested members."""
    if len(selected_refs) > MAX_DENSE_MEMBERS_PER_REQUEST:
        raise DenseAssetError("dense request exceeds the member budget")
    total = sum(ref.payload_bytes for ref in selected_refs)
    if total > MAX_DENSE_TOTAL_BYTES_PER_REQUEST:
        raise DenseAssetError("dense request exceeds the total byte budget")
    if any(ref.payload_bytes > MAX_DENSE_MEMBER_BYTES for ref in selected_refs):
        raise DenseAssetError("dense member exceeds the byte budget")
    expected_by_archive: dict[str, dict[str, DenseAssetRef]] = defaultdict(dict)
    for ref in all_refs:
        by_name = expected_by_archive[ref.shard_path]
        if ref.member_path in by_name:
            raise DenseAssetError("dense index contains a duplicate archive member")
        by_name[ref.member_path] = ref
    selected_by_archive: dict[str, list[DenseAssetRef]] = defaultdict(list)
    for ref in selected_refs:
        selected_by_archive[ref.shard_path].append(ref)

    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    result: list[MaterializedDenseAsset] = []
    for shard_path in sorted(selected_by_archive):
        archive_path = archive_paths.get(shard_path)
        if archive_path is None:
            raise DenseAssetError("selected dense archive is unavailable")
        with _open_archive(archive_path) as archive:
            members = archive.getmembers()
            counts = Counter(member.name for member in members)
            if any(count != 1 for count in counts.values()):
                raise DenseAssetError("dense archive contains duplicate member names")
            expected = expected_by_archive.get(shard_path, {})
            for member in members:
                if not _safe_member_name(member.name):
                    raise DenseAssetError("dense archive contains an unsafe member path")
                if not member.isreg() or member.issym() or member.islnk():
                    raise DenseAssetError("dense archive contains a non-regular member")
                indexed_ref = expected.get(member.name)
                if indexed_ref is None:
                    raise DenseAssetError("dense archive contains an unexpected member")
                if member.size != indexed_ref.payload_bytes:
                    raise DenseAssetError("dense archive member size differs from its index")
                if member.size > MAX_DENSE_MEMBER_BYTES:
                    raise DenseAssetError("dense archive member exceeds the byte budget")
            if set(counts) != set(expected):
                raise DenseAssetError("dense archive is missing an indexed member")

            for ref in sorted(
                selected_by_archive[shard_path], key=lambda item: item.member_path
            ):
                member = archive.getmember(ref.member_path)
                stream = archive.extractfile(member)
                if stream is None:
                    raise DenseAssetError("dense archive member is unreadable")
                suffix = Path(ref.member_path).suffix
                directory = output_dir / ref.payload_sha256
                directory.mkdir(mode=0o700, exist_ok=True)
                final = directory / f"asset{suffix}"
                if final.exists():
                    data = final.read_bytes()
                    digest = hashlib.sha256(data).hexdigest()
                    if len(data) != ref.payload_bytes or digest != ref.payload_sha256:
                        raise DenseAssetError("cached dense member identity differs")
                    _validate_decoded_payload(final, ref)
                    result.append(MaterializedDenseAsset(ref, final, len(data), digest))
                    continue
                temporary = directory / f".asset.{secrets.token_hex(12)}.tmp"
                flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(temporary, flags, 0o600)
                digest_state = hashlib.sha256()
                size = 0
                try:
                    with os.fdopen(descriptor, "wb") as output:
                        while chunk := stream.read(1024 * 1024):
                            size += len(chunk)
                            if size > ref.payload_bytes or size > MAX_DENSE_MEMBER_BYTES:
                                raise DenseAssetError("dense member exceeds its declared size")
                            digest_state.update(chunk)
                            output.write(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                    digest = digest_state.hexdigest()
                    if size != ref.payload_bytes or digest != ref.payload_sha256:
                        raise DenseAssetError("dense member digest or size differs")
                    _validate_decoded_payload(temporary, ref)
                    os.replace(temporary, final)
                finally:
                    temporary.unlink(missing_ok=True)
                result.append(MaterializedDenseAsset(ref, final, size, digest))
    return tuple(result)
