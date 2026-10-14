from __future__ import annotations

import io
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

from dataset_devkit.indus_ad_annotations import (
    parse_camera_annotations,
    parse_dense_indexes,
)
from dataset_devkit.indus_ad_dense import DenseAssetError, materialize_selected_dense_assets
from dataset_devkit.indus_ad_models import DenseAssetRef
from indus_ad_fixture import IndusAdFixture, write_indus_ad_fixture


def _refs(
    tmp_path: Path,
) -> tuple[IndusAdFixture, tuple[DenseAssetRef, ...], dict[str, Path]]:
    fixture = write_indus_ad_fixture(tmp_path)
    camera = parse_camera_annotations(
        fixture.camera_annotations,
        expected_recordings={fixture.recording_id: fixture.duration_ns},
    ).camera
    refs = parse_dense_indexes([fixture.dense_index], camera_annotations=camera)
    archives = {ref.shard_path: fixture.root / ref.shard_path for ref in refs}
    return fixture, refs, archives


def test_dense_materializer_verifies_and_decodes_selected_members(tmp_path: Path) -> None:
    _, refs, archives = _refs(tmp_path / "fixture")

    result = materialize_selected_dense_assets(
        selected_refs=refs,
        all_refs=refs,
        archive_paths=archives,
        output_dir=tmp_path / "output",
    )

    assert len(result) == 2
    assert all(item.path.is_file() for item in result)
    assert {item.sha256 for item in result} == {item.ref.payload_sha256 for item in result}


@pytest.mark.parametrize("case", ["traversal", "symlink", "unexpected"])
def test_dense_materializer_rejects_unsafe_archive_members(
    tmp_path: Path, case: str
) -> None:
    _, refs, archives = _refs(tmp_path / "fixture")
    target_ref = refs[0]
    target = archives[target_ref.shard_path]
    with tarfile.open(target, "a") as archive:
        if case == "traversal":
            info = tarfile.TarInfo("../escape")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
        elif case == "symlink":
            info = tarfile.TarInfo("link")
            info.type = tarfile.SYMTYPE
            info.linkname = target_ref.member_path
            archive.addfile(info)
        else:
            info = tarfile.TarInfo("unexpected.bin")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))

    with pytest.raises(DenseAssetError):
        materialize_selected_dense_assets(
            selected_refs=[target_ref],
            all_refs=refs,
            archive_paths=archives,
            output_dir=tmp_path / "output",
        )


def test_dense_materializer_rejects_wrong_digest(tmp_path: Path) -> None:
    _, refs, archives = _refs(tmp_path / "fixture")
    wrong = replace(refs[0], payload_sha256="f" * 64)

    with pytest.raises(DenseAssetError, match="digest"):
        materialize_selected_dense_assets(
            selected_refs=[wrong],
            all_refs=[wrong, *refs[1:]],
            archive_paths=archives,
            output_dir=tmp_path / "output",
        )
