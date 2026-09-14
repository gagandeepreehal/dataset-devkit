from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from dataset_devkit.decoded_acquisition import DecodedHfAcquirer
from dataset_devkit.huggingface_acquisition import AcquisitionError, IntegrityError
from decoded_v2_fixture import DecodedFixture, decoded_source_config, write_decoded_fixture


class FixtureDownload:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[str] = []

    def __call__(self, **kwargs: object) -> str:
        filename = str(kwargs["filename"])
        self.calls.append(filename)
        target = Path(str(kwargs["local_dir"])) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.root / filename, target)
        return str(target)


def make_decoded_acquirer(
    fixture: DecodedFixture, cache: Path, calls: list[str]
) -> DecodedHfAcquirer:
    download = FixtureDownload(fixture.root)

    def tracked(**kwargs: object) -> str:
        result = download(**kwargs)
        calls.append(str(kwargs["filename"]))
        return result

    return DecodedHfAcquirer(
        source=decoded_source_config(fixture.revision, [fixture.recording_id]),
        cache_dir=cache,
        download_file=tracked,
    )


def test_acquirer_downloads_only_selected_artifacts(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path / "remote")
    calls: list[str] = []
    acquirer = make_decoded_acquirer(fixture, tmp_path / "cache", calls)

    plan = acquirer.load_plan()
    bundle = acquirer.acquire(plan.recordings[0])

    assert bundle.entry.recording_id == fixture.recording_id
    assert set(calls) == {
        "data/recordings.parquet",
        "data/manifests/output-files.parquet",
        "data/manifests/processing-config.json",
        "data/manifests/model-receipt.json",
        "data/manifests/schema-audit.parquet",
        *(artifact.path for artifact in bundle.entry.artifacts),
        *(video.path for video in bundle.entry.videos),
    }
    assert all(bundle.path(path).is_file() for path in bundle.files)


def test_acquirer_rejects_downloaded_digest_mismatch(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path / "remote")
    acquirer = make_decoded_acquirer(fixture, tmp_path / "cache", [])
    plan = acquirer.load_plan()
    target = fixture.root / plan.recordings[0].artifacts[0].path
    target.write_bytes(target.read_bytes() + b"corrupt")

    with pytest.raises(IntegrityError):
        acquirer.acquire(plan.recordings[0])


def test_decoded_verified_cache_avoids_repeated_payload_downloads(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path / "remote")
    calls: list[str] = []
    acquirer = make_decoded_acquirer(fixture, tmp_path / "cache", calls)
    entry = acquirer.load_plan().recordings[0]

    first = acquirer.acquire(entry)
    payload_call_count = len(calls)
    second = acquirer.acquire(entry)

    assert first.files == second.files
    assert len(calls) == payload_call_count


def test_decoded_concurrent_acquisition_downloads_each_payload_once(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path / "remote")
    calls: list[str] = []
    acquirer = make_decoded_acquirer(fixture, tmp_path / "cache", calls)
    entry = acquirer.load_plan().recordings[0]
    control_count = len(calls)

    with ThreadPoolExecutor(max_workers=2) as executor:
        bundles = tuple(executor.map(lambda _: acquirer.acquire(entry), range(2)))

    assert bundles[0].files == bundles[1].files
    assert len(calls) - control_count == len(entry.videos) + len(entry.artifacts)


def test_decoded_acquirer_rejects_zero_byte_or_short_payload(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path / "remote")
    acquirer = make_decoded_acquirer(fixture, tmp_path / "cache", [])
    entry = acquirer.load_plan().recordings[0]
    target = fixture.root / entry.artifacts[0].path
    target.write_bytes(b"")

    with pytest.raises(IntegrityError, match="size"):
        acquirer.acquire(entry)


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_decoded_acquirer_rejects_linked_download(tmp_path: Path, kind: str) -> None:
    fixture = write_decoded_fixture(tmp_path / "remote")
    source = decoded_source_config(fixture.revision, [fixture.recording_id])
    normal = FixtureDownload(fixture.root)
    acquirer = DecodedHfAcquirer(
        source=source,
        cache_dir=tmp_path / "cache",
        download_file=normal,
    )
    entry = acquirer.load_plan().recordings[0]
    outside = tmp_path / "outside"
    outside.write_bytes((fixture.root / entry.artifacts[0].path).read_bytes())

    def linked(**kwargs: object) -> str:
        target = Path(str(kwargs["local_dir"])) / str(kwargs["filename"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if kind == "symlink":
            target.symlink_to(outside)
        else:
            os.link(outside, target)
        return str(target)

    acquirer.download_file = linked
    with pytest.raises(AcquisitionError, match="owned regular file"):
        acquirer.acquire(entry)


def test_decoded_acquirer_rejects_returned_path_escape(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path / "remote")
    source = decoded_source_config(fixture.revision, [fixture.recording_id])
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    acquirer = DecodedHfAcquirer(
        source=source,
        cache_dir=tmp_path / "cache",
        download_file=FixtureDownload(fixture.root),
    )
    entry = acquirer.load_plan().recordings[0]
    acquirer.download_file = lambda **_: str(outside)

    with pytest.raises(AcquisitionError, match="outside"):
        acquirer.acquire(entry)
