from __future__ import annotations

import fcntl
import json
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import dataset_devkit.quarantine as quarantine_module
from dataset_devkit.config import GlobalConfig, InvalidationRulesConfig, SanityChecksConfig
from dataset_devkit.coordinator import (
    CoordinatorInputError,
    PublicationBlockedError,
    RecordingCoordinator,
    RecordingRequest,
)
from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.service import RecordingExtractor
from dataset_devkit.publication import (
    OwnedDirectoryAuthority,
    OwnedDirectoryCleanupError,
)
from dataset_devkit.quarantine import (
    QuarantineReport,
    write_quarantine_report,
    write_rejection_manifest,
)
from dataset_devkit.sanity import SANITY_CHECK_CODES
from dataset_devkit.validity import INVALIDITY_CODES
from mcap_fixture import camera_message, write_mcap
from test_extraction_service import DeterministicDecoder
from test_validity import _result


def _report() -> QuarantineReport:
    return QuarantineReport(
        recording_id="recording-a",
        source_path="/source/recording-a.mcap",
        status="quarantined",
        category="structural",
        exception_type="StructuralExtractionError",
        exception_message="malformed protobuf",
        stage="extraction",
        deterministic_details={"attempt": 1},
        observed_context=({"code": "grid_miss", "enabled_as_invalidator": True},),
        source_config_hash="a" * 64,
        extraction_config_hash="b" * 64,
        artifact_handling="no_owned_artifacts",
    )


def test_quarantine_json_is_versioned_deterministic_and_collision_safe(tmp_path: Path) -> None:
    report = _report()

    with ThreadPoolExecutor(max_workers=8) as pool:
        artifacts = tuple(pool.map(lambda _: write_quarantine_report(tmp_path, report), range(8)))

    assert len({artifact.path for artifact in artifacts}) == 8
    payloads = {artifact.path.read_bytes() for artifact in artifacts}
    assert len(payloads) == 1
    decoded = json.loads(payloads.pop())
    assert decoded["schema_version"] == "1.0"
    assert decoded["recording_id"] == "recording-a"
    assert "created_at" not in decoded
    assert all(path.parent == tmp_path.resolve() for path in (item.path for item in artifacts))
    assert all(path.stat().st_nlink == 1 for path in (item.path for item in artifacts))


def test_quarantine_refuses_symlinked_ancestor(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)

    with pytest.raises(StructuralExtractionError, match="quarantine.*ancestor|symlink"):
        write_quarantine_report(linked, _report())
    assert not list(actual.iterdir())


def test_quarantine_report_is_deeply_immutable() -> None:
    nested = {"outer": {"items": [1, {"value": 2}]}}
    report = replace(_report(), deterministic_details=nested)

    nested["outer"]["items"][1]["value"] = 99  # type: ignore[index]

    assert report.as_dict()["deterministic_details"] == {
        "outer": {"items": [1, {"value": 2}]}
    }
    with pytest.raises(TypeError):
        report.deterministic_details["outer"]["items"][1]["value"] = 3


def test_quarantine_final_name_is_invisible_until_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written = threading.Event()
    release = threading.Event()
    original = quarantine_module._write_all

    def blocked_write(file_descriptor: int, content: bytes) -> None:
        original(file_descriptor, content)
        written.set()
        assert release.wait(timeout=5)

    monkeypatch.setattr(quarantine_module, "_write_all", blocked_write)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(write_quarantine_report, tmp_path / "nested", _report())
        assert written.wait(timeout=5)
        assert not list((tmp_path / "nested").glob("*.quarantine.json"))
        release.set()
        artifact = future.result(timeout=5)

    assert artifact.path.read_bytes().endswith(b"\n")


@pytest.mark.parametrize("fault", ["write", "fsync", "link", "unlink"])
def test_quarantine_fault_never_exposes_partial_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    directory = tmp_path / fault
    if fault == "write":
        monkeypatch.setattr(
            quarantine_module,
            "_write_all",
            lambda *_: (_ for _ in ()).throw(OSError("injected write")),
        )
    elif fault == "fsync":
        original_fsync = os.fsync
        failed = False

        def fail_file_fsync(file_descriptor: int) -> None:
            nonlocal failed
            if not failed and os.path.isfile(f"/dev/fd/{file_descriptor}"):
                failed = True
                raise OSError("injected fsync")
            original_fsync(file_descriptor)

        monkeypatch.setattr("dataset_devkit.quarantine.os.fsync", fail_file_fsync)
    elif fault == "link":
        monkeypatch.setattr(
            "dataset_devkit.quarantine.os.link",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("injected link")),
        )
    else:
        original_unlink = os.unlink
        failed = False

        def fail_first_unlink(path: str, *, dir_fd: int | None = None) -> None:
            nonlocal failed
            if not failed:
                failed = True
                raise OSError("injected unlink")
            original_unlink(path, dir_fd=dir_fd)

        monkeypatch.setattr("dataset_devkit.quarantine.os.unlink", fail_first_unlink)

    with pytest.raises(OSError, match=f"injected {fault}"):
        write_quarantine_report(directory, _report())

    assert not list(directory.glob("*.quarantine.json"))
    assert not list(directory.glob(".*.quarantine.tmp"))


def test_quarantine_final_name_collision_retries_without_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collision = "a" * 32
    fresh = "b" * 32
    existing = tmp_path / f"recording-a-{collision}.quarantine.json"
    existing.write_bytes(b"external")
    tokens = iter((collision, fresh))
    monkeypatch.setattr(
        "dataset_devkit.quarantine.uuid.uuid4",
        lambda: SimpleNamespace(hex=next(tokens)),
    )

    artifact = write_quarantine_report(tmp_path, _report())

    assert artifact.path.name == f"recording-a-{fresh}.quarantine.json"
    assert existing.read_bytes() == b"external"


def test_new_quarantine_ancestors_are_fsynced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_fsync = os.fsync
    directory_fsyncs = 0

    def record_fsync(file_descriptor: int) -> None:
        nonlocal directory_fsyncs
        if os.path.isdir(f"/dev/fd/{file_descriptor}"):
            directory_fsyncs += 1
        original_fsync(file_descriptor)

    monkeypatch.setattr("dataset_devkit.quarantine.os.fsync", record_fsync)

    write_quarantine_report(tmp_path / "a" / "b" / "c", _report())

    assert directory_fsyncs >= 4


def test_rejection_manifest_lock_file_replacement_cannot_lose_concurrent_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_flock = fcntl.flock
    first_locked = threading.Event()
    second_attempted = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def replace_lock_after_flock(file_descriptor: int, operation: int) -> None:
        nonlocal calls
        if operation != fcntl.LOCK_EX:
            original_flock(file_descriptor, operation)
            return
        with calls_lock:
            calls += 1
            call = calls
        if call == 2:
            second_attempted.set()
        original_flock(file_descriptor, operation)
        if call == 1:
            lock = tmp_path / ".rejections.jsonl.lock"
            if lock.exists():
                lock.rename(tmp_path / ".rejections.jsonl.lock.stale")
            lock.write_bytes(b"")
            first_locked.set()
            assert second_attempted.wait(timeout=5)

    monkeypatch.setattr(
        "dataset_devkit.quarantine.fcntl.flock", replace_lock_after_flock
    )
    first = replace(_report(), recording_id="first")
    second = replace(_report(), recording_id="second")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(
            write_rejection_manifest, tmp_path, "rejections.jsonl", (first,)
        )
        assert first_locked.wait(timeout=5)
        second_future = pool.submit(
            write_rejection_manifest, tmp_path, "rejections.jsonl", (second,)
        )
        first_future.result(timeout=5)
        second_future.result(timeout=5)

    rows = [
        json.loads(line)
        for line in (tmp_path / "rejections.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["recording_id"] for row in rows} == {"first", "second"}


@pytest.mark.parametrize("replaced_component", ["directory", "ancestor"])
def test_rejection_manifest_binds_configured_directory_path_for_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replaced_component: str,
) -> None:
    ancestor = tmp_path / "configured"
    directory = ancestor / "quarantine"
    manifest = write_rejection_manifest(directory, "rejections.jsonl", (_report(),))
    original_content = manifest.read_bytes()
    original_write_all = quarantine_module._write_all
    raced = False

    def replace_path_after_temporary_write(file_descriptor: int, content: bytes) -> None:
        nonlocal raced
        original_write_all(file_descriptor, content)
        if raced:
            return
        raced = True
        if replaced_component == "directory":
            directory.rename(ancestor / "quarantine-stale")
            directory.mkdir()
        else:
            ancestor.rename(tmp_path / "configured-stale")
            ancestor.mkdir()
            (tmp_path / "configured-stale" / "quarantine").rename(directory)

    monkeypatch.setattr(
        quarantine_module,
        "_write_all",
        replace_path_after_temporary_write,
    )

    with pytest.raises(StructuralExtractionError, match="quarantine directory.*identity"):
        write_rejection_manifest(
            directory,
            "rejections.jsonl",
            (replace(_report(), recording_id="raced"),),
        )

    if replaced_component == "directory":
        assert (ancestor / "quarantine-stale" / "rejections.jsonl").read_bytes() == (
            original_content
        )
        assert not (directory / "rejections.jsonl").exists()
    else:
        assert (directory / "rejections.jsonl").read_bytes() == original_content


def test_rejection_manifest_revalidates_directory_path_after_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "configured" / "quarantine"
    write_rejection_manifest(directory, "rejections.jsonl", (_report(),))
    original_fsync = os.fsync
    raced = False

    def replace_path_after_directory_fsync(file_descriptor: int) -> None:
        nonlocal raced
        original_fsync(file_descriptor)
        if raced or not stat.S_ISDIR(os.fstat(file_descriptor).st_mode):
            return
        raced = True
        directory.rename(directory.parent / "quarantine-stale")
        directory.mkdir()

    monkeypatch.setattr(
        "dataset_devkit.quarantine.os.fsync", replace_path_after_directory_fsync
    )

    with pytest.raises(StructuralExtractionError, match="quarantine directory.*identity"):
        write_rejection_manifest(
            directory,
            "rejections.jsonl",
            (replace(_report(), recording_id="published-before-race"),),
        )

    assert raced
    assert not (directory / "rejections.jsonl").exists()


def test_rejection_manifest_read_is_bound_to_opened_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = write_rejection_manifest(tmp_path, "rejections.jsonl", (_report(),))
    original_content = manifest.read_bytes()
    original_open = os.open
    raced = False

    def replace_manifest_before_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal raced
        if path == "rejections.jsonl" and flags & os.O_ACCMODE == os.O_RDONLY and not raced:
            raced = True
            manifest.rename(tmp_path / "rejections.jsonl.stale")
            manifest.write_text('{"forged":true}\n', encoding="utf-8")
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(
        "dataset_devkit.quarantine.os.open", replace_manifest_before_open
    )

    with pytest.raises(StructuralExtractionError, match="manifest.*identity"):
        write_rejection_manifest(tmp_path, "rejections.jsonl", (_report(),))

    assert manifest.read_bytes() == b'{"forged":true}\n'
    assert (tmp_path / "rejections.jsonl.stale").read_bytes() == original_content


def test_rejection_manifest_write_is_bound_to_temporary_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = write_rejection_manifest(tmp_path, "rejections.jsonl", (_report(),))
    original_content = manifest.read_bytes()
    original_write_all = quarantine_module._write_all
    raced = False

    def replace_temporary_after_write(file_descriptor: int, content: bytes) -> None:
        nonlocal raced
        original_write_all(file_descriptor, content)
        if not raced:
            raced = True
            temporary = next(tmp_path.glob(".rejections.jsonl.*.tmp"))
            temporary.rename(tmp_path / "owned-written-manifest.tmp")
            temporary.write_text('{"forged":true}\n', encoding="utf-8")

    monkeypatch.setattr(quarantine_module, "_write_all", replace_temporary_after_write)

    with pytest.raises(StructuralExtractionError, match="manifest.*identity"):
        write_rejection_manifest(tmp_path, "rejections.jsonl", (_report(),))

    assert manifest.read_bytes() == original_content
    assert next(tmp_path.glob(".rejections.jsonl.*.tmp")).read_bytes() == b'{"forged":true}\n'


def test_rejection_manifest_write_refuses_replaced_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = write_rejection_manifest(tmp_path, "rejections.jsonl", (_report(),))
    original_content = manifest.read_bytes()
    original_write_all = quarantine_module._write_all
    raced = False

    def replace_manifest_after_temporary_write(
        file_descriptor: int, content: bytes
    ) -> None:
        nonlocal raced
        original_write_all(file_descriptor, content)
        if not raced:
            raced = True
            manifest.rename(tmp_path / "rejections.jsonl.stale")
            manifest.write_text('{"forged":true}\n', encoding="utf-8")

    monkeypatch.setattr(
        quarantine_module,
        "_write_all",
        replace_manifest_after_temporary_write,
    )

    with pytest.raises(StructuralExtractionError, match="manifest.*identity"):
        write_rejection_manifest(tmp_path, "rejections.jsonl", (_report(),))

    assert manifest.read_bytes() == b'{"forged":true}\n'
    assert (tmp_path / "rejections.jsonl.stale").read_bytes() == original_content


def _policy_config(config_factory: object, *, sanity_off: bool = True) -> GlobalConfig:
    config = cast(GlobalConfig, config_factory())  # type: ignore[operator]
    rules = InvalidationRulesConfig(**{code: False for code in INVALIDITY_CODES})
    sanity = SanityChecksConfig.model_validate(
        {code: "off" if sanity_off else "error" for code in SANITY_CHECK_CODES}
    )
    return config.model_copy(
        update={
            "frame_validity": config.frame_validity.model_copy(
                update={"required_cameras": [], "invalidate_on": rules}
            ),
            "sanity_checks": sanity,
        }
    )


def test_coordinator_finishes_all_inputs_then_blocks_or_authorizes_partial(
    tmp_path: Path, config_factory: object
) -> None:
    calls: list[str] = []
    attempts = 0

    def extract(path: Path) -> object:
        nonlocal attempts
        attempts += 1
        calls.append(path.stem)
        if path.stem == "first":
            raise StructuralExtractionError("bad descriptor")
        if path.stem == "third":
            raise RuntimeError("boom")
        return replace(_result(tmp_path / f"{path.stem}-{attempts}"), source_path=path)

    requests = tuple(
        RecordingRequest(name, tmp_path / f"{name}.mcap")
        for name in ("first", "second", "third")
    )
    config = _policy_config(config_factory)
    coordinator = RecordingCoordinator(
        config=config,
        quarantine_directory=tmp_path / "quarantine-blocked",
        extractor=extract,  # type: ignore[arg-type]
    )

    with pytest.raises(PublicationBlockedError) as caught:
        coordinator.process(requests, allow_partial_export=False)

    blocked = caught.value.result
    assert calls == ["first", "second", "third"]
    assert [item.recording_id for item in blocked.successes] == ["second"]
    assert [item.recording_id for item in blocked.failures] == ["first", "third"]
    assert [item.category for item in blocked.failures] == ["structural", "unexpected"]
    assert not caught.value.publish_authorized
    assert caught.value.authorized_recording_ids == ()
    assert all(
        item.quarantine is not None and item.quarantine.path.is_file()
        for item in blocked.failures
    )

    calls.clear()
    partial = RecordingCoordinator(
        config=config,
        quarantine_directory=tmp_path / "quarantine-partial",
        extractor=extract,  # type: ignore[arg-type]
    ).process(requests, allow_partial_export=True)
    assert calls == ["first", "second", "third"]
    assert partial.publish_authorized
    assert partial.authorized_recording_ids == ("second",)


def test_partial_export_never_authorizes_incomplete_owned_cleanup(
    tmp_path: Path, config_factory: object
) -> None:
    working = tmp_path / "working" / "bad-invocation"
    working.mkdir(parents=True)
    (working / "owned.jpg").write_bytes(b"owned")
    authority = OwnedDirectoryAuthority.capture(working)

    def extract(path: Path) -> object:
        if path.stem == "bad":
            original = RuntimeError("cache materialization failed")
            cleanup = OwnedDirectoryCleanupError((authority.cleanup_failure(),))
            raise cleanup from original
        return replace(_result(tmp_path / "good-owned"), source_path=path)

    coordinator = RecordingCoordinator(
        config=_policy_config(config_factory),
        quarantine_directory=tmp_path / "quarantine",
        extractor=extract,  # type: ignore[arg-type]
    )
    requests = (
        RecordingRequest("good", tmp_path / "good.mcap"),
        RecordingRequest("bad", tmp_path / "bad.mcap"),
    )

    with pytest.raises(PublicationBlockedError) as caught:
        coordinator.process(requests, allow_partial_export=True)

    result = caught.value.result
    assert [item.recording_id for item in result.successes] == ["good"]
    assert not result.cleanup_complete
    assert not result.publish_authorized
    assert result.authorized_recording_ids == ()
    failure = result.failures[0]
    assert not failure.cleanup_complete
    assert failure.quarantine is not None
    payload = json.loads(failure.quarantine.path.read_text(encoding="utf-8"))
    assert payload["artifact_handling"] == "preserved_in_place"
    tree = payload["deterministic_details"]["owned_working_trees"][0]
    assert tree["path"] == str(working)
    assert tree["expected_parent_chain"]


def test_quarantine_failure_retains_original_and_continues_but_blocks_partial(
    tmp_path: Path, config_factory: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    persistence_calls = 0

    def extract(path: Path) -> object:
        calls.append(path.stem)
        if path.stem == "first":
            raise StructuralExtractionError("original descriptor failure")
        return replace(_result(tmp_path / path.stem), source_path=path)

    original_write = write_quarantine_report

    def fail_first_write(*args: object, **kwargs: object) -> object:
        nonlocal persistence_calls
        persistence_calls += 1
        if persistence_calls == 1:
            raise OSError("quarantine fsync failed")
        return original_write(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        "dataset_devkit.coordinator.write_quarantine_report", fail_first_write
    )
    coordinator = RecordingCoordinator(
        config=_policy_config(config_factory),
        quarantine_directory=tmp_path / "quarantine",
        extractor=extract,  # type: ignore[arg-type]
    )

    with pytest.raises(PublicationBlockedError) as caught:
        coordinator.process(
            (
                RecordingRequest("first", tmp_path / "first.mcap"),
                RecordingRequest("second", tmp_path / "second.mcap"),
            ),
            allow_partial_export=True,
        )

    result = caught.value.result
    assert calls == ["first", "second"]
    assert [item.recording_id for item in result.outcomes] == ["first", "second"]
    failure = result.failures[0]
    assert failure.exception_type == "StructuralExtractionError"
    assert failure.exception_message == "original descriptor failure"
    assert not failure.quarantine_persisted
    assert failure.quarantine is None
    assert failure.quarantine_report_path is None
    assert failure.quarantine_error is not None
    assert failure.quarantine_error.exception_type == "OSError"
    assert failure.quarantine_error.exception_message == "quarantine fsync failed"
    assert failure.quarantine_error.details["stage"] == "quarantine_persistence"
    assert not result.publish_authorized
    assert result.authorized_recording_ids == ()


def test_committed_drop_cleanup_failure_becomes_structural_quarantine(
    tmp_path: Path, config_factory: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _policy_config(config_factory)
    toggles = {code: False for code in INVALIDITY_CODES}
    toggles["gnss_source_invalid"] = True
    config = config.model_copy(
        update={
            "frame_validity": config.frame_validity.model_copy(
                update={
                    "invalid_sample_policy": "drop",
                    "invalidate_on": InvalidationRulesConfig.model_validate(toggles),
                }
            )
        }
    )
    original_unlink = os.unlink
    failed = False

    def fail_first_unlink(path: str, *, dir_fd: int | None = None) -> None:
        nonlocal failed
        if not failed and ".drop-" in path:
            failed = True
            raise OSError("injected cleanup")
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr("dataset_devkit.extraction.staging.os.unlink", fail_first_unlink)
    coordinator = RecordingCoordinator(
        config=config,
        quarantine_directory=tmp_path / "quarantine",
        extractor=lambda path: replace(_result(tmp_path / "owned"), source_path=path),
    )
    with pytest.raises(PublicationBlockedError) as caught:
        coordinator.process(
            (RecordingRequest("cleanup", tmp_path / "cleanup.mcap"),),
            allow_partial_export=True,
        )

    result = caught.value.result
    failure = result.failures[0]
    assert failure.category == "structural"
    assert failure.stage == "validity"
    assert "tombstone cleanup" in failure.exception_message
    assert failure.quarantine_persisted
    assert not failure.cleanup_complete
    assert not result.cleanup_complete
    assert not result.publish_authorized
    assert result.authorized_recording_ids == ()
    assert failure.quarantine is not None
    payload = json.loads(failure.quarantine.path.read_text(encoding="utf-8"))
    assert payload["artifact_handling"] == "preserved_in_place"
    record = payload["deterministic_details"]["owned_tombstones"][0]
    assert record["invocation_root"].endswith("/staging/recording-owned")
    assert record["tombstone_name"].startswith(".")
    assert record["original_name"].endswith(".jpg")
    assert isinstance(record["device"], int)
    assert isinstance(record["inode"], int)


def test_coordinator_does_not_follow_replaced_tombstone_for_artifact_detection(
    tmp_path: Path, config_factory: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _policy_config(config_factory)
    toggles = {code: False for code in INVALIDITY_CODES}
    toggles["gnss_source_invalid"] = True
    config = config.model_copy(
        update={
            "frame_validity": config.frame_validity.model_copy(
                update={
                    "invalid_sample_policy": "drop",
                    "invalidate_on": InvalidationRulesConfig.model_validate(toggles),
                }
            )
        }
    )
    external = tmp_path / "external-owned-by-someone-else"
    external.write_bytes(b"external")
    original_unlink = os.unlink
    replaced_name: str | None = None

    def replace_first_tombstone(path: str, *, dir_fd: int | None = None) -> None:
        nonlocal replaced_name
        if replaced_name is None and ".drop-" in path:
            original_unlink(path, dir_fd=dir_fd)
            os.symlink(external, path, dir_fd=dir_fd)
            replaced_name = path
            raise OSError("injected replaced cleanup entry")
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(
        "dataset_devkit.extraction.staging.os.unlink", replace_first_tombstone
    )

    def single_sample_result(path: Path) -> object:
        extracted = _result(tmp_path / "owned-replaced")
        sample = extracted.samples[0]
        selected_batch = replace(
            extracted.camera_batches[1],
            frames=(extracted.camera_batches[1].frames[0],),
        )
        return replace(
            extracted,
            source_path=path,
            camera_batches=(extracted.camera_batches[0], selected_batch),
            samples=(sample,),
            ego_poses_by_timestamp={sample.camera_timestamp_ns: sample.ego_pose},
        )

    coordinator = RecordingCoordinator(
        config=config,
        quarantine_directory=tmp_path / "quarantine-replaced",
        extractor=single_sample_result,  # type: ignore[arg-type]
    )
    with pytest.raises(PublicationBlockedError) as caught:
        coordinator.process(
            (RecordingRequest("cleanup", tmp_path / "cleanup.mcap"),),
            allow_partial_export=True,
        )

    result = caught.value.result
    failure = result.failures[0]
    assert not failure.cleanup_complete
    assert not result.publish_authorized
    assert result.authorized_recording_ids == ()
    assert failure.quarantine is not None
    payload = json.loads(failure.quarantine.path.read_text(encoding="utf-8"))
    assert payload["artifact_handling"] == "no_owned_artifacts"
    assert replaced_name is not None
    replaced = tmp_path / "owned-replaced" / "staging" / "recording-owned" / replaced_name
    assert replaced.is_symlink()
    assert external.read_bytes() == b"external"


def test_structural_failure_is_never_downgraded_by_sanity_policy(
    tmp_path: Path, config_factory: object
) -> None:
    def broken(path: Path) -> object:
        del path
        raise StructuralExtractionError("undecodable video")

    coordinator = RecordingCoordinator(
        config=_policy_config(config_factory),
        quarantine_directory=tmp_path / "quarantine",
        extractor=broken,  # type: ignore[arg-type]
    )

    result = coordinator.process(
        (RecordingRequest("broken", tmp_path / "broken.mcap"),),
        allow_partial_export=True,
    )

    assert result.failures[0].category == "structural"
    assert result.failures[0].stage == "extraction"


def test_coordinator_honors_worker_bound_and_preserves_request_order(
    tmp_path: Path, config_factory: object
) -> None:
    rendezvous = threading.Barrier(2)
    threads: set[int] = set()
    lock = threading.Lock()

    def extract(path: Path) -> object:
        with lock:
            threads.add(threading.get_ident())
        rendezvous.wait(timeout=5)
        return replace(_result(tmp_path / f"owned-{path.stem}"), source_path=path)

    requests = (
        RecordingRequest("second", tmp_path / "second.mcap"),
        RecordingRequest("first", tmp_path / "first.mcap"),
    )
    result = RecordingCoordinator(
        config=_policy_config(config_factory),
        quarantine_directory=tmp_path / "quarantine",
        extractor=extract,  # type: ignore[arg-type]
    ).process(requests, allow_partial_export=False, max_workers=2)

    assert len(threads) == 2
    assert [item.recording_id for item in result.successes] == ["second", "first"]


def test_sanity_error_is_quarantined_with_validity_audit_context(
    tmp_path: Path, config_factory: object
) -> None:
    config = _policy_config(config_factory)
    config = config.model_copy(
        update={
            "frame_validity": config.frame_validity.model_copy(
                update={
                    "required_cameras": ["front", "rear", "side"],
                    "invalidate_on": InvalidationRulesConfig(),
                }
            ),
            "sanity_checks": SanityChecksConfig.model_validate(
                {
                    **{code: "off" for code in SANITY_CHECK_CODES},
                    "empty_final_candidates": "error",
                }
            ),
        }
    )
    coordinator = RecordingCoordinator(
        config=config,
        quarantine_directory=tmp_path / "quarantine",
        extractor=lambda path: replace(_result(tmp_path / "owned"), source_path=path),
    )

    coordinated = coordinator.process(
        (RecordingRequest("sanity", tmp_path / "sanity.mcap"),),
        allow_partial_export=True,
    )

    failure = coordinated.failures[0]
    assert failure.category == "sanity"
    assert failure.stage == "sanity"
    assert failure.quarantine is not None
    payload = json.loads(failure.quarantine.path.read_text(encoding="utf-8"))
    assert payload["artifact_handling"] == "preserved_in_place"
    context_codes = {item["code"] for item in payload["observed_context"]}
    assert context_codes == {*INVALIDITY_CODES, "empty_final_candidates"}
    position = next(
        item for item in payload["observed_context"]
        if item["code"] == "position_sigma_exceeded"
    )
    assert position["scope"] == "pose"
    assert position["measured_values"]["east_sigma_m"] == pytest.approx(1.2)
    assert position["threshold"] == 0.5
    assert position["details"]["before_position_uncertainty"]["east_sigma_m"] == 1.2
    assert position["grid_target_timestamp_ns"] == 2_000_000_000
    assert position["batch_timestamp_ns"] == 2_000_000_100
    assert position["camera_timestamp_ns"] in {100, 2_000_000_100}
    assert position["camera_name"] in {"front", "rear", "extra"}
    assert position["enabled_as_invalidator"] is True
    sanity = next(
        item for item in payload["observed_context"]
        if item["code"] == "empty_final_candidates"
    )
    assert sanity["policy"] == "error"
    assert sanity["details"] == {"count": 0}


def test_coordinator_rejects_empty_and_duplicate_identities(
    tmp_path: Path, config_factory: object
) -> None:
    coordinator = RecordingCoordinator(
        config=_policy_config(config_factory),
        quarantine_directory=tmp_path / "quarantine",
        extractor=lambda path: _result(path.parent),
    )
    with pytest.raises(CoordinatorInputError, match="at least one"):
        coordinator.process((), allow_partial_export=True)
    duplicate = (
        RecordingRequest("same", tmp_path / "one.mcap"),
        RecordingRequest("same", tmp_path / "two.mcap"),
    )
    with pytest.raises(CoordinatorInputError, match="duplicate.*same"):
        coordinator.process(duplicate, allow_partial_export=True)


def test_duplicate_identity_detection_is_linear_and_deterministic(
    tmp_path: Path, config_factory: object
) -> None:
    class CountingIdentity(str):
        comparisons = 0

        def __eq__(self, other: object) -> bool:
            type(self).comparisons += 1
            return super().__eq__(other)

        __hash__ = str.__hash__

    identities = [CountingIdentity(f"recording-{index}") for index in range(2_000)]
    requests = tuple(
        RecordingRequest(identity, tmp_path / f"{identity}.mcap")
        for identity in (*identities, CountingIdentity("recording-0"))
    )
    coordinator = RecordingCoordinator(
        config=_policy_config(config_factory),
        quarantine_directory=tmp_path / "quarantine",
        extractor=lambda path: _result(path.parent),
    )

    with pytest.raises(CoordinatorInputError, match="duplicate.*recording-0"):
        coordinator.process(requests, allow_partial_export=True)

    assert CountingIdentity.comparisons < 10_000


def test_broken_exception_string_never_aborts_later_recordings(
    tmp_path: Path, config_factory: object
) -> None:
    class BrokenStringError(StructuralExtractionError):
        def __str__(self) -> str:
            raise RuntimeError("broken __str__")

    calls: list[str] = []

    def extract(path: Path) -> object:
        calls.append(path.stem)
        if path.stem == "first":
            raise BrokenStringError()
        return replace(_result(tmp_path / path.stem), source_path=path)

    result = RecordingCoordinator(
        config=_policy_config(config_factory),
        quarantine_directory=tmp_path / "quarantine",
        extractor=extract,  # type: ignore[arg-type]
    ).process(
        (
            RecordingRequest("first", tmp_path / "first.mcap"),
            RecordingRequest("second", tmp_path / "second.mcap"),
        ),
        allow_partial_export=True,
    )

    assert calls == ["first", "second"]
    failure = result.failures[0]
    assert failure.exception_type == "BrokenStringError"
    assert "unprintable" in failure.exception_message
    assert "RuntimeError" in failure.exception_message
    assert failure.quarantine_persisted
    assert result.authorized_recording_ids == ("second",)


def test_real_task3_extraction_feeds_policy_and_coordinator(
    tmp_path: Path, config_factory: object
) -> None:
    path = tmp_path / "real.mcap"
    write_mcap(
        path,
        camera_payloads=(camera_message(1_000_000_000, (1_000_000_010, 1_000_000_020)),),
    )
    config = _policy_config(config_factory)
    config = config.model_copy(
        update={
            "frame_validity": config.frame_validity.model_copy(
                update={"required_cameras": ["cam_0", "cam_1"]}
            )
        }
    )
    extractor = RecordingExtractor(
        camera_topic="rec_cameras",
        gnss_topic="gnss",
        target_fps=Fraction(2, 1),
        tolerance_ns=0,
        staging_root=tmp_path / "staging",
        decoder_factory=DeterministicDecoder,
    )
    result = RecordingCoordinator(
        config=config,
        quarantine_directory=tmp_path / "quarantine",
        extractor=extractor.extract,
    ).process((RecordingRequest("real", path),), allow_partial_export=False)

    assert result.authorized_recording_ids == ("real",)
    assert len(result.successes[0].validity.final_candidates) == 1
    assert result.successes[0].validity.final_candidates[0].camera_timestamps == (
        ("cam_0", 1_000_000_010),
        ("cam_1", 1_000_000_020),
    )
