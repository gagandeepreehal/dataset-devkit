from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import cast

import pytest

from dataset_devkit.config import GlobalConfig, InvalidationRulesConfig, SanityChecksConfig
from dataset_devkit.coordinator import (
    CoordinatorInputError,
    PublicationBlockedError,
    RecordingCoordinator,
    RecordingRequest,
)
from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.service import RecordingExtractor
from dataset_devkit.quarantine import QuarantineReport, write_quarantine_report
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
    assert all(item.quarantine.path.is_file() for item in blocked.failures)

    calls.clear()
    partial = RecordingCoordinator(
        config=config,
        quarantine_directory=tmp_path / "quarantine-partial",
        extractor=extract,  # type: ignore[arg-type]
    ).process(requests, allow_partial_export=True)
    assert calls == ["first", "second", "third"]
    assert partial.publish_authorized
    assert partial.authorized_recording_ids == ("second",)


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
    payload = json.loads(failure.quarantine.path.read_text(encoding="utf-8"))
    assert payload["artifact_handling"] == "preserved_in_place"
    context_codes = {item["code"] for item in payload["observed_context"]}
    assert context_codes == {*INVALIDITY_CODES, "empty_final_candidates"}


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
