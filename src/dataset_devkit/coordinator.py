"""Independent recording coordination and the partial-publication authorization gate."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dataset_devkit.config import GlobalConfig
from dataset_devkit.extraction.errors import (
    NonstructuralSanityError,
    StructuralExtractionError,
)
from dataset_devkit.extraction.models import RecordingExtractionResult
from dataset_devkit.quarantine import (
    FailureCategory,
    QuarantineArtifact,
    QuarantineReport,
    write_quarantine_report,
)
from dataset_devkit.sanity import SanityReport, evaluate_sanity
from dataset_devkit.validity import ValidityReport, evaluate_validity


class CoordinatorInputError(ValueError):
    """Raised before processing for an ambiguous recording request set."""


@dataclass(frozen=True)
class RecordingRequest:
    recording_id: str
    source_path: Path
    source_config_hash: str | None = None
    extraction_config_hash: str | None = None

    def __post_init__(self) -> None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.recording_id) is None:
            raise CoordinatorInputError("recording identity must be a nonblank safe segment")


@dataclass(frozen=True)
class RecordingSuccess:
    recording_id: str
    extraction: RecordingExtractionResult
    validity: ValidityReport
    sanity: SanityReport


@dataclass(frozen=True)
class RecordingFailure:
    recording_id: str
    category: FailureCategory
    stage: str
    exception_type: str
    exception_message: str
    quarantine: QuarantineArtifact


RecordingOutcome = RecordingSuccess | RecordingFailure


@dataclass(frozen=True)
class CoordinatorResult:
    outcomes: tuple[RecordingOutcome, ...]
    successes: tuple[RecordingSuccess, ...]
    failures: tuple[RecordingFailure, ...]
    publish_authorized: bool
    authorized_recording_ids: tuple[str, ...]


class PublicationBlockedError(RuntimeError):
    """All recordings completed, but failures prohibit any publication."""

    def __init__(self, result: CoordinatorResult) -> None:
        self.result = result
        self.successes = result.successes
        self.failures = result.failures
        self.publish_authorized = False
        self.authorized_recording_ids: tuple[str, ...] = ()
        super().__init__(
            f"publication blocked after {len(result.failures)} recording failure(s); "
            "zero recordings are authorized"
        )


def _observed_context(
    validity: ValidityReport | None, error: Exception
) -> tuple[dict[str, object], ...]:
    context: list[dict[str, object]] = [] if validity is None else [
        {
            "code": item.code,
            "scope": item.scope,
            "enabled_as_invalidator": item.enabled_as_invalidator,
            "grid_target_timestamp_ns": item.grid_target_timestamp_ns,
            "batch_timestamp_ns": item.batch_timestamp_ns,
            "camera_timestamp_ns": item.camera_timestamp_ns,
            "camera_name": item.camera_name,
        }
        for item in validity.observations
    ]
    if isinstance(error, NonstructuralSanityError):
        context.extend(
            {
                "code": item.code,
                "scope": item.scope,
                "policy": item.policy,
                "message": item.message,
                "details": item.details,
            }
            for item in error.observations
        )
    return tuple(context)


def _has_owned_artifacts(extraction: RecordingExtractionResult | None) -> bool:
    if extraction is None:
        return False
    for sample in extraction.samples:
        staged = sample.staged_image
        try:
            current = staged.path.stat(follow_symlinks=False)
        except OSError:
            continue
        if (
            staged.path.parent == extraction.staging_root
            and stat.S_ISREG(current.st_mode)
            and current.st_nlink == 1
            and (current.st_dev, current.st_ino) == (staged.device, staged.inode)
        ):
            return True
    return False


class RecordingCoordinator:
    """Run each requested source independently and authorize, but never publish, results."""

    def __init__(
        self,
        *,
        config: GlobalConfig,
        quarantine_directory: Path | None = None,
        extractor: Callable[[Path], RecordingExtractionResult],
    ) -> None:
        self.config = config
        self.quarantine_directory = (
            config.quarantine.directory if quarantine_directory is None else quarantine_directory
        )
        self.extractor = extractor
        extraction_config = config.model_dump(
            mode="json",
            include={
                "topics",
                "downsampling",
                "image",
                "gnss",
                "frame_validity",
                "sanity_checks",
            },
        )
        self._config_hash = hashlib.sha256(
            json.dumps(extraction_config, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    def _failure(
        self,
        request: RecordingRequest,
        error: Exception,
        category: FailureCategory,
        stage: str,
        extraction: RecordingExtractionResult | None,
        validity: ValidityReport | None,
    ) -> RecordingFailure:
        source_details: dict[str, object] = {}
        try:
            source_stat = request.source_path.stat(follow_symlinks=False)
        except OSError:
            source_details["source_available"] = False
        else:
            source_details.update(
                {
                    "source_available": True,
                    "source_size": source_stat.st_size,
                    "source_device": source_stat.st_dev,
                    "source_inode": source_stat.st_ino,
                }
            )
        artifact_handling: Literal["no_owned_artifacts", "preserved_in_place"] = (
            "preserved_in_place" if _has_owned_artifacts(extraction) else "no_owned_artifacts"
        )
        report = QuarantineReport(
            recording_id=request.recording_id,
            source_path=str(request.source_path),
            status="quarantined",
            category=category,
            exception_type=type(error).__name__,
            exception_message=str(error),
            stage=stage,
            deterministic_details=source_details,
            observed_context=_observed_context(validity, error),
            source_config_hash=request.source_config_hash,
            extraction_config_hash=request.extraction_config_hash or self._config_hash,
            artifact_handling=artifact_handling,
        )
        artifact = write_quarantine_report(self.quarantine_directory, report)
        return RecordingFailure(
            request.recording_id,
            category,
            stage,
            type(error).__name__,
            str(error),
            artifact,
        )

    def process(
        self,
        requests: Sequence[RecordingRequest],
        *,
        allow_partial_export: bool | None = None,
    ) -> CoordinatorResult:
        if not requests:
            raise CoordinatorInputError("at least one recording is required")
        identities = [request.recording_id for request in requests]
        duplicates = sorted({item for item in identities if identities.count(item) > 1})
        if duplicates:
            raise CoordinatorInputError(f"duplicate recording identity: {duplicates[0]}")
        outcomes: list[RecordingOutcome] = []
        for request in requests:
            extraction: RecordingExtractionResult | None = None
            validity: ValidityReport | None = None
            stage = "extraction"
            try:
                extraction = self.extractor(request.source_path)
                stage = "validity"
                validity = evaluate_validity(extraction, self.config)
                stage = "sanity"
                sanity = evaluate_sanity(extraction, validity, self.config)
            except StructuralExtractionError as error:
                outcomes.append(
                    self._failure(request, error, "structural", stage, extraction, validity)
                )
            except NonstructuralSanityError as error:
                outcomes.append(
                    self._failure(request, error, "sanity", stage, extraction, validity)
                )
            except Exception as error:
                outcomes.append(
                    self._failure(request, error, "unexpected", stage, extraction, validity)
                )
            else:
                outcomes.append(
                    RecordingSuccess(request.recording_id, extraction, validity, sanity)
                )
        successes = tuple(item for item in outcomes if isinstance(item, RecordingSuccess))
        failures = tuple(item for item in outcomes if isinstance(item, RecordingFailure))
        partial = (
            self.config.execution.allow_partial_export
            if allow_partial_export is None
            else allow_partial_export
        )
        if failures and not partial:
            blocked = CoordinatorResult(tuple(outcomes), successes, failures, False, ())
            raise PublicationBlockedError(blocked)
        return CoordinatorResult(
            tuple(outcomes),
            successes,
            failures,
            True,
            tuple(item.recording_id for item in successes),
        )
