"""Independent recording coordination and the partial-publication authorization gate."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from dataset_devkit.config import GlobalConfig
from dataset_devkit.extraction.errors import (
    NonstructuralSanityError,
    StructuralExtractionError,
)
from dataset_devkit.extraction.models import RecordingExtractionResult
from dataset_devkit.extraction.staging import (
    StagedImageCleanupError,
    TombstoneRecord,
    owned_tombstone_record_matches,
)
from dataset_devkit.publication import OwnedDirectoryCleanupError
from dataset_devkit.quarantine import (
    FailureCategory,
    QuarantineArtifact,
    QuarantineReport,
    write_quarantine_report,
)
from dataset_devkit.sanity import SanityReport, evaluate_sanity
from dataset_devkit.validity import (
    ValidityReport,
    evaluate_validity,
    validate_validity_enforcement,
)


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
class QuarantinePersistenceFailure:
    exception_type: str
    exception_message: str
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


@dataclass(frozen=True)
class RecordingFailure:
    recording_id: str
    category: FailureCategory
    stage: str
    exception_type: str
    exception_message: str
    quarantine: QuarantineArtifact | None
    quarantine_persisted: bool
    quarantine_report_path: Path | None
    quarantine_error: QuarantinePersistenceFailure | None


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


def _safe_exception_message(error: BaseException) -> str:
    try:
        return str(error)
    except BaseException as formatting_error:
        return (
            f"<unprintable {type(error).__name__}: __str__ raised "
            f"{type(formatting_error).__name__}>"
        )


def _observed_context(
    validity: ValidityReport | None, error: Exception
) -> tuple[dict[str, object], ...]:
    context: list[dict[str, object]] = [] if validity is None else [
        {
            "code": item.code,
            "scope": item.scope,
            "measured_values": item.measured_values,
            "threshold": item.threshold,
            "details": item.details,
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


def _has_owned_artifacts(
    extraction: RecordingExtractionResult | None,
    error: Exception,
) -> bool:
    if isinstance(error, OwnedDirectoryCleanupError) and error.failures:
        return True
    if isinstance(error, StagedImageCleanupError) and any(
        owned_tombstone_record_matches(record) for record in error.tombstones
    ):
        return True
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


def _tombstone_details(record: TombstoneRecord) -> dict[str, object]:
    return {
        "invocation_root": str(record.invocation_root),
        "directory_device": record.directory_device,
        "directory_inode": record.directory_inode,
        "directory_chain_identities": record.directory_chain_identities,
        "tombstone_name": record.tombstone_name,
        "original_name": record.original_name,
        "device": record.device,
        "inode": record.inode,
        "expected_regular": record.expected_regular,
        "expected_single_link": record.expected_single_link,
    }


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
        exception_message = _safe_exception_message(error)
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
            "preserved_in_place"
            if _has_owned_artifacts(extraction, error)
            else "no_owned_artifacts"
        )
        if isinstance(error, StagedImageCleanupError):
            source_details["owned_tombstones"] = tuple(
                _tombstone_details(record) for record in error.tombstones
            )
        if isinstance(error, OwnedDirectoryCleanupError):
            source_details["owned_working_trees"] = tuple(
                item.to_dict() for item in error.failures
            )
            if error.__cause__ is not None:
                source_details["cleanup_original_cause"] = {
                    "exception_type": type(error.__cause__).__name__,
                    "exception_message": _safe_exception_message(error.__cause__),
                }
        artifact: QuarantineArtifact | None = None
        quarantine_error: QuarantinePersistenceFailure | None = None
        try:
            report = QuarantineReport(
                recording_id=request.recording_id,
                source_path=str(request.source_path),
                status="quarantined",
                category=category,
                exception_type=type(error).__name__,
                exception_message=exception_message,
                stage=stage,
                deterministic_details=source_details,
                observed_context=_observed_context(validity, error),
                source_config_hash=request.source_config_hash,
                extraction_config_hash=request.extraction_config_hash or self._config_hash,
                artifact_handling=artifact_handling,
            )
            artifact = write_quarantine_report(self.quarantine_directory, report)
        except Exception as persistence_error:
            quarantine_error = QuarantinePersistenceFailure(
                type(persistence_error).__name__,
                _safe_exception_message(persistence_error),
                {
                    "stage": "quarantine_persistence",
                    "directory": str(self.quarantine_directory),
                },
            )
        return RecordingFailure(
            recording_id=request.recording_id,
            category=category,
            stage=stage,
            exception_type=type(error).__name__,
            exception_message=exception_message,
            quarantine=artifact,
            quarantine_persisted=artifact is not None,
            quarantine_report_path=None if artifact is None else artifact.path,
            quarantine_error=quarantine_error,
        )

    def quarantine_failure(
        self,
        request: RecordingRequest,
        error: Exception,
        *,
        category: FailureCategory,
        stage: str,
        extraction: RecordingExtractionResult | None = None,
        validity: ValidityReport | None = None,
    ) -> RecordingFailure:
        """Persist one failure outside the standard extraction/validity/sanity path."""
        return self._failure(request, error, category, stage, extraction, validity)

    def _process_one(self, request: RecordingRequest) -> RecordingOutcome:
        extraction: RecordingExtractionResult | None = None
        validity: ValidityReport | None = None
        stage = "extraction"
        try:
            extraction = self.extractor(request.source_path)
            stage = "validity"
            validity = evaluate_validity(extraction, self.config)
            validate_validity_enforcement(extraction, validity, self.config)
            stage = "sanity"
            sanity = evaluate_sanity(extraction, validity, self.config)
        except StructuralExtractionError as error:
            return self._failure(request, error, "structural", stage, extraction, validity)
        except NonstructuralSanityError as error:
            return self._failure(request, error, "sanity", stage, extraction, validity)
        except Exception as error:
            return self._failure(request, error, "unexpected", stage, extraction, validity)
        return RecordingSuccess(request.recording_id, extraction, validity, sanity)

    def process(
        self,
        requests: Sequence[RecordingRequest],
        *,
        allow_partial_export: bool | None = None,
        max_workers: int = 1,
    ) -> CoordinatorResult:
        if not requests:
            raise CoordinatorInputError("at least one recording is required")
        identities = [request.recording_id for request in requests]
        duplicates = sorted(
            identity for identity, count in Counter(identities).items() if count > 1
        )
        if duplicates:
            raise CoordinatorInputError(f"duplicate recording identity: {duplicates[0]}")
        if max_workers < 1:
            raise CoordinatorInputError("max_workers must be positive")
        if max_workers == 1:
            outcomes = [self._process_one(request) for request in requests]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                outcomes = list(executor.map(self._process_one, requests))
        successes = tuple(item for item in outcomes if isinstance(item, RecordingSuccess))
        failures = tuple(item for item in outcomes if isinstance(item, RecordingFailure))
        partial = (
            self.config.execution.allow_partial_export
            if allow_partial_export is None
            else allow_partial_export
        )
        quarantine_incomplete = any(not failure.quarantine_persisted for failure in failures)
        if failures and (not partial or quarantine_incomplete):
            blocked = CoordinatorResult(tuple(outcomes), successes, failures, False, ())
            raise PublicationBlockedError(blocked)
        return CoordinatorResult(
            tuple(outcomes),
            successes,
            failures,
            True,
            tuple(item.recording_id for item in successes),
        )
