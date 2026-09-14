"""Backend-neutral preparation of immutable recording inputs."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import Literal, Protocol, cast

from dataset_devkit.config import (
    GlobalConfig,
    GlobalConfigV1,
    GlobalConfigV2,
    PrivacyClassification,
    normalized_source,
)
from dataset_devkit.decoded_acquisition import (
    AcquiredDecodedRecording,
    DecodedHfAcquirer,
)
from dataset_devkit.decoded_manifest import DecodedRecordingEntry, DecodedSelectionPlan
from dataset_devkit.extraction.camera import HevcDecoder
from dataset_devkit.extraction.decoded import DecodedRecordingExtractor
from dataset_devkit.extraction.models import PoseFrame
from dataset_devkit.extraction.service import OwnedRecordingExtraction, RecordingExtractor
from dataset_devkit.huggingface_acquisition import AcquisitionResult, HuggingFaceAcquirer
from dataset_devkit.huggingface_manifest import ManifestEntry
from dataset_devkit.provenance import (
    RecordingFingerprint,
    SourceFingerprint,
    canonical_hash,
)

SourceType = Literal["mcap_hf", "decoded_hf"]


class McapAcquirerProtocol(Protocol):
    def load_entries(self) -> tuple[ManifestEntry, ...]: ...

    def acquire(self, entry: ManifestEntry) -> AcquisitionResult: ...

    def extraction_cache_reusable(
        self, source: SourceFingerprint, expected_extraction_config_hash: str
    ) -> bool: ...

    def record_extraction_complete(
        self, source: SourceFingerprint, completed_extraction_config_hash: str
    ) -> Path: ...


class DecodedAcquirerProtocol(Protocol):
    def load_plan(self) -> DecodedSelectionPlan: ...

    def acquire(self, entry: DecodedRecordingEntry) -> AcquiredDecodedRecording: ...


class SourceRuntimeProtocol(Protocol):
    @property
    def mcap_acquirer_factory(
        self,
    ) -> Callable[[GlobalConfigV1], McapAcquirerProtocol]: ...

    @property
    def decoded_acquirer_factory(
        self,
    ) -> Callable[[GlobalConfigV2], DecodedAcquirerProtocol]: ...

    @property
    def decoder_factory(self) -> Callable[[], HevcDecoder] | None: ...


@dataclass(frozen=True)
class PreparedRecording:
    recording_id: str
    locator: str
    fingerprint: RecordingFingerprint
    working_path: Path
    extract_owned: Callable[[], OwnedRecordingExtraction]
    preparation_error: Exception | None = None
    cache_reusable: Callable[[str], bool] | None = None
    record_extraction_complete: Callable[[str], Path] | None = None


@dataclass(frozen=True)
class PreparedSourceBatch:
    source_type: SourceType
    privacy_classification: PrivacyClassification
    pose_frame: PoseFrame
    source_config_hash: str
    recordings: tuple[PreparedRecording, ...]


def _raise_preparation_error(error: Exception) -> OwnedRecordingExtraction:
    raise error


def _prepare_mcap(
    config: GlobalConfigV1, runtime: SourceRuntimeProtocol
) -> PreparedSourceBatch:
    acquirer = runtime.mcap_acquirer_factory(config)
    entries = acquirer.load_entries()
    extractor_kwargs: dict[str, object] = {}
    if runtime.decoder_factory is not None:
        extractor_kwargs["decoder_factory"] = runtime.decoder_factory
    extractor = RecordingExtractor(
        camera_topic=config.topics.camera,
        gnss_topic=config.topics.gnss,
        target_fps=Fraction(str(config.downsampling.target_fps)),
        tolerance_ns=int(config.downsampling.tolerance_ms * 1_000_000),
        staging_root=config.paths.work_dir,
        **extractor_kwargs,  # type: ignore[arg-type]
    )

    def acquire_one(entry: ManifestEntry) -> AcquisitionResult | Exception:
        try:
            return acquirer.acquire(entry)
        except Exception as error:
            return error

    with ThreadPoolExecutor(max_workers=config.execution.workers) as executor:
        outcomes = tuple(executor.map(acquire_one, entries))
    prepared: list[PreparedRecording] = []
    source = normalized_source(config)
    for index, (entry, outcome) in enumerate(zip(entries, outcomes, strict=True)):
        expected_fingerprint = SourceFingerprint(
            source.repo_id,
            source.revision,
            entry.repo_path,
            entry.sha256,
            entry.size,
        )
        recording_id = f"recording-{index:06d}"
        if isinstance(outcome, Exception):
            prepared.append(
                PreparedRecording(
                    recording_id,
                    entry.repo_path,
                    expected_fingerprint,
                    Path(entry.repo_path),
                    partial(_raise_preparation_error, outcome),
                    preparation_error=outcome,
                )
            )
            continue
        fingerprint = outcome.manifest.source
        prepared.append(
            PreparedRecording(
                recording_id,
                entry.repo_path,
                fingerprint,
                outcome.artifact_path,
                partial(extractor.extract_owned, outcome.artifact_path),
                cache_reusable=partial(
                    acquirer.extraction_cache_reusable, fingerprint
                ),
                record_extraction_complete=partial(
                    acquirer.record_extraction_complete, fingerprint
                ),
            )
        )
    return PreparedSourceBatch(
        "mcap_hf",
        "restricted_raw",
        "web_mercator_v1",
        canonical_hash(source.model_dump(mode="json")),
        tuple(prepared),
    )


def _prepare_decoded(
    config: GlobalConfigV2, runtime: SourceRuntimeProtocol
) -> PreparedSourceBatch:
    acquirer = runtime.decoded_acquirer_factory(config)
    plan = acquirer.load_plan()
    extractor = DecodedRecordingExtractor(
        target_fps=Fraction(str(config.downsampling.target_fps)),
        tolerance_ns=int(config.downsampling.tolerance_ms * 1_000_000),
        staging_root=config.paths.work_dir,
    )

    def acquire_one(
        entry: DecodedRecordingEntry,
    ) -> AcquiredDecodedRecording | Exception:
        try:
            return acquirer.acquire(entry)
        except Exception as error:
            return error

    with ThreadPoolExecutor(max_workers=config.execution.workers) as executor:
        outcomes = tuple(executor.map(acquire_one, plan.recordings))
    prepared: list[PreparedRecording] = []
    for entry, outcome in zip(plan.recordings, outcomes, strict=True):
        if isinstance(outcome, Exception):
            prepared.append(
                PreparedRecording(
                    entry.recording_id,
                    entry.recording_id,
                    entry.fingerprint,
                    Path(entry.recording_id),
                    partial(_raise_preparation_error, outcome),
                    preparation_error=outcome,
                )
            )
            continue
        working_path = outcome.path(outcome.entry.videos[0].path)
        prepared.append(
            PreparedRecording(
                entry.recording_id,
                entry.recording_id,
                entry.fingerprint,
                working_path,
                partial(extractor.extract_owned, outcome),
            )
        )
    return PreparedSourceBatch(
        "decoded_hf",
        "privacy_transformed",
        "recording_local_enu_v1",
        canonical_hash(config.source.model_dump(mode="json")),
        tuple(prepared),
    )


def prepare_source(
    config: GlobalConfig, runtime: SourceRuntimeProtocol
) -> PreparedSourceBatch:
    """Acquire and bind exactly the backend selected by the versioned config."""
    if isinstance(config, GlobalConfigV1):
        return _prepare_mcap(config, runtime)
    if config.source.type == "decoded_hf":
        return _prepare_decoded(config, runtime)
    raise ValueError("schema 2.0 currently requires a decoded_hf source")


DEFAULT_MCAP_ACQUIRER_FACTORY = cast(
    Callable[[GlobalConfigV1], McapAcquirerProtocol], HuggingFaceAcquirer.from_config
)
DEFAULT_DECODED_ACQUIRER_FACTORY = cast(
    Callable[[GlobalConfigV2], DecodedAcquirerProtocol], DecodedHfAcquirer.from_config
)
