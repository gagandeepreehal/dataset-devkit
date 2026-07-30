"""Public standalone build, validation, and inspection services."""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Protocol

from dataset_devkit.acquisition import AcquisitionResult, AzureBlobAcquirer
from dataset_devkit.blob_list import parse_blob_list
from dataset_devkit.config import GlobalConfig
from dataset_devkit.coordinator import (
    PublicationBlockedError,
    RecordingCoordinator,
    RecordingFailure,
    RecordingRequest,
    RecordingSuccess,
)
from dataset_devkit.dataset import Dataset, DatasetFormatError
from dataset_devkit.export import (
    NUSCENES_VERSION,
    ExportEvidence,
    export_dataset,
    pipeline_graph_scene_sequence,
    preflight_recording_export,
)
from dataset_devkit.extraction.cache import ExtractionResultCache
from dataset_devkit.extraction.camera import HevcDecoder
from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.models import RecordingExtractionResult
from dataset_devkit.extraction.service import RecordingExtractor
from dataset_devkit.features import SceneFeatures, compute_recording_features
from dataset_devkit.filtering import filter_scenes
from dataset_devkit.provenance import SourceFingerprint, canonical_hash, extraction_config_hash
from dataset_devkit.publication import StagingLease, publish_staging
from dataset_devkit.quarantine import write_rejection_manifest
from dataset_devkit.scenario_selection import select_scenarios
from dataset_devkit.scenes import build_recording_scenes
from dataset_devkit.split import split_selected_scenes
from dataset_devkit.validation import ValidationReport, finalize_dataset
from dataset_devkit.validation import validate_dataset as _validate_dataset


class BuildOperationalError(RuntimeError):
    """Raised for a safe, user-facing pipeline failure."""


class AcquirerProtocol(Protocol):
    def acquire(self, blob_path: str) -> AcquisitionResult: ...

    def extraction_cache_reusable(
        self, source: SourceFingerprint, expected_extraction_config_hash: str
    ) -> bool: ...

    def record_extraction_complete(
        self, source: SourceFingerprint, completed_extraction_config_hash: str
    ) -> Path: ...


@dataclass(frozen=True)
class BuildRuntime:
    """Injectable external-runtime boundary used by deterministic tests."""

    acquirer_factory: Callable[[GlobalConfig], AcquirerProtocol] = AzureBlobAcquirer.from_config
    decoder_factory: Callable[[], HevcDecoder] | None = None
    extraction_cache_factory: Callable[[Path], ExtractionResultCache] = ExtractionResultCache
    official_smoke: bool = True


@dataclass(frozen=True)
class BuildResult:
    dataroot: Path
    version: str
    scene_count: int
    sample_count: int
    sample_data_count: int
    content_hash: str
    partial: bool
    failed_recordings: tuple[str, ...]


@dataclass(frozen=True)
class InspectionSummary:
    version: str
    table_counts: tuple[tuple[str, int], ...]
    scene_count: int
    sample_count: int
    sample_data_count: int
    image_count: int
    channels: tuple[str, ...]
    recording_count: int
    train_scene_count: int
    test_scene_count: int
    validation_state: str
    content_hash: str

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "table_counts": {key: value for key, value in self.table_counts},
            "scene_count": self.scene_count,
            "sample_count": self.sample_count,
            "sample_data_count": self.sample_data_count,
            "image_count": self.image_count,
            "channels": list(self.channels),
            "recording_count": self.recording_count,
            "split_counts": {"train": self.train_scene_count, "test": self.test_scene_count},
            "validation_state": self.validation_state,
            "content_hash": self.content_hash,
        }


def _build_evidence(
    config: GlobalConfig, runtime: BuildRuntime
) -> tuple[ExportEvidence, tuple[str, ...]]:
    blobs = parse_blob_list(config.azure.blob_list)
    if not blobs:
        raise BuildOperationalError("blob list contains no recordings")
    acquirer = runtime.acquirer_factory(config)
    decoder_factory = runtime.decoder_factory
    extractor_kwargs: dict[str, object] = {}
    if decoder_factory is not None:
        extractor_kwargs["decoder_factory"] = decoder_factory
    extractor = RecordingExtractor(
        camera_topic=config.topics.camera,
        gnss_topic=config.topics.gnss,
        target_fps=Fraction(str(config.downsampling.target_fps)),
        tolerance_ns=int(config.downsampling.tolerance_ms * 1_000_000),
        staging_root=config.paths.work_dir,
        **extractor_kwargs,  # type: ignore[arg-type]
    )

    def acquire_one(blob: str) -> AcquisitionResult | Exception:
        try:
            return acquirer.acquire(blob)
        except Exception as error:
            return error

    with ThreadPoolExecutor(max_workers=config.execution.workers) as executor:
        acquired_outcomes = tuple(executor.map(acquire_one, blobs))
    acquired: list[tuple[int, str, AcquisitionResult]] = []
    acquisition_errors: list[tuple[int, str, Exception]] = []
    for index, (blob, outcome) in enumerate(zip(blobs, acquired_outcomes, strict=True)):
        if isinstance(outcome, Exception):
            acquisition_errors.append((index, blob, outcome))
        else:
            acquired.append((index, blob, outcome))
    acquisition_by_path = {
        item.artifact_path.resolve(): item for _, _, item in acquired
    }
    extraction_cache = runtime.extraction_cache_factory(config.paths.cache_dir)

    def extract_cached(path: Path) -> RecordingExtractionResult:
        acquisition = acquisition_by_path[path.resolve()]
        source = acquisition.manifest.source
        if acquirer.extraction_cache_reusable(source, extraction_hash):
            cached = extraction_cache.load(source, extraction_hash, path)
            if cached is not None:
                return cached
        extracted = extractor.extract(path)
        cached = extraction_cache.store(
            source,
            extraction_hash,
            extracted,
            force_refresh=True,
        )
        acquirer.record_extraction_complete(source, extraction_hash)
        return cached

    coordinator = RecordingCoordinator(config=config, extractor=extract_cached)
    source_config_hash = canonical_hash(config.azure.model_dump(mode="json"))
    extraction_hash = extraction_config_hash(config)
    acquisition_failures: list[RecordingFailure] = []
    for index, blob, error in acquisition_errors:
        request = RecordingRequest(
            f"recording-{index:06d}",
            Path(blob),
            source_config_hash,
            extraction_hash,
        )
        acquisition_failures.append(
            coordinator.quarantine_failure(
                request,
                error,
                category=(
                    "structural" if isinstance(error, StructuralExtractionError) else "unexpected"
                ),
                stage="acquisition",
            )
        )
    requests = tuple(
        RecordingRequest(
            f"recording-{index:06d}",
            acquisition.artifact_path,
            canonical_hash(acquisition.manifest.source.to_dict()),
            extraction_hash,
        )
        for index, _, acquisition in acquired
    )
    try:
        coordinator_result = (
            coordinator.process(
                requests,
                allow_partial_export=True,
                max_workers=config.execution.workers,
            )
            if requests
            else None
        )
    except PublicationBlockedError as error:
        raise BuildOperationalError(str(error)) from error
    successes = () if coordinator_result is None else coordinator_result.successes
    processing_failures = () if coordinator_result is None else coordinator_result.failures
    request_by_id = {item.recording_id: item for item in requests}
    graphs = []
    success_by_source: dict[str, RecordingSuccess] = {}
    scene_failures: list[RecordingFailure] = []
    for success in successes:
        acquisition = acquisition_by_path[success.extraction.source_path.resolve()]
        try:
            graph = build_recording_scenes(
                success.validity, acquisition.manifest.source, config
            )
        except Exception as error:
            request = request_by_id[success.recording_id]
            scene_failures.append(
                coordinator.quarantine_failure(
                    request,
                    error,
                    category=(
                        "structural"
                        if isinstance(error, StructuralExtractionError)
                        else "unexpected"
                    ),
                    stage="scene_building",
                    extraction=success.extraction,
                    validity=success.validity,
                )
            )
            continue
        graphs.append(graph)
        success_by_source[graph.source.digest] = success

    features: list[SceneFeatures] = []
    feature_graphs = []
    feature_failures: list[RecordingFailure] = []
    for graph in graphs:
        success = success_by_source[graph.source.digest]
        try:
            result = compute_recording_features(graph, config.tags)
        except Exception as error:
            feature_failures.append(
                coordinator.quarantine_failure(
                    request_by_id[success.recording_id],
                    error,
                    category=(
                        "structural"
                        if isinstance(error, StructuralExtractionError)
                        else "unexpected"
                    ),
                    stage="feature_computation",
                    extraction=success.extraction,
                    validity=success.validity,
                )
            )
            continue
        feature_graphs.append(graph)
        features.extend(result.scenes)

    exportable_graphs = []
    preflight_failures: list[RecordingFailure] = []
    for graph in feature_graphs:
        success = success_by_source[graph.source.digest]
        try:
            preflight_recording_export(graph)
        except Exception as error:
            preflight_failures.append(
                coordinator.quarantine_failure(
                    request_by_id[success.recording_id],
                    error,
                    category=(
                        "structural"
                        if isinstance(error, StructuralExtractionError)
                        else "unexpected"
                    ),
                    stage="export_preflight",
                    extraction=success.extraction,
                    validity=success.validity,
                )
            )
            continue
        exportable_graphs.append(graph)
    graphs = exportable_graphs
    exportable_sources = {item.source.digest for item in graphs}
    features = [item for item in features if item.source.digest in exportable_sources]

    all_failures = tuple(
        sorted(
            (
                *acquisition_failures,
                *processing_failures,
                *scene_failures,
                *feature_failures,
                *preflight_failures,
            ),
            key=lambda item: item.recording_id,
        )
    )
    quarantine_complete = all(item.quarantine_persisted for item in all_failures)
    if all_failures and quarantine_complete:
        try:
            write_rejection_manifest(
                config.quarantine.directory,
                config.quarantine.manifest_name,
                tuple(
                    item.quarantine.report
                    for item in all_failures
                    if item.quarantine is not None
                ),
            )
        except Exception:
            quarantine_complete = False
    failures = tuple(
        blobs[int(item.recording_id.removeprefix("recording-"))]
        for item in all_failures
    )
    if failures and (not config.execution.allow_partial_export or not quarantine_complete):
        raise BuildOperationalError(
            f"publication blocked after {len(failures)} quarantined recording failure(s)"
        )
    if not graphs:
        raise BuildOperationalError("no successful recordings are available for publication")
    filtered = filter_scenes(features, config.filters)
    selection = select_scenarios(filtered.accepted, config.scenarios)
    if not selection.selected_scenes:
        raise BuildOperationalError("scenario selection produced an empty dataset")
    selected_sources = {item.source_digest for item in selection.assignments}
    selected_graphs = tuple(
        graph for graph in graphs if graph.source.digest in selected_sources
    )
    selected_validity = tuple(
        (graph.source, success_by_source[graph.source.digest].validity)
        for graph in selected_graphs
    )
    split = split_selected_scenes(
        selection,
        filtered.accepted,
        config.scenarios,
        selected_graphs,
        config.split,
    )
    evidence = ExportEvidence(
        filtered.accepted,
        config.scenarios,
        selection,
        selected_graphs,
        config.split,
        split,
        config,
        {"schema_version": 1, "state": "pending_finalization"},
        {
            "schema_version": 1,
            "blob_order": list(blobs),
            "failed_recordings": list(failures),
            "filter": {
                "accepted": [
                    {
                        "scene_token": item.scene_token,
                        "source_digest": item.source.digest,
                    }
                    for item in filtered.accepted
                ],
                "rejected": [
                    {
                        "scene_token": item.feature.scene_token,
                        "source_digest": item.feature.source.digest,
                        "reasons": [asdict(reason) for reason in item.reasons],
                    }
                    for item in filtered.rejected
                ],
            },
            "selection": {
                "candidate_fingerprint": selection.candidate_fingerprint,
                "config_fingerprint": selection.config_fingerprint,
                "rules_fingerprint": selection.rules_fingerprint,
                "assignments": [asdict(item) for item in selection.assignments],
                "rule_audits": [asdict(item) for item in selection.rule_audits],
                "unselected": [asdict(item) for item in selection.unselected],
            },
            "graph_scene_sequence": pipeline_graph_scene_sequence(
                tuple(graphs), selection
            ),
        },
        selected_validity,
    )
    return evidence, tuple(failures)


def build_dataset(config: GlobalConfig, *, runtime: BuildRuntime | None = None) -> BuildResult:
    """Run the native pipeline into sibling staging, validate, and atomically publish."""
    if config.publication.version != NUSCENES_VERSION or not config.publication.refuse_overwrite:
        raise BuildOperationalError(
            "v1 publication requires v1.0-trainval and refuse_overwrite=true"
        )
    selected_runtime = BuildRuntime() if runtime is None else runtime
    output = config.paths.output_dir
    final = output / config.publication.version
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing final dataset: {final}")
    evidence, failures = _build_evidence(config, selected_runtime)
    lease = StagingLease.create(output, f".{config.publication.version}.staging-")
    staging = lease.root
    try:
        exported = export_dataset(staging, evidence, lease=lease)
        report = finalize_dataset(
            staging,
            config.publication.version,
            official_smoke=selected_runtime.official_smoke,
            lease=lease,
        )
        if not report.succeeded or report.content_hash is None:
            raise BuildOperationalError("final validation failed")
        published = publish_staging(
            lease,
            final,
            expected_content_hash=report.content_hash,
        )
    except Exception:
        lease.cleanup()
        raise
    finally:
        lease.close()
    return BuildResult(
        published,
        config.publication.version,
        exported.scene_count,
        exported.sample_count,
        exported.sample_data_count,
        report.content_hash,
        bool(failures),
        failures,
    )


def validate_dataset(
    dataroot: Path,
    version: str,
    *,
    official_smoke: bool = True,
) -> ValidationReport:
    """Validate one published dataset and raise for any error."""
    return _validate_dataset(
        dataroot,
        version,
        official_smoke=official_smoke,
        verify_manifest=True,
        raise_on_error=True,
    )


def inspect_dataset(
    dataroot: Path,
    version: str,
    *,
    official_smoke: bool = True,
) -> InspectionSummary:
    """Return a deterministic read-only summary after strict validation."""
    report = validate_dataset(dataroot, version, official_smoke=official_smoke)
    dataset = Dataset(dataroot, version)
    sensors = dataset.table("sensor")
    validation = dataset.validation_report()
    manifest = json.loads((Path(dataroot) / "mz_extensions/content_manifest.json").read_text())
    try:
        state = validation["state"]
        content_hash = manifest["root_sha256"]
    except (KeyError, TypeError) as error:
        raise DatasetFormatError("validation or content manifest is malformed") from error
    if state != "succeeded" or not isinstance(content_hash, str):
        raise DatasetFormatError("dataset is not successfully finalized")
    return InspectionSummary(
        version,
        report.table_counts,
        len(dataset.table("scene")),
        len(dataset.table("sample")),
        len(dataset.table("sample_data")),
        len(dataset.table("sample_data")),
        tuple(sorted(str(item["channel"]) for item in sensors)),
        len(dataset.recordings()),
        len(dataset.scenes_in_split("train")),
        len(dataset.scenes_in_split("test")),
        state,
        content_hash,
    )
