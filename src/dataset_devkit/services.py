"""Public standalone build, validation, and inspection services."""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Protocol

from dataset_devkit.acquisition import AcquisitionResult, AzureBlobAcquirer
from dataset_devkit.blob_list import parse_blob_list
from dataset_devkit.config import GlobalConfig
from dataset_devkit.dataset import Dataset, DatasetFormatError
from dataset_devkit.export import NUSCENES_VERSION, ExportEvidence, export_dataset
from dataset_devkit.extraction.camera import HevcDecoder
from dataset_devkit.extraction.service import RecordingExtractor
from dataset_devkit.features import SceneFeatures, compute_recording_features
from dataset_devkit.filtering import filter_scenes
from dataset_devkit.publication import publish_staging
from dataset_devkit.quarantine import QuarantineReport, write_quarantine_report
from dataset_devkit.sanity import evaluate_sanity
from dataset_devkit.scenario_selection import select_scenarios
from dataset_devkit.scenes import build_recording_scenes
from dataset_devkit.split import split_selected_scenes
from dataset_devkit.validation import ValidationReport, finalize_dataset
from dataset_devkit.validation import validate_dataset as _validate_dataset


class BuildOperationalError(RuntimeError):
    """Raised for a safe, user-facing pipeline failure."""


class AcquirerProtocol(Protocol):
    def acquire(self, blob_path: str) -> AcquisitionResult: ...


@dataclass(frozen=True)
class BuildRuntime:
    """Injectable external-runtime boundary used by deterministic tests."""

    acquirer_factory: Callable[[GlobalConfig], AcquirerProtocol] = AzureBlobAcquirer.from_config
    decoder_factory: Callable[[], HevcDecoder] | None = None
    evidence_builder: Callable[
        [GlobalConfig], tuple[ExportEvidence, tuple[str, ...]]
    ] | None = None
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


def _quarantine_failure(config: GlobalConfig, blob: str, stage: str, error: Exception) -> bool:
    try:
        write_quarantine_report(
            config.quarantine.directory,
            QuarantineReport(
                recording_id=Path(blob).stem.replace(" ", "_") or "recording",
                source_path=blob,
                status="quarantined",
                category="unexpected",
                exception_type=type(error).__name__,
                exception_message=str(error),
                stage=stage,
                deterministic_details={"blob_path": blob},
                observed_context=(),
                source_config_hash=None,
                extraction_config_hash=None,
                artifact_handling="no_owned_artifacts",
            ),
        )
    except Exception:
        return False
    return True


def _cleanup_owned_staging(staging: Path, identity: tuple[int, int]) -> bool:
    """Delete only the invocation-owned staging entry through its pinned parent."""
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(staging.parent, os.O_RDONLY | os.O_DIRECTORY | nofollow)
    try:
        current = os.stat(staging.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            (current.st_dev, current.st_ino) != identity
            or not stat.S_ISDIR(current.st_mode)
        ):
            return False
        shutil.rmtree(staging.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
        return True
    except OSError:
        return False
    finally:
        os.close(parent_fd)


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
    graphs = []
    failures: list[str] = []
    quarantine_complete = True
    for blob in blobs:
        try:
            acquisition = acquirer.acquire(blob)
        except Exception as error:
            failures.append(blob)
            quarantine_complete &= _quarantine_failure(config, blob, "acquisition", error)
            continue
        try:
            extraction = extractor.extract(acquisition.artifact_path)
            from dataset_devkit.validity import evaluate_validity

            validity = evaluate_validity(extraction, config)
            evaluate_sanity(extraction, validity, config)
            graph = build_recording_scenes(validity, acquisition.manifest.source, config)
        except Exception as error:
            failures.append(blob)
            quarantine_complete &= _quarantine_failure(config, blob, "extraction", error)
            continue
        graphs.append(graph)
    if failures and (not config.execution.allow_partial_export or not quarantine_complete):
        raise BuildOperationalError(
            f"publication blocked after {len(failures)} quarantined recording failure(s)"
        )
    if not graphs:
        raise BuildOperationalError("no successful recordings are available for publication")
    features: list[SceneFeatures] = []
    for graph in graphs:
        features.extend(compute_recording_features(graph, config.tags).scenes)
    filtered = filter_scenes(features, config.filters)
    selection = select_scenarios(filtered.accepted, config.scenarios)
    if not selection.selected_scenes:
        raise BuildOperationalError("scenario selection produced an empty dataset")
    split = split_selected_scenes(
        selection,
        filtered.accepted,
        config.scenarios,
        tuple(graphs),
        config.split,
    )
    evidence = ExportEvidence(
        filtered.accepted,
        config.scenarios,
        selection,
        tuple(graphs),
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
                "assignments": [asdict(item) for item in selection.assignments],
                "rule_audits": [asdict(item) for item in selection.rule_audits],
                "unselected": [asdict(item) for item in selection.unselected],
            },
        },
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
    output.mkdir(parents=True, exist_ok=True)
    final = output / config.publication.version
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing final dataset: {final}")
    if selected_runtime.evidence_builder is None:
        evidence, failures = _build_evidence(config, selected_runtime)
    else:
        evidence, failures = selected_runtime.evidence_builder(config)
        if failures and not config.execution.allow_partial_export:
            raise BuildOperationalError(
                f"publication blocked after {len(failures)} recording failure(s)"
            )
    if evidence.pipeline_audit is None:
        raise BuildOperationalError("build evidence is missing the pipeline audit report")
    staging = Path(tempfile.mkdtemp(prefix=f".{config.publication.version}.staging-", dir=output))
    identity = staging.stat().st_dev, staging.stat().st_ino
    try:
        exported = export_dataset(staging, evidence)
        report = finalize_dataset(
            staging, config.publication.version, official_smoke=selected_runtime.official_smoke
        )
        if not report.succeeded or report.content_hash is None:
            raise BuildOperationalError("final validation failed")
        published = publish_staging(staging, final)
    except Exception:
        _cleanup_owned_staging(staging, identity)
        raise
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
