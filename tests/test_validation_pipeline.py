from __future__ import annotations

import hashlib
import json
import os
import threading
import tracemalloc
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import cast

import pytest

from conftest import FeatureFactory
from dataset_devkit import publication as publication_module
from dataset_devkit import validation as validation_module
from dataset_devkit.acquisition import AcquisitionResult
from dataset_devkit.config import (
    FiltersConfig,
    GlobalConfig,
    ScenarioRuleConfig,
    ScenariosConfig,
    TagsConfig,
)
from dataset_devkit.dataset import Dataset
from dataset_devkit.export import export_dataset, preflight_recording_export
from dataset_devkit.extraction.cache import (
    CacheStoreResult,
    ExtractionResultCache,
    MaterializedExtraction,
)
from dataset_devkit.extraction.camera import DecoderOutput
from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.models import RecordingExtractionResult
from dataset_devkit.extraction.service import OwnedRecordingExtraction, RecordingExtractor
from dataset_devkit.features import RecordingFeatureResult, compute_recording_features
from dataset_devkit.provenance import (
    AcquisitionManifest,
    ArtifactIdentity,
    IntegrityVerification,
    SourceFingerprint,
    extraction_config_hash,
)
from dataset_devkit.publication import (
    OwnedDirectoryAuthority,
    OwnedDirectoryCleanupError,
    StagingLease,
    publish_staging,
)
from dataset_devkit.scene_models import RecordingSceneResult
from dataset_devkit.services import (
    BuildOperationalError,
    BuildResult,
    BuildRuntime,
    _WorkingExtractionRegistry,
    build_dataset,
    inspect_dataset,
)
from dataset_devkit.validation import (
    DatasetValidationError,
    ValidationReport,
    build_content_manifest,
    finalize_dataset,
    validate_dataset,
)
from dataset_devkit.validity import evaluate_validity
from mcap_fixture import camera_message, encode_hevc_access_units, write_mcap
from test_export_dataset import _evidence
from test_extraction_service import DeterministicDecoder


class _FakeAcquirer:
    def __init__(self, artifacts: dict[str, Path | Exception]) -> None:
        self.artifacts = artifacts
        self.completed: set[tuple[str, str]] = set()
        self.lock = threading.Lock()

    def acquire(self, blob_path: str) -> AcquisitionResult:
        value = self.artifacts[blob_path]
        if isinstance(value, Exception):
            raise value
        content = value.read_bytes()
        source = SourceFingerprint(
            "https://example.blob.core.windows.net",
            "recordings",
            blob_path,
            f'"{hashlib.sha256(content).hexdigest()[:16]}"',
            len(content),
        )
        digest = hashlib.sha256(content).hexdigest()
        manifest = AcquisitionManifest(
            source,
            "downloaded",
            ArtifactIdentity(value.name, len(content), digest),
            IntegrityVerification("size_etag", True, None),
            "0" * 64,
        )
        return AcquisitionResult(
            value,
            value.with_suffix(".manifest.json"),
            value.with_suffix(".extraction.json"),
            manifest,
        )

    def extraction_cache_reusable(
        self, source: SourceFingerprint, expected_extraction_config_hash: str
    ) -> bool:
        with self.lock:
            return (source.digest, expected_extraction_config_hash) in self.completed

    def record_extraction_complete(
        self, source: SourceFingerprint, completed_extraction_config_hash: str
    ) -> Path:
        with self.lock:
            self.completed.add((source.digest, completed_extraction_config_hash))
        return Path("marker")


def _pipeline_config(
    base: GlobalConfig,
    tmp_path: Path,
    blobs: tuple[str, ...],
    *,
    partial: bool,
) -> GlobalConfig:
    blob_list = tmp_path / "mcap_blobs.txt"
    blob_list.write_text("".join(f"{blob}\n" for blob in blobs))
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text("")
    paths = base.paths.model_copy(
        update={
            "work_dir": tmp_path / "work",
            "cache_dir": tmp_path / "cache",
            "output_dir": tmp_path / "output",
        }
    )
    scenes = base.scenes.model_copy(
        update={
            "mode": "automatic",
            "min_duration_s": Decimal("0.1"),
            "max_duration_s": Decimal("10"),
            "min_samples": 2,
            "max_sample_gap_ms": Decimal("1000"),
            "skip_between_scenes_s": Decimal("0"),
        }
    )
    gnss = base.gnss.model_copy(
        update={
            "position_sigma_max_m": 10.0,
            "orientation_variance_max": 10.0,
            "sync_gap_max_ms": 2_000.0,
        }
    )
    return base.model_copy(
        update={
            "azure": base.azure.model_copy(update={"blob_list": blob_list}),
            "paths": paths,
            "annotations": base.annotations.model_copy(update={"path": annotations}),
            "scenes": scenes,
            "gnss": gnss,
            "filters": FiltersConfig(),
            "scenarios": ScenariosConfig(
                seed=7,
                strict_quotas=True,
                rules=[ScenarioRuleConfig(name="all", quota=1)],
            ),
            "execution": base.execution.model_copy(
                update={"workers": 2, "allow_partial_export": partial}
            ),
            "quarantine": base.quarantine.model_copy(
                update={"directory": tmp_path / "quarantine"}
            ),
        }
    )


def _export(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> Path:
    root = tmp_path / "dataset"
    export_dataset(root, _evidence(tmp_path, config_factory, feature_factory))
    return root


def test_finalize_validates_and_creates_deterministic_manifest(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    evidence = _evidence(tmp_path / "input", config_factory, feature_factory)
    first = tmp_path / "a"
    second = tmp_path / "b"
    export_dataset(first, evidence)
    export_dataset(second, evidence)

    first_report = finalize_dataset(first)
    second_report = finalize_dataset(second)

    assert first_report.succeeded is True
    assert first_report.content_hash == second_report.content_hash
    assert validate_dataset(first).succeeded is True
    manifest = json.loads((first / "mz_extensions/content_manifest.json").read_text())
    assert manifest["excluded_paths"] == ["mz_extensions/content_manifest.json"]
    assert manifest["root_sha256"] == first_report.content_hash
    assert not any("timestamp" in key for key in manifest)


@pytest.mark.parametrize(
    "mutation",
    (
        "empty_report",
        "schema_version",
        "top_level_keys",
        "report_keys",
        "checks",
        "table_counts",
        "findings",
        "resolved_config_hash",
    ),
)
def test_finalized_validation_rejects_forged_success_evidence(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
    mutation: str,
) -> None:
    root = _export(tmp_path, config_factory, feature_factory)
    finalize_dataset(root, official_smoke=False)
    validation_path = root / "mz_extensions/validation.json"
    forged = json.loads(validation_path.read_text(encoding="utf-8"))
    if mutation == "empty_report":
        forged["report"] = {}
    elif mutation == "schema_version":
        forged["schema_version"] = 2
    elif mutation == "top_level_keys":
        forged["extra"] = True
    elif mutation == "report_keys":
        forged["report"]["extra"] = True
    elif mutation == "checks":
        forged["report"]["checks"] = []
    elif mutation == "table_counts":
        forged["report"]["table_counts"] = {}
    elif mutation == "findings":
        forged["report"]["findings"] = [
            {
                "severity": "warning",
                "code": "forged",
                "location": "validation",
                "message": "not recomputed",
            }
        ]
    else:
        forged["report"]["resolved_config_sha256"] = "0" * 64
    validation_path.write_text(json.dumps(forged), encoding="utf-8")
    manifest_path = root / "mz_extensions/content_manifest.json"
    manifest_path.write_text(
        json.dumps(build_content_manifest(root)),
        encoding="utf-8",
    )

    report = validate_dataset(root, official_smoke=False)

    assert report.succeeded is False
    assert any(item.code == "validation_state" for item in report.findings)


def test_atomic_json_retries_short_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension_dir = tmp_path / "mz_extensions"
    extension_dir.mkdir()
    destination = extension_dir / "validation.json"
    destination.write_text("{}\n", encoding="utf-8")
    original_write = os.write
    writes = 0

    def short_write(descriptor: int, content: bytes) -> int:
        nonlocal writes
        writes += 1
        return original_write(descriptor, content[:3])

    monkeypatch.setattr(os, "write", short_write)

    validation_module._atomic_json(
        tmp_path,
        "mz_extensions/validation.json",
        {"schema_version": 1, "state": "succeeded"},
    )

    assert writes > 1
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "state": "succeeded",
    }


@pytest.mark.parametrize(
    ("table", "mutation", "code"),
    [
        ("sample", lambda rows: rows[0].update(next=rows[0]["token"]), "sample_chain"),
        ("sample_data", lambda rows: rows[0].update(sample_token="missing"), "foreign_key"),
        ("ego_pose", lambda rows: rows[0].update(rotation=[2.0, 0.0, 0.0, 0.0]), "quaternion"),
    ],
)
def test_validator_rejects_one_mutation_per_contract(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
    table: str,
    mutation: Callable[[list[dict[str, object]]], None],
    code: str,
) -> None:
    root = _export(tmp_path, config_factory, feature_factory)
    path = root / "v1.0-trainval" / f"{table}.json"
    rows = json.loads(path.read_text())
    mutation(rows)
    path.write_text(json.dumps(rows), encoding="utf-8")

    report = validate_dataset(root, official_smoke=False)

    assert report.succeeded is False
    assert any(item.code == code for item in report.findings)


def test_manifest_detects_tamper_extra_and_symlink(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    root = _export(tmp_path, config_factory, feature_factory)
    finalize_dataset(root, official_smoke=False)
    image = next((root / "samples").rglob("*.jpg"))
    image.write_bytes(image.read_bytes() + b"tamper")
    with pytest.raises(DatasetValidationError, match="manifest|GNSS evidence"):
        validate_dataset(root, official_smoke=False, raise_on_error=True)

    other = _export(tmp_path / "extra", config_factory, feature_factory)
    finalize_dataset(other, official_smoke=False)
    (other / "unexpected.txt").write_text("x")
    with pytest.raises(DatasetValidationError, match="manifest"):
        validate_dataset(other, official_smoke=False, raise_on_error=True)

    linked = _export(tmp_path / "link", config_factory, feature_factory)
    (linked / "unsafe").symlink_to(linked / "v1.0-trainval")
    with pytest.raises(DatasetValidationError, match="symlink"):
        validate_dataset(linked, official_smoke=False, raise_on_error=True)


@pytest.mark.parametrize(
    ("relative", "mutate", "code"),
    [
        (
            "v1.0-trainval/sample.json",
            lambda value: value.append(dict(value[0])),
            "token_unique",
        ),
        (
            "v1.0-trainval/scene.json",
            lambda value: value[0].update(nbr_samples=999),
            "scene_endpoint",
        ),
        (
            "v1.0-trainval/sample.json",
            lambda value: [item.update(timestamp=0) for item in value],
            "timestamp",
        ),
        (
            "v1.0-trainval/sample_data.json",
            lambda value: value[0].update(next="missing"),
            "sample_data_chain",
        ),
        (
            "v1.0-trainval/sample_data.json",
            lambda value: value[0].update(width=999),
            "image",
        ),
        (
            "v1.0-trainval/calibrated_sensor.json",
            lambda value: value[0].update(translation=[float("inf"), 0.0, 0.0]),
            "finite_pose",
        ),
        (
            "v1.0-trainval/map.json",
            lambda value: value[0].update(log_tokens=["missing"]),
            "foreign_key",
        ),
        (
            "mz_extensions/tags.json",
            lambda value: value.pop(),
            "extension_reference",
        ),
        (
            "mz_extensions/tags.json",
            lambda value: value.append("malformed"),
            "extension_reference",
        ),
        (
            "mz_extensions/tags.json",
            lambda value: value[0].update(computed_tags=["duplicate", "duplicate"]),
            "extension_value",
        ),
        (
            "mz_extensions/tags.json",
            lambda value: value[0].update(source_digest="0" * 64),
            "extension_reference",
        ),
        (
            "mz_extensions/validity.json",
            lambda value: value["scenes"].append(dict(value["scenes"][0])),
            "extension_reference",
        ),
        (
            "mz_extensions/validity.json",
            lambda value: value["scenes"][0].update(scene_valid_ratio=2.0),
            "extension_value",
        ),
        (
            "mz_extensions/validity.json",
            lambda value: value["scenes"][0].update(source_digest="0" * 64),
            "extension_reference",
        ),
        (
            "mz_extensions/validity.json",
            lambda value: value["scenes"][0]["samples"].append(
                dict(value["scenes"][0]["samples"][0])
            ),
            "extension_value",
        ),
        (
            "mz_extensions/validity.json",
            lambda value: value["scenes"][0]["sample_data"][0].update(
                sample_data_token="foreign"
            ),
            "extension_value",
        ),
        (
            "mz_extensions/validity.json",
            lambda value: value["recordings"][0].update(policy="unknown"),
            "extension_value",
        ),
        (
            "mz_extensions/validity.json",
            lambda value: value["recordings"][0]["sample_audits"][0].update(
                final_candidate=False
            ),
            "extension_value",
        ),
        (
            "mz_extensions/gnss.json",
            lambda value: value.append(dict(value[0])),
            "extension_reference",
        ),
        (
            "mz_extensions/gnss.json",
            lambda value: value[0].update(scene_token="foreign"),
            "extension_reference",
        ),
        (
            "mz_extensions/gnss.json",
            lambda value: value[0].update(normalized_channel="CAM_FOREIGN"),
            "extension_value",
        ),
        (
            "mz_extensions/gnss.json",
            lambda value: value[0].update(image_sha256="0" * 64),
            "extension_value",
        ),
        (
            "mz_extensions/gnss.json",
            lambda value: value[0].update(position_uncertainty={"sigma": float("nan")}),
            "extension_value",
        ),
        (
            "mz_extensions/recordings.json",
            lambda value: value[0].update(unexpected=True),
            "extension_reference",
        ),
        (
            "mz_extensions/annotations.json",
            lambda value: value["scenes"].append(dict(value["scenes"][0])),
            "extension_reference",
        ),
        (
            "mz_extensions/split.json",
            lambda value: value["assignments"].append(dict(value["assignments"][0])),
            "split_integrity",
        ),
        (
            "mz_extensions/split.json",
            lambda value: value.update(seed=value["seed"] + 1),
            "split_integrity",
        ),
        (
            "mz_extensions/split.json",
            lambda value: value["assignments"][0].update(rank="0" * 64),
            "split_integrity",
        ),
        (
            "mz_extensions/split.json",
            lambda value: value["strata"][0].update(population_count=999),
            "split_integrity",
        ),
        (
            "mz_extensions/split.json",
            lambda value: value.update(candidate_fingerprint="0" * 64),
            "split_integrity",
        ),
        (
            "mz_extensions/split.json",
            lambda value: value["adjacent_scene_leakage"].update(warning="changed"),
            "split_integrity",
        ),
        (
            "v1.0-trainval/sample_data.json",
            lambda value: value[0].update(filename="../escape.jpg"),
            "filename",
        ),
    ],
)
def test_validator_contract_mutation_matrix(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
    relative: str,
    mutate: Callable[[object], None],
    code: str,
) -> None:
    root = _export(tmp_path, config_factory, feature_factory)
    path = root / relative
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value, allow_nan=True), encoding="utf-8")

    report = validate_dataset(
        root,
        official_smoke=False,
        verify_manifest=False,
    )

    assert report.succeeded is False
    assert any(item.code == code for item in report.findings)


def test_validator_rejects_missing_manifest_and_incomplete_required_camera(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    root = _export(tmp_path, config_factory, feature_factory)
    finalize_dataset(root, official_smoke=False)
    (root / "mz_extensions/content_manifest.json").unlink()
    report = validate_dataset(root, official_smoke=False)
    assert report.succeeded is False
    assert any(item.code in {"json", "manifest"} for item in report.findings)

    coverage = _export(tmp_path / "coverage", config_factory, feature_factory)
    config_path = coverage / "mz_extensions/config.json"
    config = json.loads(config_path.read_text())
    config["frame_validity"]["required_cameras"].append("side")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    report = validate_dataset(
        coverage,
        official_smoke=False,
        verify_manifest=False,
    )
    assert any(item.code == "camera_coverage" for item in report.findings)


def test_inspect_and_atomic_publication_refuses_overwrite(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    staging = _export(tmp_path / "staging", config_factory, feature_factory)
    finalize_dataset(staging, official_smoke=False)
    final = staging.parent / "v1.0-trainval"

    staging_stat = staging.stat()
    published = publish_staging(
        staging,
        final,
        expected_identity=(staging_stat.st_dev, staging_stat.st_ino),
    )
    summary = inspect_dataset(published, "v1.0-trainval", official_smoke=False)

    assert summary.validation_state == "succeeded"
    assert summary.scene_count > 0
    assert summary.content_hash
    with pytest.raises(FileExistsError, match="overwrite"):
        publish_staging(tmp_path / "other", final)


def test_complete_stage_injected_build_is_repeatable_and_cli_loadable(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = tmp_path / "recording.mcap"
    write_mcap(
        recording,
        camera_payloads=tuple(
            camera_message(
                timestamp,
                (timestamp + 10, timestamp + 20),
                camera_names=("front", "rear"),
            )
            for timestamp in (1_000_000_000, 1_500_000_000, 2_000_000_000)
        ),
    )
    blob = "mcap-h265/recording.mcap"
    config = _pipeline_config(config_factory(), tmp_path, (blob,), partial=False)
    acquirer = _FakeAcquirer({blob: recording})
    decoder_creations = 0

    def decoder_factory() -> DeterministicDecoder:
        nonlocal decoder_creations
        decoder_creations += 1
        return DeterministicDecoder()

    runtime = BuildRuntime(
        acquirer_factory=lambda _config: acquirer,
        decoder_factory=decoder_factory,
        official_smoke=True,
    )

    first = build_dataset(config, runtime=runtime)
    assert not config.paths.work_dir.exists() or not tuple(config.paths.work_dir.iterdir())
    first_hash = first.content_hash
    first_decoder_creations = decoder_creations
    assert first_decoder_creations > 0
    assert Dataset(first.dataroot).validation_report()["succeeded"] is True
    source = acquirer.acquire(blob).manifest.source
    extraction_hash = extraction_config_hash(config)
    extraction_cache = ExtractionResultCache(config.paths.cache_dir)
    assert extraction_cache.contains(source, extraction_hash)
    first_cached_path = (
        extraction_cache.path_for(source, extraction_hash) / "images" / "00000000.jpg"
    )
    first_cache_identity = first_cached_path.stat().st_ino
    # A sealed dataroot is read/execute-only; same-parent rename avoids changing
    # its POSIX `..` entry while preserving the first result for comparison.
    archived = first.dataroot.with_name("first-published")
    first.dataroot.rename(archived)
    second = build_dataset(config, runtime=runtime)
    assert not tuple(config.paths.work_dir.iterdir())

    assert second.content_hash == first_hash
    assert decoder_creations == first_decoder_creations
    assert acquirer.extraction_cache_reusable(
        source,
        extraction_hash,
    )
    assert extraction_cache.contains(source, extraction_hash)
    assert first_cached_path.stat().st_ino == first_cache_identity

    second.dataroot.rename(second.dataroot.with_name("second-published"))
    acquirer.completed.clear()
    third = build_dataset(config, runtime=runtime)
    assert not tuple(config.paths.work_dir.iterdir())
    assert decoder_creations > first_decoder_creations
    assert extraction_cache.contains(source, extraction_hash)
    assert first_cached_path.stat().st_ino != first_cache_identity
    from dataset_devkit.cli import main

    assert main(
        [
            "validate",
            "--dataroot",
            str(third.dataroot),
            "--version",
            "v1.0-trainval",
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "succeeded"
    assert main(
        [
            "inspect",
            "--dataroot",
            str(third.dataroot),
            "--version",
            "v1.0-trainval",
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["content_hash"] == third.content_hash

    monkeypatch.setattr("dataset_devkit.cli.build_dataset", lambda _config: third)
    assert main(["build", "--config", "examples/dataset_config.json"]) == 0
    assert json.loads(capsys.readouterr().out)["dataroot"] == str(third.dataroot)


def test_true_pyav_multicamera_build_is_officially_loadable_and_repeatable(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
) -> None:
    from nuscenes.nuscenes import NuScenes  # type: ignore[import-untyped]
    from PIL import Image

    frame_count = 10
    front = encode_hevc_access_units(
        tuple((20 + index * 10, 30, 40) for index in range(frame_count)),
        b_frames=3,
    )
    rear = encode_hevc_access_units(
        tuple((40, 30, 20 + index * 10) for index in range(frame_count)),
        b_frames=3,
    )
    recording = tmp_path / "true-pyav.mcap"
    timestamps = tuple(1_000_000_000 + index * 100_000_000 for index in range(frame_count))
    write_mcap(
        recording,
        camera_payloads=tuple(
            camera_message(
                timestamp,
                (timestamp + 10, timestamp + 20),
                payloads=(front[index], rear[index]),
                dimensions=(32, 32),
                camera_names=("front", "rear"),
            )
            for index, timestamp in enumerate(timestamps)
        ),
    )
    blob = "mcap-h265/true-pyav.mcap"
    config = _pipeline_config(config_factory(), tmp_path, (blob,), partial=False)
    acquirer = _FakeAcquirer({blob: recording})

    def build_once() -> BuildResult:
        return build_dataset(
            config,
            runtime=BuildRuntime(
                acquirer_factory=lambda _config: acquirer,
                official_smoke=True,
            ),
        )

    first = build_once()
    first_root = first.dataroot.with_name("true-pyav-first")
    first.dataroot.rename(first_root)
    first_tables = {
        path.name: path.read_bytes()
        for path in sorted((first_root / first.version).glob("*.json"))
    }

    second = build_once()
    second_tables = {
        path.name: path.read_bytes()
        for path in sorted((second.dataroot / second.version).glob("*.json"))
    }

    assert first.content_hash == second.content_hash
    assert first_tables == second_tables
    assert first.sample_count == second.sample_count == 3
    assert first.sample_data_count == second.sample_data_count == 6

    dataset = NuScenes(
        version=second.version,
        dataroot=str(second.dataroot),
        verbose=False,
    )
    sample = dataset.get("sample", dataset.scene[0]["first_sample_token"])
    assert set(sample["data"]) == {"CAM_FRONT", "CAM_REAR"}
    for channel in ("CAM_FRONT", "CAM_REAR"):
        sample_data = dataset.get("sample_data", sample["data"][channel])
        with Image.open(second.dataroot / sample_data["filename"]) as image:
            image.load()
            assert image.size == (32, 32)


def test_stage_injected_partial_failure_requires_persisted_quarantine(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
) -> None:
    recording = tmp_path / "recording.mcap"
    write_mcap(
        recording,
        camera_payloads=tuple(
            camera_message(
                timestamp,
                (timestamp + 10, timestamp + 20),
                camera_names=("front", "rear"),
            )
            for timestamp in (1_000_000_000, 1_500_000_000, 2_000_000_000)
        ),
    )
    good = "mcap-h265/good.mcap"
    bad = "mcap-h265/bad.mcap"
    config = _pipeline_config(config_factory(), tmp_path, (good, bad), partial=False)
    acquirer = _FakeAcquirer({good: recording, bad: RuntimeError("transport failed")})
    runtime = BuildRuntime(
        acquirer_factory=lambda _config: acquirer,
        decoder_factory=DeterministicDecoder,
        official_smoke=False,
    )

    with pytest.raises(BuildOperationalError, match="blocked"):
        build_dataset(config, runtime=runtime)

    partial_execution = config.execution.model_copy(update={"allow_partial_export": True})
    partial_config = config.model_copy(update={"execution": partial_execution})
    result = build_dataset(partial_config, runtime=runtime)
    assert result.partial is True
    assert result.failed_recordings == (bad,)
    manifest = partial_config.quarantine.directory / partial_config.quarantine.manifest_name
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["stage"] == "acquisition"
    assert rows[0]["category"] == "unexpected"
    assert rows[0]["source_config_hash"]
    assert rows[0]["extraction_config_hash"] == extraction_config_hash(config)
    assert rows[0]["artifact_handling"] == "no_owned_artifacts"


def _two_pipeline_recordings(tmp_path: Path) -> tuple[Path, Path]:
    good = tmp_path / "good.mcap"
    bad = tmp_path / "bad.mcap"
    payloads = tuple(
        camera_message(
            timestamp,
            (timestamp + 10, timestamp + 20),
            camera_names=("front", "rear"),
        )
        for timestamp in (1_000_000_000, 1_500_000_000, 2_000_000_000)
    )
    write_mcap(good, camera_payloads=payloads)
    write_mcap(bad, camera_payloads=payloads)
    return good, bad


def test_feature_failure_blocks_default_and_partial_export_keeps_good_source(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good_path, bad_path = _two_pipeline_recordings(tmp_path)
    good_blob = "mcap-h265/good.mcap"
    bad_blob = "mcap-h265/bad.mcap"
    config = _pipeline_config(
        config_factory(), tmp_path, (good_blob, bad_blob), partial=False
    )
    runtime = BuildRuntime(
        acquirer_factory=lambda _config: _FakeAcquirer(
            {good_blob: good_path, bad_blob: bad_path}
        ),
        decoder_factory=DeterministicDecoder,
        official_smoke=False,
    )
    original_compute = compute_recording_features

    def fail_bad_feature(
        graph: RecordingSceneResult, tags: TagsConfig
    ) -> RecordingFeatureResult:
        if graph.source.blob_path == bad_blob:
            raise StructuralExtractionError("injected per-source feature failure")
        return original_compute(graph, tags)

    monkeypatch.setattr(
        "dataset_devkit.services.compute_recording_features", fail_bad_feature
    )
    with pytest.raises(BuildOperationalError, match="blocked"):
        build_dataset(config, runtime=runtime)

    partial_config = config.model_copy(
        update={
            "execution": config.execution.model_copy(
                update={"allow_partial_export": True}
            )
        }
    )
    result = build_dataset(partial_config, runtime=runtime)
    assert result.partial is True
    assert result.failed_recordings == (bad_blob,)
    rows = [
        json.loads(line)
        for line in (
            partial_config.quarantine.directory
            / partial_config.quarantine.manifest_name
        ).read_text().splitlines()
    ]
    assert {row["stage"] for row in rows} == {"feature_computation"}
    assert all(row["category"] == "structural" for row in rows)


@pytest.mark.parametrize("failure_kind", ["image", "calibration"])
def test_export_preflight_failure_blocks_default_and_partial_keeps_good_source(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    good_path, bad_path = _two_pipeline_recordings(tmp_path)
    good_blob = "mcap-h265/good.mcap"
    bad_blob = "mcap-h265/bad.mcap"
    config = _pipeline_config(
        config_factory(), tmp_path, (good_blob, bad_blob), partial=False
    )
    runtime = BuildRuntime(
        acquirer_factory=lambda _config: _FakeAcquirer(
            {good_blob: good_path, bad_blob: bad_path}
        ),
        decoder_factory=DeterministicDecoder,
        official_smoke=False,
    )

    def fail_bad_preflight(graph: RecordingSceneResult) -> None:
        if graph.source.blob_path != bad_blob:
            preflight_recording_export(graph)
            return
        if failure_kind == "image":
            graph.sample_data[0].staged_image.path.write_bytes(b"not-jpeg")
            preflight_recording_export(graph)
        else:
            data = graph.sample_data
            damaged = replace(data[0], calibration=None)
            preflight_recording_export(
                replace(graph, sample_data=(damaged, *data[1:]))
            )

    monkeypatch.setattr(
        "dataset_devkit.services.preflight_recording_export", fail_bad_preflight
    )
    with pytest.raises(BuildOperationalError, match="blocked"):
        build_dataset(config, runtime=runtime)

    partial_config = config.model_copy(
        update={
            "execution": config.execution.model_copy(
                update={"allow_partial_export": True}
            )
        }
    )
    result = build_dataset(partial_config, runtime=runtime)
    assert result.partial is True
    assert result.failed_recordings == (bad_blob,)
    rows = [
        json.loads(line)
        for line in (
            partial_config.quarantine.directory
            / partial_config.quarantine.manifest_name
        ).read_text().splitlines()
    ]
    assert {row["stage"] for row in rows} == {"export_preflight"}
    assert all(row["category"] == "structural" for row in rows)


def test_build_never_publishes_or_deletes_a_replacement_staging_entry(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = tmp_path / "recording.mcap"
    write_mcap(
        recording,
        camera_payloads=tuple(
            camera_message(
                timestamp,
                (timestamp + 10, timestamp + 20),
                camera_names=("front", "rear"),
            )
            for timestamp in (1_000_000_000, 1_500_000_000, 2_000_000_000)
        ),
    )
    blob = "mcap-h265/recording.mcap"
    config = _pipeline_config(config_factory(), tmp_path, (blob,), partial=False)
    runtime = BuildRuntime(
        acquirer_factory=lambda _config: _FakeAcquirer({blob: recording}),
        decoder_factory=DeterministicDecoder,
        official_smoke=False,
    )
    replacement: Path | None = None
    sentinel: Path | None = None

    def displace_after_validation(
        dataroot: str | Path,
        version: str = "v1.0-trainval",
        *,
        official_smoke: bool = True,
        lease: StagingLease | None = None,
    ) -> ValidationReport:
        nonlocal replacement, sentinel
        report = finalize_dataset(
            dataroot,
            version,
            official_smoke=official_smoke,
            lease=lease,
        )
        staging = Path(dataroot)
        staging.rename(staging.with_name(f"{staging.name}.displaced"))
        staging.mkdir()
        replacement = staging
        sentinel = staging / "unrelated"
        sentinel.write_text("survives", encoding="utf-8")
        return report

    monkeypatch.setattr(
        "dataset_devkit.services.finalize_dataset", displace_after_validation
    )

    with pytest.raises(ValueError, match="entry no longer names"):
        build_dataset(config, runtime=runtime)

    assert replacement is not None and replacement.is_dir()
    assert sentinel is not None and sentinel.read_text(encoding="utf-8") == "survives"
    assert not (config.paths.output_dir / "v1.0-trainval").exists()


def test_exclusive_publish_race_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "final"
    original = publication_module._rename_exclusive

    def race(parent_fd: int, source_name: str, destination_name: str) -> None:
        destination.mkdir()
        original(parent_fd, source_name, destination_name)

    monkeypatch.setattr(publication_module, "_rename_exclusive", race)
    source_stat = source.stat()
    with pytest.raises(FileExistsError, match="overwrite"):
        publish_staging(
            source,
            destination,
            expected_identity=(source_stat.st_dev, source_stat.st_ino),
        )
    assert source.is_dir()
    assert destination.is_dir()


def test_extraction_result_cache_is_materialized_and_tamper_evident(
    tmp_path: Path,
) -> None:
    recording = tmp_path / "recording.mcap"
    write_mcap(
        recording,
        camera_payloads=(
            camera_message(1_000_000_000, (1_000_000_010, 1_000_000_020)),
            camera_message(1_500_000_000, (1_500_000_010, 1_500_000_020)),
        ),
    )
    extracted = RecordingExtractor(
        camera_topic="rec_cameras",
        gnss_topic="gnss",
        target_fps=2,
        tolerance_ns=0,
        staging_root=tmp_path / "staging",
        decoder_factory=DeterministicDecoder,
    ).extract(recording)
    source = SourceFingerprint(
        "https://example.blob.core.windows.net",
        "recordings",
        "mcap-h265/recording.mcap",
        '"etag"',
        recording.stat().st_size,
    )
    config_hash = "a" * 64
    cache = ExtractionResultCache(tmp_path / "cache")

    stored = cache.store(source, config_hash, extracted)
    materialized = cache.materialize(
        source,
        config_hash,
        recording,
        tmp_path / "working",
        "recording-000000",
    )

    assert isinstance(stored, CacheStoreResult)
    assert not isinstance(stored, RecordingExtractionResult)
    assert not hasattr(cache, "load")
    assert cache.contains(source, config_hash)
    assert materialized is not None
    assert materialized.camera_batches == extracted.camera_batches
    assert materialized.selected_grid == extracted.selected_grid
    assert len(materialized.samples) == len(extracted.samples)
    stored_path = cache.path_for(source, config_hash) / "images" / "00000000.jpg"
    assert stored_path.is_file()
    assert materialized.staging_root != stored.path
    assert materialized.samples[0].staged_image.inode != stored_path.stat().st_ino
    assert materialized.samples[0].staged_image.sha256 == hashlib.sha256(
        stored_path.read_bytes()
    ).hexdigest()
    assert not cache.contains(source, "b" * 64)
    changed_source = replace(source, etag='"changed"')
    assert not cache.contains(changed_source, config_hash)

    assert stored_path.stat().st_mode & 0o222 == 0
    stored_path.chmod(0o600)
    stored_path.write_bytes(b"corrupt")
    assert not cache.contains(source, config_hash)


def _cache_security_case(
    tmp_path: Path,
) -> tuple[Path, RecordingExtractionResult, SourceFingerprint]:
    recording = tmp_path / "security-recording.mcap"
    write_mcap(
        recording,
        camera_payloads=(
            camera_message(1_000_000_000, (1_000_000_010, 1_000_000_020)),
        ),
    )
    extracted = RecordingExtractor(
        camera_topic="rec_cameras",
        gnss_topic="gnss",
        target_fps=1,
        tolerance_ns=0,
        staging_root=tmp_path / "security-staging",
        decoder_factory=DeterministicDecoder,
    ).extract(recording)
    source = SourceFingerprint(
        "https://example.blob.core.windows.net",
        "recordings",
        "mcap-h265/security-recording.mcap",
        '"etag"',
        recording.stat().st_size,
    )
    return recording, extracted, source


def test_cache_hit_drop_uses_independent_working_trees_and_preserves_cache(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
) -> None:
    recording, extracted, source = _cache_security_case(tmp_path)
    cache = ExtractionResultCache(tmp_path / "cache")
    config_hash = "a" * 64
    cache.store(source, config_hash, extracted)
    first = cache.materialize(
        source, config_hash, recording, tmp_path / "work", "recording-000000"
    )
    second = cache.materialize(
        source, config_hash, recording, tmp_path / "work", "recording-000000"
    )
    assert first is not None
    assert second is not None
    assert first.staging_root != second.staging_root
    assert first.samples[0].staged_image.inode != second.samples[0].staged_image.inode
    base = config_factory()
    config = base.model_copy(
        update={
            "frame_validity": base.frame_validity.model_copy(
                update={"invalid_sample_policy": "drop"}
            )
        }
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        reports = tuple(executor.map(lambda item: evaluate_validity(item, config), (first, second)))

    assert all(report.observations for report in reports)
    assert cache.contains(source, config_hash)
    third = cache.materialize(
        source, config_hash, recording, tmp_path / "work", "recording-000000"
    )
    assert third is not None
    assert all(sample.staged_image.path.is_file() for sample in third.samples)


def test_cache_materialization_memory_is_bounded_by_one_image(tmp_path: Path) -> None:
    recording, extracted, source = _cache_security_case(tmp_path)
    payload = b"x" * (2 * 1024 * 1024)
    expanded_samples = []
    for sample in extracted.samples:
        sample.staged_image.path.write_bytes(payload)
        current = sample.staged_image.path.stat()
        expanded_samples.append(
            replace(
                sample,
                staged_image=replace(
                    sample.staged_image,
                    device=current.st_dev,
                    inode=current.st_ino,
                    size=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                ),
            )
        )
    expanded = replace(extracted, samples=tuple(expanded_samples) * 8)
    cache = ExtractionResultCache(tmp_path / "cache")
    config_hash = "a" * 64
    cache.store(source, config_hash, expanded)

    tracemalloc.start()
    try:
        materialized = cache.materialize(
            source,
            config_hash,
            recording,
            tmp_path / "working",
            "recording-000000",
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert materialized is not None
    assert len(materialized.samples) == len(expanded.samples)
    assert peak < 6 * 1024 * 1024


def test_cache_materialization_uses_transactionally_preestablished_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording, extracted, source = _cache_security_case(tmp_path)
    cache = ExtractionResultCache(tmp_path / "cache")
    config_hash = "a" * 64
    cache.store(source, config_hash, extracted)

    def reject_late_capture(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("late authority capture is forbidden")

    monkeypatch.setattr(OwnedDirectoryAuthority, "capture", reject_late_capture)
    owned = cache._materialize_owned(
        source,
        config_hash,
        recording,
        tmp_path / "working",
        "recording-000000",
    )

    assert owned is not None
    assert owned.authority.root == owned.result.staging_root
    assert owned.authority.is_bound()


def test_cache_materialization_rolls_back_a_partial_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording, extracted, source = _cache_security_case(tmp_path)
    cache = ExtractionResultCache(tmp_path / "cache")
    config_hash = "a" * 64
    cache.store(source, config_hash, extracted)
    from dataset_devkit.extraction import cache as cache_module

    original_copy = cache_module._copy_verified_image_at
    copied = 0

    def fail_second_copy(*args: object, **kwargs: object) -> os.stat_result:
        nonlocal copied
        copied += 1
        if copied == 2:
            raise ValueError("injected streamed copy failure")
        return original_copy(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cache_module, "_copy_verified_image_at", fail_second_copy)
    working = tmp_path / "working"

    assert (
        cache.materialize(
            source,
            config_hash,
            recording,
            working,
            "recording-000000",
        )
        is None
    )
    assert not working.exists() or not tuple(working.iterdir())


def test_cache_materialization_rollback_failure_preserves_structured_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording, extracted, source = _cache_security_case(tmp_path)
    cache = ExtractionResultCache(tmp_path / "cache")
    config_hash = "a" * 64
    cache.store(source, config_hash, extracted)
    from dataset_devkit.extraction import cache as cache_module

    original_copy = cache_module._copy_verified_image_at
    copied = 0

    def fail_second_copy(*args: object, **kwargs: object) -> os.stat_result:
        nonlocal copied
        copied += 1
        if copied == 2:
            raise ValueError("injected materialization failure")
        return original_copy(*args, **kwargs)  # type: ignore[arg-type]

    def fail_rollback(_invocation: object) -> None:
        raise StructuralExtractionError("injected rollback failure")

    monkeypatch.setattr(cache_module, "_copy_verified_image_at", fail_second_copy)
    monkeypatch.setattr(cache_module, "rollback_staging_invocation", fail_rollback)
    working = tmp_path / "working"

    with pytest.raises(OwnedDirectoryCleanupError) as captured:
        cache.materialize(
            source,
            config_hash,
            recording,
            working,
            "recording-000000",
        )

    assert isinstance(captured.value.__cause__, ValueError)
    assert str(captured.value.__cause__) == "injected materialization failure"
    failure = captured.value.failures[0]
    assert failure.path.parent == working
    assert failure.expected_inode > 0
    assert failure.expected_parent_chain
    assert failure.path.is_dir()


@pytest.mark.parametrize("failure_point", ["store", "marker"])
def test_failed_cache_completion_cleans_owned_working_extraction(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    recording = tmp_path / "recording.mcap"
    write_mcap(
        recording,
        camera_payloads=(
            camera_message(1_000_000_000, (1_000_000_010, 1_000_000_020)),
        ),
    )
    blob = "mcap-h265/recording.mcap"
    config = _pipeline_config(config_factory(), tmp_path, (blob,), partial=False)
    acquirer = _FakeAcquirer({blob: recording})
    cache = ExtractionResultCache(config.paths.cache_dir)

    def fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError(f"injected {failure_point} failure")

    if failure_point == "store":
        monkeypatch.setattr(cache, "store", fail)
    else:
        monkeypatch.setattr(acquirer, "record_extraction_complete", fail)
    runtime = BuildRuntime(
        acquirer_factory=lambda _config: acquirer,
        decoder_factory=DeterministicDecoder,
        extraction_cache_factory=lambda _path: cache,
        official_smoke=False,
    )

    with pytest.raises(BuildOperationalError, match="publication blocked"):
        build_dataset(config, runtime=runtime)

    assert not config.paths.work_dir.exists() or not tuple(config.paths.work_dir.iterdir())


@pytest.mark.parametrize("failure_point", ["store", "marker"])
def test_failed_cache_completion_reports_uncleaned_owned_working_tree(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    recording = tmp_path / "recording.mcap"
    write_mcap(
        recording,
        camera_payloads=(
            camera_message(1_000_000_000, (1_000_000_010, 1_000_000_020)),
        ),
    )
    blob = "mcap-h265/recording.mcap"
    config = _pipeline_config(config_factory(), tmp_path, (blob,), partial=False)
    acquirer = _FakeAcquirer({blob: recording})
    cache = ExtractionResultCache(config.paths.cache_dir)

    def fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError(f"injected {failure_point} failure")

    if failure_point == "store":
        monkeypatch.setattr(cache, "store", fail)
    else:
        monkeypatch.setattr(acquirer, "record_extraction_complete", fail)
    monkeypatch.setattr(OwnedDirectoryAuthority, "cleanup", lambda _self: False)
    runtime = BuildRuntime(
        acquirer_factory=lambda _config: acquirer,
        decoder_factory=DeterministicDecoder,
        extraction_cache_factory=lambda _path: cache,
        official_smoke=False,
    )

    with pytest.raises(OwnedDirectoryCleanupError) as captured:
        build_dataset(config, runtime=runtime)

    assert isinstance(captured.value.__cause__, BuildOperationalError)
    reports = tuple(config.quarantine.directory.glob("*.quarantine.json"))
    row = json.loads(reports[0].read_text(encoding="utf-8"))
    assert row["exception_type"] == "OwnedDirectoryCleanupError"
    assert row["artifact_handling"] == "preserved_in_place"
    tree = row["deterministic_details"]["owned_working_trees"][0]
    assert tree["path"].startswith(str(config.paths.work_dir))
    assert tree["expected_device"] >= 0
    assert tree["expected_inode"] > 0
    original = row["deterministic_details"]["cleanup_original_cause"]
    assert original == {
        "exception_type": "RuntimeError",
        "exception_message": f"injected {failure_point} failure",
    }
    assert tuple(config.paths.work_dir.iterdir())


def test_partial_pipeline_never_publishes_after_owned_cleanup_failure(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
) -> None:
    good_recording = tmp_path / "good.mcap"
    bad_recording = tmp_path / "bad.mcap"
    payloads = tuple(
        camera_message(
            timestamp,
            (timestamp + 10, timestamp + 20),
            camera_names=("front", "rear"),
        )
        for timestamp in (1_000_000_000, 1_500_000_000, 2_000_000_000)
    )
    write_mcap(good_recording, camera_payloads=payloads)
    write_mcap(bad_recording, camera_payloads=payloads)
    good_blob = "mcap-h265/good.mcap"
    bad_blob = "mcap-h265/bad.mcap"
    config = _pipeline_config(
        config_factory(), tmp_path, (good_blob, bad_blob), partial=True
    )
    acquirer = _FakeAcquirer(
        {good_blob: good_recording, bad_blob: bad_recording}
    )
    extraction_hash = extraction_config_hash(config)
    for blob in (good_blob, bad_blob):
        acquirer.record_extraction_complete(
            acquirer.acquire(blob).manifest.source, extraction_hash
        )
    good_extraction = RecordingExtractor(
        camera_topic=config.topics.camera,
        gnss_topic=config.topics.gnss,
        target_fps=Fraction(str(config.downsampling.target_fps)),
        tolerance_ns=int(config.downsampling.tolerance_ms * 1_000_000),
        staging_root=config.paths.work_dir,
        decoder_factory=DeterministicDecoder,
    ).extract_owned(good_recording)
    failed_working = config.paths.work_dir / "bad-owned-invocation"
    failed_working.mkdir()
    (failed_working / "partial.jpg").write_bytes(b"owned")
    failed_authority = OwnedDirectoryAuthority.capture(failed_working)

    class CleanupFailingCache:
        def _materialize_owned(
            self,
            source: SourceFingerprint,
            _config_hash: str,
            _source_path: Path,
            _working_root: Path,
            _recording_id: str,
        ) -> MaterializedExtraction:
            if source.blob_path == bad_blob:
                original = ValueError("injected cache materialization failure")
                cleanup = OwnedDirectoryCleanupError(
                    (failed_authority.cleanup_failure(),)
                )
                raise cleanup from original
            return MaterializedExtraction(
                good_extraction.result, good_extraction.authority
            )

    fake_cache = cast(ExtractionResultCache, CleanupFailingCache())
    runtime = BuildRuntime(
        acquirer_factory=lambda _config: acquirer,
        decoder_factory=DeterministicDecoder,
        extraction_cache_factory=lambda _path: fake_cache,
        official_smoke=False,
    )

    with pytest.raises(BuildOperationalError, match="zero recordings are authorized"):
        build_dataset(config, runtime=runtime)

    assert not (config.paths.output_dir / config.publication.version).exists()
    assert failed_working.is_dir()
    reports = tuple(config.quarantine.directory.glob("*.quarantine.json"))
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert payload["artifact_handling"] == "preserved_in_place"
    assert payload["deterministic_details"]["owned_working_trees"][0]["path"] == str(
        failed_working
    )


def test_partial_pipeline_blocks_fresh_extraction_rollback_failure(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailSecondDecode(DeterministicDecoder):
        def decode(
            self, payload: bytes, pts: int, time_base: Fraction
        ) -> list[DecoderOutput]:
            if self.calls == 1:
                raise StructuralExtractionError("injected fresh extraction failure")
            return super().decode(payload, pts, time_base)

    good_recording = tmp_path / "good-fresh.mcap"
    bad_recording = tmp_path / "bad-fresh.mcap"
    payloads = tuple(
        camera_message(
            timestamp,
            (timestamp + 10, timestamp + 20),
            camera_names=("front", "rear"),
        )
        for timestamp in (1_000_000_000, 1_500_000_000, 2_000_000_000)
    )
    write_mcap(good_recording, camera_payloads=payloads)
    write_mcap(bad_recording, camera_payloads=payloads)
    good_blob = "mcap-h265/good-fresh.mcap"
    bad_blob = "mcap-h265/bad-fresh.mcap"
    config = _pipeline_config(
        config_factory(), tmp_path, (good_blob, bad_blob), partial=True
    )
    acquirer = _FakeAcquirer(
        {good_blob: good_recording, bad_blob: bad_recording}
    )
    original_extract_owned = RecordingExtractor.extract_owned

    def extract_with_bad_decoder(
        extractor: RecordingExtractor, path: Path
    ) -> OwnedRecordingExtraction:
        if path == bad_recording:
            failing = RecordingExtractor(
                camera_topic=extractor.camera_topic,
                gnss_topic=extractor.gnss_topic,
                target_fps=extractor.target_fps,
                tolerance_ns=extractor.tolerance_ns,
                staging_root=extractor.staging_root,
                decoder_factory=FailSecondDecode,
            )
            return original_extract_owned(failing, path)
        return original_extract_owned(extractor, path)

    def fail_rollback(_invocation: object) -> None:
        raise StructuralExtractionError("injected fresh rollback failure")

    monkeypatch.setattr(
        RecordingExtractor, "extract_owned", extract_with_bad_decoder
    )
    monkeypatch.setattr(
        "dataset_devkit.extraction.service.rollback_staging_invocation",
        fail_rollback,
    )
    runtime = BuildRuntime(
        acquirer_factory=lambda _config: acquirer,
        decoder_factory=DeterministicDecoder,
        official_smoke=False,
    )

    with pytest.raises(BuildOperationalError, match="zero recordings are authorized"):
        build_dataset(config, runtime=runtime)

    assert not (config.paths.output_dir / config.publication.version).exists()
    reports = tuple(config.quarantine.directory.glob("*.quarantine.json"))
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert payload["artifact_handling"] == "preserved_in_place"
    details = payload["deterministic_details"]
    assert details["owned_working_trees"][0]["expected_parent_chain"]
    assert details["cleanup_original_cause"] == {
        "exception_type": "StructuralExtractionError",
        "exception_message": "injected fresh extraction failure",
    }


def test_partial_pipeline_blocks_failed_authority_handoff(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good_recording = tmp_path / "good-handoff.mcap"
    bad_recording = tmp_path / "bad-handoff.mcap"
    payloads = tuple(
        camera_message(
            timestamp,
            (timestamp + 10, timestamp + 20),
            camera_names=("front", "rear"),
        )
        for timestamp in (1_000_000_000, 1_500_000_000, 2_000_000_000)
    )
    write_mcap(good_recording, camera_payloads=payloads)
    write_mcap(bad_recording, camera_payloads=payloads)
    good_blob = "mcap-h265/good-handoff.mcap"
    bad_blob = "mcap-h265/bad-handoff.mcap"
    config = _pipeline_config(
        config_factory(), tmp_path, (good_blob, bad_blob), partial=True
    )
    acquirer = _FakeAcquirer(
        {good_blob: good_recording, bad_blob: bad_recording}
    )
    bad_roots: set[Path] = set()
    original_validate = _WorkingExtractionRegistry._validate_handoff
    original_cleanup = OwnedDirectoryAuthority.cleanup

    def fail_bad_handoff(
        result: RecordingExtractionResult, authority: OwnedDirectoryAuthority
    ) -> None:
        if result.source_path == bad_recording.resolve():
            bad_roots.add(authority.root)
            raise ValueError("injected authority handoff failure")
        original_validate(result, authority)

    def fail_bad_cleanup(authority: OwnedDirectoryAuthority) -> bool:
        if authority.root in bad_roots:
            return False
        return original_cleanup(authority)

    monkeypatch.setattr(
        _WorkingExtractionRegistry,
        "_validate_handoff",
        staticmethod(fail_bad_handoff),
    )
    monkeypatch.setattr(OwnedDirectoryAuthority, "cleanup", fail_bad_cleanup)
    runtime = BuildRuntime(
        acquirer_factory=lambda _config: acquirer,
        decoder_factory=DeterministicDecoder,
        official_smoke=False,
    )

    with pytest.raises(BuildOperationalError, match="zero recordings are authorized"):
        build_dataset(config, runtime=runtime)

    assert not (config.paths.output_dir / config.publication.version).exists()
    assert len(bad_roots) == 1
    assert next(iter(bad_roots)).is_dir()
    reports = tuple(config.quarantine.directory.glob("*.quarantine.json"))
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert payload["artifact_handling"] == "preserved_in_place"
    assert payload["deterministic_details"]["cleanup_original_cause"] == {
        "exception_type": "ValueError",
        "exception_message": "injected authority handoff failure",
    }


def test_partial_pipeline_blocks_unpinned_fresh_invocation(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good_recording = tmp_path / "good-preauthority.mcap"
    bad_recording = tmp_path / "bad-preauthority.mcap"
    payloads = tuple(
        camera_message(
            timestamp,
            (timestamp + 10, timestamp + 20),
            camera_names=("front", "rear"),
        )
        for timestamp in (1_000_000_000, 1_500_000_000, 2_000_000_000)
    )
    write_mcap(good_recording, camera_payloads=payloads)
    write_mcap(bad_recording, camera_payloads=payloads)
    good_blob = "mcap-h265/good-preauthority.mcap"
    bad_blob = "mcap-h265/bad-preauthority.mcap"
    config = _pipeline_config(
        config_factory(), tmp_path, (good_blob, bad_blob), partial=True
    )
    acquirer = _FakeAcquirer(
        {good_blob: good_recording, bad_blob: bad_recording}
    )
    original_open = os.open

    def fail_bad_child_open(
        path: str | bytes | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if (
            isinstance(path, str)
            and path.startswith("bad-preauthority-")
            and dir_fd is not None
        ):
            raise OSError("injected child open failure")
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(
        "dataset_devkit.extraction.staging.os.open", fail_bad_child_open
    )
    runtime = BuildRuntime(
        acquirer_factory=lambda _config: acquirer,
        decoder_factory=DeterministicDecoder,
        official_smoke=False,
    )

    with pytest.raises(BuildOperationalError, match="zero recordings are authorized"):
        build_dataset(config, runtime=runtime)

    assert not (config.paths.output_dir / config.publication.version).exists()
    unpinned = tuple(config.paths.work_dir.glob("bad-preauthority-*"))
    assert len(unpinned) == 1
    reports = tuple(config.quarantine.directory.glob("*.quarantine.json"))
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert payload["artifact_handling"] == "preserved_in_place"
    details = payload["deterministic_details"]
    assert details["owned_working_trees"][0]["path"] == str(unpinned[0])
    assert details["cleanup_original_cause"] == {
        "exception_type": "OSError",
        "exception_message": "injected child open failure",
    }


def test_post_export_working_cleanup_failure_blocks_publication(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = tmp_path / "recording.mcap"
    write_mcap(
        recording,
        camera_payloads=tuple(
            camera_message(
                timestamp,
                (timestamp + 10, timestamp + 20),
                camera_names=("front", "rear"),
            )
            for timestamp in (1_000_000_000, 1_500_000_000, 2_000_000_000)
        ),
    )
    blob = "mcap-h265/recording.mcap"
    config = _pipeline_config(config_factory(), tmp_path, (blob,), partial=False)
    runtime = BuildRuntime(
        acquirer_factory=lambda _config: _FakeAcquirer({blob: recording}),
        decoder_factory=DeterministicDecoder,
        official_smoke=False,
    )
    monkeypatch.setattr(OwnedDirectoryAuthority, "cleanup", lambda _self: False)

    with pytest.raises(OwnedDirectoryCleanupError) as captured:
        build_dataset(config, runtime=runtime)

    assert captured.value.failures[0].path.parent == config.paths.work_dir
    assert not (config.paths.output_dir / config.publication.version).exists()
    assert not tuple(config.paths.output_dir.glob(".*.staging-*"))


def test_global_failure_surfaces_working_cleanup_failure_with_original_cause(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = tmp_path / "recording.mcap"
    write_mcap(
        recording,
        camera_payloads=tuple(
            camera_message(
                timestamp,
                (timestamp + 10, timestamp + 20),
                camera_names=("front", "rear"),
            )
            for timestamp in (1_000_000_000, 1_500_000_000, 2_000_000_000)
        ),
    )
    blob = "mcap-h265/recording.mcap"
    config = _pipeline_config(config_factory(), tmp_path, (blob,), partial=False)
    runtime = BuildRuntime(
        acquirer_factory=lambda _config: _FakeAcquirer({blob: recording}),
        decoder_factory=DeterministicDecoder,
        official_smoke=False,
    )

    def fail_selection(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected global failure")

    monkeypatch.setattr("dataset_devkit.services.select_scenarios", fail_selection)
    monkeypatch.setattr(OwnedDirectoryAuthority, "cleanup", lambda _self: False)

    with pytest.raises(OwnedDirectoryCleanupError) as captured:
        build_dataset(config, runtime=runtime)

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert str(captured.value.__cause__) == "injected global failure"
    assert captured.value.failures[0].path.parent == config.paths.work_dir


@pytest.mark.parametrize("unsafe_component", ["ancestor", "root", "generation"])
def test_extraction_result_cache_rejects_symlinked_directory_components(
    tmp_path: Path,
    unsafe_component: str,
) -> None:
    _, extracted, source = _cache_security_case(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    config_hash = "a" * 64
    if unsafe_component == "ancestor":
        link = tmp_path / "cache-link"
        link.symlink_to(outside, target_is_directory=True)
        cache = ExtractionResultCache(link / "cache")
    else:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        if unsafe_component == "root":
            (cache_dir / "extraction-results").symlink_to(
                outside, target_is_directory=True
            )
        else:
            source_root = cache_dir / "extraction-results" / source.digest
            source_root.mkdir(parents=True)
            (source_root / config_hash).symlink_to(outside, target_is_directory=True)
        cache = ExtractionResultCache(cache_dir)

    with pytest.raises((OSError, ValueError)):
        cache.store(source, config_hash, extracted)

    assert tuple(outside.iterdir()) == ()


def test_extraction_result_cache_detects_root_replacement_without_escaping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, extracted, source = _cache_security_case(tmp_path)
    cache = ExtractionResultCache(tmp_path / "cache")
    outside = tmp_path / "outside"
    outside.mkdir()
    displaced = tmp_path / "displaced-extraction-results"
    from dataset_devkit.extraction import cache as cache_module

    original_lock = cache_module._cache_lock

    @contextmanager
    def replace_root(parent_fd: int, name: str) -> Iterator[None]:
        with original_lock(parent_fd, name):
            cache.root.rename(displaced)
            cache.root.symlink_to(outside, target_is_directory=True)
            yield

    monkeypatch.setattr(cache_module, "_cache_lock", replace_root)

    with pytest.raises(ValueError, match="cache directory binding changed"):
        cache.store(source, "a" * 64, extracted)

    assert tuple(outside.iterdir()) == ()


def test_extraction_result_cache_force_refresh_replaces_an_existing_generation(
    tmp_path: Path,
) -> None:
    recording = tmp_path / "recording.mcap"
    write_mcap(
        recording,
        camera_payloads=(
            camera_message(1_000_000_000, (1_000_000_010, 1_000_000_020)),
            camera_message(1_500_000_000, (1_500_000_010, 1_500_000_020)),
        ),
    )
    extractor = RecordingExtractor(
        camera_topic="rec_cameras",
        gnss_topic="gnss",
        target_fps=2,
        tolerance_ns=0,
        staging_root=tmp_path / "staging",
        decoder_factory=DeterministicDecoder,
    )
    extracted = extractor.extract(recording)
    source = SourceFingerprint(
        "https://example.blob.core.windows.net",
        "recordings",
        "mcap-h265/recording.mcap",
        '"etag"',
        recording.stat().st_size,
    )
    config_hash = "a" * 64
    cache = ExtractionResultCache(tmp_path / "cache")
    first = cache.store(source, config_hash, extracted)
    first_image = first.path / "images" / "00000000.jpg"
    first_identity = first_image.stat().st_ino

    refreshed = cache.store(source, config_hash, extracted, force_refresh=True)

    assert (refreshed.path / "images" / "00000000.jpg").stat().st_ino != first_identity
    assert cache.contains(source, config_hash)


def test_extraction_result_cache_interrupted_refresh_preserves_previous_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = tmp_path / "recording.mcap"
    write_mcap(
        recording,
        camera_payloads=(
            camera_message(1_000_000_000, (1_000_000_010, 1_000_000_020)),
        ),
    )
    extracted = RecordingExtractor(
        camera_topic="rec_cameras",
        gnss_topic="gnss",
        target_fps=1,
        tolerance_ns=0,
        staging_root=tmp_path / "staging",
        decoder_factory=DeterministicDecoder,
    ).extract(recording)
    source = SourceFingerprint(
        "https://example.blob.core.windows.net",
        "recordings",
        "mcap-h265/recording.mcap",
        '"etag"',
        recording.stat().st_size,
    )
    config_hash = "a" * 64
    cache = ExtractionResultCache(tmp_path / "cache")
    first = cache.store(source, config_hash, extracted)
    first_identity = (first.path / "images" / "00000000.jpg").stat().st_ino

    def interrupt(*_args: object) -> None:
        raise RuntimeError("interrupted refresh")

    monkeypatch.setattr(cache, "_publish_refresh", interrupt, raising=False)
    with pytest.raises(RuntimeError, match="interrupted refresh"):
        cache.store(source, config_hash, extracted, force_refresh=True)

    assert cache.contains(source, config_hash)
    assert (first.path / "images" / "00000000.jpg").stat().st_ino == first_identity


def test_extraction_result_cache_refresh_cleanup_preserves_replacement_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = tmp_path / "recording.mcap"
    write_mcap(
        recording,
        camera_payloads=(
            camera_message(1_000_000_000, (1_000_000_010, 1_000_000_020)),
        ),
    )
    extracted = RecordingExtractor(
        camera_topic="rec_cameras",
        gnss_topic="gnss",
        target_fps=1,
        tolerance_ns=0,
        staging_root=tmp_path / "staging",
        decoder_factory=DeterministicDecoder,
    ).extract(recording)
    source = SourceFingerprint(
        "https://example.blob.core.windows.net",
        "recordings",
        "mcap-h265/recording.mcap",
        '"etag"',
        recording.stat().st_size,
    )
    config_hash = "a" * 64
    cache = ExtractionResultCache(tmp_path / "cache")
    cache.store(source, config_hash, extracted)
    original_cleanup = publication_module.cleanup_pinned_directory
    replacement: Path | None = None
    displaced: Path | None = None

    def replace_exchanged_predecessor_before_cleanup(
        parent_fd: int,
        name: str,
        directory_fd: int,
        expected_identity: tuple[int, int],
    ) -> bool:
        nonlocal displaced, replacement
        predecessor = cache.path_for(source, config_hash).parent / name
        displaced = predecessor.with_name(f"{predecessor.name}.owned-stale")
        predecessor.rename(displaced)
        predecessor.mkdir()
        replacement = predecessor
        (predecessor / "keep.txt").write_text("unrelated", encoding="utf-8")
        return original_cleanup(parent_fd, name, directory_fd, expected_identity)

    monkeypatch.setattr(
        "dataset_devkit.extraction.cache.cleanup_pinned_directory",
        replace_exchanged_predecessor_before_cleanup,
    )

    cache.store(source, config_hash, extracted, force_refresh=True)

    assert replacement is not None
    assert (replacement / "keep.txt").read_text(encoding="utf-8") == "unrelated"
    assert displaced is not None
    assert tuple(displaced.iterdir()) == ()


def test_extraction_result_cache_post_exchange_failure_never_cleans_new_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = tmp_path / "recording.mcap"
    write_mcap(
        recording,
        camera_payloads=(
            camera_message(1_000_000_000, (1_000_000_010, 1_000_000_020)),
        ),
    )
    extracted = RecordingExtractor(
        camera_topic="rec_cameras",
        gnss_topic="gnss",
        target_fps=1,
        tolerance_ns=0,
        staging_root=tmp_path / "staging",
        decoder_factory=DeterministicDecoder,
    ).extract(recording)
    source = SourceFingerprint(
        "https://example.blob.core.windows.net",
        "recordings",
        "mcap-h265/recording.mcap",
        '"etag"',
        recording.stat().st_size,
    )
    config_hash = "a" * 64
    cache = ExtractionResultCache(tmp_path / "cache")
    cache.store(source, config_hash, extracted)
    from dataset_devkit.extraction import cache as cache_module

    original_exchange = cache_module._exchange_directories

    def fail_after_exchange(parent_fd: int, left: str, right: str) -> None:
        original_exchange(parent_fd, left, right)
        raise RuntimeError("injected after exchange")

    monkeypatch.setattr(cache_module, "_exchange_directories", fail_after_exchange)
    with pytest.raises(RuntimeError, match="injected after exchange"):
        cache.store(source, config_hash, extracted, force_refresh=True)

    assert cache.contains(source, config_hash)
    assert (cache.path_for(source, config_hash) / "images" / "00000000.jpg").read_bytes()


def test_extraction_result_cache_concurrent_refreshes_publish_complete_generations(
    tmp_path: Path,
) -> None:
    recording = tmp_path / "recording.mcap"
    write_mcap(
        recording,
        camera_payloads=(
            camera_message(1_000_000_000, (1_000_000_010, 1_000_000_020)),
        ),
    )
    extracted = RecordingExtractor(
        camera_topic="rec_cameras",
        gnss_topic="gnss",
        target_fps=1,
        tolerance_ns=0,
        staging_root=tmp_path / "staging",
        decoder_factory=DeterministicDecoder,
    ).extract(recording)
    source = SourceFingerprint(
        "https://example.blob.core.windows.net",
        "recordings",
        "mcap-h265/recording.mcap",
        '"etag"',
        recording.stat().st_size,
    )
    config_hash = "a" * 64
    cache = ExtractionResultCache(tmp_path / "cache")
    cache.store(source, config_hash, extracted)

    with ThreadPoolExecutor(max_workers=2) as executor:
        refreshed = tuple(
            executor.map(
                lambda _: cache.store(
                    source,
                    config_hash,
                    extracted,
                    force_refresh=True,
                ),
                range(2),
            )
        )

    assert all(result.image_count == len(extracted.samples) for result in refreshed)
    assert all(result.refreshed for result in refreshed)
    assert cache.contains(source, config_hash)
    stale = tuple(cache.path_for(source, config_hash).parent.glob("*.staging-*"))
    assert stale == ()
