from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

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
from dataset_devkit.extraction.cache import ExtractionResultCache
from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.service import RecordingExtractor
from dataset_devkit.features import RecordingFeatureResult, compute_recording_features
from dataset_devkit.provenance import (
    AcquisitionManifest,
    ArtifactIdentity,
    IntegrityVerification,
    SourceFingerprint,
    extraction_config_hash,
)
from dataset_devkit.publication import StagingLease, publish_staging
from dataset_devkit.scene_models import RecordingSceneResult
from dataset_devkit.services import (
    BuildOperationalError,
    BuildResult,
    BuildRuntime,
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
    first_hash = first.content_hash
    first_decoder_creations = decoder_creations
    assert first_decoder_creations > 0
    assert Dataset(first.dataroot).validation_report()["succeeded"] is True
    source = acquirer.acquire(blob).manifest.source
    extraction_hash = extraction_config_hash(config)
    extraction_cache = ExtractionResultCache(config.paths.cache_dir)
    first_cached = extraction_cache.load(source, extraction_hash, recording)
    assert first_cached is not None
    first_cache_identity = first_cached.samples[0].staged_image.path.stat().st_ino
    # A sealed dataroot is read/execute-only; same-parent rename avoids changing
    # its POSIX `..` entry while preserving the first result for comparison.
    archived = first.dataroot.with_name("first-published")
    first.dataroot.rename(archived)
    second = build_dataset(config, runtime=runtime)

    assert second.content_hash == first_hash
    assert decoder_creations == first_decoder_creations
    assert acquirer.extraction_cache_reusable(
        source,
        extraction_hash,
    )
    second_cached = extraction_cache.load(source, extraction_hash, recording)
    assert second_cached is not None
    assert second_cached.samples[0].staged_image.path.stat().st_ino == first_cache_identity

    second.dataroot.rename(second.dataroot.with_name("second-published"))
    acquirer.completed.clear()
    third = build_dataset(config, runtime=runtime)
    refreshed = extraction_cache.load(source, extraction_hash, recording)
    assert decoder_creations > first_decoder_creations
    assert refreshed is not None
    assert refreshed.samples[0].staged_image.path.stat().st_ino != first_cache_identity
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
    loaded = cache.load(source, config_hash, recording)

    assert loaded is not None
    assert loaded.camera_batches == extracted.camera_batches
    assert loaded.selected_grid == extracted.selected_grid
    assert len(loaded.samples) == len(extracted.samples)
    assert all(item.staged_image.path.is_file() for item in stored.samples)
    assert cache.load(source, "b" * 64, recording) is None
    changed_source = replace(source, etag='"changed"')
    assert cache.load(changed_source, config_hash, recording) is None

    stored.samples[0].staged_image.path.write_bytes(b"corrupt")
    assert cache.load(source, config_hash, recording) is None


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
    first_identity = first.samples[0].staged_image.path.stat().st_ino

    refreshed = cache.store(source, config_hash, extracted, force_refresh=True)

    assert refreshed.samples[0].staged_image.path.stat().st_ino != first_identity
    assert cache.load(source, config_hash, recording) is not None


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
    first_identity = first.samples[0].staged_image.path.stat().st_ino

    def interrupt(*_args: object) -> None:
        raise RuntimeError("interrupted refresh")

    monkeypatch.setattr(cache, "_publish_refresh", interrupt, raising=False)
    with pytest.raises(RuntimeError, match="interrupted refresh"):
        cache.store(source, config_hash, extracted, force_refresh=True)

    loaded = cache.load(source, config_hash, recording)
    assert loaded is not None
    assert loaded.samples[0].staged_image.path.stat().st_ino == first_identity


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

    loaded = cache.load(source, config_hash, recording)
    assert loaded is not None
    assert loaded.samples[0].staged_image.path.read_bytes()


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

    assert all(len(result.samples) == len(extracted.samples) for result in refreshed)
    refreshed_identities = {
        result.samples[0].staged_image.inode for result in refreshed
    }
    assert None not in refreshed_identities
    assert len(refreshed_identities) == 2
    loaded = cache.load(source, config_hash, recording)
    assert loaded is not None
    assert len(loaded.samples) == len(extracted.samples)
    assert loaded.samples[0].staged_image.inode in refreshed_identities
    stale = tuple(cache.path_for(source, config_hash).parent.glob("*.staging-*"))
    assert stale == ()
