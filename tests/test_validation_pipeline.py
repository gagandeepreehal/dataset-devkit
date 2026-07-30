from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from conftest import FeatureFactory
from dataset_devkit import publication as publication_module
from dataset_devkit import services as services_module
from dataset_devkit.config import GlobalConfig
from dataset_devkit.dataset import Dataset
from dataset_devkit.export import export_dataset
from dataset_devkit.publication import publish_staging
from dataset_devkit.services import (
    BuildOperationalError,
    BuildRuntime,
    build_dataset,
    inspect_dataset,
)
from dataset_devkit.validation import (
    DatasetValidationError,
    finalize_dataset,
    validate_dataset,
)
from test_export_dataset import _evidence


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
    with pytest.raises(DatasetValidationError, match="manifest"):
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
            "mz_extensions/split.json",
            lambda value: value["assignments"].append(dict(value["assignments"][0])),
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

    published = publish_staging(staging, final)
    summary = inspect_dataset(published, "v1.0-trainval", official_smoke=False)

    assert summary.validation_state == "succeeded"
    assert summary.scene_count > 0
    assert summary.content_hash
    with pytest.raises(FileExistsError, match="overwrite"):
        publish_staging(tmp_path / "other", final)


def test_complete_injected_build_is_repeatable_and_cli_loadable(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence(tmp_path / "input", config_factory, feature_factory)
    output = tmp_path / "published"
    paths = evidence.resolved_config.paths.model_copy(update={"output_dir": output})
    config = evidence.resolved_config.model_copy(update={"paths": paths})
    evidence = replace(evidence, resolved_config=config)
    runtime = BuildRuntime(
        evidence_builder=lambda supplied: (
            replace(
                evidence,
                resolved_config=supplied,
                pipeline_audit={"schema_version": 1, "synthetic": True},
            ),
            (),
        ),
        official_smoke=True,
    )

    first = build_dataset(config, runtime=runtime)
    first_hash = first.content_hash
    assert Dataset(first.dataroot).validation_report()["succeeded"] is True
    archived = tmp_path / "first-published"
    first.dataroot.rename(archived)
    second = build_dataset(config, runtime=runtime)

    assert second.content_hash == first_hash
    from dataset_devkit.cli import main

    assert main(
        [
            "validate",
            "--dataroot",
            str(second.dataroot),
            "--version",
            "v1.0-trainval",
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "succeeded"
    assert main(
        [
            "inspect",
            "--dataroot",
            str(second.dataroot),
            "--version",
            "v1.0-trainval",
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["content_hash"] == first_hash

    monkeypatch.setattr("dataset_devkit.cli.build_dataset", lambda _config: second)
    assert main(["build", "--config", "examples/dataset_config.json"]) == 0
    assert json.loads(capsys.readouterr().out)["dataroot"] == str(second.dataroot)


def test_injected_partial_failure_requires_explicit_authorization(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    evidence = _evidence(tmp_path / "input", config_factory, feature_factory)
    output = tmp_path / "published"
    paths = evidence.resolved_config.paths.model_copy(update={"output_dir": output})
    config = evidence.resolved_config.model_copy(update={"paths": paths})
    evidence = replace(evidence, resolved_config=config)
    runtime = BuildRuntime(
        evidence_builder=lambda supplied: (
            replace(
                evidence,
                resolved_config=supplied,
                pipeline_audit={"schema_version": 1, "synthetic": True},
            ),
            ("mcap-h265/bad.mcap",),
        ),
        official_smoke=False,
    )

    with pytest.raises(BuildOperationalError, match="blocked"):
        build_dataset(config, runtime=runtime)

    partial_execution = config.execution.model_copy(update={"allow_partial_export": True})
    partial_config = config.model_copy(update={"execution": partial_execution})
    result = build_dataset(partial_config, runtime=runtime)
    assert result.partial is True
    assert result.failed_recordings == ("mcap-h265/bad.mcap",)


def test_identity_bound_cleanup_and_publish_race_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "owned.txt").write_text("owned")
    current = staging.stat()
    assert services_module._cleanup_owned_staging(
        staging, (current.st_dev, current.st_ino)
    )
    assert not staging.exists()

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    staging.mkdir()
    stale = staging.stat()
    staging.rmdir()
    staging.symlink_to(outside, target_is_directory=True)
    assert not services_module._cleanup_owned_staging(
        staging, (stale.st_dev, stale.st_ino)
    )
    assert (outside / "keep.txt").read_text() == "keep"

    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "final"
    original = publication_module._rename_exclusive

    def race(parent_fd: int, source_name: str, destination_name: str) -> None:
        destination.mkdir()
        original(parent_fd, source_name, destination_name)

    monkeypatch.setattr(publication_module, "_rename_exclusive", race)
    with pytest.raises(FileExistsError, match="overwrite"):
        publish_staging(source, destination)
    assert source.is_dir()
    assert destination.is_dir()
