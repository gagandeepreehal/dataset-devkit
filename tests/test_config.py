from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from dataset_devkit.config import GlobalConfig, load_config


def minimal_config() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "azure": {
            "account_url": "https://example.blob.core.windows.net",
            "container": "ecal-batchstore",
            "blob_list": "mcap_blobs.txt",
        },
        "paths": {
            "work_dir": "../work",
            "cache_dir": "../cache",
            "output_dir": "../output/dataset",
        },
        "topics": {"camera": "rec_cameras", "gnss": "gnss"},
        "downsampling": {"target_fps": 2.0, "tolerance_ms": 100.0},
        "image": {"jpeg_quality": 95},
        "gnss": {
            "position_sigma_max_m": 0.5,
            "orientation_variance_max": 0.1,
            "sync_gap_max_ms": 30.0,
        },
        "frame_validity": {
            "invalid_sample_policy": "retain_for_audit",
            "invalidate_on": {},
        },
        "sanity_checks": {},
        "scenes": {
            "mode": "hybrid",
            "min_duration_s": 10.0,
            "max_duration_s": 40.0,
            "min_samples": 20,
            "max_sample_gap_ms": 650.0,
            "skip_between_scenes_s": 0.0,
        },
        "annotations": {
            "path": "annotations.jsonl",
            "match_tolerance_ms": 500.0,
            "before_s": 20.0,
            "after_s": 20.0,
        },
        "tags": {"stationary_speed_mps": 0.2, "turn_angle_deg": 45.0},
        "filters": {"min_valid_ratio": 1.0, "required_tags": []},
        "scenarios": {"seed": 42, "rules": []},
        "split": {"test_fraction": 0.2, "seed": 42, "stratify": True},
        "execution": {"workers": 2, "allow_partial_export": False},
        "publication": {"version": "v1.0-trainval", "refuse_overwrite": True},
    }


def write_config(tmp_path: Path, data: dict[str, object]) -> Path:
    config_path = tmp_path / "config" / "dataset_config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(data), encoding="utf-8")
    return config_path


def set_nested(data: dict[str, object], dotted_key: str, value: object) -> None:
    target: dict[str, Any] = data
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def test_load_config_is_strict_and_resolves_relative_paths(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, minimal_config())

    config = load_config(config_path)

    assert isinstance(config, GlobalConfig)
    assert config.azure.blob_list == (config_path.parent / "mcap_blobs.txt").resolve()
    assert config.paths.work_dir == (config_path.parent / "../work").resolve()
    assert config.annotations.path == (config_path.parent / "annotations.jsonl").resolve()

    invalid = minimal_config()
    invalid["unexpected"] = True
    invalid_path = write_config(tmp_path / "invalid", invalid)
    with pytest.raises(ValidationError, match="unexpected"):
        load_config(invalid_path)


@pytest.mark.parametrize("quality", [0, 101])
def test_jpeg_quality_must_be_in_valid_range(tmp_path: Path, quality: int) -> None:
    data = minimal_config()
    data["image"]["jpeg_quality"] = quality  # type: ignore[index]

    with pytest.raises(ValidationError, match="jpeg_quality"):
        load_config(write_config(tmp_path / str(quality), data))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("downsampling.target_fps", 0),
        ("downsampling.tolerance_ms", -1),
        ("gnss.position_sigma_max_m", -1),
        ("gnss.orientation_variance_max", -1),
        ("gnss.sync_gap_max_ms", -1),
        ("scenes.min_duration_s", 0),
        ("scenes.min_samples", 0),
        ("scenes.max_sample_gap_ms", -1),
        ("scenes.skip_between_scenes_s", -1),
        ("annotations.match_tolerance_ms", -1),
        ("annotations.before_s", -1),
        ("annotations.after_s", -1),
        ("tags.stationary_speed_mps", -1),
        ("tags.turn_angle_deg", 181),
        ("filters.min_valid_ratio", -0.1),
        ("filters.min_valid_ratio", 1.1),
        ("split.test_fraction", 0),
        ("split.test_fraction", 1),
        ("execution.workers", 0),
    ],
)
def test_invalid_numeric_thresholds_are_rejected(tmp_path: Path, field: str, value: object) -> None:
    data = minimal_config()
    set_nested(data, field, value)

    with pytest.raises(ValidationError, match=field.rsplit(".", 1)[-1]):
        load_config(write_config(tmp_path / field.replace(".", "_"), data))


def test_scene_max_duration_cannot_be_less_than_minimum(tmp_path: Path) -> None:
    data = minimal_config()
    set_nested(data, "scenes.min_duration_s", 30)
    set_nested(data, "scenes.max_duration_s", 20)

    with pytest.raises(ValidationError, match="max_duration_s"):
        load_config(write_config(tmp_path, data))


@pytest.mark.parametrize(
    ("first", "second", "first_value", "second_value"),
    [
        ("work_dir", "cache_dir", "../runtime", "../runtime"),
        ("work_dir", "cache_dir", "../runtime", "../runtime/cache"),
        ("cache_dir", "work_dir", "../runtime", "../runtime/work"),
        ("work_dir", "output_dir", "../runtime", "../runtime/export"),
        ("output_dir", "cache_dir", "../runtime", "../runtime/cache"),
    ],
)
def test_work_cache_and_output_paths_must_not_overlap(
    tmp_path: Path,
    first: str,
    second: str,
    first_value: str,
    second_value: str,
) -> None:
    data = minimal_config()
    paths: dict[str, object] = data["paths"]  # type: ignore[assignment]
    paths[first] = first_value
    paths[second] = second_value

    with pytest.raises(ValidationError, match="overlap"):
        load_config(write_config(tmp_path, data))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("azure.client_secret", "do-not-store-this"),
        (
            "azure.account_url",
            "https://example.blob.core.windows.net?sv=2024-11-04&sig=signature",
        ),
        (
            "azure.container",
            "DefaultEndpointsProtocol=https;AccountName=x;AccountKey=secret",
        ),
        ("azure.access_token", "do-not-store-this"),
        ("azure.container", "sk-abcdefghijklmnopqrstuvwxyz012345"),
    ],
)
def test_embedded_credentials_are_rejected(tmp_path: Path, field: str, value: str) -> None:
    data = minimal_config()
    set_nested(data, field, value)

    with pytest.raises(ValidationError, match="credential"):
        load_config(write_config(tmp_path, data))


def test_ordinary_urls_and_secret_named_paths_are_allowed(tmp_path: Path) -> None:
    data = minimal_config()
    set_nested(data, "annotations.path", "secret-camera/annotations.jsonl")

    config = load_config(write_config(tmp_path, data))

    assert config.azure.account_url == "https://example.blob.core.windows.net"
    assert config.annotations.path.name == "annotations.jsonl"


@pytest.mark.parametrize(
    "account_url",
    [
        "https://user@example.blob.core.windows.net",
        "https://user:password@example.blob.core.windows.net",
    ],
)
def test_account_url_rejects_url_userinfo(tmp_path: Path, account_url: str) -> None:
    data = minimal_config()
    set_nested(data, "azure.account_url", account_url)

    with pytest.raises(ValidationError, match="credential|userinfo"):
        load_config(write_config(tmp_path, data))


@pytest.mark.parametrize(
    "secret",
    [
        "A" * 86 + "==",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.c2lnbmF0dXJl",
    ],
)
def test_raw_azure_keys_and_jwts_are_rejected_in_values(tmp_path: Path, secret: str) -> None:
    data = minimal_config()
    set_nested(data, "publication.version", secret)

    with pytest.raises(ValidationError, match="credential"):
        load_config(write_config(tmp_path, data))


@pytest.mark.parametrize(
    "credential_field",
    [
        "auth",
        "password",
        "secret",
        "account_key",
        "azure_storage_account_key",
        "access_token",
    ],
)
def test_explicit_credential_field_names_are_rejected(
    tmp_path: Path, credential_field: str
) -> None:
    data = minimal_config()
    azure: dict[str, object] = data["azure"]  # type: ignore[assignment]
    azure[credential_field] = "embedded-value"

    with pytest.raises(ValidationError, match="credential key"):
        load_config(write_config(tmp_path, data))


def test_bearer_prefixed_ordinary_strings_and_paths_are_allowed(tmp_path: Path) -> None:
    data = minimal_config()
    set_nested(data, "azure.container", "bearer recordings")
    set_nested(data, "annotations.path", "bearer archive/annotations.jsonl")
    set_nested(data, "publication.version", "bearer migration archive for July recordings")

    config = load_config(write_config(tmp_path, data))

    assert config.azure.container == "bearer recordings"
    assert config.annotations.path.name == "annotations.jsonl"
    assert config.publication.version == "bearer migration archive for July recordings"


@pytest.mark.parametrize("scheme", ["Bearer", "bearer", "BEARER"])
def test_opaque_bearer_tokens_are_rejected(tmp_path: Path, scheme: str) -> None:
    data = minimal_config()
    set_nested(
        data,
        "publication.version",
        f"{scheme} abcdefghijklmnopqrstuvwxyz123456",
    )

    with pytest.raises(ValidationError, match="credential"):
        load_config(write_config(tmp_path, data))


@pytest.mark.parametrize(
    "token",
    [
        "abcdefghijklmnopqrstuvwxyz123456.",
        "abcdefghijklmnopqrstuvwxyz123456~",
        "abcdefghijklmnopqrstuvwxyz123456+",
        "abcdefghijklmnopqrstuvwxyz123456/",
        "abcdefghijklmnop/qrstuvwxyzABCDEFGH",
        "abcdefghijklmnopqrstuvwxyz123456==",
    ],
)
def test_rfc6750_b64token_characters_are_rejected(tmp_path: Path, token: str) -> None:
    data = minimal_config()
    set_nested(data, "publication.version", f"Bearer {token}")

    with pytest.raises(ValidationError, match="credential"):
        load_config(write_config(tmp_path, data))


def test_bearer_prefixed_path_with_digits_is_allowed(tmp_path: Path) -> None:
    data = minimal_config()
    set_nested(data, "publication.version", "bearer archive/2026/annotations.jsonl")

    config = load_config(write_config(tmp_path, data))

    assert config.publication.version.endswith("annotations.jsonl")


@pytest.mark.parametrize(
    "token",
    [
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "abcdefghijklmnopqrstuvwxyz123456.abc",
        "abcdefghijklmnop/qrstuvwxyzABCDEFGH",
    ],
)
def test_standalone_bearer_token_shapes_are_rejected(tmp_path: Path, token: str) -> None:
    data = minimal_config()
    set_nested(data, "publication.version", f"Bearer {token}")

    with pytest.raises(ValidationError, match="credential"):
        load_config(write_config(tmp_path, data))


@pytest.mark.parametrize(
    "value",
    [
        "bearer archive/2026/recording12345678901234567890",
        "bearer /var/archive/recording12345678901234567890",
        "bearer ../archive/recording12345678901234567890",
        r"bearer C:\archive\recording12345678901234567890",
    ],
)
def test_clearly_path_like_bearer_values_are_allowed(tmp_path: Path, value: str) -> None:
    data = minimal_config()
    set_nested(data, "publication.version", value)

    config = load_config(write_config(tmp_path, data))

    assert config.publication.version == value


@pytest.mark.parametrize(
    ("field", "url"),
    [
        (
            "azure.account_url",
            "https://example.blob.core.windows.net?password=embedded",
        ),
        (
            "publication.version",
            "https://example.test/data?access_token=embedded",
        ),
        (
            "azure.account_url",
            "https://example.blob.core.windows.net/container?sv=2024-11-04&sp=r&se=2030-01-01&sig=embedded",
        ),
    ],
)
def test_credential_bearing_url_queries_are_rejected(tmp_path: Path, field: str, url: str) -> None:
    data = minimal_config()
    set_nested(data, field, url)

    with pytest.raises(ValidationError, match="credential"):
        load_config(write_config(tmp_path, data))


def test_ordinary_url_query_parameters_are_allowed(tmp_path: Path) -> None:
    data = minimal_config()
    set_nested(
        data,
        "azure.account_url",
        "https://example.blob.core.windows.net?api-version=2023-11-03&comp=list",
    )

    config = load_config(write_config(tmp_path, data))

    assert "api-version=2023-11-03" in config.azure.account_url


@pytest.mark.parametrize(
    ("query_key", "query_value"),
    [
        ("sp", "robotics"),
        ("st", "2026-01-01T00:00:00Z"),
        ("se", "2030-01-01T00:00:00Z"),
        ("sr", "search"),
        ("sv", "2024-11-04"),
    ],
)
def test_standalone_sas_metadata_query_parameters_are_allowed(
    tmp_path: Path, query_key: str, query_value: str
) -> None:
    data = minimal_config()
    set_nested(
        data,
        "azure.account_url",
        f"https://example.test/search?{query_key}={query_value}",
    )

    config = load_config(write_config(tmp_path, data))

    assert f"{query_key}={query_value}" in config.azure.account_url


def test_extended_policy_models_are_typed_and_resolve_paths(tmp_path: Path) -> None:
    data = minimal_config()
    data["frame_validity"] = {
        "invalid_sample_policy": "retain_for_audit",
        "invalidate_on": {
            "missing_camera": True,
            "invalid_gnss": True,
            "sync_gap_exceeded": True,
        },
    }
    data["sanity_checks"] = {
        "timestamp_policy": "quarantine",
        "max_speed_mps": 70.0,
        "max_position_jump_m": 20.0,
    }
    data["scenarios"] = {
        "seed": 42,
        "rules": [
            {
                "name": "turns",
                "required_tags": ["turn"],
                "sampling": {"fraction": 0.5, "max_scenes": 100},
            }
        ],
    }
    data["quarantine"] = {
        "enabled": True,
        "directory": "../quarantine",
        "manifest_name": "rejected.jsonl",
    }

    config = load_config(write_config(tmp_path, data))

    assert config.scenarios.rules[0].sampling.fraction == 0.5
    assert config.quarantine.directory == (tmp_path / "quarantine").resolve()


@pytest.mark.parametrize(
    "section",
    [
        "azure",
        "paths",
        "topics",
        "downsampling",
        "image",
        "gnss",
        "frame_validity",
        "frame_validity.invalidate_on",
        "sanity_checks",
        "scenes",
        "annotations",
        "tags",
        "filters",
        "scenarios",
        "split",
        "execution",
        "publication",
    ],
)
def test_unknown_keys_are_rejected_in_every_nested_section(tmp_path: Path, section: str) -> None:
    data = minimal_config()
    set_nested(data, f"{section}.unknown_key", True)

    with pytest.raises(ValidationError, match="unknown_key"):
        load_config(write_config(tmp_path, data))


@pytest.mark.parametrize("nested", ["rule", "sampling", "quarantine"])
def test_unknown_keys_are_rejected_in_extended_nested_models(tmp_path: Path, nested: str) -> None:
    data = minimal_config()
    rule: dict[str, object] = {
        "name": "turns",
        "required_tags": ["turn"],
        "sampling": {"fraction": 0.5},
    }
    data["scenarios"] = {"seed": 42, "rules": [rule]}
    data["quarantine"] = {"unknown_key": True}
    if nested == "rule":
        rule["unknown_key"] = True
        data["quarantine"] = {}
    elif nested == "sampling":
        rule["sampling"] = {"fraction": 0.5, "unknown_key": True}
        data["quarantine"] = {}

    with pytest.raises(ValidationError, match="unknown_key"):
        load_config(write_config(tmp_path, data))
