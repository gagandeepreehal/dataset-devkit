"""Versioned global configuration models."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_AZURE_ACCOUNT_KEY_PATTERN = re.compile(
    r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{86}==(?=$|[^A-Za-z0-9+/=])"
)
_JWT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{5,}"
    r"\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}(?![A-Za-z0-9_-])"
)
_BEARER_VALUE_PATTERN = re.compile(r"^bearer[ \t]+(?P<payload>.+)$", re.IGNORECASE)
_BEARER_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9._~+/\-]{32,}=*$")
_CREDENTIAL_KEY_NAMES = {
    "auth",
    "authentication",
    "authorization",
    "key",
    "passwd",
    "password",
    "secret",
    "token",
    "accesstoken",
    "accountkey",
    "apikey",
    "authtoken",
    "bearertoken",
    "clientsecret",
    "connectionstring",
    "credential",
    "credentials",
    "idtoken",
    "privatekey",
    "refreshtoken",
    "sastoken",
    "secretkey",
    "sharedaccesskey",
    "signingkey",
    "storagekey",
}
_CREDENTIAL_QUERY_KEYS = _CREDENTIAL_KEY_NAMES | {"sig", "signature"}


def _is_credential_key(key: object) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
    return normalized in _CREDENTIAL_KEY_NAMES or normalized.endswith(
        (
            "password",
            "passwd",
            "secret",
            "token",
            "credential",
            "credentials",
            "accountkey",
            "accesskey",
            "apikey",
            "clientkey",
            "encryptionkey",
            "privatekey",
            "secretkey",
            "signingkey",
            "storagekey",
        )
    )


def _has_credential_url_query(value: str) -> bool:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return False
    return any(
        query_value and re.sub(r"[^a-z0-9]", "", query_key.lower()) in _CREDENTIAL_QUERY_KEYS
        for query_key, query_value in parse_qsl(parsed.query, keep_blank_values=True)
    )


def _looks_like_opaque_bearer_token(value: str) -> bool:
    bearer_match = _BEARER_VALUE_PATTERN.fullmatch(value.strip())
    if bearer_match is None:
        return False
    payload = bearer_match.group("payload")
    explicit_path_prefixes = ("/", "\\", "./", "../", ".\\", "..\\")
    has_drive_prefix = re.match(r"^[A-Za-z]:[\\/]", payload) is not None
    separator_count = payload.count("/") + payload.count("\\")
    if payload.startswith(explicit_path_prefixes) or has_drive_prefix or separator_count >= 2:
        return False
    return _BEARER_TOKEN_PATTERN.fullmatch(payload) is not None


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class AzureConfig(StrictModel):
    account_url: str
    container: str
    blob_list: Path

    @field_validator("account_url")
    @classmethod
    def reject_url_userinfo(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("account_url must not contain credential userinfo")
        if _has_credential_url_query(value):
            raise ValueError("account_url must not contain credential query parameters")
        return value


class PathsConfig(StrictModel):
    work_dir: Path
    cache_dir: Path
    output_dir: Path

    @model_validator(mode="after")
    def validate_isolated_directories(self) -> PathsConfig:
        named_paths = {
            "work_dir": self.work_dir,
            "cache_dir": self.cache_dir,
            "output_dir": self.output_dir,
        }
        items = list(named_paths.items())
        for index, (first_name, first_path) in enumerate(items):
            for second_name, second_path in items[index + 1 :]:
                overlaps = (
                    first_path == second_path
                    or first_path in second_path.parents
                    or second_path in first_path.parents
                )
                if overlaps:
                    raise ValueError(f"unsafe path overlap between {first_name} and {second_name}")
        return self


class TopicsConfig(StrictModel):
    camera: str
    gnss: str


class DownsamplingConfig(StrictModel):
    target_fps: float = Field(gt=0)
    tolerance_ms: float = Field(ge=0)


class ImageConfig(StrictModel):
    jpeg_quality: int = Field(ge=1, le=100)


class GnssConfig(StrictModel):
    position_sigma_max_m: float = Field(ge=0)
    orientation_variance_max: float = Field(ge=0)
    sync_gap_max_ms: float = Field(ge=0)


class FrameValidityConfig(StrictModel):
    invalid_sample_policy: Literal["retain_for_audit", "drop"]
    invalidate_on: InvalidationRulesConfig


class InvalidationRulesConfig(StrictModel):
    missing_camera: bool = True
    invalid_gnss: bool = True
    sync_gap_exceeded: bool = True


class SanityChecksConfig(StrictModel):
    timestamp_policy: Literal["error", "quarantine", "warn"] = "quarantine"
    max_speed_mps: float = Field(default=70.0, gt=0)
    max_position_jump_m: float = Field(default=20.0, gt=0)


class ScenesConfig(StrictModel):
    mode: Literal["hybrid", "fixed", "annotation"]
    min_duration_s: float = Field(gt=0)
    max_duration_s: float = Field(gt=0)
    min_samples: int = Field(ge=1)
    max_sample_gap_ms: float = Field(ge=0)
    skip_between_scenes_s: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_duration_range(self) -> ScenesConfig:
        if self.max_duration_s < self.min_duration_s:
            raise ValueError("max_duration_s must be greater than or equal to min_duration_s")
        return self


class AnnotationsConfig(StrictModel):
    path: Path
    match_tolerance_ms: float = Field(ge=0)
    before_s: float = Field(ge=0)
    after_s: float = Field(ge=0)


class TagsConfig(StrictModel):
    stationary_speed_mps: float = Field(ge=0)
    turn_angle_deg: float = Field(ge=0, le=180)


class FiltersConfig(StrictModel):
    min_valid_ratio: float = Field(ge=0, le=1)
    required_tags: list[str]


class SamplingConfig(StrictModel):
    fraction: float = Field(gt=0, le=1)
    max_scenes: int | None = Field(default=None, ge=1)


class ScenarioRuleConfig(StrictModel):
    name: str = Field(min_length=1)
    required_tags: list[str] = Field(default_factory=list)
    excluded_tags: list[str] = Field(default_factory=list)
    sampling: SamplingConfig


class ScenariosConfig(StrictModel):
    seed: int
    rules: list[ScenarioRuleConfig]


class SplitConfig(StrictModel):
    test_fraction: float = Field(gt=0, lt=1)
    seed: int
    stratify: bool


class ExecutionConfig(StrictModel):
    workers: int = Field(ge=1)
    allow_partial_export: bool


class QuarantineConfig(StrictModel):
    enabled: bool = True
    directory: Path = Path("quarantine")
    manifest_name: str = Field(default="rejected.jsonl", min_length=1)


class PublicationConfig(StrictModel):
    version: str
    refuse_overwrite: bool


class GlobalConfig(StrictModel):
    schema_version: Literal["1.0"]
    azure: AzureConfig
    paths: PathsConfig
    topics: TopicsConfig
    downsampling: DownsamplingConfig
    image: ImageConfig
    gnss: GnssConfig
    frame_validity: FrameValidityConfig
    sanity_checks: SanityChecksConfig
    scenes: ScenesConfig
    annotations: AnnotationsConfig
    tags: TagsConfig
    filters: FiltersConfig
    scenarios: ScenariosConfig
    split: SplitConfig
    execution: ExecutionConfig
    quarantine: QuarantineConfig = Field(default_factory=QuarantineConfig)
    publication: PublicationConfig

    @model_validator(mode="before")
    @classmethod
    def reject_embedded_credentials(cls, value: object) -> object:
        def inspect(item: object, location: str = "config") -> None:
            if isinstance(item, dict):
                for key, nested in item.items():
                    if _is_credential_key(key):
                        raise ValueError(
                            f"embedded credential key is prohibited at {location}.{key}"
                        )
                    inspect(nested, f"{location}.{key}")
            elif isinstance(item, list):
                for index, nested in enumerate(item):
                    inspect(nested, f"{location}[{index}]")
            elif isinstance(item, str):
                lowered = item.lower()
                has_secret_assignment = any(
                    marker in lowered
                    for marker in (
                        "accountkey=",
                        "sharedaccesssignature=",
                        "client_secret=",
                        "api_key=",
                    )
                )
                if (
                    has_secret_assignment
                    or _has_credential_url_query(item)
                    or _looks_like_opaque_bearer_token(item)
                    or _AZURE_ACCOUNT_KEY_PATTERN.search(item) is not None
                    or _JWT_PATTERN.search(item) is not None
                    or re.fullmatch(r"sk-[A-Za-z0-9_-]{20,}", item) is not None
                    or "-----begin private key-----" in lowered
                ):
                    raise ValueError(f"embedded credential value is prohibited at {location}")

        inspect(value)
        return value


def load_config(path: Path) -> GlobalConfig:
    """Load configuration and resolve paths relative to its JSON file."""
    config_path = path.resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    quarantine_data = data.setdefault("quarantine", {})
    if isinstance(quarantine_data, dict):
        quarantine_data.setdefault("directory", "quarantine")
    for section, field in (
        ("azure", "blob_list"),
        ("paths", "work_dir"),
        ("paths", "cache_dir"),
        ("paths", "output_dir"),
        ("annotations", "path"),
        ("quarantine", "directory"),
    ):
        section_data = data.get(section)
        if isinstance(section_data, dict) and isinstance(section_data.get(field), str):
            value = Path(section_data[field])
            section_data[field] = str(value if value.is_absolute() else (base / value).resolve())
    return GlobalConfig.model_validate(data)
