"""Versioned global configuration models."""

from __future__ import annotations

import json
import math
import re
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import parse_qsl, urlsplit
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema, field_validator, model_validator

from dataset_devkit.identifiers import SafeSegment

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
_AZURE_BLOB_HOST_SUFFIXES = (
    ".blob.core.windows.net",
    ".blob.core.usgovcloudapi.net",
    ".blob.core.chinacloudapi.cn",
    ".blob.core.cloudapi.de",
)
_PATH_FIELD_LOCATIONS = {
    "config.azure.blob_list",
    "config.paths.work_dir",
    "config.paths.cache_dir",
    "config.paths.output_dir",
    "config.annotations.path",
    "config.quarantine.directory",
}
_HOST_LABEL_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


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
    return _BEARER_TOKEN_PATTERN.fullmatch(payload) is not None


def _is_azure_blob_service_host(hostname: str | None) -> bool:
    if hostname is None:
        return False
    for suffix in _AZURE_BLOB_HOST_SUFFIXES:
        if not hostname.endswith(suffix):
            continue
        prefix = hostname[: -len(suffix)]
        labels = prefix.split(".")
        if len(labels) == 2 and labels[1] == "privatelink":
            labels.pop()
        return len(labels) == 1 and _HOST_LABEL_PATTERN.fullmatch(labels[0]) is not None
    return False


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _validate_tag_list(values: list[str]) -> list[str]:
    if any(not value.strip() or value != value.strip() for value in values):
        raise ValueError("tags must be nonempty and have no surrounding whitespace")
    if len(values) != len(set(values)):
        raise ValueError("tags must be unique")
    return values


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class ConfigRootError(ValueError):
    """Raised when a JSON configuration does not contain an object root."""


class AzureConfig(StrictModel):
    account_url: str
    container: str
    blob_list: Path

    @field_validator("account_url")
    @classmethod
    def reject_url_userinfo(cls, value: str) -> str:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        try:
            _ = parsed.port
        except ValueError as error:
            raise ValueError("account_url must contain a valid port") from error
        if parsed.scheme != "https" or not _is_azure_blob_service_host(hostname):
            raise ValueError("account_url must be an HTTPS Azure Blob service URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("account_url must not contain credential userinfo")
        if parsed.path not in {"", "/"}:
            raise ValueError("account_url must not contain a container or blob path")
        if "#" in value:
            raise ValueError("account_url must not contain a fragment")
        if _has_credential_url_query(value):
            raise ValueError("account_url must not contain credential query parameters")
        return value

    @field_validator("container")
    @classmethod
    def validate_container_name(cls, value: str) -> str:
        valid = (
            3 <= len(value) <= 63
            and re.fullmatch(r"[a-z0-9-]+", value) is not None
            and value[0].isalnum()
            and value[-1].isalnum()
            and "--" not in value
        )
        if not valid:
            raise ValueError("container must follow Azure container naming rules")
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
                if _paths_overlap(first_path, second_path):
                    raise ValueError(f"unsafe path overlap between {first_name} and {second_name}")
        return self


class TopicsConfig(StrictModel):
    camera: str
    gnss: str

    @field_validator("camera", "gnss")
    @classmethod
    def validate_nonblank_topic(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("topic name must be nonblank")
        return value


class DownsamplingConfig(StrictModel):
    target_fps: float = Field(gt=0)
    tolerance_ms: float = Field(ge=0)


class ImageConfig(StrictModel):
    jpeg_quality: int = Field(ge=1, le=100)


class GnssConfig(StrictModel):
    position_sigma_max_m: float = Field(ge=0)
    orientation_variance_max: float = Field(ge=0)
    sync_gap_max_ms: float = Field(ge=0)


class InvalidationRulesConfig(StrictModel):
    gnss_source_invalid: bool = True
    position_sigma_exceeded: bool = True
    orientation_variance_exceeded: bool = True
    gnss_sync_gap_exceeded: bool = True
    camera_timestamp_non_monotonic: bool = True
    camera_timestamp_gap_exceeded: bool = True
    missing_required_camera: bool = True
    grid_miss: bool = True


class FrameValidityConfig(StrictModel):
    invalid_sample_policy: Literal["retain_for_audit", "drop"] = "retain_for_audit"
    required_cameras: list[SafeSegment] = Field(
        default_factory=list, json_schema_extra={"uniqueItems": True}
    )
    camera_timestamp_gap_max_ms: float = Field(gt=0)
    invalidate_on: InvalidationRulesConfig = Field(default_factory=InvalidationRulesConfig)

    @field_validator("required_cameras")
    @classmethod
    def validate_required_cameras(cls, values: list[SafeSegment]) -> list[SafeSegment]:
        if len(values) != len(set(values)):
            raise ValueError("required_cameras must be unique exact camera identities")
        return values


class SanityChecksConfig(StrictModel):
    empty_selected_grid: Literal["error", "warn", "off"] = "error"
    empty_final_candidates: Literal["error", "warn", "off"] = "error"
    all_gnss_sources_invalid: Literal["error", "warn", "off"] = "warn"
    zero_required_camera_coverage: Literal["error", "warn", "off"] = "error"


def _exact_nanoseconds(value: Decimal, multiplier: int, field_name: str) -> int:
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")
    converted = value * multiplier
    if converted != converted.to_integral_value():
        raise ValueError(f"{field_name} must resolve to an exact integer number of nanoseconds")
    return int(converted)


PositiveExactDecimal = Annotated[
    Decimal,
    Field(gt=0),
    WithJsonSchema({"type": "number", "exclusiveMinimum": 0}),
]
NonnegativeExactDecimal = Annotated[
    Decimal,
    Field(ge=0),
    WithJsonSchema({"type": "number", "minimum": 0}),
]


class ScenesConfig(StrictModel):
    mode: Literal["automatic", "annotation_only", "hybrid"] = "hybrid"
    dataset_namespace: UUID
    min_duration_s: PositiveExactDecimal
    max_duration_s: PositiveExactDecimal
    min_samples: int = Field(ge=1)
    max_sample_gap_ms: NonnegativeExactDecimal
    skip_between_scenes_s: NonnegativeExactDecimal

    @model_validator(mode="after")
    def validate_duration_range(self) -> ScenesConfig:
        if self.max_duration_s < self.min_duration_s:
            raise ValueError("max_duration_s must be greater than or equal to min_duration_s")
        return self

    @field_validator("dataset_namespace", mode="before")
    @classmethod
    def validate_namespace(cls, value: object) -> UUID:
        if isinstance(value, UUID):
            return value
        if not isinstance(value, str):
            raise ValueError("dataset_namespace must be a UUID string")
        try:
            return UUID(value)
        except ValueError as error:
            raise ValueError("dataset_namespace must be a valid UUID") from error

    @property
    def min_duration_ns(self) -> int:
        return _exact_nanoseconds(self.min_duration_s, 1_000_000_000, "min_duration_s")

    @property
    def max_duration_ns(self) -> int:
        return _exact_nanoseconds(self.max_duration_s, 1_000_000_000, "max_duration_s")

    @property
    def max_sample_gap_ns(self) -> int:
        return _exact_nanoseconds(self.max_sample_gap_ms, 1_000_000, "max_sample_gap_ms")

    @property
    def skip_between_scenes_ns(self) -> int:
        return _exact_nanoseconds(
            self.skip_between_scenes_s, 1_000_000_000, "skip_between_scenes_s"
        )

    @model_validator(mode="after")
    def validate_exact_nanoseconds(self) -> ScenesConfig:
        _ = (
            self.min_duration_ns,
            self.max_duration_ns,
            self.max_sample_gap_ns,
            self.skip_between_scenes_ns,
        )
        return self


class AnnotationsConfig(StrictModel):
    path: Path
    match_tolerance_ms: NonnegativeExactDecimal
    before_s: NonnegativeExactDecimal
    after_s: NonnegativeExactDecimal

    @property
    def match_tolerance_ns(self) -> int:
        return _exact_nanoseconds(self.match_tolerance_ms, 1_000_000, "match_tolerance_ms")

    @property
    def before_ns(self) -> int:
        return _exact_nanoseconds(self.before_s, 1_000_000_000, "before_s")

    @property
    def after_ns(self) -> int:
        return _exact_nanoseconds(self.after_s, 1_000_000_000, "after_s")

    @model_validator(mode="after")
    def validate_exact_nanoseconds(self) -> AnnotationsConfig:
        _ = self.match_tolerance_ns, self.before_ns, self.after_ns
        return self


class TagsConfig(StrictModel):
    reference_camera_channel: SafeSegment | None = None
    reference_camera_policy: Literal["require", "lexicographic_fallback"] = (
        "lexicographic_fallback"
    )
    stationary_speed_mps: float = Field(ge=0)
    minimum_movement_m: float = Field(ge=0)
    straight_max_heading_change_deg: float = Field(ge=0, lt=180)
    curvature_min_heading_change_deg: float = Field(gt=0, lt=180)
    turn_min_heading_change_deg: float = Field(gt=0, le=180)

    @model_validator(mode="after")
    def validate_heading_threshold_order(self) -> TagsConfig:
        if self.reference_camera_channel is None and self.reference_camera_policy == "require":
            raise ValueError(
                "reference_camera_channel is required when reference_camera_policy is require"
            )
        if not (
            self.straight_max_heading_change_deg
            < self.curvature_min_heading_change_deg
            < self.turn_min_heading_change_deg
        ):
            raise ValueError(
                "straight_max_heading_change_deg < curvature_min_heading_change_deg "
                "< turn_min_heading_change_deg is required"
            )
        return self


class FiltersConfig(StrictModel):
    min_duration_s: float | None = Field(default=None, ge=0)
    max_duration_s: float | None = Field(default=None, ge=0)
    min_scene_valid_ratio: float | None = Field(default=None, ge=0, le=1)
    max_scene_valid_ratio: float | None = Field(default=None, ge=0, le=1)
    min_source_gnss_valid_ratio: float | None = Field(default=None, ge=0, le=1)
    max_source_gnss_valid_ratio: float | None = Field(default=None, ge=0, le=1)
    min_camera_coverage_ratio: float | None = Field(default=None, ge=0, le=1)
    max_camera_coverage_ratio: float | None = Field(default=None, ge=0, le=1)
    min_camera_coverage_by_channel: dict[SafeSegment, float] = Field(
        default_factory=dict, json_schema_extra={"additionalProperties": False}
    )
    max_camera_coverage_by_channel: dict[SafeSegment, float] = Field(
        default_factory=dict, json_schema_extra={"additionalProperties": False}
    )
    max_sync_error_ms: float | None = Field(default=None, ge=0)
    min_distance_m: float | None = Field(default=None, ge=0)
    max_distance_m: float | None = Field(default=None, ge=0)
    required_any_tags: list[str] = Field(
        default_factory=list, json_schema_extra={"uniqueItems": True}
    )
    required_all_tags: list[str] = Field(
        default_factory=list, json_schema_extra={"uniqueItems": True}
    )
    excluded_tags: list[str] = Field(
        default_factory=list, json_schema_extra={"uniqueItems": True}
    )
    required_any_labels: list[str] = Field(
        default_factory=list, json_schema_extra={"uniqueItems": True}
    )
    required_all_labels: list[str] = Field(
        default_factory=list, json_schema_extra={"uniqueItems": True}
    )
    excluded_labels: list[str] = Field(
        default_factory=list, json_schema_extra={"uniqueItems": True}
    )
    blacklisted_scene_tokens: list[str] = Field(
        default_factory=list, json_schema_extra={"uniqueItems": True}
    )
    blacklisted_source_digests: list[str] = Field(
        default_factory=list, json_schema_extra={"uniqueItems": True}
    )
    blacklisted_blob_paths: list[str] = Field(
        default_factory=list, json_schema_extra={"uniqueItems": True}
    )

    @field_validator(
        "required_any_tags",
        "required_all_tags",
        "excluded_tags",
        "required_any_labels",
        "required_all_labels",
        "excluded_labels",
        "blacklisted_scene_tokens",
        "blacklisted_source_digests",
        "blacklisted_blob_paths",
    )
    @classmethod
    def validate_string_lists(cls, values: list[str]) -> list[str]:
        return _validate_tag_list(values)

    @field_validator("min_camera_coverage_by_channel", "max_camera_coverage_by_channel")
    @classmethod
    def validate_channel_ratios(cls, values: dict[SafeSegment, float]) -> dict[SafeSegment, float]:
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
            raise ValueError("per-channel camera coverage ratios must be finite in [0, 1]")
        return values

    @model_validator(mode="after")
    def validate_ranges_and_predicates(self) -> FiltersConfig:
        for minimum_name, maximum_name in (
            ("min_duration_s", "max_duration_s"),
            ("min_scene_valid_ratio", "max_scene_valid_ratio"),
            ("min_source_gnss_valid_ratio", "max_source_gnss_valid_ratio"),
            ("min_camera_coverage_ratio", "max_camera_coverage_ratio"),
            ("min_distance_m", "max_distance_m"),
        ):
            minimum = getattr(self, minimum_name)
            maximum = getattr(self, maximum_name)
            if minimum is not None and maximum is not None and minimum > maximum:
                raise ValueError(f"{minimum_name} must be <= {maximum_name}")
        for channel in set(self.min_camera_coverage_by_channel) & set(
            self.max_camera_coverage_by_channel
        ):
            if (
                self.min_camera_coverage_by_channel[channel]
                > self.max_camera_coverage_by_channel[channel]
            ):
                raise ValueError(f"camera coverage minimum exceeds maximum for {channel}")
        for kind in ("tags", "labels"):
            required_any = set(getattr(self, f"required_any_{kind}"))
            required_all = set(getattr(self, f"required_all_{kind}"))
            excluded = set(getattr(self, f"excluded_{kind}"))
            if (required_any | required_all) & excluded:
                raise ValueError(f"required and excluded {kind} overlap")
        return self


class ScenarioRuleConfig(StrictModel):
    name: str = Field(min_length=1)
    quota: int = Field(ge=0)
    required_any_tags: list[str] = Field(
        default_factory=list, json_schema_extra={"uniqueItems": True}
    )
    required_all_tags: list[str] = Field(
        default_factory=list, json_schema_extra={"uniqueItems": True}
    )
    excluded_tags: list[str] = Field(
        default_factory=list, json_schema_extra={"uniqueItems": True}
    )
    required_any_labels: list[str] = Field(
        default_factory=list, json_schema_extra={"uniqueItems": True}
    )
    required_all_labels: list[str] = Field(
        default_factory=list, json_schema_extra={"uniqueItems": True}
    )
    excluded_labels: list[str] = Field(
        default_factory=list, json_schema_extra={"uniqueItems": True}
    )
    filters: FiltersConfig | None = None

    @field_validator("name")
    @classmethod
    def validate_rule_name(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("rule name must be nonblank with no surrounding whitespace")
        return value

    @field_validator(
        "required_any_tags",
        "required_all_tags",
        "excluded_tags",
        "required_any_labels",
        "required_all_labels",
        "excluded_labels",
    )
    @classmethod
    def validate_tags(cls, values: list[str]) -> list[str]:
        return _validate_tag_list(values)

    @model_validator(mode="after")
    def validate_tag_sets_do_not_overlap(self) -> ScenarioRuleConfig:
        for kind in ("tags", "labels"):
            required = set(getattr(self, f"required_any_{kind}")) | set(
                getattr(self, f"required_all_{kind}")
            )
            if required & set(getattr(self, f"excluded_{kind}")):
                raise ValueError(f"required and excluded {kind} overlap")
        return self


class ScenariosConfig(StrictModel):
    seed: int
    strict_quotas: bool = True
    rules: list[ScenarioRuleConfig]

    @model_validator(mode="after")
    def validate_unique_rule_names(self) -> ScenariosConfig:
        names = [rule.name for rule in self.rules]
        if len(names) != len(set(names)):
            raise ValueError("scenario rule names must be unique")
        return self


class SplitConfig(StrictModel):
    test_fraction: float = Field(gt=0, lt=1)
    seed: int
    stratify: bool


class ExecutionConfig(StrictModel):
    workers: int = Field(ge=1)
    allow_partial_export: bool


class QuarantineConfig(StrictModel):
    enabled: Literal[True] = True
    directory: Path = Path("quarantine")
    manifest_name: SafeSegment = "rejected.jsonl"


class PublicationConfig(StrictModel):
    version: SafeSegment
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

    @model_validator(mode="after")
    def validate_quarantine_isolation(self) -> GlobalConfig:
        quarantine_dir = self.quarantine.directory
        for name, destructive_path in (
            ("work_dir", self.paths.work_dir),
            ("cache_dir", self.paths.cache_dir),
            ("output_dir", self.paths.output_dir),
        ):
            if _paths_overlap(quarantine_dir, destructive_path):
                raise ValueError(f"quarantine directory overlaps {name}")
        return self

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
            elif isinstance(item, (str, Path)):
                text = str(item)
                lowered = text.lower()
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
                    or _has_credential_url_query(text)
                    or (
                        location not in _PATH_FIELD_LOCATIONS
                        and _looks_like_opaque_bearer_token(text)
                    )
                    or _AZURE_ACCOUNT_KEY_PATTERN.search(text) is not None
                    or _JWT_PATTERN.search(text) is not None
                    or re.fullmatch(r"sk-[A-Za-z0-9_-]{20,}", text) is not None
                    or "-----begin private key-----" in lowered
                ):
                    raise ValueError(f"embedded credential value is prohibited at {location}")

        inspect(value)
        return value


def load_config(path: Path) -> GlobalConfig:
    """Load configuration and resolve paths relative to its JSON file."""
    config_path = path.resolve()
    text = config_path.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ConfigRootError("configuration must be a top-level JSON object")
    exact_data = json.loads(text, parse_float=Decimal)
    if isinstance(exact_data, dict):
        for section, fields in (
            (
                "scenes",
                (
                    "min_duration_s",
                    "max_duration_s",
                    "max_sample_gap_ms",
                    "skip_between_scenes_s",
                ),
            ),
            ("annotations", ("match_tolerance_ms", "before_s", "after_s")),
        ):
            normal_section = data.get(section)
            exact_section = exact_data.get(section)
            if isinstance(normal_section, dict) and isinstance(exact_section, dict):
                for field in fields:
                    exact_value = exact_section.get(field)
                    if isinstance(exact_value, int) and not isinstance(exact_value, bool):
                        exact_value = Decimal(exact_value)
                    normal_section[field] = exact_value
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
            section_data[field] = value if value.is_absolute() else (base / value).resolve()
    return GlobalConfig.model_validate(data)
