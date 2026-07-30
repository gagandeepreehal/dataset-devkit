"""Comprehensive deterministic validation and secure dataset finalization."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from contextlib import suppress
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from PIL import Image

from dataset_devkit.config import GlobalConfig
from dataset_devkit.export import NUSCENES_VERSION, OFFICIAL_TABLES
from dataset_devkit.provenance import canonical_hash, canonical_json

_MANIFEST = "mz_extensions/content_manifest.json"
_QUATERNION_TOLERANCE = 1e-9
_LEAKAGE_WARNING = (
    "Scene-level splitting can leak neighboring temporal context when chronologically "
    "adjacent scenes from one recording are assigned to different splits."
)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


@dataclass(frozen=True, order=True)
class ValidationFinding:
    """One deterministic validation observation."""

    severity: Literal["error", "warning"]
    code: str
    location: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    """Immutable validation outcome with deterministic findings and counts."""

    succeeded: bool
    findings: tuple[ValidationFinding, ...]
    checks: tuple[str, ...]
    table_counts: tuple[tuple[str, int], ...]
    resolved_config_hash: str | None
    content_hash: str | None

    def to_extension(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "state": "succeeded" if self.succeeded else "failed",
            "succeeded": self.succeeded,
            "report": {
                "checks": list(self.checks),
                "resolved_config_sha256": self.resolved_config_hash,
                "table_counts": {key: value for key, value in self.table_counts},
                "findings": [
                    {
                        "severity": item.severity,
                        "code": item.code,
                        "location": item.location,
                        "message": item.message,
                    }
                    for item in self.findings
                ],
            },
        }


class DatasetValidationError(RuntimeError):
    """Raised when strict validation or finalization fails."""

    def __init__(self, report: ValidationReport, message: str | None = None) -> None:
        self.report = report
        first = next((item for item in report.findings if item.severity == "error"), None)
        super().__init__(
            message or ("dataset validation failed" if first is None else first.message)
        )


def _load_json(path: Path, findings: list[ValidationFinding], location: str) -> object | None:
    try:
        current = path.lstat()
        if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
            raise OSError("not a single-link regular file")
        return cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        findings.append(ValidationFinding("error", "json", location, f"malformed JSON: {error}"))
        return None


def _records(
    value: object | None, table: str, findings: list[ValidationFinding]
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        findings.append(
            ValidationFinding("error", "table_shape", table, "table must be an array of objects")
        )
        return []
    return cast(list[dict[str, Any]], value)


def _tokens(
    tables: dict[str, list[dict[str, Any]]], findings: list[ValidationFinding]
) -> dict[str, dict[str, dict[str, Any]]]:
    indexes: dict[str, dict[str, dict[str, Any]]] = {}
    global_tokens: dict[str, str] = {}
    for name in OFFICIAL_TABLES:
        index: dict[str, dict[str, Any]] = {}
        for ordinal, record in enumerate(tables[name]):
            token = record.get("token")
            location = f"{name}[{ordinal}].token"
            if not isinstance(token, str) or not token:
                findings.append(
                    ValidationFinding("error", "token", location, "token must be nonempty string")
                )
                continue
            if token in index or token in global_tokens:
                findings.append(
                    ValidationFinding(
                        "error", "token_unique", location, "token is not globally unique"
                    )
                )
                continue
            index[token] = record
            global_tokens[token] = name
        indexes[name] = index
    return indexes


def _foreign_keys(
    tables: dict[str, list[dict[str, Any]]],
    indexes: dict[str, dict[str, dict[str, Any]]],
    findings: list[ValidationFinding],
) -> None:
    rules = {
        "calibrated_sensor": (("sensor_token", "sensor"),),
        "scene": (
            ("log_token", "log"),
            ("first_sample_token", "sample"),
            ("last_sample_token", "sample"),
        ),
        "sample": (("scene_token", "scene"),),
        "sample_data": (
            ("sample_token", "sample"),
            ("ego_pose_token", "ego_pose"),
            ("calibrated_sensor_token", "calibrated_sensor"),
        ),
    }
    for table, fields in rules.items():
        for ordinal, record in enumerate(tables[table]):
            for field, target in fields:
                value = record.get(field)
                if not isinstance(value, str) or value not in indexes[target]:
                    findings.append(
                        ValidationFinding(
                            "error",
                            "foreign_key",
                            f"{table}[{ordinal}].{field}",
                            f"missing {target} reference",
                        )
                    )
    log_tokens = indexes["log"]
    for ordinal, record in enumerate(tables["map"]):
        values = record.get("log_tokens")
        if not isinstance(values, list) or any(
            not isinstance(item, str) or item not in log_tokens for item in values
        ):
            findings.append(
                ValidationFinding(
                    "error",
                    "foreign_key",
                    f"map[{ordinal}].log_tokens",
                    "map references missing logs",
                )
            )


def _chain(
    *,
    records: list[dict[str, Any]],
    ownership_field: str,
    owners: dict[str, dict[str, Any]],
    label: str,
    findings: list[ValidationFinding],
) -> None:
    by_token = {
        record.get("token"): record for record in records if isinstance(record.get("token"), str)
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        owner = record.get(ownership_field)
        if isinstance(owner, str):
            grouped.setdefault(owner, []).append(record)
    for owner, values in sorted(grouped.items()):
        if owner not in owners:
            continue
        for record in values:
            token = record.get("token")
            prev = record.get("prev")
            next_value = record.get("next")
            if not isinstance(prev, str) or not isinstance(next_value, str):
                findings.append(
                    ValidationFinding(
                        "error", f"{label}_chain", str(token), "prev and next must be strings"
                    )
                )
                continue
            if prev and (
                prev not in by_token
                or by_token[prev].get("next") != token
                or by_token[prev].get(ownership_field) != owner
            ):
                findings.append(
                    ValidationFinding(
                        "error", f"{label}_chain", str(token), "broken prev/next symmetry"
                    )
                )
            if next_value and (
                next_value not in by_token
                or by_token[next_value].get("prev") != token
                or by_token[next_value].get(ownership_field) != owner
            ):
                findings.append(
                    ValidationFinding(
                        "error", f"{label}_chain", str(token), "broken next/prev symmetry"
                    )
                )
        starts = [item for item in values if item.get("prev") == ""]
        if len(starts) != 1:
            findings.append(
                ValidationFinding(
                    "error", f"{label}_chain", owner, "chain must have exactly one start"
                )
            )
            continue
        seen: set[str] = set()
        timestamps: list[int] = []
        token = cast(str, starts[0].get("token"))
        while token:
            if token in seen or token not in by_token:
                findings.append(
                    ValidationFinding(
                        "error", f"{label}_chain", owner, "chain is cyclic or dangling"
                    )
                )
                break
            seen.add(token)
            item = by_token[token]
            timestamp = item.get("timestamp")
            if not isinstance(timestamp, int) or isinstance(timestamp, bool):
                findings.append(
                    ValidationFinding(
                        "error", "timestamp", token, "timestamp must be integer microseconds"
                    )
                )
            else:
                timestamps.append(timestamp)
            token = item.get("next") if isinstance(item.get("next"), str) else ""
        if len(seen) != len(values):
            findings.append(
                ValidationFinding(
                    "error", f"{label}_chain", owner, "chain does not cover every owned row"
                )
            )
        if any(
            current <= previous
            for previous, current in zip(timestamps, timestamps[1:], strict=False)
        ):
            findings.append(
                ValidationFinding(
                    "error", "timestamp", owner, "chain timestamps are not strictly increasing"
                )
            )


def _numeric(value: object, shape: tuple[int, ...]) -> bool:
    if not shape:
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        )
    return (
        isinstance(value, list)
        and len(value) == shape[0]
        and all(_numeric(item, shape[1:]) for item in value)
    )


def _geometry(tables: dict[str, list[dict[str, Any]]], findings: list[ValidationFinding]) -> None:
    for table in ("ego_pose", "calibrated_sensor"):
        for ordinal, record in enumerate(tables[table]):
            location = f"{table}[{ordinal}]"
            if not _numeric(record.get("translation"), (3,)):
                findings.append(
                    ValidationFinding(
                        "error",
                        "finite_pose",
                        location,
                        "translation must contain three finite numbers",
                    )
                )
            rotation = record.get("rotation")
            if not _numeric(rotation, (4,)):
                findings.append(
                    ValidationFinding(
                        "error", "quaternion", location, "rotation must contain four finite numbers"
                    )
                )
            else:
                norm = math.sqrt(sum(float(item) ** 2 for item in cast(list[float], rotation)))
                if abs(norm - 1.0) > _QUATERNION_TOLERANCE:
                    findings.append(
                        ValidationFinding(
                            "error", "quaternion", location, "rotation quaternion is not normalized"
                        )
                    )
            if table == "calibrated_sensor" and not _numeric(
                record.get("camera_intrinsic"), (3, 3)
            ):
                findings.append(
                    ValidationFinding(
                        "error",
                        "finite_calibration",
                        location,
                        "camera intrinsic must be finite 3x3",
                    )
                )


def _safe_relative(value: object, *, prefix: str, parts: int) -> str | None:
    if not isinstance(value, str):
        return None
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or len(path.parts) != parts
        or path.parts[0] != prefix
        or any(item in {"", ".", ".."} for item in path.parts)
    ):
        return None
    return value


def _images(
    root: Path, tables: dict[str, list[dict[str, Any]]], findings: list[ValidationFinding]
) -> None:
    for ordinal, record in enumerate(tables["sample_data"]):
        location = f"sample_data[{ordinal}]"
        filename = _safe_relative(record.get("filename"), prefix="samples", parts=3)
        if filename is None:
            findings.append(
                ValidationFinding("error", "filename", location, "unsafe sample_data filename")
            )
            continue
        path = root / filename
        try:
            current = path.lstat()
            if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
                raise OSError("image is not a single-link regular file")
            data = path.read_bytes()
            after = path.lstat()
            if (current.st_dev, current.st_ino, current.st_size) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
            ):
                raise OSError("image identity changed")
            with Image.open(BytesIO(data)) as image:
                image.load()
                if image.format != "JPEG" or image.size != (
                    record.get("width"),
                    record.get("height"),
                ):
                    raise OSError("image format or dimensions differ")
        except (OSError, ValueError) as error:
            findings.append(
                ValidationFinding("error", "image", location, f"invalid image: {error}")
            )
    for ordinal, record in enumerate(tables["map"]):
        location = f"map[{ordinal}]"
        filename = _safe_relative(record.get("filename"), prefix="maps", parts=2)
        if filename is None:
            findings.append(
                ValidationFinding("error", "filename", location, "unsafe map filename")
            )
            continue
        try:
            path = root / filename
            current = path.lstat()
            if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
                raise OSError("map is not a single-link regular file")
            with Image.open(path) as image:
                image.load()
                if image.format != "PNG" or image.mode != "L" or image.size != (1, 1):
                    raise OSError("compatibility mask must be a 1x1 grayscale PNG")
        except (OSError, ValueError) as error:
            findings.append(
                ValidationFinding("error", "compatibility_map", location, str(error))
            )


def _sha256_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_split_extension(
    split: object,
    config: GlobalConfig | None,
    tables: dict[str, list[dict[str, Any]]],
    scene_to_source: dict[str, str],
    source_blob_paths: dict[str, str],
    pipeline_audit: object,
    findings: list[ValidationFinding],
) -> None:
    def fail(message: str) -> None:
        findings.append(
            ValidationFinding("error", "split_integrity", "split", message)
        )

    expected_keys = {
        "schema_version",
        "assignments",
        "strata",
        "adjacent_scene_leakage",
        "seed",
        "test_fraction",
        "stratify",
        "population_count",
        "train_count",
        "test_count",
        "rounding_rule",
        "config_fingerprint",
        "upstream_fingerprint",
        "candidate_fingerprint",
        "graph_fingerprint",
    }
    if not isinstance(split, dict) or set(split) != expected_keys:
        fail("split schema fields differ")
        return
    assignments = split["assignments"]
    if not isinstance(assignments, list) or any(
        not isinstance(item, dict)
        or set(item)
        != {"scene_token", "source_digest", "primary_scenario", "split", "rank"}
        for item in assignments
    ):
        fail("split assignments are malformed")
        return
    typed = cast(list[dict[str, Any]], assignments)
    identities = [(item.get("scene_token"), item.get("source_digest")) for item in typed]
    scene_tokens = {item.get("token") for item in tables["scene"]}
    if (
        len(identities) != len(set(identities))
        or {item[0] for item in identities} != scene_tokens
        or any(
            not isinstance(item[0], str)
            or not isinstance(item[1], str)
            or scene_to_source.get(item[0]) != item[1]
            for item in identities
        )
    ):
        fail("split assignment identities are not unique and complete")
        return
    seed = split.get("seed")
    fraction = split.get("test_fraction")
    stratify = split.get("stratify")
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or not isinstance(fraction, (int, float))
        or isinstance(fraction, bool)
        or not 0 < float(fraction) < 1
        or not isinstance(stratify, bool)
    ):
        fail("split seed, fraction, or stratify value is malformed")
        return
    if config is not None and (
        seed != config.split.seed
        or float(fraction) != config.split.test_fraction
        or stratify != config.split.stratify
        or split.get("config_fingerprint")
        != canonical_hash(config.split.model_dump(mode="json"))
    ):
        fail("split configuration commitment differs from resolved config")
    population = len(typed)
    target = int(
        (Decimal(population) * Decimal(str(fraction)) + Decimal("0.5")).to_integral_value(
            rounding=ROUND_FLOOR
        )
    )
    actual_test = sum(item.get("split") == "test" for item in typed)
    if (
        split.get("schema_version") != 1
        or split.get("population_count") != population
        or split.get("test_count") != target
        or split.get("train_count") != population - target
        or actual_test != target
        or split.get("rounding_rule") != "floor(n * test_fraction + 0.5)"
        or any(item.get("split") not in {"train", "test"} for item in typed)
    ):
        fail("split counts, rounding, or train/test values differ")
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in typed:
        scenario = item.get("primary_scenario")
        if not isinstance(scenario, str) or not scenario:
            fail("split primary scenario is malformed")
            continue
        groups.setdefault(scenario, []).append(item)
        expected_rank = canonical_hash(
            {
                "seed": seed,
                "primary_scenario": scenario if stratify else "__all__",
                "scene_token": item["scene_token"],
                "source_digest": item["source_digest"],
            }
        )
        if item.get("rank") != expected_rank:
            fail("split assignment rank differs")
    strata = split.get("strata")
    stratum_keys = {
        "primary_scenario",
        "population_count",
        "expected_test_count",
        "target_test_count",
        "actual_test_count",
        "actual_train_count",
        "stratification_applied",
        "fallback_reason",
    }
    if not isinstance(strata, list) or any(
        not isinstance(item, dict) or set(item) != stratum_keys for item in strata
    ):
        fail("split strata audit is malformed")
    else:
        by_scenario = {
            item.get("primary_scenario"): item for item in cast(list[dict[str, Any]], strata)
        }
        if len(by_scenario) != len(strata) or set(by_scenario) != set(groups):
            fail("split strata identities differ")
        for scenario, items in groups.items():
            audit = by_scenario.get(scenario)
            if audit is None:
                continue
            count = len(items)
            tests = sum(item.get("split") == "test" for item in items)
            applied = bool(stratify and count >= 2 and 0 < tests < count)
            fallback = (
                "stratification_disabled"
                if not stratify
                else "stratum_too_small"
                if count < 2
                else "global_target_prevents_both_sides"
                if not applied
                else None
            )
            if (
                audit.get("population_count") != count
                or audit.get("expected_test_count")
                != str(Decimal(count) * Decimal(str(fraction)))
                or audit.get("target_test_count") != tests
                or audit.get("actual_test_count") != tests
                or audit.get("actual_train_count") != count - tests
                or audit.get("stratification_applied") is not applied
                or audit.get("fallback_reason") != fallback
            ):
                fail("split stratum audit differs from assignments")
    for key in ("upstream_fingerprint", "candidate_fingerprint", "graph_fingerprint"):
        if not _sha256_text(split.get(key)):
            fail(f"split {key} is not a SHA-256 commitment")
    if isinstance(pipeline_audit, dict):
        selection = pipeline_audit.get("selection")
        if isinstance(selection, dict) and selection.get("candidate_fingerprint") != split.get(
            "candidate_fingerprint"
        ):
            fail("split candidate fingerprint differs from pipeline audit")
    leakage = split.get("adjacent_scene_leakage")
    if (
        not isinstance(leakage, dict)
        or set(leakage) != {"checked", "warning", "cross_split_pairs"}
        or leakage.get("checked") is not True
        or leakage.get("warning") != _LEAKAGE_WARNING
        or not isinstance(leakage.get("cross_split_pairs"), list)
    ):
        fail("adjacent-scene leakage audit is malformed")
        return
    split_by_scene = {cast(str, item["scene_token"]): item["split"] for item in typed}
    sample_by_token = {item.get("token"): item for item in tables["sample"]}
    expected_pairs: list[dict[str, object]] = []
    for source in sorted(set(scene_to_source.values())):
        source_scenes = [
            scene
            for scene in tables["scene"]
            if scene_to_source.get(cast(str, scene.get("token"))) == source
        ]
        ordered = sorted(
            source_scenes,
            key=lambda scene: cast(
                int, sample_by_token[scene.get("first_sample_token")]["timestamp"]
            ),
        )
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            earlier_token = cast(str, earlier["token"])
            later_token = cast(str, later["token"])
            if split_by_scene[earlier_token] == split_by_scene[later_token]:
                continue
            expected_pairs.append(
                {
                    "source_digest": source,
                    "source_blob_path": source_blob_paths.get(source),
                    "earlier_scene_token": earlier_token,
                    "later_scene_token": later_token,
                    "earlier_last_timestamp_us": sample_by_token[
                        earlier["last_sample_token"]
                    ]["timestamp"],
                    "later_first_timestamp_us": sample_by_token[
                        later["first_sample_token"]
                    ]["timestamp"],
                    "earlier_split": split_by_scene[earlier_token],
                    "later_split": split_by_scene[later_token],
                }
            )
    actual_pairs = cast(list[object], leakage["cross_split_pairs"])
    if len(actual_pairs) != len(expected_pairs):
        fail("adjacent-scene leakage pair count differs")
    else:
        for actual, expected in zip(actual_pairs, expected_pairs, strict=True):
            earlier_ns = (
                actual.get("earlier_last_timestamp_ns") if isinstance(actual, dict) else None
            )
            later_ns = actual.get("later_first_timestamp_ns") if isinstance(actual, dict) else None
            if not isinstance(actual, dict) or (
                actual.get("source_digest") != expected["source_digest"]
                or actual.get("source_blob_path") != expected["source_blob_path"]
                or actual.get("earlier_scene_token") != expected["earlier_scene_token"]
                or actual.get("later_scene_token") != expected["later_scene_token"]
                or actual.get("earlier_split") != expected["earlier_split"]
                or actual.get("later_split") != expected["later_split"]
                or not isinstance(earlier_ns, int)
                or isinstance(earlier_ns, bool)
                or earlier_ns // 1000 != expected["earlier_last_timestamp_us"]
                or not isinstance(later_ns, int)
                or isinstance(later_ns, bool)
                or later_ns // 1000 != expected["later_first_timestamp_us"]
            ):
                fail("adjacent-scene leakage pair differs from chronology")


def _extensions(
    root: Path,
    tables: dict[str, list[dict[str, Any]]],
    config: GlobalConfig | None,
    findings: list[ValidationFinding],
    *,
    finalized: bool,
) -> None:
    scenes = {
        cast(str, item.get("token"))
        for item in tables["scene"]
        if isinstance(item.get("token"), str)
    }
    samples = {
        cast(str, item.get("token"))
        for item in tables["sample"]
        if isinstance(item.get("token"), str)
    }
    sample_data = {
        cast(str, item.get("token"))
        for item in tables["sample_data"]
        if isinstance(item.get("token"), str)
    }
    loaded: dict[str, object | None] = {}
    for name in (
        "recordings",
        "gnss",
        "validity",
        "validation",
        "tags",
        "annotations",
        "split",
        "config",
        "content_manifest",
        "pipeline_audit",
    ):
        loaded[name] = _load_json(
            root / "mz_extensions" / f"{name}.json", findings, f"mz_extensions/{name}.json"
        )
    recordings = loaded["recordings"]
    log_tokens = {
        cast(str, item.get("token"))
        for item in tables["log"]
        if isinstance(item.get("token"), str)
    }
    recording_sources: set[str] = set()
    log_to_source: dict[str, str] = {}
    source_blob_paths: dict[str, str] = {}
    recording_keys = {"source", "source_digest", "log_token", "channels"}
    source_keys = {"account_url", "container", "blob_path", "etag", "size"}
    channel_keys = {"original", "normalized"}
    if isinstance(recordings, list) and all(
        isinstance(item, dict) and set(item) == recording_keys for item in recordings
    ):
        for item in cast(list[dict[str, Any]], recordings):
            source = item.get("source_digest")
            log_token = item.get("log_token")
            if (
                not isinstance(source, str)
                or not source
                or source in recording_sources
                or not isinstance(log_token, str)
                or log_token not in log_tokens
                or log_token in log_to_source
            ):
                findings.append(
                    ValidationFinding(
                        "error",
                        "extension_reference",
                        "recordings",
                        "recording source/log ownership is malformed",
                    )
                )
                continue
            recording_sources.add(source)
            log_to_source[log_token] = source
            source_value = item.get("source")
            blob_path = source_value.get("blob_path") if isinstance(source_value, dict) else None
            channels = item.get("channels")
            channel_rows = (
                cast(list[dict[str, Any]], channels) if isinstance(channels, list) else []
            )
            channel_identities = [row.get("normalized") for row in channel_rows]
            if (
                not isinstance(source_value, dict)
                or set(source_value) != source_keys
                or not isinstance(blob_path, str)
                or source in source_blob_paths
                or not isinstance(channels, list)
                or any(not isinstance(row, dict) or set(row) != channel_keys for row in channels)
                or any(
                    not isinstance(row.get("original"), str)
                    or not isinstance(row.get("normalized"), str)
                    for row in channel_rows
                )
                or len(channel_identities) != len(set(channel_identities))
            ):
                findings.append(
                    ValidationFinding(
                        "error",
                        "extension_reference",
                        "recordings",
                        "recording source payload is malformed or duplicated",
                    )
                )
            else:
                source_blob_paths[source] = blob_path
    else:
        findings.append(
            ValidationFinding(
                "error",
                "extension_reference",
                "recordings",
                "recordings must contain exact-shaped unique rows",
            )
        )
    if set(log_to_source) != log_tokens:
        findings.append(
            ValidationFinding(
                "error",
                "extension_reference",
                "recordings",
                "recording log coverage differs from official logs",
            )
        )
    scene_to_source = {
        cast(str, scene["token"]): log_to_source.get(cast(str, scene.get("log_token")), "")
        for scene in tables["scene"]
        if isinstance(scene.get("token"), str)
    }
    scene_extension_keys = {
        "validity": {
            "scene_token",
            "source_digest",
            "scene_valid_ratio",
            "source_gnss_valid_ratio",
            "camera_coverage_ratio",
            "camera_coverage_by_channel",
            "max_abs_sync_error_ms",
            "mean_abs_sync_error_ms",
            "sample_data",
            "samples",
        },
        "tags": {"scene_token", "source_digest", "computed_tags"},
    }
    for name in ("validity", "tags"):
        value = loaded[name]
        typed_rows = cast(list[dict[str, Any]], value) if isinstance(value, list) else []
        identities = [item.get("scene_token") for item in typed_rows if isinstance(item, dict)]
        if (
            not isinstance(value, list)
            or any(
                not isinstance(item, dict) or set(item) != scene_extension_keys[name]
                for item in value
            )
            or len(identities) != len(value)
            or len(identities) != len(set(identities))
            or set(identities) != scenes
        ):
            findings.append(
                ValidationFinding(
                    "error",
                    "extension_reference",
                    name,
                    "scene extension coverage differs from scenes",
                )
            )
        if isinstance(value, list) and any(
            not isinstance(item, dict)
            or (
                item.get("source_digest") not in recording_sources
                or scene_to_source.get(cast(str, item.get("scene_token")))
                != item.get("source_digest")
            )
            for item in value
        ):
            findings.append(
                ValidationFinding(
                    "error",
                    "extension_reference",
                    name,
                    "scene extension source differs from recording ownership",
                )
            )
    gnss = loaded["gnss"]
    gnss_keys = {
        "sample_data_token",
        "scene_token",
        "source_digest",
        "original_channel",
        "normalized_channel",
        "timestamp_ns",
        "image_sha256",
        "available",
        "latitude_deg",
        "longitude_deg",
        "height_m",
        "quaternion_wxyz",
        "fraction",
        "sync_gap_before_ns",
        "sync_gap_after_ns",
        "source_validity",
        "position_uncertainty",
        "orientation_uncertainty",
        "before",
        "after",
    }
    gnss_identities = (
        [item.get("sample_data_token") for item in gnss if isinstance(item, dict)]
        if isinstance(gnss, list)
        else []
    )
    if (
        not isinstance(gnss, list)
        or any(not isinstance(item, dict) or set(item) != gnss_keys for item in gnss)
        or len(gnss_identities) != len(gnss)
        or len(gnss_identities) != len(set(gnss_identities))
        or set(gnss_identities) != sample_data
    ):
        findings.append(
            ValidationFinding(
                "error", "extension_reference", "gnss", "GNSS coverage differs from sample_data"
            )
        )
    elif any(
        not isinstance(item.get("scene_token"), str)
        or item.get("scene_token") not in scenes
        or item.get("source_digest") != scene_to_source.get(cast(str, item.get("scene_token")))
        for item in cast(list[dict[str, Any]], gnss)
    ):
        findings.append(
            ValidationFinding(
                "error",
                "extension_reference",
                "gnss",
                "GNSS scene/source reference differs from official ownership",
            )
        )
    split = loaded["split"]
    assignments = split.get("assignments") if isinstance(split, dict) else None
    assigned = (
        [item.get("scene_token") for item in assignments if isinstance(item, dict)]
        if isinstance(assignments, list)
        else []
    )
    if (
        set(assigned) != scenes
        or len(assigned) != len(set(assigned))
        or any(
            item.get("split") not in {"train", "test"}
            for item in assignments
            if isinstance(item, dict)
        )
        if isinstance(assignments, list)
        else True
    ):
        findings.append(
            ValidationFinding(
                "error", "split_integrity", "split", "split must fully and disjointly assign scenes"
            )
        )
    elif isinstance(assignments, list) and any(
        isinstance(item, dict)
        and item.get("source_digest")
        != scene_to_source.get(cast(str, item.get("scene_token")))
        for item in assignments
    ):
        findings.append(
            ValidationFinding(
                "error",
                "split_integrity",
                "split",
                "split scene/source ownership differs",
            )
        )
    annotations = loaded["annotations"]
    annotation_scenes = annotations.get("scenes") if isinstance(annotations, dict) else None
    annotation_scene_keys = {
        "scene_token",
        "source_digest",
        "human_labels",
        "annotation_refs",
        "annotation_window_ref",
    }
    annotation_scene_ids = (
        [item.get("scene_token") for item in annotation_scenes if isinstance(item, dict)]
        if isinstance(annotation_scenes, list)
        else []
    )
    if (
        not isinstance(annotations, dict)
        or set(annotations) != {"scenes", "records", "matches", "windows"}
        or
        not isinstance(annotation_scenes, list)
        or any(
            not isinstance(item, dict) or set(item) != annotation_scene_keys
            for item in annotation_scenes
        )
        or len(annotation_scene_ids) != len(annotation_scenes)
        or len(annotation_scene_ids) != len(set(annotation_scene_ids))
        or set(annotation_scene_ids) != scenes
    ):
        findings.append(
            ValidationFinding(
                "error", "extension_reference", "annotations", "annotation scene coverage differs"
            )
        )
    elif any(
        isinstance(item, dict)
        and item.get("source_digest")
        != scene_to_source.get(cast(str, item.get("scene_token")))
        for item in annotation_scenes
    ):
        findings.append(
            ValidationFinding(
                "error",
                "extension_reference",
                "annotations",
                "annotation scene/source ownership differs",
            )
        )
    if isinstance(annotations, dict):
        records = annotations.get("records")
        matches = annotations.get("matches")
        windows = annotations.get("windows")
        record_keys = {
            "source_digest",
            "token",
            "line_number",
            "blob_path",
            "timestamp_ns",
            "labels",
        }
        match_keys = {
            "source_digest",
            "annotation_token",
            "line_number",
            "matched",
            "sample_timestamp_ns",
            "signed_error_ns",
            "absolute_error_ns",
            "reason",
        }
        window_keys = {
            "source_digest",
            "token",
            "annotation_tokens",
            "first_timestamp_ns",
            "last_timestamp_ns",
            "first_sample_timestamp_ns",
            "last_sample_timestamp_ns",
            "labels",
        }
        records_typed = cast(list[dict[str, Any]], records) if isinstance(records, list) else []
        matches_typed = cast(list[dict[str, Any]], matches) if isinstance(matches, list) else []
        windows_typed = cast(list[dict[str, Any]], windows) if isinstance(windows, list) else []
        record_ids = [item.get("token") for item in records_typed if isinstance(item, dict)]
        match_ids = [
            (item.get("source_digest"), item.get("annotation_token"))
            for item in matches_typed
            if isinstance(item, dict)
        ]
        window_ids = [item.get("token") for item in windows_typed if isinstance(item, dict)]
        malformed_annotations = (
            not isinstance(records, list)
            or any(not isinstance(item, dict) or set(item) != record_keys for item in records)
            or len(record_ids) != len(records)
            or len(record_ids) != len(set(record_ids))
            or not isinstance(matches, list)
            or any(not isinstance(item, dict) or set(item) != match_keys for item in matches)
            or len(match_ids) != len(matches)
            or len(match_ids) != len(set(match_ids))
            or not isinstance(windows, list)
            or any(not isinstance(item, dict) or set(item) != window_keys for item in windows)
            or len(window_ids) != len(windows)
            or len(window_ids) != len(set(window_ids))
        )
        record_source = {
            cast(str, item.get("token")): item.get("source_digest") for item in records_typed
        }
        window_source = {
            cast(str, item.get("token")): item.get("source_digest") for item in windows_typed
        }
        malformed_annotations = malformed_annotations or any(
            item.get("source_digest") not in recording_sources
            or source_blob_paths.get(cast(str, item.get("source_digest")))
            != item.get("blob_path")
            for item in records_typed
        )
        malformed_annotations = malformed_annotations or any(
            item.get("source_digest") not in recording_sources
            or record_source.get(cast(str, item.get("annotation_token")))
            != item.get("source_digest")
            for item in matches_typed
        )
        malformed_annotations = malformed_annotations or any(
            item.get("source_digest") not in recording_sources
            or not isinstance(item.get("annotation_tokens"), list)
            or len(item.get("annotation_tokens", []))
            != len(set(item.get("annotation_tokens", [])))
            or any(
                record_source.get(cast(str, token)) != item.get("source_digest")
                for token in item.get("annotation_tokens", [])
            )
            for item in windows_typed
        )
        if isinstance(annotation_scenes, list):
            malformed_annotations = malformed_annotations or any(
                not isinstance(item, dict)
                or not isinstance(item.get("annotation_refs"), list)
                or len(item.get("annotation_refs", []))
                != len(set(item.get("annotation_refs", [])))
                or any(
                    record_source.get(cast(str, token)) != item.get("source_digest")
                    for token in item.get("annotation_refs", [])
                )
                or (
                    item.get("annotation_window_ref") != ""
                    and window_source.get(cast(str, item.get("annotation_window_ref")))
                    != item.get("source_digest")
                )
                for item in annotation_scenes
            )
        if malformed_annotations:
            findings.append(
                ValidationFinding(
                    "error",
                    "extension_reference",
                    "annotations",
                    "annotation rows are malformed, duplicated, or cross recording ownership",
                )
            )
    validation = loaded["validation"]
    if finalized and (
        not isinstance(validation, dict)
        or validation.get("state") != "succeeded"
        or validation.get("succeeded") is not True
        or not isinstance(validation.get("report"), dict)
    ):
        findings.append(
            ValidationFinding(
                "error",
                "validation_state",
                "validation",
                "finalized dataset must contain succeeded validation evidence",
            )
        )
    pipeline_audit = loaded["pipeline_audit"]
    audit_keys = (
        set(pipeline_audit) if isinstance(pipeline_audit, dict) else set()
    )
    selection_audit = (
        pipeline_audit.get("selection") if isinstance(pipeline_audit, dict) else None
    )
    if (
        not isinstance(pipeline_audit, dict)
        or audit_keys
        not in (
            {"schema_version", "filter", "selection"},
            {"schema_version", "blob_order", "failed_recordings", "filter", "selection"},
        )
        or pipeline_audit.get("schema_version") != 1
        or not isinstance(pipeline_audit.get("filter"), dict)
        or not isinstance(selection_audit, dict)
        or set(selection_audit)
        != {
            "candidate_fingerprint",
            "config_fingerprint",
            "rules_fingerprint",
            "assignments",
            "rule_audits",
            "unselected",
        }
        or any(
            not _sha256_text(selection_audit.get(key))
            for key in ("candidate_fingerprint", "config_fingerprint", "rules_fingerprint")
        )
        or any(
            not isinstance(selection_audit.get(key), list)
            for key in ("assignments", "rule_audits", "unselected")
        )
    ):
        findings.append(
            ValidationFinding(
                "error",
                "pipeline_audit",
                "pipeline_audit",
                "pipeline audit report is malformed",
            )
        )
    _validate_split_extension(
        split,
        config,
        tables,
        scene_to_source,
        source_blob_paths,
        pipeline_audit,
        findings,
    )
    if config is not None:
        required = {
            f"CAM_{str(channel).upper().replace('-', '_')}"
            for channel in config.frame_validity.required_cameras
        }
        calibration_by_token = {item.get("token"): item for item in tables["calibrated_sensor"]}
        sensor_by_token = {item.get("token"): item for item in tables["sensor"]}
        channels_by_sample: dict[str, set[str]] = {token: set() for token in samples}
        for item in tables["sample_data"]:
            calibration = calibration_by_token.get(item.get("calibrated_sensor_token"), {})
            sensor = sensor_by_token.get(calibration.get("sensor_token"), {})
            channel = sensor.get("channel")
            sample_token = item.get("sample_token")
            if (
                isinstance(channel, str)
                and isinstance(sample_token, str)
                and sample_token in channels_by_sample
            ):
                channels_by_sample[sample_token].add(channel)
        for token, channels in channels_by_sample.items():
            if not required <= channels:
                findings.append(
                    ValidationFinding(
                        "error", "camera_coverage", token, "required camera coverage is incomplete"
                    )
                )


def _walk_regular_files(root: Path) -> dict[str, tuple[int, str]]:
    entries: dict[str, tuple[int, str]] = {}
    root_stat = root.lstat()
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("dataset root is not a directory")
    for directory, names, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in names:
            value = base / name
            if not stat.S_ISDIR(value.lstat().st_mode):
                raise ValueError(
                    f"symlink or unsafe directory: {value.relative_to(root).as_posix()}"
                )
        for name in filenames:
            path = base / name
            relative = path.relative_to(root).as_posix()
            if relative == _MANIFEST:
                continue
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ValueError(f"symlink or unsafe hardlink: {relative}")
            digest = hashlib.sha256()
            descriptor = os.open(path, os.O_RDONLY | _NOFOLLOW)
            try:
                opened = os.fstat(descriptor)
                while chunk := os.read(descriptor, 1024 * 1024):
                    digest.update(chunk)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            listed = path.lstat()
            def identity(item: os.stat_result) -> tuple[int, int, int, int]:
                return item.st_dev, item.st_ino, item.st_size, item.st_nlink
            if (
                identity(before) != identity(opened)
                or identity(opened) != identity(after)
                or identity(after) != identity(listed)
            ):
                raise ValueError(f"path changed while hashing: {relative}")
            entries[relative] = (before.st_size, digest.hexdigest())
    return entries


def build_content_manifest(root: Path) -> dict[str, object]:
    """Hash every published regular file except the manifest itself."""
    entries = _walk_regular_files(root)
    serialized = [
        {"path": path, "size": size, "sha256": digest}
        for path, (size, digest) in sorted(entries.items())
    ]
    return {
        "schema_version": 1,
        "algorithm": "sha256",
        "excluded_paths": [_MANIFEST],
        "entries": serialized,
        "root_sha256": canonical_hash(serialized),
    }


def _verify_manifest(root: Path, findings: list[ValidationFinding]) -> str | None:
    value = _load_json(root / _MANIFEST, findings, _MANIFEST)
    try:
        actual = build_content_manifest(root)
    except ValueError as error:
        findings.append(ValidationFinding("error", "manifest", _MANIFEST, str(error)))
        return None
    if value != actual:
        findings.append(
            ValidationFinding(
                "error",
                "manifest",
                _MANIFEST,
                "manifest entries or root hash differ from published content",
            )
        )
        return None
    return cast(str, actual["root_sha256"])


def _official_smoke(root: Path, version: str, tables: dict[str, list[dict[str, Any]]]) -> None:
    if not tables["scene"]:
        raise ValueError("official smoke requires at least one selected scene")
    from nuscenes.nuscenes import NuScenes  # type: ignore[import-untyped]

    dataset = NuScenes(version=version, dataroot=str(root), verbose=False)
    scene = dataset.get("scene", cast(str, tables["scene"][0]["token"]))
    sample = dataset.get("sample", cast(str, scene["first_sample_token"]))
    camera_tokens = [value for key, value in sample["data"].items() if key.startswith("CAM_")]
    if not camera_tokens:
        raise ValueError("official SDK query found no camera sample data")
    row = dataset.get("sample_data", camera_tokens[0])
    with Image.open(root / row["filename"]) as image:
        image.load()


def validate_dataset(
    dataroot: str | Path,
    version: str = NUSCENES_VERSION,
    *,
    official_smoke: bool = True,
    verify_manifest: bool = True,
    raise_on_error: bool = False,
) -> ValidationReport:
    """Validate a complete exported dataset, accumulating deterministic findings."""
    root = Path(dataroot).absolute()
    findings: list[ValidationFinding] = []
    if version != NUSCENES_VERSION:
        findings.append(
            ValidationFinding("error", "version", "version", f"version must be {NUSCENES_VERSION}")
        )
    tables: dict[str, list[dict[str, Any]]] = {}
    for name in OFFICIAL_TABLES:
        tables[name] = _records(
            _load_json(root / version / f"{name}.json", findings, f"{version}/{name}.json"),
            name,
            findings,
        )
    indexes = _tokens(tables, findings)
    _foreign_keys(tables, indexes, findings)
    for ordinal, sensor_record in enumerate(tables["sensor"]):
        channel = sensor_record.get("channel")
        if (
            not isinstance(channel, str)
            or not channel.startswith("CAM_")
            or sensor_record.get("modality") != "camera"
        ):
            findings.append(
                ValidationFinding(
                    "error", "sensor", f"sensor[{ordinal}]", "invalid camera sensor"
                )
            )
    sensor_channels = [item.get("channel") for item in tables["sensor"]]
    if len(sensor_channels) != len(set(map(str, sensor_channels))):
        findings.append(
            ValidationFinding("error", "sensor", "sensor", "camera channels are not unique")
        )
    pose_references = [item.get("ego_pose_token") for item in tables["sample_data"]]
    if (
        len(pose_references) != len(set(map(str, pose_references)))
        or set(pose_references) != set(indexes["ego_pose"])
    ):
        findings.append(
            ValidationFinding(
                "error",
                "ownership",
                "ego_pose",
                "ego poses must be owned one-to-one by sample_data",
            )
        )
    for ordinal, item in enumerate(tables["sample_data"]):
        width = item.get("width")
        height = item.get("height")
        if (
            item.get("fileformat") != "jpg"
            or item.get("is_key_frame") is not True
            or not isinstance(width, int)
            or isinstance(width, bool)
            or width <= 0
            or not isinstance(height, int)
            or isinstance(height, bool)
            or height <= 0
        ):
            findings.append(
                ValidationFinding(
                    "error",
                    "sample_data_shape",
                    f"sample_data[{ordinal}]",
                    "camera sample_data metadata is malformed",
                )
            )
    _chain(
        records=tables["sample"],
        ownership_field="scene_token",
        owners=indexes["scene"],
        label="sample",
        findings=findings,
    )
    scenes_by_token = indexes["scene"]
    for token, scene in sorted(scenes_by_token.items()):
        owned = [item for item in tables["sample"] if item.get("scene_token") == token]
        starts = [item for item in owned if item.get("prev") == ""]
        ends = [item for item in owned if item.get("next") == ""]
        if (
            len(owned) != scene.get("nbr_samples")
            or len(starts) != 1
            or len(ends) != 1
            or starts[0].get("token") != scene.get("first_sample_token")
            or ends[0].get("token") != scene.get("last_sample_token")
        ):
            findings.append(
                ValidationFinding(
                    "error",
                    "scene_endpoint",
                    token,
                    "scene endpoints or sample count differ from chain",
                )
            )
    calibrations = indexes["calibrated_sensor"]
    sensors = indexes["sensor"]
    sample_data_by_channel: dict[str, list[dict[str, Any]]] = {}
    for item in tables["sample_data"]:
        calibration_key = item.get("calibrated_sensor_token")
        calibration = (
            calibrations.get(calibration_key)
            if isinstance(calibration_key, str)
            else None
        )
        sensor_key = None if calibration is None else calibration.get("sensor_token")
        sensor = sensors.get(sensor_key) if isinstance(sensor_key, str) else None
        channel = None if sensor is None else sensor.get("channel")
        if isinstance(channel, str):
            sample_data_by_channel.setdefault(channel, []).append(item)
    samples_by_token = indexes["sample"]
    for channel, records in sorted(sample_data_by_channel.items()):
        # A calibrated sensor can be reused by several scenes. Official sample_data
        # chains restart at every scene, so validate ownership by scene and channel.
        chain_records: list[dict[str, Any]] = []
        owners: dict[str, dict[str, Any]] = {}
        for record in records:
            sample_key = record.get("sample_token")
            sample = samples_by_token.get(sample_key) if isinstance(sample_key, str) else None
            if sample is None:
                continue
            owner = f"{sample.get('scene_token')}:{channel}"
            owners[owner] = {}
            chain_records.append({**record, "chain_owner": owner})
        _chain(
            records=chain_records,
            ownership_field="chain_owner",
            owners=owners,
            label="sample_data",
            findings=findings,
        )
    _geometry(tables, findings)
    _images(root, tables, findings)
    config_value = _load_json(
        root / "mz_extensions/config.json", findings, "mz_extensions/config.json"
    )
    config: GlobalConfig | None = None
    if isinstance(config_value, dict):
        try:
            # The exported JSON representation contains JSON strings for Path and
            # exact Decimal values; validate that representation using JSON-mode
            # coercions while the model itself remains strict for Python callers.
            config = GlobalConfig.model_validate(config_value, strict=False)
        except Exception as error:
            findings.append(
                ValidationFinding(
                    "error",
                    "resolved_config",
                    "mz_extensions/config.json",
                    f"invalid resolved config: {error}",
                )
            )
    _extensions(root, tables, config, findings, finalized=verify_manifest)
    content_hash = _verify_manifest(root, findings) if verify_manifest else None
    if official_smoke and not any(item.severity == "error" for item in findings):
        try:
            _official_smoke(root, version, tables)
        except Exception as error:
            findings.append(
                ValidationFinding(
                    "error", "official_smoke", "NuScenes", f"official SDK smoke failed: {error}"
                )
            )
    ordered = tuple(sorted(set(findings)))
    resolved_config_hash = (
        canonical_hash(config.model_dump(mode="json")) if config is not None else None
    )
    report = ValidationReport(
        not any(item.severity == "error" for item in ordered),
        ordered,
        ("tables", "foreign_keys", "chains", "images", "geometry", "extensions", "official_sdk"),
        tuple((name, len(tables[name])) for name in OFFICIAL_TABLES),
        resolved_config_hash,
        content_hash,
    )
    if raise_on_error and not report.succeeded:
        raise DatasetValidationError(report)
    return report


def _atomic_json(root: Path, relative: str, value: object) -> None:
    parts = PurePosixPath(relative).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("unsafe finalization path")
    directory = root.joinpath(*parts[:-1])
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | _NOFOLLOW)
    temporary = f".{parts[-1]}.{os.getpid()}.tmp"
    descriptor = -1
    try:
        destination = directory / parts[-1]
        current = destination.lstat()
        if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
            raise ValueError("finalization destination is not a safe regular file")
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW, 0o600, dir_fd=directory_fd
        )
        content = (canonical_json(value) + "\n").encode()
        os.write(descriptor, content)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, parts[-1], src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=directory_fd)
        os.close(directory_fd)


def finalize_dataset(
    dataroot: str | Path,
    version: str = NUSCENES_VERSION,
    *,
    official_smoke: bool = True,
) -> ValidationReport:
    """Validate, record success, create the manifest last, and revalidate."""
    root = Path(dataroot).absolute()
    preliminary = validate_dataset(
        root, version, official_smoke=official_smoke, verify_manifest=False
    )
    if not preliminary.succeeded:
        raise DatasetValidationError(preliminary)
    _atomic_json(root, "mz_extensions/validation.json", preliminary.to_extension())
    _atomic_json(root, _MANIFEST, build_content_manifest(root))
    final = validate_dataset(root, version, official_smoke=official_smoke, verify_manifest=True)
    if not final.succeeded:
        failed = ValidationReport(
            False,
            final.findings,
            final.checks,
            final.table_counts,
            final.resolved_config_hash,
            None,
        )
        _atomic_json(root, "mz_extensions/validation.json", failed.to_extension())
        raise DatasetValidationError(failed)
    return final
