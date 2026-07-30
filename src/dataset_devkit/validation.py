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
from dataset_devkit.publication import StagingLease, hash_regular_files_fd

_MANIFEST = "mz_extensions/content_manifest.json"
_QUATERNION_TOLERANCE = 1e-9
_LEAKAGE_WARNING = (
    "Scene-level splitting can leak neighboring temporal context when chronologically "
    "adjacent scenes from one recording are assigned to different splits."
)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_CHECKS = (
    "tables",
    "foreign_keys",
    "chains",
    "images",
    "geometry",
    "extensions",
    "official_sdk",
)


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


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _canonical_strings(value: object) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(item, str) and item for item in value)
        and value == sorted(set(value))
    )


def _finite_json_tree(value: object) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if _finite_number(value):
        return True
    if isinstance(value, list):
        return all(_finite_json_tree(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _finite_json_tree(item)
            for key, item in value.items()
        )
    return False


def _regular_file_sha256(path: Path) -> str | None:
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            return None
        descriptor = os.open(path, os.O_RDONLY | _NOFOLLOW)
        try:
            opened = os.fstat(descriptor)
            digest = hashlib.sha256()
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        def identity(item: os.stat_result) -> tuple[int, int, int, int]:
            return item.st_dev, item.st_ino, item.st_size, item.st_nlink
        if identity(before) != identity(opened) or identity(opened) != identity(after):
            return None
        return digest.hexdigest()
    except OSError:
        return None


def _validate_pipeline_audit(
    pipeline_audit: object,
    config: GlobalConfig | None,
    scene_to_source: dict[str, str],
    source_blob_paths: dict[str, str],
    split: object,
    findings: list[ValidationFinding],
) -> list[dict[str, Any]] | None:
    def fail(message: str) -> None:
        findings.append(
            ValidationFinding("error", "pipeline_audit", "pipeline_audit", message)
        )

    if not isinstance(pipeline_audit, dict):
        fail("pipeline audit report is malformed")
        return None
    expected_keys = {"schema_version", "filter", "selection", "graph_scene_sequence"}
    if "blob_order" in pipeline_audit or "failed_recordings" in pipeline_audit:
        expected_keys.update({"blob_order", "failed_recordings"})
    filter_audit = pipeline_audit.get("filter")
    selection = pipeline_audit.get("selection")
    sequence = pipeline_audit.get("graph_scene_sequence")
    if (
        set(pipeline_audit) != expected_keys
        or pipeline_audit.get("schema_version") != 1
        or not isinstance(filter_audit, dict)
        or set(filter_audit) != {"accepted", "rejected"}
        or not isinstance(selection, dict)
        or set(selection)
        != {
            "candidate_fingerprint",
            "config_fingerprint",
            "rules_fingerprint",
            "assignments",
            "rule_audits",
            "unselected",
        }
        or not isinstance(sequence, list)
    ):
        fail("pipeline audit report is malformed")
        return None
    accepted = filter_audit.get("accepted")
    rejected = filter_audit.get("rejected")
    assignments = selection.get("assignments")
    unselected = selection.get("unselected")
    if (
        any(
            not _sha256_text(selection.get(key))
            for key in ("candidate_fingerprint", "config_fingerprint", "rules_fingerprint")
        )
        or config is not None
        and selection.get("config_fingerprint")
        != canonical_hash(config.scenarios.model_dump(mode="json"))
        or not isinstance(accepted, list)
        or not isinstance(rejected, list)
        or not isinstance(assignments, list)
        or not isinstance(selection.get("rule_audits"), list)
        or not isinstance(unselected, list)
    ):
        fail("pipeline audit selection or filter evidence is malformed")
        return None
    if any(
        not isinstance(item, dict)
        or set(item) != {"scene_token", "source_digest"}
        or not all(isinstance(item.get(key), str) for key in item)
        for item in accepted
    ) or any(
        not isinstance(item, dict)
        or set(item) != {"scene_token", "source_digest", "reasons"}
        or not isinstance(item.get("scene_token"), str)
        or not isinstance(item.get("source_digest"), str)
        or not isinstance(item.get("reasons"), list)
        for item in rejected
    ):
        fail("pipeline filter membership evidence is malformed")
        return None
    assignment_keys = {
        "scene_token",
        "source_digest",
        "primary_scenario",
        "rule_index",
        "rank",
    }
    if any(
        not isinstance(item, dict)
        or set(item) != assignment_keys
        or not isinstance(item.get("scene_token"), str)
        or not isinstance(item.get("source_digest"), str)
        or not isinstance(item.get("primary_scenario"), str)
        or not isinstance(item.get("rule_index"), int)
        or isinstance(item.get("rule_index"), bool)
        or not _sha256_text(item.get("rank"))
        for item in assignments
    ) or any(
        not isinstance(item, dict)
        or set(item) != {"scene_token", "source_digest", "reason", "matching_rules"}
        or not isinstance(item.get("scene_token"), str)
        or not isinstance(item.get("source_digest"), str)
        or not isinstance(item.get("reason"), str)
        or not isinstance(item.get("matching_rules"), list)
        or any(not isinstance(rule, str) for rule in item.get("matching_rules", []))
        for item in unselected
    ):
        fail("pipeline selection membership evidence is malformed")
        return None
    sequence_keys = {
        "source_digest",
        "source_blob_path",
        "scene_token",
        "ordinal",
        "first_timestamp_ns",
        "last_timestamp_ns",
    }
    if any(
        not isinstance(item, dict)
        or set(item) != sequence_keys
        or not isinstance(item.get("source_digest"), str)
        or not isinstance(item.get("source_blob_path"), str)
        or not isinstance(item.get("scene_token"), str)
        or not isinstance(item.get("ordinal"), int)
        or isinstance(item.get("ordinal"), bool)
        or not isinstance(item.get("first_timestamp_ns"), int)
        or isinstance(item.get("first_timestamp_ns"), bool)
        or not isinstance(item.get("last_timestamp_ns"), int)
        or isinstance(item.get("last_timestamp_ns"), bool)
        or item.get("first_timestamp_ns", 0) > item.get("last_timestamp_ns", 0)
        for item in sequence
    ):
        fail("pipeline complete graph scene sequence is malformed")
        return None
    typed_sequence = cast(list[dict[str, Any]], sequence)
    canonical_sequence = sorted(
        typed_sequence,
        key=lambda item: (
            item["source_digest"],
            item["first_timestamp_ns"],
            item["last_timestamp_ns"],
            item["ordinal"],
            item["scene_token"],
        ),
    )
    accepted_ids = {(item["scene_token"], item["source_digest"]) for item in accepted}
    rejected_ids = {(item["scene_token"], item["source_digest"]) for item in rejected}
    selected_ids = {(item["scene_token"], item["source_digest"]) for item in assignments}
    unselected_ids = {(item["scene_token"], item["source_digest"]) for item in unselected}
    sequence_ids = {
        (item["scene_token"], item["source_digest"]) for item in typed_sequence
    }
    selected_sources = {source for _, source in selected_ids}
    sequence_sources = {source for _, source in sequence_ids}
    filtered_ids = accepted_ids | rejected_ids
    expected_sequence_ids = {
        identity for identity in filtered_ids if identity[1] in selected_sources
    }
    published_ids = {(token, source) for token, source in scene_to_source.items()}
    split_assignments = split.get("assignments") if isinstance(split, dict) else None
    split_ids = (
        {
            (item.get("scene_token"), item.get("source_digest"))
            for item in split_assignments
            if isinstance(item, dict)
        }
        if isinstance(split_assignments, list)
        else set()
    )
    duplicate = any(
        len(rows) != len(identities)
        for rows, identities in (
            (accepted, accepted_ids),
            (rejected, rejected_ids),
            (assignments, selected_ids),
            (unselected, unselected_ids),
            (typed_sequence, sequence_ids),
        )
    )
    if (
        duplicate
        or typed_sequence != canonical_sequence
        or accepted_ids & rejected_ids
        or selected_ids & unselected_ids
        or sequence_sources != selected_sources
        or sequence_ids != expected_sequence_ids
        or selected_ids | unselected_ids != accepted_ids
        or selected_ids != published_ids
        or selected_ids != split_ids
        or any(
            source_blob_paths.get(item["source_digest"]) != item["source_blob_path"]
            for item in typed_sequence
        )
        or any(
            scene_to_source.get(item["scene_token"]) != item["source_digest"]
            for item in assignments
        )
    ):
        fail("published, filtered, selected, split, and graph memberships differ")
        return None
    typed_split_assignments = cast(list[object], split_assignments)
    split_by_id = {
        (item.get("scene_token"), item.get("source_digest")): item
        for item in typed_split_assignments
        if isinstance(item, dict)
    }
    if any(
        split_by_id.get((item["scene_token"], item["source_digest"]), {}).get(
            "primary_scenario"
        )
        != item["primary_scenario"]
        for item in assignments
    ):
        fail("selection and split primary-scenario membership differs")
        return None
    return typed_sequence


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

    expected_test_identities: set[tuple[str, str]] = set()
    quotas: dict[str, int] = {}
    if len(groups) > 0 and sum(map(len, groups.values())) == population:
        if stratify:
            remainders: list[tuple[Decimal, str, str]] = []
            for scenario in sorted(groups):
                ideal = Decimal(len(groups[scenario])) * Decimal(str(fraction))
                floor = int(ideal.to_integral_value(rounding=ROUND_FLOOR))
                quotas[scenario] = floor
                tie = canonical_hash(
                    {
                        "seed": seed,
                        "primary_scenario": scenario,
                        "purpose": "apportion",
                    }
                )
                remainders.append((ideal - Decimal(floor), tie, scenario))
            remaining = target - sum(quotas.values())
            for _, _, scenario in sorted(
                remainders, key=lambda item: (-item[0], item[1], item[2])
            )[:remaining]:
                quotas[scenario] += 1

            for scenario in sorted(groups):
                count = len(groups[scenario])
                if count >= 2 and quotas[scenario] == 0:
                    donors = [
                        name
                        for name in groups
                        if quotas[name] > (1 if len(groups[name]) >= 2 else 0)
                    ]
                    if donors:
                        donor = min(
                            donors,
                            key=lambda name: canonical_hash(
                                {
                                    "seed": seed,
                                    "from": name,
                                    "to": scenario,
                                    "purpose": "rebalance",
                                }
                            ),
                        )
                        quotas[donor] -= 1
                        quotas[scenario] += 1
            for scenario in sorted(groups):
                count = len(groups[scenario])
                if count >= 2 and quotas[scenario] == count:
                    recipients = [
                        name
                        for name in groups
                        if quotas[name]
                        < (len(groups[name]) - 1 if len(groups[name]) >= 2 else 1)
                    ]
                    if recipients:
                        recipient = min(
                            recipients,
                            key=lambda name: canonical_hash(
                                {
                                    "seed": seed,
                                    "from": scenario,
                                    "to": name,
                                    "purpose": "rebalance",
                                }
                            ),
                        )
                        quotas[scenario] -= 1
                        quotas[recipient] += 1

            for scenario, items in groups.items():
                ranked = sorted(
                    items,
                    key=lambda item: (
                        canonical_hash(
                            {
                                "seed": seed,
                                "primary_scenario": scenario,
                                "scene_token": item["scene_token"],
                                "source_digest": item["source_digest"],
                            }
                        ),
                        (item["scene_token"], item["source_digest"]),
                    ),
                )
                expected_test_identities.update(
                    (item["scene_token"], item["source_digest"])
                    for item in ranked[: quotas[scenario]]
                )
        else:
            ranked = sorted(
                typed,
                key=lambda item: (
                    canonical_hash(
                        {
                            "seed": seed,
                            "primary_scenario": "__all__",
                            "scene_token": item["scene_token"],
                            "source_digest": item["source_digest"],
                        }
                    ),
                    (item["scene_token"], item["source_digest"]),
                ),
            )
            expected_test_identities.update(
                (item["scene_token"], item["source_digest"])
                for item in ranked[:target]
            )
        submitted_test_identities = {
            (item["scene_token"], item["source_digest"])
            for item in typed
            if item.get("split") == "test"
        }
        if submitted_test_identities != expected_test_identities:
            fail("split membership differs from deterministic apportionment")
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
            expected_tests = sum(
                (item["scene_token"], item["source_digest"])
                in expected_test_identities
                for item in items
            )
            target_tests = quotas.get(scenario, expected_tests)
            applied = bool(stratify and count >= 2 and 0 < expected_tests < count)
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
                or audit.get("target_test_count") != target_tests
                or audit.get("actual_test_count") != expected_tests
                or audit.get("actual_train_count") != count - expected_tests
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
    expected_pairs: list[dict[str, object]] = []
    graph_sequence = (
        pipeline_audit.get("graph_scene_sequence")
        if isinstance(pipeline_audit, dict)
        else None
    )
    sequence_fields = {
        "source_digest",
        "source_blob_path",
        "scene_token",
        "ordinal",
        "first_timestamp_ns",
        "last_timestamp_ns",
    }
    if not isinstance(graph_sequence, list) or any(
        not isinstance(item, dict)
        or set(item) != sequence_fields
        or not isinstance(item.get("source_digest"), str)
        or not isinstance(item.get("scene_token"), str)
        or not isinstance(item.get("first_timestamp_ns"), int)
        or isinstance(item.get("first_timestamp_ns"), bool)
        or not isinstance(item.get("last_timestamp_ns"), int)
        or isinstance(item.get("last_timestamp_ns"), bool)
        for item in graph_sequence
    ):
        return
    typed_sequence = cast(list[dict[str, Any]], graph_sequence)
    by_source: dict[str, list[dict[str, Any]]] = {}
    for item in typed_sequence:
        source = item.get("source_digest")
        if isinstance(source, str):
            by_source.setdefault(source, []).append(item)
    for source in sorted(by_source):
        for earlier, later in zip(by_source[source], by_source[source][1:], strict=False):
            earlier_token = cast(str, earlier["scene_token"])
            later_token = cast(str, later["scene_token"])
            if earlier_token not in split_by_scene or later_token not in split_by_scene:
                continue
            if split_by_scene[earlier_token] == split_by_scene[later_token]:
                continue
            expected_pairs.append(
                {
                    "source_digest": source,
                    "source_blob_path": source_blob_paths.get(source),
                    "earlier_scene_token": earlier_token,
                    "later_scene_token": later_token,
                    "earlier_last_timestamp_ns": earlier["last_timestamp_ns"],
                    "later_first_timestamp_ns": later["first_timestamp_ns"],
                    "earlier_split": split_by_scene[earlier_token],
                    "later_split": split_by_scene[later_token],
                }
            )
    actual_pairs = cast(list[object], leakage["cross_split_pairs"])
    if len(actual_pairs) != len(expected_pairs):
        fail("adjacent-scene leakage pair count differs")
    else:
        for actual, expected in zip(actual_pairs, expected_pairs, strict=True):
            if not isinstance(actual, dict) or (
                set(actual)
                != {
                    "source_digest",
                    "source_blob_path",
                    "earlier_scene_token",
                    "later_scene_token",
                    "earlier_last_timestamp_ns",
                    "later_first_timestamp_ns",
                    "earlier_split",
                    "later_split",
                }
                or actual.get("source_digest") != expected["source_digest"]
                or actual.get("source_blob_path") != expected["source_blob_path"]
                or actual.get("earlier_scene_token") != expected["earlier_scene_token"]
                or actual.get("later_scene_token") != expected["later_scene_token"]
                or actual.get("earlier_split") != expected["earlier_split"]
                or actual.get("later_split") != expected["later_split"]
                or actual.get("earlier_last_timestamp_ns")
                != expected["earlier_last_timestamp_ns"]
                or actual.get("later_first_timestamp_ns")
                != expected["later_first_timestamp_ns"]
            ):
                fail("adjacent-scene leakage pair differs from chronology")


def _extensions(
    root: Path,
    tables: dict[str, list[dict[str, Any]]],
    config: GlobalConfig | None,
    findings: list[ValidationFinding],
    *,
    finalized: bool,
) -> object | None:
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
    channels_by_source: dict[str, dict[str, str]] = {}
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
                channels_by_source[source] = {
                    cast(str, row["normalized"]): cast(str, row["original"])
                    for row in channel_rows
                }
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
            isinstance(item, dict)
            and (
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
    sample_by_token = {cast(str, item["token"]): item for item in tables["sample"]}
    data_by_token = {cast(str, item["token"]): item for item in tables["sample_data"]}
    calibration_by_token = {
        cast(str, item["token"]): item for item in tables["calibrated_sensor"]
    }
    sensor_by_token = {cast(str, item["token"]): item for item in tables["sensor"]}
    samples_by_scene: dict[str, set[str]] = {scene: set() for scene in scenes}
    data_by_scene: dict[str, set[str]] = {scene: set() for scene in scenes}
    normalized_channel_by_data: dict[str, str] = {}
    for token, sample in sample_by_token.items():
        scene_token = sample.get("scene_token")
        if isinstance(scene_token, str) and scene_token in samples_by_scene:
            samples_by_scene[scene_token].add(token)
    for token, data in data_by_token.items():
        sample_token = data.get("sample_token")
        sample = sample_by_token.get(cast(str, sample_token), {})
        scene_token = sample.get("scene_token")
        if isinstance(scene_token, str) and scene_token in data_by_scene:
            data_by_scene[scene_token].add(token)
        calibration = calibration_by_token.get(
            cast(str, data.get("calibrated_sensor_token")), {}
        )
        sensor = sensor_by_token.get(cast(str, calibration.get("sensor_token")), {})
        channel = sensor.get("channel")
        if isinstance(channel, str):
            normalized_channel_by_data[token] = channel
    official_channels_by_source: dict[str, set[str]] = {
        source: set() for source in recording_sources
    }
    for token, channel in normalized_channel_by_data.items():
        sample = sample_by_token.get(cast(str, data_by_token[token].get("sample_token")), {})
        source = scene_to_source.get(cast(str, sample.get("scene_token")))
        if source in official_channels_by_source:
            official_channels_by_source[source].add(channel)
    if any(
        set(channels_by_source.get(source, {})) != official_channels
        for source, official_channels in official_channels_by_source.items()
    ):
        findings.append(
            ValidationFinding(
                "error",
                "extension_reference",
                "recordings",
                "recording channel mapping differs from official sample data",
            )
        )

    tags_value = loaded["tags"]
    if isinstance(tags_value, list) and any(
        not isinstance(item, dict) or not _canonical_strings(item.get("computed_tags"))
        for item in tags_value
    ):
        findings.append(
            ValidationFinding(
                "error",
                "extension_value",
                "tags",
                "computed tags must be a canonical unique string list",
            )
        )

    validity_value = loaded["validity"]
    validity_malformed = False
    if isinstance(validity_value, list):
        for item in validity_value:
            if not isinstance(item, dict):
                validity_malformed = True
                continue
            scene_token = item.get("scene_token")
            if not isinstance(scene_token, str) or scene_token not in scenes:
                validity_malformed = True
                continue
            ratios = (
                item.get("scene_valid_ratio"),
                item.get("source_gnss_valid_ratio"),
                item.get("camera_coverage_ratio"),
            )
            sync_values = (
                item.get("max_abs_sync_error_ms"),
                item.get("mean_abs_sync_error_ms"),
            )
            coverage = item.get("camera_coverage_by_channel")
            coverage_rows = (
                cast(list[dict[str, Any]], coverage) if isinstance(coverage, list) else []
            )
            coverage_channels = [row.get("channel") for row in coverage_rows]
            nested_samples = item.get("samples")
            nested_data = item.get("sample_data")
            sample_rows = (
                cast(list[dict[str, Any]], nested_samples)
                if isinstance(nested_samples, list)
                else []
            )
            data_rows = (
                cast(list[dict[str, Any]], nested_data)
                if isinstance(nested_data, list)
                else []
            )
            sample_ids = [row.get("sample_token") for row in sample_rows]
            data_ids = [row.get("sample_data_token") for row in data_rows]
            validity_malformed = validity_malformed or (
                any(
                    not _finite_number(value) or not 0 <= cast(float, value) <= 1
                    for value in ratios
                )
                or any(
                    not _finite_number(value) or cast(float, value) < 0
                    for value in sync_values
                )
                or not isinstance(coverage, list)
                or any(
                    not isinstance(row, dict)
                    or set(row) != {"channel", "present", "expected", "ratio"}
                    or not isinstance(row.get("channel"), str)
                    or not isinstance(row.get("present"), int)
                    or isinstance(row.get("present"), bool)
                    or not isinstance(row.get("expected"), int)
                    or isinstance(row.get("expected"), bool)
                    or cast(int, row.get("present")) < 0
                    or cast(int, row.get("expected")) <= 0
                    or cast(int, row.get("present")) > cast(int, row.get("expected"))
                    or not _finite_number(row.get("ratio"))
                    or cast(float, row.get("ratio"))
                    != cast(int, row.get("present")) / cast(int, row.get("expected"))
                    for row in coverage_rows
                )
                or not all(isinstance(channel, str) for channel in coverage_channels)
                or cast(list[str], coverage_channels)
                != sorted(set(cast(list[str], coverage_channels)))
                or not isinstance(nested_samples, list)
                or any(
                    not isinstance(row, dict)
                    or set(row)
                    != {"sample_token", "timestamp_ns", "grid_timestamp_ns", "batch_timestamp_ns"}
                    or not isinstance(row.get("timestamp_ns"), int)
                    or isinstance(row.get("timestamp_ns"), bool)
                    or not isinstance(row.get("grid_timestamp_ns"), int)
                    or isinstance(row.get("grid_timestamp_ns"), bool)
                    or not isinstance(row.get("batch_timestamp_ns"), int)
                    or isinstance(row.get("batch_timestamp_ns"), bool)
                    for row in sample_rows
                )
                or len(sample_ids) != len(set(sample_ids))
                or set(sample_ids) != samples_by_scene[scene_token]
                or any(
                    cast(int, row.get("timestamp_ns")) // 1000
                    != sample_by_token[cast(str, row.get("sample_token"))].get("timestamp")
                    for row in sample_rows
                )
                or not isinstance(nested_data, list)
                or any(
                    not isinstance(row, dict)
                    or set(row)
                    != {
                        "sample_data_token",
                        "timestamp_ns",
                        "grid_signed_sync_error_ns",
                        "camera_signed_sync_error_ns",
                        "gnss_source_validity",
                    }
                    or not isinstance(row.get("timestamp_ns"), int)
                    or isinstance(row.get("timestamp_ns"), bool)
                    or not isinstance(row.get("grid_signed_sync_error_ns"), int)
                    or isinstance(row.get("grid_signed_sync_error_ns"), bool)
                    or not isinstance(row.get("camera_signed_sync_error_ns"), int)
                    or isinstance(row.get("camera_signed_sync_error_ns"), bool)
                    or not isinstance(row.get("gnss_source_validity"), list)
                    or len(row.get("gnss_source_validity", [])) != 2
                    or any(
                        not isinstance(value, bool)
                        for value in row.get("gnss_source_validity", [])
                    )
                    for row in data_rows
                )
                or len(data_ids) != len(set(data_ids))
                or set(data_ids) != data_by_scene[scene_token]
                or any(
                    cast(int, row.get("timestamp_ns")) // 1000
                    != data_by_token[cast(str, row.get("sample_data_token"))].get("timestamp")
                    for row in data_rows
                )
            )
    else:
        validity_malformed = True
    if validity_malformed:
        findings.append(
            ValidationFinding(
                "error",
                "extension_value",
                "validity",
                "validity evidence is malformed or differs from official scene ownership",
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
    gnss_malformed = False
    if isinstance(gnss, list):
        for item in gnss:
            if not isinstance(item, dict):
                gnss_malformed = True
                continue
            gnss_token = item.get("sample_data_token")
            scene_token = item.get("scene_token")
            source = item.get("source_digest")
            if (
                not isinstance(gnss_token, str)
                or not isinstance(scene_token, str)
                or not isinstance(source, str)
            ):
                gnss_malformed = True
                continue
            official = data_by_token.get(gnss_token)
            if official is None:
                gnss_malformed = True
                continue
            official_sample = sample_by_token.get(cast(str, official.get("sample_token")), {})
            normalized = normalized_channel_by_data.get(gnss_token)
            original = channels_by_source.get(source, {}).get(cast(str, normalized))
            filename = official.get("filename")
            image_digest = (
                _regular_file_sha256(root / filename) if isinstance(filename, str) else None
            )
            available = item.get("available")
            source_validity = item.get("source_validity")
            quaternion = item.get("quaternion_wxyz")
            sync_before = item.get("sync_gap_before_ns")
            sync_after = item.get("sync_gap_after_ns")
            fraction = item.get("fraction")
            nullable_scalars = (
                item.get("latitude_deg"),
                item.get("longitude_deg"),
                item.get("height_m"),
            )
            gnss_malformed = gnss_malformed or (
                official_sample.get("scene_token") != scene_token
                or scene_to_source.get(scene_token) != source
                or item.get("normalized_channel") != normalized
                or item.get("original_channel") != original
                or not isinstance(item.get("timestamp_ns"), int)
                or isinstance(item.get("timestamp_ns"), bool)
                or cast(int, item.get("timestamp_ns")) // 1000
                != official.get("timestamp")
                or item.get("image_sha256") != image_digest
                or not _sha256_text(item.get("image_sha256"))
                or not isinstance(available, bool)
                or not isinstance(item.get("position_uncertainty"), dict)
                or not _finite_json_tree(item.get("position_uncertainty"))
                or not isinstance(item.get("orientation_uncertainty"), dict)
                or not _finite_json_tree(item.get("orientation_uncertainty"))
                or not _finite_json_tree(item.get("before"))
                or not _finite_json_tree(item.get("after"))
                or any(
                    value is not None and not _finite_number(value)
                    for value in nullable_scalars
                )
                or (
                    fraction is not None
                    and (not _finite_number(fraction) or not 0 <= fraction <= 1)
                )
                or (
                    sync_before is not None
                    and (
                        not isinstance(sync_before, int)
                        or isinstance(sync_before, bool)
                        or sync_before < 0
                    )
                )
                or (
                    sync_after is not None
                    and (
                        not isinstance(sync_after, int)
                        or isinstance(sync_after, bool)
                        or sync_after < 0
                    )
                )
                or (
                    source_validity is not None
                    and (
                        not isinstance(source_validity, list)
                        or len(source_validity) != 2
                        or any(not isinstance(value, bool) for value in source_validity)
                    )
                )
                or (
                    quaternion is not None
                    and (
                        not isinstance(quaternion, list)
                        or len(quaternion) != 4
                        or any(not _finite_number(value) for value in quaternion)
                        or abs(sum(value * value for value in quaternion) - 1.0) > 1e-6
                    )
                )
                or (item.get("before") is not None and not isinstance(item.get("before"), dict))
                or (item.get("after") is not None and not isinstance(item.get("after"), dict))
            )
    else:
        gnss_malformed = True
    if gnss_malformed:
        findings.append(
            ValidationFinding(
                "error",
                "extension_value",
                "gnss",
                "GNSS evidence is malformed or differs from official image/channel ownership",
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
    pipeline_audit = loaded["pipeline_audit"]
    _validate_pipeline_audit(
        pipeline_audit,
        config,
        scene_to_source,
        source_blob_paths,
        split,
        findings,
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
        calibration_by_token = {
            cast(str, item["token"]): item
            for item in tables["calibrated_sensor"]
            if isinstance(item.get("token"), str)
        }
        sensor_by_token = {
            cast(str, item["token"]): item
            for item in tables["sensor"]
            if isinstance(item.get("token"), str)
        }
        channels_by_sample: dict[str, set[str]] = {token: set() for token in samples}
        for item in tables["sample_data"]:
            calibration = calibration_by_token.get(
                cast(str, item.get("calibrated_sensor_token")), {}
            )
            sensor = sensor_by_token.get(cast(str, calibration.get("sensor_token")), {})
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
    return validation


def _walk_regular_files(
    root: Path, *, root_fd: int | None = None
) -> dict[str, tuple[int, str]]:
    if root_fd is not None:
        return hash_regular_files_fd(root_fd, excluded=frozenset({_MANIFEST}))
    before = root.lstat()
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError("dataset root is not a directory")
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | _NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("dataset root identity changed")
        return hash_regular_files_fd(descriptor, excluded=frozenset({_MANIFEST}))
    finally:
        os.close(descriptor)


def build_content_manifest(root: Path, *, root_fd: int | None = None) -> dict[str, object]:
    """Hash every published regular file except the manifest itself."""
    entries = _walk_regular_files(root, root_fd=root_fd)
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


def _verify_manifest(
    root: Path,
    findings: list[ValidationFinding],
    *,
    lease: StagingLease | None = None,
) -> str | None:
    if lease is None:
        value = _load_json(root / _MANIFEST, findings, _MANIFEST)
        root_fd = None
    else:
        lease.assert_bound()
        root_fd = lease.duplicate_root_fd()
        try:
            value = _read_pinned_json(
                root_fd, ("mz_extensions", "content_manifest.json")
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            findings.append(
                ValidationFinding("error", "json", _MANIFEST, f"malformed JSON: {error}")
            )
            value = None
    try:
        actual = build_content_manifest(root, root_fd=root_fd)
    except ValueError as error:
        findings.append(ValidationFinding("error", "manifest", _MANIFEST, str(error)))
        return None
    finally:
        if root_fd is not None:
            os.close(root_fd)
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


def _read_pinned_json(root_fd: int, parts: tuple[str, ...]) -> object:
    directory_fd = os.dup(root_fd)
    try:
        for component in parts[:-1]:
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | _NOFOLLOW,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = child
        descriptor = os.open(parts[-1], os.O_RDONLY | _NOFOLLOW, dir_fd=directory_fd)
        try:
            before = os.fstat(descriptor)
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
            after = os.fstat(descriptor)
            listed = os.stat(parts[-1], dir_fd=directory_fd, follow_symlinks=False)
            def identity(item: os.stat_result) -> tuple[int, int, int, int, int, int]:
                return (
                    item.st_dev,
                    item.st_ino,
                    item.st_size,
                    item.st_nlink,
                    item.st_mtime_ns,
                    item.st_ctime_ns,
                )
            if identity(before) != identity(after) or identity(after) != identity(listed):
                raise ValueError("pinned JSON changed while reading")
            return cast(object, json.loads(b"".join(chunks)))
        finally:
            os.close(descriptor)
    finally:
        os.close(directory_fd)


def verify_publication_manifest_fd(root_fd: int, expected_content_hash: str) -> None:
    """Rebind finalized bytes to their manifest through pinned root authority."""
    manifest = _read_pinned_json(
        root_fd, ("mz_extensions", "content_manifest.json")
    )
    actual = build_content_manifest(Path("."), root_fd=root_fd)
    if manifest != actual or actual.get("root_sha256") != expected_content_hash:
        raise DatasetValidationError(
            ValidationReport(
                False,
                (
                    ValidationFinding(
                        "error",
                        "manifest",
                        _MANIFEST,
                        "finalized bytes differ from the authorized publication manifest",
                    ),
                ),
                (),
                (),
                None,
                None,
            )
        )


def verify_publication_manifest(lease: StagingLease, expected_content_hash: str) -> None:
    """Rebind finalized bytes to their manifest immediately before publication."""
    lease.assert_bound()
    root_fd = lease.duplicate_root_fd()
    try:
        verify_publication_manifest_fd(root_fd, expected_content_hash)
    finally:
        os.close(root_fd)
    lease.assert_bound()


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
    lease: StagingLease | None = None,
) -> ValidationReport:
    """Validate a complete exported dataset, accumulating deterministic findings."""
    root = Path(dataroot).absolute()
    if lease is not None:
        if root != lease.root:
            raise ValueError("validation root differs from staging lease")
        lease.assert_bound()
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
    validation_evidence = _extensions(
        root, tables, config, findings, finalized=verify_manifest
    )
    content_hash = (
        _verify_manifest(root, findings, lease=lease) if verify_manifest else None
    )
    if official_smoke and not any(item.severity == "error" for item in findings):
        try:
            _official_smoke(root, version, tables)
        except Exception as error:
            findings.append(
                ValidationFinding(
                    "error", "official_smoke", "NuScenes", f"official SDK smoke failed: {error}"
                )
            )
    resolved_config_hash = (
        canonical_hash(config.model_dump(mode="json")) if config is not None else None
    )
    table_counts = tuple((name, len(tables[name])) for name in OFFICIAL_TABLES)
    if verify_manifest:
        recomputed_findings = tuple(sorted(set(findings)))
        recomputed = ValidationReport(
            not any(item.severity == "error" for item in recomputed_findings),
            recomputed_findings,
            _CHECKS,
            table_counts,
            resolved_config_hash,
            None,
        )
        if not recomputed.succeeded or validation_evidence != recomputed.to_extension():
            findings.append(
                ValidationFinding(
                    "error",
                    "validation_state",
                    "validation",
                    "finalized validation evidence differs from recomputed validation",
                )
            )
    ordered = tuple(sorted(set(findings)))
    report = ValidationReport(
        not any(item.severity == "error" for item in ordered),
        ordered,
        _CHECKS,
        table_counts,
        resolved_config_hash,
        content_hash,
    )
    if lease is not None:
        lease.assert_bound()
    if raise_on_error and not report.succeeded:
        raise DatasetValidationError(report)
    return report


def _atomic_json(
    root: Path,
    relative: str,
    value: object,
    *,
    root_fd: int | None = None,
) -> None:
    parts = PurePosixPath(relative).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("unsafe finalization path")
    directory_fd = (
        os.dup(root_fd)
        if root_fd is not None
        else os.open(root, os.O_RDONLY | os.O_DIRECTORY | _NOFOLLOW)
    )
    for component in parts[:-1]:
        child = os.open(
            component,
            os.O_RDONLY | os.O_DIRECTORY | _NOFOLLOW,
            dir_fd=directory_fd,
        )
        os.close(directory_fd)
        directory_fd = child
    temporary = f".{parts[-1]}.{os.getpid()}.tmp"
    descriptor = -1
    try:
        current = os.stat(parts[-1], dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
            raise ValueError("finalization destination is not a safe regular file")
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW, 0o600, dir_fd=directory_fd
        )
        content = (canonical_json(value) + "\n").encode()
        offset = 0
        while offset < len(content):
            written = os.write(descriptor, content[offset:])
            if written <= 0:
                raise OSError("short write while finalizing dataset")
            offset += written
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


def _finalize_dataset_locked(
    dataroot: str | Path,
    version: str = NUSCENES_VERSION,
    *,
    official_smoke: bool = True,
    lease: StagingLease | None = None,
) -> ValidationReport:
    """Validate, record success, create the manifest last, and revalidate."""
    root = Path(dataroot).absolute()
    if lease is not None:
        if root != lease.root:
            raise ValueError("finalization root differs from staging lease")
        lease.assert_bound()
    preliminary = validate_dataset(
        root,
        version,
        official_smoke=official_smoke,
        verify_manifest=False,
        lease=lease,
    )
    if not preliminary.succeeded:
        raise DatasetValidationError(preliminary)
    root_fd = lease._root_fd if lease is not None else None
    _atomic_json(
        root,
        "mz_extensions/validation.json",
        preliminary.to_extension(),
        root_fd=root_fd,
    )
    _atomic_json(
        root,
        _MANIFEST,
        build_content_manifest(root, root_fd=root_fd),
        root_fd=root_fd,
    )
    final = validate_dataset(
        root,
        version,
        official_smoke=official_smoke,
        verify_manifest=True,
        lease=lease,
    )
    if not final.succeeded:
        failed = ValidationReport(
            False,
            final.findings,
            final.checks,
            final.table_counts,
            final.resolved_config_hash,
            None,
        )
        _atomic_json(
            root,
            "mz_extensions/validation.json",
            failed.to_extension(),
            root_fd=root_fd,
        )
        raise DatasetValidationError(failed)
    return final


def finalize_dataset(
    dataroot: str | Path,
    version: str = NUSCENES_VERSION,
    *,
    official_smoke: bool = True,
    lease: StagingLease | None = None,
) -> ValidationReport:
    """Finalize staging while excluding cooperative concurrent mutation."""
    if lease is not None:
        with lease.mutation_guard():
            return _finalize_dataset_locked(
                dataroot,
                version,
                official_smoke=official_smoke,
                lease=lease,
            )
    return _finalize_dataset_locked(
        dataroot,
        version,
        official_smoke=official_smoke,
        lease=None,
    )
