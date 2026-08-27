"""Strict, line-audited human annotation JSONL parsing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dataset_devkit.repository_paths import RepositoryPathError, validate_repo_mcap_path


class AnnotationFormatError(ValueError):
    """Raised when a human annotation JSONL record violates its exact contract."""


class _DuplicateKey(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateKey(key)
        value[key] = item
    return value


@dataclass(frozen=True)
class ParsedAnnotation:
    line_number: int
    repo_path: str
    timestamp_ns: int
    labels: tuple[str, ...]


@dataclass(frozen=True)
class AnnotationBudgets:
    """Stable bounds for streaming annotation input."""

    max_total_bytes: int = 64 * 1024 * 1024
    max_line_bytes: int = 256 * 1024
    max_records: int = 250_000
    max_labels_per_record: int = 256
    max_label_chars: int = 256
    max_label_bytes: int = 1024
    max_repo_path_chars: int = 2048
    max_repo_path_bytes: int = 4096

    def __post_init__(self) -> None:
        if any(value <= 0 for value in self.__dict__.values()):
            raise ValueError("annotation budgets must be positive integers")


DEFAULT_ANNOTATION_BUDGETS = AnnotationBudgets()


def _parse_record(value: object, line_number: int, budgets: AnnotationBudgets) -> ParsedAnnotation:
    if not isinstance(value, dict) or set(value) != {"repo_path", "timestamp_ns", "labels"}:
        raise AnnotationFormatError(
            f"invalid annotation object at line {line_number}: exact keys are required"
        )
    repo_path = value["repo_path"]
    timestamp = value["timestamp_ns"]
    labels = value["labels"]
    if not isinstance(repo_path, str):
        raise AnnotationFormatError(f"invalid repo_path at line {line_number}")
    if len(repo_path) > budgets.max_repo_path_chars:
        raise AnnotationFormatError(
            f"repository path characters exceed budget at line {line_number}"
        )
    if len(repo_path.encode("utf-8")) > budgets.max_repo_path_bytes:
        raise AnnotationFormatError(f"repository path bytes exceed budget at line {line_number}")
    try:
        validate_repo_mcap_path(repo_path, line_number=line_number)
    except RepositoryPathError as error:
        raise AnnotationFormatError(str(error)) from error
    if not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp < 0:
        raise AnnotationFormatError(
            f"timestamp_ns must be a nonnegative integer at line {line_number}"
        )
    if (
        not isinstance(labels, list)
        or not labels
        or len(labels) > budgets.max_labels_per_record
        or any(
            not isinstance(label, str) or not label.strip() or label != label.strip()
            for label in labels
        )
        or len(labels) != len(set(labels))
    ):
        if isinstance(labels, list) and len(labels) > budgets.max_labels_per_record:
            raise AnnotationFormatError(f"label count exceeds budget at line {line_number}")
        raise AnnotationFormatError(
            f"labels must be a nonempty unique array of nonblank strings at line {line_number}"
        )
    for label in labels:
        if len(label) > budgets.max_label_chars:
            raise AnnotationFormatError(f"label characters exceed budget at line {line_number}")
        if len(label.encode("utf-8")) > budgets.max_label_bytes:
            raise AnnotationFormatError(f"label bytes exceed budget at line {line_number}")
    return ParsedAnnotation(line_number, repo_path, timestamp, tuple(labels))


def parse_annotations(
    path: Path, *, budgets: AnnotationBudgets = DEFAULT_ANNOTATION_BUDGETS
) -> tuple[ParsedAnnotation, ...]:
    """Parse JSONL; blank and comment-only lines are explicitly ignored."""
    records: list[ParsedAnnotation] = []
    identities: dict[tuple[str, int, tuple[str, ...]], int] = {}
    total_bytes = 0
    with path.open("rb") as stream:
        for line_number, raw_line in enumerate(
            iter(lambda: stream.readline(budgets.max_line_bytes + 2), b""), 1
        ):
            total_bytes += len(raw_line)
            if total_bytes > budgets.max_total_bytes:
                raise AnnotationFormatError(
                    f"annotation total bytes exceed budget at line {line_number}"
                )
            content = raw_line[:-1] if raw_line.endswith(b"\n") else raw_line
            if content.endswith(b"\r"):
                content = content[:-1]
            if len(content) > budgets.max_line_bytes:
                raise AnnotationFormatError(
                    f"annotation line bytes exceed budget at line {line_number}"
                )
            try:
                line = content.decode("utf-8")
            except UnicodeDecodeError as error:
                raise AnnotationFormatError(
                    f"invalid UTF-8 at line {line_number} byte {error.start + 1}"
                ) from error
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if len(records) >= budgets.max_records:
                raise AnnotationFormatError(
                    f"annotation record count exceeds budget at line {line_number}"
                )
            try:
                value = json.loads(line, object_pairs_hook=_unique_object)
            except _DuplicateKey as error:
                raise AnnotationFormatError(
                    f"duplicate JSON key {error.args[0]!r} at line {line_number}"
                ) from error
            except json.JSONDecodeError as error:
                raise AnnotationFormatError(
                    f"invalid JSON at line {line_number}: {error}"
                ) from error
            record = _parse_record(value, line_number, budgets)
            identity = (record.repo_path, record.timestamp_ns, record.labels)
            if identity in identities:
                raise AnnotationFormatError(
                    f"duplicate annotation at line {line_number} "
                    f"(first seen at line {identities[identity]})"
                )
            identities[identity] = line_number
            records.append(record)
    return tuple(records)
