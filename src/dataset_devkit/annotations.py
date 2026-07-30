"""Strict, line-audited human annotation JSONL parsing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dataset_devkit.blob_list import BlobListError, validate_blob_path


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
    blob_path: str
    timestamp_ns: int
    labels: tuple[str, ...]


def _parse_record(value: object, line_number: int) -> ParsedAnnotation:
    if not isinstance(value, dict) or set(value) != {"blob_path", "timestamp_ns", "labels"}:
        raise AnnotationFormatError(
            f"invalid annotation object at line {line_number}: exact keys are required"
        )
    blob_path = value["blob_path"]
    timestamp = value["timestamp_ns"]
    labels = value["labels"]
    if not isinstance(blob_path, str):
        raise AnnotationFormatError(f"invalid blob_path at line {line_number}")
    try:
        validate_blob_path(blob_path, line_number=line_number)
    except BlobListError as error:
        raise AnnotationFormatError(str(error)) from error
    if not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp < 0:
        raise AnnotationFormatError(
            f"timestamp_ns must be a nonnegative integer at line {line_number}"
        )
    if (
        not isinstance(labels, list)
        or not labels
        or any(
            not isinstance(label, str) or not label.strip() or label != label.strip()
            for label in labels
        )
        or len(labels) != len(set(labels))
    ):
        raise AnnotationFormatError(
            f"labels must be a nonempty unique array of nonblank strings at line {line_number}"
        )
    return ParsedAnnotation(line_number, blob_path, timestamp, tuple(labels))


def parse_annotations(path: Path) -> tuple[ParsedAnnotation, ...]:
    """Parse JSONL; blank and comment-only lines are explicitly ignored."""
    records: list[ParsedAnnotation] = []
    identities: dict[tuple[str, int, tuple[str, ...]], int] = {}
    raw_file = path.read_bytes()
    try:
        text = raw_file.decode("utf-8")
    except UnicodeDecodeError as error:
        line_number = raw_file[: error.start].count(b"\n") + 1
        raise AnnotationFormatError(f"invalid UTF-8 at line {line_number}") from error
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            value = json.loads(line, object_pairs_hook=_unique_object)
        except _DuplicateKey as error:
            raise AnnotationFormatError(
                f"duplicate JSON key {error.args[0]!r} at line {line_number}"
            ) from error
        except json.JSONDecodeError as error:
            raise AnnotationFormatError(f"invalid JSON at line {line_number}: {error}") from error
        record = _parse_record(value, line_number)
        identity = (record.blob_path, record.timestamp_ns, record.labels)
        if identity in identities:
            raise AnnotationFormatError(
                f"duplicate annotation at line {line_number} "
                f"(first seen at line {identities[identity]})"
            )
        identities[identity] = line_number
        records.append(record)
    return tuple(records)
