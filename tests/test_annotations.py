from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_devkit.annotations import AnnotationFormatError, parse_annotations


def _write(path: Path, lines: list[object | str]) -> Path:
    path.write_text(
        "\n".join(line if isinstance(line, str) else json.dumps(line) for line in lines),
        encoding="utf-8",
    )
    return path


def test_parser_preserves_lines_exact_paths_and_all_unique_labels(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "annotations.jsonl",
        [
            "# recording review",
            "",
            {
                "blob_path": "mcap-h265/day/a.mcap",
                "timestamp_ns": 10,
                "labels": ["turn", "rain"],
            },
        ],
    )

    records = parse_annotations(path)

    assert len(records) == 1
    assert records[0].line_number == 3
    assert records[0].blob_path == "mcap-h265/day/a.mcap"
    assert records[0].timestamp_ns == 10
    assert records[0].labels == ("turn", "rain")


@pytest.mark.parametrize(
    "record",
    [
        {"blob_path": "mcap-h265/a.mcap", "timestamp_ns": True, "labels": ["x"]},
        {"blob_path": "mcap-h265/a.mcap", "timestamp_ns": 1.0, "labels": ["x"]},
        {"blob_path": "mcap-h265/a.mcap", "timestamp_ns": "1", "labels": ["x"]},
        {"blob_path": "mcap-h265/a.mcap", "timestamp_ns": -1, "labels": ["x"]},
        {"blob_path": "a.mcap", "timestamp_ns": 1, "labels": ["x"]},
        {"blob_path": "mcap-h265/a.mcap", "timestamp_ns": 1, "labels": []},
        {"blob_path": "mcap-h265/a.mcap", "timestamp_ns": 1, "labels": [" "]},
        {"blob_path": "mcap-h265/a.mcap", "timestamp_ns": 1, "labels": ["x", "x"]},
        {"blob_path": "mcap-h265/a.mcap", "timestamp_ns": 1, "labels": ["x"], "x": 1},
    ],
)
def test_parser_rejects_non_strict_records_with_line_number(
    tmp_path: Path, record: dict[str, object]
) -> None:
    with pytest.raises(AnnotationFormatError, match="line 1"):
        parse_annotations(_write(tmp_path / "annotations.jsonl", [record]))


def test_parser_rejects_duplicate_record_identity(tmp_path: Path) -> None:
    record = {
        "blob_path": "mcap-h265/a.mcap",
        "timestamp_ns": 1,
        "labels": ["x"],
    }
    with pytest.raises(AnnotationFormatError, match="duplicate.*line 2.*line 1"):
        parse_annotations(_write(tmp_path / "annotations.jsonl", [record, record]))


def test_parser_rejects_duplicate_json_object_keys(tmp_path: Path) -> None:
    raw = '{"blob_path":"mcap-h265/a.mcap","timestamp_ns":1,"timestamp_ns":2,"labels":["x"]}'
    with pytest.raises(AnnotationFormatError, match="duplicate.*timestamp_ns.*line 1"):
        parse_annotations(_write(tmp_path / "annotations.jsonl", [raw]))


def test_parser_reports_invalid_utf8_line(tmp_path: Path) -> None:
    path = tmp_path / "annotations.jsonl"
    path.write_bytes(b"# ok\n\xff\n")
    with pytest.raises(AnnotationFormatError, match="UTF-8.*line 2"):
        parse_annotations(path)
