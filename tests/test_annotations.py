from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from dataset_devkit.annotations import (
    AnnotationBudgets,
    AnnotationFormatError,
    parse_annotations,
)


def _write(path: Path, lines: Sequence[object | str]) -> Path:
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


def test_parser_streams_without_path_whole_file_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(
        tmp_path / "annotations.jsonl",
        [{"blob_path": "mcap-h265/a.mcap", "timestamp_ns": 1, "labels": ["x"]}],
    )
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("read_bytes called"))
    monkeypatch.setattr(Path, "read_text", lambda self, **kwargs: pytest.fail("read_text called"))

    assert len(parse_annotations(path)) == 1


def test_annotation_stream_total_line_and_record_budgets_are_inclusive(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path / "annotations.jsonl",
        [
            {"blob_path": "mcap-h265/a.mcap", "timestamp_ns": 1, "labels": ["x"]},
            {"blob_path": "mcap-h265/a.mcap", "timestamp_ns": 2, "labels": ["y"]},
        ],
    )
    size = path.stat().st_size
    longest = max(len(line) for line in path.read_bytes().splitlines())
    accepted = AnnotationBudgets(max_total_bytes=size, max_line_bytes=longest, max_records=2)
    assert len(parse_annotations(path, budgets=accepted)) == 2

    for budgets, message in (
        (AnnotationBudgets(max_total_bytes=size - 1), "total bytes"),
        (AnnotationBudgets(max_line_bytes=longest - 1), "line bytes"),
        (AnnotationBudgets(max_records=1), "record count"),
    ):
        with pytest.raises(AnnotationFormatError, match=message):
            parse_annotations(path, budgets=budgets)


def test_annotation_label_and_string_budgets_enforce_bytes_and_characters(
    tmp_path: Path,
) -> None:
    blob_path = "mcap-h265/é.mcap"
    record = {
        "blob_path": blob_path,
        "timestamp_ns": 1,
        "labels": ["éé", "x"],
    }
    path = _write(tmp_path / "annotations.jsonl", [record])
    blob_chars = len(blob_path)
    blob_bytes = len(blob_path.encode("utf-8"))
    assert parse_annotations(
        path,
        budgets=AnnotationBudgets(
            max_labels_per_record=2,
            max_label_chars=2,
            max_label_bytes=4,
            max_blob_chars=blob_chars,
            max_blob_bytes=blob_bytes,
        ),
    )
    for budgets, message in (
        (AnnotationBudgets(max_labels_per_record=1), "label count"),
        (AnnotationBudgets(max_label_chars=1), "label characters"),
        (AnnotationBudgets(max_label_bytes=3), "label bytes"),
        (AnnotationBudgets(max_blob_chars=blob_chars - 1), "blob.*characters"),
        (AnnotationBudgets(max_blob_bytes=blob_bytes - 1), "blob.*bytes"),
    ):
        with pytest.raises(AnnotationFormatError, match=message):
            parse_annotations(path, budgets=budgets)


def test_large_annotation_file_is_streamed_with_stable_line_numbers(tmp_path: Path) -> None:
    records = [
        {"blob_path": "mcap-h265/a.mcap", "timestamp_ns": index, "labels": [f"x{index}"]}
        for index in range(2_000)
    ]
    path = _write(tmp_path / "annotations.jsonl", records)

    parsed = parse_annotations(path)

    assert len(parsed) == 2_000
    assert parsed[-1].line_number == 2_000
