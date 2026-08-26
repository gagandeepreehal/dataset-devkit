from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_devkit.huggingface_manifest import ManifestEntry, ManifestError, parse_manifest


def write_rows(path: Path, rows: list[object]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_parse_manifest_preserves_order_and_required_identity(tmp_path: Path) -> None:
    path = write_rows(
        tmp_path / "manifest.jsonl",
        [
            {
                "repo_path": "data/2025-04-11/a.mcap",
                "source_size": 12,
                "sha256": "a" * 64,
                "source_etag": "ignored",
            },
            {
                "repo_path": "data/2025-04-12/b.mcap",
                "source_size": 34,
                "sha256": "b" * 64,
            },
        ],
    )

    assert parse_manifest(path) == (
        ManifestEntry("data/2025-04-11/a.mcap", 12, "a" * 64),
        ManifestEntry("data/2025-04-12/b.mcap", 34, "b" * 64),
    )


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"repo_path": "../a.mcap", "source_size": 1, "sha256": "a" * 64}],
        [{"repo_path": "data/a.txt", "source_size": 1, "sha256": "a" * 64}],
        [{"repo_path": "data/a.mcap", "source_size": 0, "sha256": "a" * 64}],
        [{"repo_path": "data/a.mcap", "source_size": 1, "sha256": "invalid"}],
        [
            {"repo_path": "data/a.mcap", "source_size": 1, "sha256": "a" * 64},
            {"repo_path": "data/a.mcap", "source_size": 1, "sha256": "a" * 64},
        ],
    ],
)
def test_parse_manifest_rejects_invalid_rows(tmp_path: Path, rows: list[object]) -> None:
    with pytest.raises(ManifestError):
        parse_manifest(write_rows(tmp_path / "manifest.jsonl", rows))
