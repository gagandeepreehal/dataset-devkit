from __future__ import annotations

from pathlib import Path

import pytest

from dataset_devkit.blob_list import BlobListError, parse_blob_list


def test_blob_list_preserves_exact_accepted_paths_and_skips_blanks_and_comments(
    tmp_path: Path,
) -> None:
    path = tmp_path / "blobs.txt"
    path.write_text(
        "\n   \n# comment\n  # indented comment\nmcap-h265/fleet/a.mcap\n"
        "mcap-h265/fleet/name with space.mcap\r\n",
        encoding="utf-8",
    )

    assert parse_blob_list(path) == (
        "mcap-h265/fleet/a.mcap",
        "mcap-h265/fleet/name with space.mcap",
    )


def test_blob_list_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "blobs.txt"
    path.write_text("mcap-h265/a.mcap\nmcap-h265/a.mcap\n", encoding="utf-8")

    with pytest.raises(BlobListError, match="duplicate.*line 2"):
        parse_blob_list(path)


@pytest.mark.parametrize(
    "value",
    [
        "fleet/a.mcap",
        "mcap-h265/a.txt",
        "/mcap-h265/a.mcap",
        "mcap-h265/../a.mcap",
        "mcap-h265/./a.mcap",
        "mcap-h265/a//b.mcap",
        r"mcap-h265\a.mcap",
        "mcap-h265/a.mcap?sig=x",
        "mcap-h265/a.mcap#fragment",
        "mcap-h265/%2e%2e/a.mcap",
        "mcap-h265/a%2fb.mcap",
        "mcap-h265/a%5cb.mcap",
        "mcap-h265/",
        " mcap-h265/a.mcap",
        "mcap-h265/a.mcap ",
    ],
)
def test_blob_list_rejects_unsafe_or_out_of_scope_paths(tmp_path: Path, value: str) -> None:
    path = tmp_path / "blobs.txt"
    path.write_text(value + "\n", encoding="utf-8")

    with pytest.raises(BlobListError, match="line 1"):
        parse_blob_list(path)
