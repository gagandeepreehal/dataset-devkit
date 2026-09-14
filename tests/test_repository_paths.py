from __future__ import annotations

import pytest

from dataset_devkit.repository_paths import (
    RepositoryPathError,
    validate_repo_file_path,
    validate_repo_mcap_path,
)


@pytest.mark.parametrize(
    "value",
    ["data/fleet/a.mcap", "data/fleet/name with space.mcap"],
)
def test_repository_mcap_paths_preserve_exact_valid_values(value: str) -> None:
    assert validate_repo_mcap_path(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "fleet/a.mcap",
        "data/a.txt",
        "/data/a.mcap",
        "data/../a.mcap",
        "data/./a.mcap",
        "data/a//b.mcap",
        r"data\a.mcap",
        "data/a.mcap?x=1",
        "data/a.mcap#fragment",
        "data/%2e%2e/a.mcap",
        "data/",
        " data/a.mcap",
        "data/a.mcap ",
    ],
)
def test_repository_mcap_paths_reject_unsafe_or_out_of_scope_values(value: str) -> None:
    with pytest.raises(RepositoryPathError, match="line 7"):
        validate_repo_mcap_path(value, line_number=7)


@pytest.mark.parametrize(
    "value",
    [
        "data/videos/cam_front/recording.mp4",
        "data/tables/gnss/part-recording.parquet",
        "data/manifests/model-receipt.json",
        "data/metadata.jsonl",
        "data/annotations/depth.tar",
    ],
)
def test_repository_file_paths_preserve_supported_relative_values(value: str) -> None:
    assert validate_repo_file_path(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "recording.mp4",
        "data/video.avi",
        "data/../recording.mp4",
        r"data\recording.mp4",
        "data/recording.mp4?download=1",
        "data/%2e%2e/recording.mp4",
        "data/CON/recording.mp4",
        "data/recording.mp4 ",
    ],
)
def test_repository_file_paths_reject_unsafe_or_unsupported_values(value: str) -> None:
    with pytest.raises(RepositoryPathError):
        validate_repo_file_path(value)
