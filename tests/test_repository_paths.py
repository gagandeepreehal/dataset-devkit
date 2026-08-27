from __future__ import annotations

import pytest

from dataset_devkit.repository_paths import RepositoryPathError, validate_repo_mcap_path


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
