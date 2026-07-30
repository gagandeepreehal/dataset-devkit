from __future__ import annotations

import json
from pathlib import Path

import pytest

import dataset_devkit


def test_dataset_is_a_stable_import_boundary(tmp_path: Path) -> None:
    assert hasattr(dataset_devkit, "Dataset")
    dataset = dataset_devkit.Dataset(dataroot=tmp_path, version="v1.0-trainval")

    assert dataset.dataroot == tmp_path.resolve()
    assert dataset.version == "v1.0-trainval"


@pytest.mark.parametrize("version", ["", ".", "..", "versions/v1", r"versions\v1", " v1 "])
def test_dataset_rejects_unsafe_version_segments(tmp_path: Path, version: str) -> None:
    with pytest.raises(ValueError, match="version|segment"):
        dataset_devkit.Dataset(dataroot=tmp_path, version=version)


def test_cli_exposes_required_command_argument_contracts() -> None:
    from dataset_devkit.cli import create_parser

    parser = create_parser()

    build = parser.parse_args(["build", "--config", "dataset_config.json"])
    validate = parser.parse_args(
        ["validate", "--dataroot", "DATASET", "--version", "v1.0-trainval"]
    )
    inspect = parser.parse_args(["inspect", "--dataroot", "DATASET", "--version", "v1.0-trainval"])

    assert build.config == Path("dataset_config.json")
    assert validate.dataroot == Path("DATASET")
    assert validate.version == "v1.0-trainval"
    assert inspect.dataroot == Path("DATASET")
    assert inspect.version == "v1.0-trainval"


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--help"], "{build,validate,inspect}"),
        (["build", "--help"], "--config"),
        (["validate", "--help"], "--dataroot"),
        (["inspect", "--help"], "--version"),
    ],
)
def test_cli_help_smoke(argv: list[str], expected: str, capsys: pytest.CaptureFixture[str]) -> None:
    from dataset_devkit.cli import create_parser

    with pytest.raises(SystemExit, match="0"):
        create_parser().parse_args(argv)

    assert expected in capsys.readouterr().out


@pytest.mark.parametrize("command", ["validate", "inspect"])
def test_cli_rejects_unsafe_version_segments(command: str) -> None:
    from dataset_devkit.cli import create_parser

    with pytest.raises(SystemExit, match="2"):
        create_parser().parse_args([command, "--dataroot", "DATASET", "--version", "../v1"])


@pytest.mark.parametrize("case", ["missing", "malformed", "invalid"])
def test_build_main_returns_concise_config_diagnostics(
    tmp_path: Path,
    case: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dataset_devkit.cli import main

    config_path = tmp_path / "dataset_config.json"
    if case == "malformed":
        config_path.write_text("{not-json", encoding="utf-8")
    elif case == "invalid":
        config_path.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")

    result = main(["build", "--config", str(config_path)])
    error = capsys.readouterr().err

    assert result == 2
    assert "dataset-devkit: error:" in error
    assert "Traceback" not in error
    assert len(error.splitlines()) == 1
