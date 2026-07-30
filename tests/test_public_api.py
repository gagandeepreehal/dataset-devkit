from __future__ import annotations

from pathlib import Path

import pytest

import dataset_devkit


def test_dataset_is_a_stable_import_boundary(tmp_path: Path) -> None:
    assert hasattr(dataset_devkit, "Dataset")
    dataset = dataset_devkit.Dataset(dataroot=tmp_path, version="v1.0-trainval")

    assert dataset.dataroot == tmp_path.resolve()
    assert dataset.version == "v1.0-trainval"


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
