"""Command-line interface for dataset-devkit."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dataset_devkit.config import load_config
from dataset_devkit.services import (
    ServiceNotImplementedError,
    build_dataset,
    inspect_dataset,
    validate_dataset,
)


def create_parser() -> argparse.ArgumentParser:
    """Create the public CLI parser."""
    parser = argparse.ArgumentParser(
        prog="dataset-devkit",
        description="Build, validate, and inspect deterministic robotics datasets.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="build a dataset from JSON configuration")
    build.add_argument("--config", type=Path, required=True, help="configuration JSON")

    for name, help_text in (
        ("validate", "validate a published dataset"),
        ("inspect", "inspect a published dataset"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--dataroot", type=Path, required=True)
        command.add_argument("--version", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            build_dataset(load_config(args.config))
        elif args.command == "validate":
            validate_dataset(args.dataroot, args.version)
        else:
            inspect_dataset(args.dataroot, args.version)
    except ServiceNotImplementedError as error:
        parser.error(str(error))
    return 0
