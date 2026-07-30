"""Command-line interface for dataset-devkit."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from dataset_devkit.config import ConfigRootError, load_config
from dataset_devkit.identifiers import validate_safe_segment
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
        command.add_argument("--version", required=True, type=_safe_segment_argument)

    return parser


def _safe_segment_argument(value: str) -> str:
    try:
        return validate_safe_segment(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.command == "build":
        try:
            config = load_config(args.config)
        except (OSError, json.JSONDecodeError, ValidationError, ConfigRootError) as error:
            print(f"dataset-devkit: error: {_format_config_error(error)}", file=sys.stderr)
            return 2
    try:
        if args.command == "build":
            build_dataset(config)
        elif args.command == "validate":
            validate_dataset(args.dataroot, args.version)
        else:
            inspect_dataset(args.dataroot, args.version)
    except ServiceNotImplementedError as error:
        parser.error(str(error))
    return 0


def _format_config_error(
    error: OSError | json.JSONDecodeError | ValidationError | ConfigRootError,
) -> str:
    if isinstance(error, json.JSONDecodeError):
        return f"invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}"
    if isinstance(error, ValidationError):
        detail = error.errors(include_url=False)[0]
        location = ".".join(str(part) for part in detail["loc"])
        return f"invalid configuration at {location}: {detail['msg']}"
    return str(error).replace("\n", " ")
