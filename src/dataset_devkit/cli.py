"""Command-line interface for dataset-devkit."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError

from dataset_devkit.config import ConfigRootError, validate_config_schema_and_runtime
from dataset_devkit.dataset import DatasetFormatError
from dataset_devkit.identifiers import validate_safe_segment
from dataset_devkit.provenance import canonical_json
from dataset_devkit.services import (
    BuildOperationalError,
    build_dataset,
    inspect_dataset,
    validate_dataset,
)
from dataset_devkit.validation import DatasetValidationError


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
            config = validate_config_schema_and_runtime(args.config)
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            JsonSchemaValidationError,
            ValidationError,
            ConfigRootError,
        ) as error:
            print(f"dataset-devkit: error: {_format_config_error(error)}", file=sys.stderr)
            return 2
    try:
        if args.command == "build":
            result = build_dataset(config)
            print(
                canonical_json(
                    {
                        "content_hash": result.content_hash,
                        "dataroot": str(result.dataroot),
                        "failed_recordings": list(result.failed_recordings),
                        "partial": result.partial,
                        "sample_count": result.sample_count,
                        "sample_data_count": result.sample_data_count,
                        "scene_count": result.scene_count,
                        "version": result.version,
                    }
                )
            )
        elif args.command == "validate":
            report = validate_dataset(args.dataroot, args.version)
            print(
                canonical_json(
                    {
                        "content_hash": report.content_hash,
                        "state": "succeeded",
                        "table_counts": dict(report.table_counts),
                        "version": args.version,
                    }
                )
            )
        else:
            summary = inspect_dataset(args.dataroot, args.version)
            print(canonical_json(summary.to_dict()))
    except (
        OSError,
        BuildOperationalError,
        DatasetFormatError,
        DatasetValidationError,
        ValueError,
    ) as error:
        print(f"dataset-devkit: error: {str(error).replace(chr(10), ' ')}", file=sys.stderr)
        return 1
    return 0


def _format_config_error(
    error: (
        OSError
        | UnicodeDecodeError
        | json.JSONDecodeError
        | JsonSchemaValidationError
        | ValidationError
        | ConfigRootError
    ),
) -> str:
    if isinstance(error, UnicodeDecodeError):
        return f"configuration is not valid UTF-8 at byte {error.start}"
    if isinstance(error, json.JSONDecodeError):
        return f"invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}"
    if isinstance(error, JsonSchemaValidationError):
        location = ".".join(str(part) for part in error.absolute_path) or "config"
        return f"invalid configuration at {location}: {error.message}"
    if isinstance(error, ValidationError):
        detail = error.errors(include_url=False)[0]
        location = ".".join(str(part) for part in detail["loc"])
        return f"invalid configuration at {location}: {detail['msg']}"
    return str(error).replace("\n", " ")
