"""Generate the checked-in global configuration JSON Schema."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from dataset_devkit.config import GlobalConfig, validate_config_schema_and_runtime

__all__ = ["validate_config_schema_and_runtime"]

DEFAULT_OUTPUT = Path("schema/dataset_config.schema.json")


def render_schema() -> str:
    """Render a stable, human-reviewable JSON Schema document."""
    return json.dumps(GlobalConfig.model_json_schema(), indent=2, sort_keys=True) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Write the current schema to disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_schema(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
