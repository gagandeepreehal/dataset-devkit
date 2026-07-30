from __future__ import annotations

import json
from pathlib import Path

from dataset_devkit.config import GlobalConfig

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schema" / "dataset_config.schema.json"


def test_checked_in_schema_is_current_and_deterministic() -> None:
    expected = json.dumps(GlobalConfig.model_json_schema(), indent=2, sort_keys=True) + "\n"

    assert SCHEMA_PATH.read_text(encoding="utf-8") == expected
