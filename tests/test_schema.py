from __future__ import annotations

import json
import re
from pathlib import Path

from dataset_devkit.config import GlobalConfig

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schema" / "dataset_config.schema.json"


def test_checked_in_schema_is_current_and_deterministic() -> None:
    expected = json.dumps(GlobalConfig.model_json_schema(), indent=2, sort_keys=True) + "\n"

    assert SCHEMA_PATH.read_text(encoding="utf-8") == expected


def _required_camera_schema() -> dict[str, object]:
    schema = GlobalConfig.model_json_schema()
    return schema["$defs"]["FrameValidityConfig"]["properties"]["required_cameras"]  # type: ignore[no-any-return]


def _schema_accepts_required_cameras(values: list[str]) -> bool:
    schema = _required_camera_schema()
    item_schema = schema["items"]
    assert isinstance(item_schema, dict)
    pattern = item_schema["pattern"]
    minimum = item_schema["minLength"]
    assert isinstance(pattern, str)
    assert isinstance(minimum, int)
    return (
        (not schema.get("uniqueItems") or len(values) == len(set(values)))
        and all(len(value) >= minimum and re.fullmatch(pattern, value) for value in values)
    )


def test_required_camera_json_schema_directly_rejects_duplicates_and_unsafe_segments() -> None:
    assert _schema_accepts_required_cameras(["front", "rear"])
    assert not _schema_accepts_required_cameras(["front", "front"])
    for unsafe in (
        "", ".", "..", "../front", "front/rear", r"front\rear", " front",
        "rear.", "rear\t", "CON", "con.txt", "LPT1.camera",
    ):
        assert not _schema_accepts_required_cameras([unsafe])
