"""Read-only indexed access to a Task 8 exported nuScenes dataset."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, cast

from dataset_devkit.export import NUSCENES_VERSION, OFFICIAL_TABLES
from dataset_devkit.identifiers import validate_safe_segment


class DatasetFormatError(ValueError):
    """Raised when exported tables cannot be indexed or safely traversed."""


type JsonRecord = dict[str, Any]


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DatasetFormatError(f"malformed or missing JSON file: {path}") from error


def _safe_asset_filename(
    value: object, *, directory: str, part_count: int, label: str
) -> str:
    if not isinstance(value, str):
        raise DatasetFormatError(f"{label} filename must be a string")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or len(path.parts) != part_count
        or path.parts[0] != directory
    ):
        raise DatasetFormatError(f"unsafe {label} filename")
    try:
        for part in path.parts:
            validate_safe_segment(part)
    except ValueError as error:
        raise DatasetFormatError(f"unsafe {label} filename") from error
    return value


@dataclass(frozen=True)
class Dataset:
    """Eagerly load and index official tables and mz_extensions.

    ``scene_samples`` validates scene chain endpoints and counts. ``camera`` resolves
    one camera row by sample and normalized ``CAM_*`` channel. Extension helpers expose
    validity, computed tags, human annotations, splits, recordings, and validation state.
    """

    dataroot: str | Path
    version: str = NUSCENES_VERSION
    _tables: Mapping[str, tuple[JsonRecord, ...]] = field(init=False, repr=False)
    _token_index: Mapping[str, Mapping[str, JsonRecord]] = field(init=False, repr=False)
    _extensions: Mapping[str, object] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        root = Path(self.dataroot).resolve()
        object.__setattr__(self, "dataroot", root)
        try:
            validate_safe_segment(self.version)
        except ValueError as error:
            raise DatasetFormatError("unsafe dataset version") from error
        version_dir = root / self.version
        tables: dict[str, tuple[JsonRecord, ...]] = {}
        indexes: dict[str, Mapping[str, JsonRecord]] = {}
        for name in OFFICIAL_TABLES:
            value = _load_json(version_dir / f"{name}.json")
            if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
                raise DatasetFormatError(f"table {name} must contain a JSON array of objects")
            records = cast(tuple[JsonRecord, ...], tuple(value))
            index: dict[str, JsonRecord] = {}
            for record in records:
                token = record.get("token")
                if not isinstance(token, str) or not token:
                    raise DatasetFormatError(f"table {name} contains a missing or invalid token")
                if token in index:
                    raise DatasetFormatError(f"table {name} contains duplicate token {token}")
                index[token] = record
                if name == "sample_data":
                    _safe_asset_filename(
                        record.get("filename"),
                        directory="samples",
                        part_count=3,
                        label="sample_data",
                    )
                elif name == "map":
                    filename = _safe_asset_filename(
                        record.get("filename"),
                        directory="maps",
                        part_count=2,
                        label="map",
                    )
                    if not (root / filename).is_file():
                        raise DatasetFormatError("map compatibility mask is missing")
            tables[name] = records
            indexes[name] = MappingProxyType(index)
        extensions: dict[str, object] = {}
        for name in (
            "recordings",
            "gnss",
            "validity",
            "validation",
            "tags",
            "annotations",
            "split",
            "config",
            "content_manifest",
        ):
            extensions[name] = _load_json(root / "mz_extensions" / f"{name}.json")
        object.__setattr__(self, "_tables", MappingProxyType(tables))
        object.__setattr__(self, "_token_index", MappingProxyType(indexes))
        object.__setattr__(self, "_extensions", MappingProxyType(extensions))

    def table(self, table_name: str) -> tuple[JsonRecord, ...]:
        """Return one official table in deterministic file order."""
        try:
            return deepcopy(self._tables[table_name])
        except KeyError as error:
            raise DatasetFormatError(f"unknown table {table_name!r}") from error

    def get(self, table_name: str, token: str) -> JsonRecord:
        """Return one official record by token."""
        try:
            table = self._token_index[table_name]
        except KeyError as error:
            raise DatasetFormatError(f"unknown table {table_name!r}") from error
        try:
            return deepcopy(table[token])
        except KeyError as error:
            raise DatasetFormatError(f"missing token {token!r} in table {table_name}") from error

    def field2token(self, table_name: str, field_name: str, query: object) -> list[str]:
        """Return tokens whose field exactly equals ``query``, in table order."""
        return [
            cast(str, record["token"])
            for record in self.table(table_name)
            if record.get(field_name) == query
        ]

    def scene_samples(self, scene_token: str) -> tuple[JsonRecord, ...]:
        """Traverse and validate a scene's complete sample chain."""
        scene = self.get("scene", scene_token)
        first = scene.get("first_sample_token")
        last = scene.get("last_sample_token")
        count = scene.get("nbr_samples")
        if not isinstance(first, str) or not isinstance(last, str) or not isinstance(count, int):
            raise DatasetFormatError("scene chain metadata is malformed")
        values: list[JsonRecord] = []
        seen: set[str] = set()
        token = first
        previous = ""
        while token:
            if token in seen:
                raise DatasetFormatError("sample chain contains a cycle")
            seen.add(token)
            sample = self.get("sample", token)
            if sample.get("scene_token") != scene_token or sample.get("prev") != previous:
                raise DatasetFormatError("sample chain has a foreign scene or broken prev link")
            values.append(sample)
            previous = token
            next_token = sample.get("next")
            if not isinstance(next_token, str):
                raise DatasetFormatError("sample chain next link is malformed")
            token = next_token
        if not values or values[-1].get("token") != last or len(values) != count:
            raise DatasetFormatError("sample chain endpoint or count differs from scene")
        return tuple(values)

    def camera(self, sample_token: str, channel: str) -> JsonRecord:
        """Resolve exactly one sample_data row for a normalized camera channel."""
        if not channel.startswith("CAM_"):
            raise DatasetFormatError("camera channel must use normalized CAM_* form")
        matches: list[JsonRecord] = []
        for record in self.table("sample_data"):
            if record.get("sample_token") != sample_token:
                continue
            calibration = self.get(
                "calibrated_sensor", cast(str, record.get("calibrated_sensor_token"))
            )
            sensor = self.get("sensor", cast(str, calibration.get("sensor_token")))
            if sensor.get("channel") == channel:
                matches.append(record)
        if len(matches) != 1:
            qualifier = "missing" if not matches else "ambiguous"
            raise DatasetFormatError(
                f"{qualifier} camera row for sample {sample_token!r} channel {channel!r}"
            )
        return matches[0]

    def ego_pose(self, sample_data_token: str) -> JsonRecord:
        """Return the ego pose referenced by one sample_data row."""
        sample_data = self.get("sample_data", sample_data_token)
        token = sample_data.get("ego_pose_token")
        if not isinstance(token, str):
            raise DatasetFormatError("sample_data ego pose reference is malformed")
        return self.get("ego_pose", token)

    def _scene_extension(self, name: str, scene_token: str) -> JsonRecord:
        value = self._extensions[name]
        if not isinstance(value, list):
            raise DatasetFormatError(f"extension {name} must contain an array")
        matches = [
            item
            for item in value
            if isinstance(item, dict) and item.get("scene_token") == scene_token
        ]
        if len(matches) != 1:
            raise DatasetFormatError(f"extension {name} has missing or duplicate scene entry")
        return deepcopy(cast(JsonRecord, matches[0]))

    def validity(self, scene_token: str) -> JsonRecord:
        """Return validity and source-audit evidence for one scene."""
        return self._scene_extension("validity", scene_token)

    def tags(self, scene_token: str) -> JsonRecord:
        """Return computed tags for one scene (human labels are intentionally separate)."""
        return self._scene_extension("tags", scene_token)

    def annotations(self, scene_token: str) -> JsonRecord:
        """Return human-label and annotation references for one scene."""
        value = self._extensions["annotations"]
        if not isinstance(value, dict) or not isinstance(value.get("scenes"), list):
            raise DatasetFormatError("annotations extension is malformed")
        matches = [
            item
            for item in value["scenes"]
            if isinstance(item, dict) and item.get("scene_token") == scene_token
        ]
        if len(matches) != 1:
            raise DatasetFormatError("annotations extension has missing or duplicate scene entry")
        return deepcopy(cast(JsonRecord, matches[0]))

    def _annotation_collection(self, name: str, token_field: str) -> tuple[JsonRecord, ...]:
        value = self._extensions["annotations"]
        if not isinstance(value, dict) or not isinstance(value.get(name), list):
            raise DatasetFormatError(f"annotations {name} collection is malformed")
        records = value[name]
        if any(not isinstance(item, dict) for item in records):
            raise DatasetFormatError(f"annotations {name} collection is malformed")
        typed = cast(list[JsonRecord], records)
        tokens = [item.get(token_field) for item in typed]
        if any(not isinstance(token, str) or not token for token in tokens):
            raise DatasetFormatError(f"annotations {name} contains an invalid reference token")
        if len(tokens) != len(set(tokens)):
            raise DatasetFormatError(f"annotations {name} contains duplicate reference tokens")
        return deepcopy(tuple(typed))

    def annotation_records(self) -> tuple[JsonRecord, ...]:
        """Return all selected human annotation records."""
        return self._annotation_collection("records", "token")

    def annotation_matches(self) -> tuple[JsonRecord, ...]:
        """Return all selected annotation-to-sample match decisions."""
        return self._annotation_collection("matches", "annotation_token")

    def annotation_windows(self) -> tuple[JsonRecord, ...]:
        """Return all selected annotation windows."""
        return self._annotation_collection("windows", "token")

    def annotation_scene_references(self) -> tuple[JsonRecord, ...]:
        """Return scene summaries that keep human labels separate from computed tags."""
        value = self._extensions["annotations"]
        if not isinstance(value, dict) or not isinstance(value.get("scenes"), list):
            raise DatasetFormatError("annotation scene references are malformed")
        records = value["scenes"]
        if any(not isinstance(item, dict) for item in records):
            raise DatasetFormatError("annotation scene references are malformed")
        return deepcopy(cast(tuple[JsonRecord, ...], tuple(records)))

    @staticmethod
    def _one_annotation(
        records: tuple[JsonRecord, ...], field_name: str, token: str, label: str
    ) -> JsonRecord:
        matches = [item for item in records if item.get(field_name) == token]
        if not matches:
            raise DatasetFormatError(f"missing {label} {token!r}")
        if len(matches) > 1:
            raise DatasetFormatError(f"duplicate {label} {token!r}")
        return deepcopy(matches[0])

    def annotation_record(self, token: str) -> JsonRecord:
        """Resolve one annotation record token."""
        return self._one_annotation(self.annotation_records(), "token", token, "annotation record")

    def annotation_match(self, annotation_token: str) -> JsonRecord:
        """Resolve the match decision for one annotation token."""
        return self._one_annotation(
            self.annotation_matches(),
            "annotation_token",
            annotation_token,
            "annotation match",
        )

    def annotation_window(self, token: str) -> JsonRecord:
        """Resolve one annotation window token."""
        return self._one_annotation(self.annotation_windows(), "token", token, "annotation window")

    def scene_annotation_evidence(self, scene_token: str) -> JsonRecord:
        """Resolve a scene summary to its records, matches, and optional window."""
        summary = self.annotations(scene_token)
        references = summary.get("annotation_refs")
        window_reference = summary.get("annotation_window_ref")
        if not isinstance(references, list) or any(
            not isinstance(token, str) for token in references
        ):
            raise DatasetFormatError("scene annotation record references are malformed")
        if not isinstance(window_reference, str):
            raise DatasetFormatError("scene annotation window reference is malformed")
        records = [self.annotation_record(token) for token in references]
        matches = [self.annotation_match(token) for token in references]
        windows = [self.annotation_window(window_reference)] if window_reference else []
        if windows:
            window_tokens = windows[0].get("annotation_tokens")
            if not isinstance(window_tokens, list) or any(
                not isinstance(token, str) for token in window_tokens
            ):
                raise DatasetFormatError("annotation window record references are malformed")
            for token in window_tokens:
                self.annotation_record(token)
        return deepcopy(
            {
                "scene": summary,
                "records": records,
                "matches": matches,
                "windows": windows,
            }
        )

    def _split_records(self) -> list[JsonRecord]:
        value = self._extensions["split"]
        if not isinstance(value, dict) or not isinstance(value.get("assignments"), list):
            raise DatasetFormatError("split extension is malformed")
        if any(not isinstance(item, dict) for item in value["assignments"]):
            raise DatasetFormatError("split assignments are malformed")
        return cast(list[JsonRecord], value["assignments"])

    def split(self, scene_token: str) -> str:
        """Return ``train`` or ``test`` for one scene."""
        matches = [item for item in self._split_records() if item.get("scene_token") == scene_token]
        if len(matches) != 1 or matches[0].get("split") not in {"train", "test"}:
            raise DatasetFormatError(
                "split extension has missing, duplicate, or invalid assignment"
            )
        return cast(str, matches[0]["split"])

    def scenes_in_split(self, split_name: str) -> tuple[str, ...]:
        """Return scene tokens assigned to one split in canonical extension order."""
        if split_name not in {"train", "test"}:
            raise DatasetFormatError("split name must be train or test")
        return tuple(
            cast(str, item["scene_token"])
            for item in self._split_records()
            if item.get("split") == split_name
        )

    def recordings(self) -> tuple[JsonRecord, ...]:
        """Return source-fingerprint/log/channel recording metadata."""
        value = self._extensions["recordings"]
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            raise DatasetFormatError("recordings extension is malformed")
        return deepcopy(cast(tuple[JsonRecord, ...], tuple(value)))

    def validation_report(self) -> JsonRecord:
        """Return the truthful validation state/report extension."""
        value = self._extensions["validation"]
        if not isinstance(value, dict):
            raise DatasetFormatError("validation extension is malformed")
        return deepcopy(cast(JsonRecord, value))
