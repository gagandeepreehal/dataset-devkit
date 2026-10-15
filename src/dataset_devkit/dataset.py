"""Read-only indexed access to a Task 8 exported nuScenes dataset."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, cast

import pyarrow.parquet as pq

from dataset_devkit.config import CameraName
from dataset_devkit.export import NUSCENES_VERSION, OFFICIAL_TABLES
from dataset_devkit.identifiers import validate_safe_segment
from dataset_devkit.indus_ad_export import (
    CameraAnnotationRecord,
    DenseAssetRecord,
    GnssAnnotationRecord,
    camera_annotation_schema,
    dense_asset_schema,
    gnss_annotation_schema,
)
from dataset_devkit.indus_ad_models import (
    CameraAnnotation,
    DenseAssetRef,
    GnssScenarioAnnotation,
    ImageQualitySummary,
    SemanticFraction,
    WeatherWeakLabel,
)


class DatasetFormatError(ValueError):
    """Raised when exported tables cannot be indexed or safely traversed."""


type JsonRecord = dict[str, Any]

_SOURCE_METADATA_KEYS = {
    "schema_version",
    "source_type",
    "privacy_classification",
    "pose_frame",
    "global_horizontal_resolution_m",
}


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DatasetFormatError(f"malformed or missing JSON file: {path}") from error


def _required_string(record: JsonRecord, field_name: str, label: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value:
        raise DatasetFormatError(f"{label} {field_name} must be a nonempty string")
    return value


def _validated_source_metadata(value: object) -> JsonRecord:
    if not isinstance(value, dict) or set(value) != _SOURCE_METADATA_KEYS:
        raise DatasetFormatError("source metadata is malformed")
    source_type = value.get("source_type")
    expected = (
        ("privacy_transformed", "recording_local_enu_v1", 1.0)
        if source_type in {"decoded_hf", "indus_ad_hf"}
        else ("restricted_raw", "web_mercator_v1", None)
        if source_type == "mcap_hf"
        else None
    )
    actual = (
        value.get("privacy_classification"),
        value.get("pose_frame"),
        value.get("global_horizontal_resolution_m"),
    )
    if value.get("schema_version") != 1 or expected is None or actual != expected:
        raise DatasetFormatError("source privacy or pose frame is inconsistent")
    return cast(JsonRecord, value)


def _parquet_rows(path: Path, schema: object) -> list[JsonRecord]:
    try:
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.remove_metadata().equals(
            cast(Any, schema), check_metadata=False
        ):
            raise DatasetFormatError(f"INDUS-AD extension schema differs: {path.name}")
        return cast(list[JsonRecord], parquet.read().to_pylist())
    except DatasetFormatError:
        raise
    except Exception as error:
        raise DatasetFormatError(f"malformed INDUS-AD extension: {path.name}") from error


def _camera_annotation_record(row: JsonRecord) -> CameraAnnotationRecord:
    category_names = (
        "drivable",
        "lane_marking",
        "person_rider",
        "two_wheeler",
        "four_wheeler",
        "building",
        "vegetation",
        "sky",
    )
    weather_names = (
        "clear",
        "overcast",
        "rain",
        "fog",
        "low_light",
        "uncertain_mixed",
    )
    quality_names = (
        "luminance_mean",
        "luminance_p05",
        "luminance_p95",
        "contrast_std",
        "saturation_mean",
        "laplacian_variance",
        "dark_fraction",
        "overexposed_fraction",
    )
    semantic = row["semantic_fractions"]
    if not isinstance(semantic, list):
        raise DatasetFormatError("camera semantic fractions are malformed")
    annotation = CameraAnnotation(
        cast(str, row["recording_id"]),
        cast(CameraName, row["camera_name"]),
        cast(int | None, row["selected_time_of_day_ns"]),
        cast(int, row["recording_offset_ns"]),
        cast(int, row["frame_index"]),
        cast(str, row["frame_id"]),
        cast(str, row["sample_kind"]),
        cast(str, row["window_id"]),
        cast(int, row["decoded_width"]),
        cast(int, row["decoded_height"]),
        tuple(
            SemanticFraction(cast(int, item["class_id"]), cast(float, item["fraction"]))
            for item in cast(list[JsonRecord], semantic)
        ),
        {name: cast(float | None, row[f"{name}_fraction"]) for name in category_names},
        {name: cast(float | None, row[f"depth_{name}"]) for name in ("p10", "p50", "p90")},
        WeatherWeakLabel(
            {
                name: cast(float | None, row[f"weather_{name}_probability"])
                for name in weather_names
            },
            cast(str | None, row["weather_label"]),
            cast(float | None, row["weather_confidence"]),
            cast(float | None, row["weather_margin"]),
            cast(bool | None, row["weather_uncertain"]),
        ),
        ImageQualitySummary({name: cast(float | None, row[name]) for name in quality_names}),
    )
    if row.get("weak_label_provenance") != annotation.provenance:
        raise DatasetFormatError("camera weak-label provenance differs")
    return CameraAnnotationRecord(
        annotation,
        cast(str, row["source_digest"]),
        cast(str, row["scene_token"]),
        cast(str, row["sample_token"]),
        cast(str, row["sample_data_token"]),
    )


def _gnss_annotation_record(row: JsonRecord) -> GnssAnnotationRecord:
    annotation = GnssScenarioAnnotation(
        cast(str, row["recording_id"]),
        cast(int, row["window_index"]),
        cast(int, row["window_start_offset_ns"]),
        cast(int, row["window_end_offset_ns"]),
        cast(str, row["motion_label"]),
        tuple(cast(list[str], row["validity_tags"])),
        tuple(cast(list[str], row["quality_tags"])),
        cast(bool, row["anomaly_candidate"]),
        tuple(cast(list[str], row["anomaly_reasons"])),
        cast(float, row["confidence"]),
        "indus_ad_gnss_annotations_v1",
        cast(str, row["run_id"]),
    )
    if row.get("weak_label_provenance") != annotation.provenance:
        raise DatasetFormatError("GNSS weak-label provenance differs")
    return GnssAnnotationRecord(
        annotation,
        cast(str, row["source_digest"]),
        tuple(cast(list[str], row["scene_tokens"])),
        tuple(cast(list[str], row["sample_tokens"])),
    )


def _dense_asset_record(row: JsonRecord) -> DenseAssetRecord:
    ref = DenseAssetRef(
        cast(str, row["recording_id"]),
        cast(CameraName, row["camera_name"]),
        cast(int, row["frame_index"]),
        cast(int, row["recording_offset_ns"]),
        cast(str, row["frame_id"]),
        cast(Any, row["kind"]),
        cast(str, row["dtype"]),
        cast(int, row["width"]),
        cast(int, row["height"]),
        cast(str, row["member_path"]),
        cast(int, row["payload_bytes"]),
        cast(str, row["payload_sha256"]),
        cast(str, row["shard_path"]),
    )
    return DenseAssetRecord(
        ref,
        cast(str, row["source_digest"]),
        cast(str, row["scene_token"]),
        cast(str, row["sample_data_token"]),
        cast(bool, row["materialized"]),
    )


def _safe_asset_filename(value: object, *, directory: str, part_count: int, label: str) -> str:
    if not isinstance(value, str):
        raise DatasetFormatError(f"{label} filename must be a string")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != part_count or path.parts[0] != directory:
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
    _camera_index: Mapping[tuple[str, str], str] = field(init=False, repr=False)
    _indus_camera: tuple[CameraAnnotationRecord, ...] = field(init=False, repr=False)
    _indus_gnss: tuple[GnssAnnotationRecord, ...] = field(init=False, repr=False)
    _indus_dense: tuple[DenseAssetRecord, ...] = field(init=False, repr=False)

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
        camera_index: dict[tuple[str, str], str] = {}
        for calibration in tables["calibrated_sensor"]:
            sensor_token = _required_string(calibration, "sensor_token", "calibrated_sensor")
            if sensor_token not in indexes["sensor"]:
                raise DatasetFormatError("calibrated_sensor references a missing sensor")
        for item in tables["sample_data"]:
            sample_token = _required_string(item, "sample_token", "sample_data")
            calibration_token = _required_string(item, "calibrated_sensor_token", "sample_data")
            ego_pose_token = _required_string(item, "ego_pose_token", "sample_data")
            if sample_token not in indexes["sample"]:
                raise DatasetFormatError("sample_data references a missing sample")
            if calibration_token not in indexes["calibrated_sensor"]:
                raise DatasetFormatError("sample_data references a missing calibrated_sensor")
            if ego_pose_token not in indexes["ego_pose"]:
                raise DatasetFormatError("sample_data references a missing ego_pose")
            calibration = indexes["calibrated_sensor"][calibration_token]
            sensor_token = _required_string(calibration, "sensor_token", "calibrated_sensor")
            sensor = indexes["sensor"][sensor_token]
            channel = _required_string(sensor, "channel", "sensor")
            if not channel.startswith("CAM_"):
                raise DatasetFormatError("camera sensor channel must use normalized CAM_* form")
            key = (sample_token, channel)
            if key in camera_index:
                raise DatasetFormatError("duplicate or ambiguous camera sample/channel reference")
            camera_index[key] = _required_string(item, "token", "sample_data")
        extensions: dict[str, object] = {}
        for name in (
            "source",
            "recordings",
            "gnss",
            "validity",
            "validation",
            "tags",
            "annotations",
            "split",
            "config",
            "content_manifest",
            "pipeline_audit",
        ):
            extensions[name] = _load_json(root / "mz_extensions" / f"{name}.json")
        source_metadata = _validated_source_metadata(extensions["source"])
        indus_camera: tuple[CameraAnnotationRecord, ...] = ()
        indus_gnss: tuple[GnssAnnotationRecord, ...] = ()
        indus_dense: tuple[DenseAssetRecord, ...] = ()
        if source_metadata["source_type"] == "indus_ad_hf":
            directory = root / "mz_extensions" / "indus_ad"
            contract = _load_json(directory / "contract.json")
            if not isinstance(contract, dict):
                raise DatasetFormatError("INDUS-AD contract is malformed")
            extensions["indus_ad/contract"] = contract
            indus_camera = tuple(
                _camera_annotation_record(row)
                for row in _parquet_rows(
                    directory / "camera_annotations.parquet",
                    camera_annotation_schema(),
                )
            )
            indus_gnss = tuple(
                _gnss_annotation_record(row)
                for row in _parquet_rows(
                    directory / "gnss_annotations.parquet",
                    gnss_annotation_schema(),
                )
            )
            indus_dense = tuple(
                _dense_asset_record(row)
                for row in _parquet_rows(directory / "dense_assets.parquet", dense_asset_schema())
            )
            if (
                contract.get("camera_rows") != len(indus_camera)
                or contract.get("gnss_rows") != len(indus_gnss)
                or contract.get("dense_rows") != len(indus_dense)
            ):
                raise DatasetFormatError("INDUS-AD contract row counts differ")
            if len({item.frame_id for item in indus_camera}) != len(indus_camera):
                raise DatasetFormatError("duplicate INDUS-AD camera annotation identity")
            if len({(item.recording_id, item.window_index) for item in indus_gnss}) != len(
                indus_gnss
            ):
                raise DatasetFormatError("duplicate INDUS-AD GNSS annotation identity")
            if len({(item.frame_id, item.kind) for item in indus_dense}) != len(indus_dense):
                raise DatasetFormatError("duplicate INDUS-AD dense asset identity")
        object.__setattr__(self, "_tables", MappingProxyType(tables))
        object.__setattr__(self, "_token_index", MappingProxyType(indexes))
        object.__setattr__(self, "_extensions", MappingProxyType(extensions))
        object.__setattr__(self, "_camera_index", MappingProxyType(camera_index))
        object.__setattr__(self, "_indus_camera", indus_camera)
        object.__setattr__(self, "_indus_gnss", indus_gnss)
        object.__setattr__(self, "_indus_dense", indus_dense)
        if source_metadata["source_type"] == "indus_ad_hf":
            from dataset_devkit.provenance import (
                IndusAdSourceFingerprint,
                fingerprint_from_dict,
            )
            from dataset_devkit.validation import (
                ValidationFinding,
                _validate_indus_ad_extensions,
            )

            fingerprints: dict[str, IndusAdSourceFingerprint] = {}
            for recording in self.recordings():
                fingerprint = fingerprint_from_dict(recording["source"])
                if not isinstance(fingerprint, IndusAdSourceFingerprint):
                    raise DatasetFormatError("integrated recording source type differs")
                fingerprints[fingerprint.digest] = fingerprint
            integrated_findings: list[ValidationFinding] = []
            log_sources = {row["log_token"]: row["source_digest"] for row in self.recordings()}
            _validate_indus_ad_extensions(
                root,
                fingerprints=fingerprints,
                scenes=set(indexes["scene"]),
                samples=set(indexes["sample"]),
                sample_data=set(indexes["sample_data"]),
                findings=integrated_findings,
                scene_sources={
                    row["token"]: log_sources.get(row["log_token"], "") for row in tables["scene"]
                },
                sample_scenes={row["token"]: row["scene_token"] for row in tables["sample"]},
                data_samples={row["token"]: row["sample_token"] for row in tables["sample_data"]},
            )
            if integrated_findings:
                raise DatasetFormatError("INDUS-AD extension validation failed")

    def table(self, table_name: str) -> tuple[JsonRecord, ...]:
        """Return one official table in deterministic file order."""
        if not isinstance(table_name, str):
            raise DatasetFormatError("table name must be a string")
        try:
            return deepcopy(self._tables[table_name])
        except KeyError as error:
            raise DatasetFormatError(f"unknown table {table_name!r}") from error

    def source_metadata(self) -> JsonRecord:
        """Return the validated source privacy and coordinate-frame contract."""
        return deepcopy(_validated_source_metadata(self._extensions["source"]))

    def get(self, table_name: str, token: str) -> JsonRecord:
        """Return one official record by token."""
        if not isinstance(table_name, str) or not isinstance(token, str):
            raise DatasetFormatError("table name and token must be strings")
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
        if (
            not isinstance(sample_token, str)
            or not isinstance(channel, str)
            or not channel.startswith("CAM_")
        ):
            raise DatasetFormatError("camera channel must use normalized CAM_* form")
        token = self._camera_index.get((sample_token, channel))
        if token is None:
            raise DatasetFormatError(
                f"missing camera row for sample {sample_token!r} channel {channel!r}"
            )
        return self.get("sample_data", token)

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
        value = self._extensions["validity"]
        if not isinstance(value, dict) or not isinstance(value.get("scenes"), list):
            raise DatasetFormatError("validity extension is malformed")
        matches = [
            item
            for item in value["scenes"]
            if isinstance(item, dict) and item.get("scene_token") == scene_token
        ]
        if len(matches) != 1:
            raise DatasetFormatError("validity has missing or duplicate scene entry")
        return deepcopy(cast(JsonRecord, matches[0]))

    def recording_validity(self, source_digest: str) -> JsonRecord:
        """Return the complete validity report for one exported source."""
        value = self._extensions["validity"]
        if not isinstance(value, dict) or not isinstance(value.get("recordings"), list):
            raise DatasetFormatError("validity extension is malformed")
        matches = [
            item
            for item in value["recordings"]
            if isinstance(item, dict) and item.get("source_digest") == source_digest
        ]
        if len(matches) != 1:
            raise DatasetFormatError("validity has missing or duplicate recording entry")
        return deepcopy(cast(JsonRecord, matches[0]))

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
        if not isinstance(token, str):
            raise DatasetFormatError(f"{label} token must be a string")
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
        if len(references) != len(set(references)):
            raise DatasetFormatError("scene annotation record references contain duplicates")
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
            if window_tokens != references:
                raise DatasetFormatError(
                    "annotation window tokens must exactly match scene annotation references"
                )
        source_digest = summary.get("source_digest")
        if not isinstance(source_digest, str) or not source_digest:
            raise DatasetFormatError("scene annotation source digest is malformed")
        if any(item.get("source_digest") != source_digest for item in records):
            raise DatasetFormatError("annotation record source differs from scene source")
        if any(item.get("source_digest") != source_digest for item in matches):
            raise DatasetFormatError("annotation match source differs from scene source")
        if any(item.get("source_digest") != source_digest for item in windows):
            raise DatasetFormatError("annotation window source differs from scene source")
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

    def pipeline_audit(self) -> JsonRecord:
        """Return filtering, rejection, and ordered scenario-selection evidence."""
        value = self._extensions["pipeline_audit"]
        if not isinstance(value, dict):
            raise DatasetFormatError("pipeline audit extension is malformed")
        return deepcopy(cast(JsonRecord, value))

    def camera_annotation(self, frame_id: str) -> CameraAnnotationRecord:
        """Return the unique selected camera weak label for ``frame_id``."""
        matches = tuple(item for item in self._indus_camera if item.frame_id == frame_id)
        if len(matches) != 1:
            raise DatasetFormatError("camera annotation identity is missing or ambiguous")
        return matches[0]

    def camera_annotations(
        self,
        *,
        recording_id: str | None = None,
        camera_name: str | None = None,
    ) -> tuple[CameraAnnotationRecord, ...]:
        """Return camera weak labels in deterministic export order."""
        return tuple(
            item
            for item in self._indus_camera
            if (recording_id is None or item.recording_id == recording_id)
            and (camera_name is None or item.camera_name == camera_name)
        )

    def gnss_annotation(self, recording_id: str, window_index: int) -> GnssAnnotationRecord:
        """Return one unique GNSS scenario weak label."""
        matches = tuple(
            item
            for item in self._indus_gnss
            if item.recording_id == recording_id and item.window_index == window_index
        )
        if len(matches) != 1:
            raise DatasetFormatError("GNSS annotation identity is missing or ambiguous")
        return matches[0]

    def gnss_annotations(
        self, *, recording_id: str | None = None
    ) -> tuple[GnssAnnotationRecord, ...]:
        """Return GNSS scenario weak labels in deterministic export order."""
        return tuple(
            item
            for item in self._indus_gnss
            if recording_id is None or item.recording_id == recording_id
        )

    def dense_asset(self, frame_id: str, kind: str) -> DenseAssetRecord:
        """Return one semantic or depth asset identity for a selected frame."""
        if kind not in {"semantic", "depth"}:
            raise DatasetFormatError("dense asset kind must be semantic or depth")
        matches = tuple(
            item for item in self._indus_dense if item.frame_id == frame_id and item.kind == kind
        )
        if len(matches) != 1:
            raise DatasetFormatError("dense asset identity is missing or ambiguous")
        return matches[0]

    def indus_ad_contract(self) -> JsonRecord:
        """Return a defensive copy of the integrated source/export contract."""
        value = self._extensions.get("indus_ad/contract")
        if not isinstance(value, dict):
            raise DatasetFormatError("INDUS-AD contract is unavailable")
        return deepcopy(cast(JsonRecord, value))
