from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataset_devkit.indus_ad_annotations import (
    IndusAdAnnotationError,
    gnss_scenario_schema,
    parse_camera_annotations,
    parse_dense_indexes,
    parse_gnss_scenarios,
)
from dataset_devkit.indus_ad_contract import camera_annotation_schema
from dataset_devkit.indus_ad_models import SemanticFraction
from indus_ad_fixture import write_indus_ad_fixture


def _rewrite_camera(path: Path, mutation: object) -> None:
    table = pq.ParquetFile(path).read()
    rows = table.to_pylist()
    assert callable(mutation)
    mutation(rows[0])
    pq.write_table(pa.Table.from_pylist(rows, schema=camera_annotation_schema()), path)


def test_camera_parser_preserves_weak_label_values(tmp_path: Path) -> None:
    fixture = write_indus_ad_fixture(tmp_path)
    bundle = parse_camera_annotations(
        fixture.camera_annotations,
        expected_recordings={fixture.recording_id: fixture.duration_ns},
    )

    row = bundle.camera[0]
    assert row.identity == (fixture.recording_id, "frame-fixture")
    assert row.semantic_fractions == (SemanticFraction(class_id=13, fraction=0.25),)
    assert row.weather.label == "clear"
    assert row.weather.probabilities["clear"] == 1.0
    assert row.provenance == "machine_generated_weak_pseudo_label"
    assert bundle.availability["gnss"].state == "not_selected"


@pytest.mark.parametrize("label", ["rain_wet_road", "fog_haze", "low_light_night"])
def test_camera_parser_accepts_published_weather_vocabulary(tmp_path: Path, label: str) -> None:
    fixture = write_indus_ad_fixture(tmp_path)
    _rewrite_camera(fixture.camera_annotations, lambda row: row.update(weather_label=label))
    bundle = parse_camera_annotations(
        fixture.camera_annotations,
        expected_recordings={fixture.recording_id: fixture.duration_ns},
    )
    assert bundle.camera[0].weather.label == label


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.__setitem__("depth_p10", -1.0),
        lambda row: row.__setitem__("depth_p10", 2.0),
        lambda row: row.__setitem__("depth_p50", 1.0),
        lambda row: row.__setitem__("depth_p90", 0.25),
        lambda row: row.__setitem__("weather_clear_probability", float("nan")),
        lambda row: row.__setitem__("decoded_width", 0),
        lambda row: row.__setitem__("recording_offset_ns", 2_000_000_000),
        lambda row: row.__setitem__("recording_id", "foreign-recording"),
        lambda row: row.__setitem__(
            "semantic_fractions",
            [{"class_id": 1, "fraction": 0.8}, {"class_id": 2, "fraction": 0.3}],
        ),
    ],
)
def test_camera_parser_rejects_invalid_weak_label_rows(tmp_path: Path, mutation: object) -> None:
    fixture = write_indus_ad_fixture(tmp_path)
    _rewrite_camera(fixture.camera_annotations, mutation)

    with pytest.raises(IndusAdAnnotationError):
        parse_camera_annotations(
            fixture.camera_annotations,
            expected_recordings={fixture.recording_id: fixture.duration_ns},
        )


def test_camera_parser_rejects_duplicate_frame_ids(tmp_path: Path) -> None:
    fixture = write_indus_ad_fixture(tmp_path)
    table = pq.ParquetFile(fixture.camera_annotations).read()
    rows = table.to_pylist()
    pq.write_table(
        pa.Table.from_pylist([rows[0], dict(rows[0])], schema=camera_annotation_schema()),
        fixture.camera_annotations,
    )

    with pytest.raises(IndusAdAnnotationError, match="duplicate camera frame"):
        parse_camera_annotations(
            fixture.camera_annotations,
            expected_recordings={fixture.recording_id: fixture.duration_ns},
        )


def test_dense_parser_binds_refs_without_opening_archives(tmp_path: Path) -> None:
    fixture = write_indus_ad_fixture(tmp_path)
    bundle = parse_camera_annotations(
        fixture.camera_annotations,
        expected_recordings={fixture.recording_id: fixture.duration_ns},
    )

    refs = parse_dense_indexes([fixture.dense_index], camera_annotations=bundle.camera)

    assert {item.kind for item in refs} == {"semantic", "depth"}
    assert {item.frame_id for item in refs} == {"frame-fixture"}


def test_gnss_parser_accepts_relative_privacy_minimized_rows(tmp_path: Path) -> None:
    fixture = write_indus_ad_fixture(tmp_path)
    path = tmp_path / "gnss-annotations.parquet"
    row = {
        "schema_version": "indus_ad_gnss_annotations_v1",
        "source_revision": fixture.source.revision,
        "recording_id": fixture.recording_id,
        "window_index": 0,
        "window_start_offset_ns": 0,
        "window_end_offset_ns": fixture.duration_ns,
        "motion_label": "moving",
        "validity_tags": ["valid"],
        "quality_tags": ["good"],
        "anomaly_candidate": False,
        "anomaly_reasons": [],
        "confidence": 0.9,
        "run_id": "fixture-run",
    }
    pq.write_table(pa.Table.from_pylist([row], schema=gnss_scenario_schema()), path)

    parsed = parse_gnss_scenarios(
        path,
        expected_recordings={fixture.recording_id: fixture.duration_ns},
        expected_source_revision=fixture.source.revision,
    )

    assert parsed[0].window_index == 0
    assert parsed[0].provenance == "machine_generated_weak_pseudo_label"


def test_gnss_parser_rejects_foreign_revision_and_forbidden_schema(
    tmp_path: Path,
) -> None:
    fixture = write_indus_ad_fixture(tmp_path)
    path = tmp_path / "gnss-annotations.parquet"
    schema = gnss_scenario_schema().append(pa.field("latitude", pa.float64()))
    row = {
        "schema_version": "indus_ad_gnss_annotations_v1",
        "source_revision": "f" * 40,
        "recording_id": fixture.recording_id,
        "window_index": 0,
        "window_start_offset_ns": 0,
        "window_end_offset_ns": 1,
        "motion_label": "moving",
        "validity_tags": [],
        "quality_tags": [],
        "anomaly_candidate": False,
        "anomaly_reasons": [],
        "confidence": 0.5,
        "run_id": "fixture-run",
        "latitude": 12.0,
    }
    pq.write_table(pa.Table.from_pylist([row], schema=schema), path)

    with pytest.raises(IndusAdAnnotationError, match="schema"):
        parse_gnss_scenarios(
            path,
            expected_recordings={fixture.recording_id: fixture.duration_ns},
            expected_source_revision=fixture.source.revision,
        )
