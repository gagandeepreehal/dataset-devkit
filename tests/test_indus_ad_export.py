from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import FeatureFactory
from dataset_devkit.config import GlobalConfig, GlobalConfigV21
from dataset_devkit.dataset import Dataset, DatasetFormatError
from dataset_devkit.export import export_dataset
from dataset_devkit.indus_ad_annotations import parse_camera_annotations, parse_dense_indexes
from dataset_devkit.indus_ad_join import (
    AnnotationJoinAudit,
    JoinedCameraAnnotation,
    JoinedDenseAsset,
    JoinedIndusAdAnnotations,
)
from dataset_devkit.provenance import IndusAdSourceFingerprint
from dataset_devkit.scenario_selection import select_scenarios
from dataset_devkit.scenes import build_recording_scenes
from dataset_devkit.split import split_selected_scenes
from dataset_devkit.validation import validate_dataset
from indus_ad_fixture import write_indus_ad_fixture
from test_export_dataset import _evidence


@pytest.mark.parametrize("mutation", [None, "provenance", "revision", "token", "schema"])
def test_export_and_sdk_expose_selected_indus_weak_labels(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
    mutation: str | None,
) -> None:
    evidence = _evidence(tmp_path, config_factory, feature_factory)
    fixture = write_indus_ad_fixture(tmp_path / "indus")
    source = IndusAdSourceFingerprint(
        fixture.source.repo_id,
        fixture.source.revision,
        "indus_ad_integrated_v1",
        fixture.recording_id,
        "a" * 64,
        "b" * 64,
        "c" * 64,
        100,
    )
    graph = build_recording_scenes(
        evidence.validity_reports[0][1], source, evidence.resolved_config
    )
    population = tuple(
        replace(
            item,
            scene_token=new_scene.token,
            scene_name=new_scene.name,
            source=source,
            source_repo_path=fixture.recording_id,
        )
        for item, new_scene in zip(evidence.features_population, graph.scenes, strict=True)
    )
    selection = select_scenarios(population, evidence.scenarios_config)
    split = split_selected_scenes(
        selection,
        population,
        evidence.scenarios_config,
        (graph,),
        evidence.split_config,
    )
    common = evidence.resolved_config.model_dump()
    common.pop("schema_version")
    common.pop("huggingface")
    common.pop("topics")
    resolved = GlobalConfigV21(schema_version="2.1", source=fixture.source, **common)
    scene = graph.scenes[0]
    sample_data = next(item for item in graph.sample_data if item.scene_token == scene.token)
    camera_bundle = parse_camera_annotations(
        fixture.camera_annotations,
        expected_recordings={fixture.recording_id: fixture.duration_ns},
    )
    dense_refs = parse_dense_indexes([fixture.dense_index], camera_annotations=camera_bundle.camera)
    camera = JoinedCameraAnnotation(
        camera_bundle.camera[0],
        graph.source.digest,
        scene.token,
        sample_data.sample_token,
        sample_data.token,
    )
    dense = tuple(
        JoinedDenseAsset(ref, graph.source.digest, scene.token, sample_data.token)
        for ref in dense_refs
    )
    annotations = JoinedIndusAdAnnotations(
        camera=(camera,),
        dense=dense,
        audits=(AnnotationJoinAudit(fixture.recording_id, 1, 1, 1, 0, 0, 2, 2, 0),),
    )
    evidence = replace(
        evidence,
        features_population=population,
        selection=selection,
        graphs=(graph,),
        split=split,
        resolved_config=resolved,
        validity_reports=((source, evidence.validity_reports[0][1]),),
        source_type="indus_ad_hf",
        privacy_classification="privacy_transformed",
        pose_frame="recording_local_enu_v1",
        global_horizontal_resolution_m=1.0,
        indus_ad_annotations=annotations,
    )
    root = tmp_path / "export"

    export_dataset(root, evidence)

    camera_rows = pq.read_table(
        root / "mz_extensions/indus_ad/camera_annotations.parquet"
    ).to_pylist()
    assert camera_rows[0]["weak_label_provenance"] == ("machine_generated_weak_pseudo_label")
    dataset = Dataset(root)
    assert dataset.camera_annotation("frame-fixture").sample_data_token == sample_data.token
    assert {dataset.dense_asset("frame-fixture", kind).kind for kind in ("semantic", "depth")} == {
        "semantic",
        "depth",
    }
    assert dataset.indus_ad_contract()["dense_rows"] == 2
    report = validate_dataset(root, official_smoke=False, verify_manifest=False)
    assert not any(
        item.code in {"extension_schema", "extension_value", "extension_reference"}
        and "indus" in item.location
        for item in report.findings
    )
    if mutation is not None:
        path = root / "mz_extensions/indus_ad/camera_annotations.parquet"
        if mutation == "revision":
            path = root / "mz_extensions/indus_ad/contract.json"
            payload = json.loads(path.read_text())
            payload["source_revision"] = "f" * 40
            path.write_text(json.dumps(payload))
        else:
            table = pq.ParquetFile(path).read()
            rows = table.to_pylist()
            schema = table.schema
            if mutation == "provenance":
                rows[0]["weak_label_provenance"] = "ground_truth"
            elif mutation == "token":
                rows[0]["sample_data_token"] = "unknown"
            else:
                schema = schema.append(pa.field("latitude", pa.float64()))
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
        changed = validate_dataset(root, official_smoke=False, verify_manifest=False)
        assert not changed.succeeded
        assert any(
            item.code in {"extension_schema", "extension_reference", "extension_value"}
            and "indus" in item.location
            for item in changed.findings
        )
        with pytest.raises((DatasetFormatError, ValueError)):
            Dataset(root)
