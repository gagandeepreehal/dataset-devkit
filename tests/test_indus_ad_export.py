from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from conftest import FeatureFactory
from dataset_devkit.config import GlobalConfig
from dataset_devkit.dataset import Dataset
from dataset_devkit.export import export_dataset
from dataset_devkit.indus_ad_annotations import (
    parse_camera_annotations,
    parse_dense_indexes,
)
from dataset_devkit.indus_ad_join import (
    AnnotationJoinAudit,
    JoinedCameraAnnotation,
    JoinedDenseAsset,
    JoinedIndusAdAnnotations,
)
from indus_ad_fixture import write_indus_ad_fixture
from test_export_dataset import _evidence


def test_export_and_sdk_expose_selected_indus_weak_labels(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    evidence = _evidence(tmp_path, config_factory, feature_factory)
    graph = evidence.graphs[0]
    scene = graph.scenes[0]
    sample_data = next(item for item in graph.sample_data if item.scene_token == scene.token)
    fixture = write_indus_ad_fixture(tmp_path / "indus")
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
        JoinedDenseAsset(
            ref,
            graph.source.digest,
            scene.token,
            sample_data.token,
        )
        for ref in dense_refs
    )
    annotations = JoinedIndusAdAnnotations(
        camera=(camera,),
        dense=dense,
        audits=(
            AnnotationJoinAudit(
                fixture.recording_id,
                1,
                1,
                1,
                0,
                0,
                2,
                2,
                0,
            ),
        ),
    )
    evidence = replace(
        evidence,
        source_type="indus_ad_hf",
        privacy_classification="privacy_transformed",
        pose_frame="recording_local_enu_v1",
        global_horizontal_resolution_m=1.0,
        indus_ad_annotations=annotations,
    )
    root = tmp_path / "export"

    export_dataset(root, evidence)

    camera_payload = json.loads(
        (root / "mz_extensions/indus_ad/camera.json").read_text(encoding="utf-8")
    )
    assert camera_payload["label_kind"] == "weak_pseudo_label"
    assert camera_payload["rows"][0]["provenance"] == ("machine_generated_weak_pseudo_label")
    dataset = Dataset(root)
    assert dataset.indus_ad_annotations("camera")[0]["sample_data_token"] == (sample_data.token)
    dense_rows = dataset.indus_ad_annotations("dense_assets")
    assert {item["kind"] for item in dense_rows} == {"semantic", "depth"}
    assert dataset.indus_ad_dense_asset(dense_rows[0]) is None
    assert dataset.indus_ad_join_audit()[0]["dense_joined"] == 2
