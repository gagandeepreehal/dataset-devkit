from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from dataset_devkit.config import GlobalConfig
from dataset_devkit.indus_ad_annotations import (
    parse_camera_annotations,
    parse_dense_indexes,
)
from dataset_devkit.indus_ad_join import join_indus_ad_annotations
from dataset_devkit.indus_ad_models import (
    GnssScenarioAnnotation,
    IndusAdAnnotationBundle,
)
from dataset_devkit.provenance import IndusAdSourceFingerprint
from dataset_devkit.scenes import build_recording_scenes
from indus_ad_fixture import write_indus_ad_fixture
from test_scenes import _config, _report


def test_join_binds_exact_selected_camera_dense_and_gnss_rows(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    fixture = write_indus_ad_fixture(tmp_path / "fixture")
    report = _report(tmp_path / "recording", (0, 1_000_000))
    audits = tuple(
        replace(
            audit,
            camera_timestamps=tuple(
                ("cam_front" if channel == "front" else channel, timestamp)
                for channel, timestamp in audit.camera_timestamps
            ),
            samples=tuple(
                replace(
                    sample,
                    camera_name=(
                        "cam_front" if sample.camera_name == "front" else sample.camera_name
                    ),
                    staged_image=replace(
                        sample.staged_image,
                        camera_name=(
                            "cam_front" if sample.camera_name == "front" else sample.camera_name
                        ),
                    ),
                )
                for sample in audit.samples
            ),
        )
        for audit in report.final_candidates
    )
    report = replace(report, sample_audits=audits, final_candidates=audits)
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
        report,
        source,
        _config(config_factory(), min_samples=2, max_duration_s=10),
    )
    selected = frozenset({graph.scenes[0].token})
    sample_data = next(item for item in graph.sample_data if item.channel == "cam_front")
    camera_bundle = parse_camera_annotations(
        fixture.camera_annotations,
        expected_recordings={fixture.recording_id: fixture.duration_ns},
    )
    camera = replace(camera_bundle.camera[0], recording_offset_ns=sample_data.timestamp_ns)
    dense = parse_dense_indexes([fixture.dense_index], camera_annotations=camera_bundle.camera)
    gnss = GnssScenarioAnnotation(
        fixture.recording_id,
        0,
        graph.scenes[0].first_timestamp_ns,
        graph.scenes[0].last_timestamp_ns + 1,
        "moving",
        ("valid",),
        ("good",),
        False,
        (),
        0.9,
        "indus_ad_gnss_annotations_v1",
        "fixture-run",
    )
    bundle = IndusAdAnnotationBundle((camera,), (gnss,), dense)

    joined = join_indus_ad_annotations(
        graph=graph,
        selected_scene_tokens=selected,
        annotations=bundle,
    )

    assert joined.camera[0].sample_data_token == sample_data.token
    assert joined.camera[0].annotation.provenance == ("machine_generated_weak_pseudo_label")
    assert {item.ref.kind for item in joined.dense} == {"semantic", "depth"}
    assert joined.gnss[0].scene_tokens == (graph.scenes[0].token,)
    assert joined.audits[0].camera_joined == 1
    assert joined.audits[0].dense_materialized == 0


def test_join_excludes_annotations_outside_selected_scenes(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    fixture = write_indus_ad_fixture(tmp_path / "fixture")
    report = _report(tmp_path / "recording", (0, 1_000_000))
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
        report,
        source,
        _config(config_factory(), min_samples=2, max_duration_s=10),
    )
    camera_bundle = parse_camera_annotations(
        fixture.camera_annotations,
        expected_recordings={fixture.recording_id: fixture.duration_ns},
    )

    joined = join_indus_ad_annotations(
        graph=graph,
        selected_scene_tokens=frozenset(),
        annotations=camera_bundle,
    )

    assert joined.camera == ()
    assert joined.audits[0].selected_scene_count == 0
