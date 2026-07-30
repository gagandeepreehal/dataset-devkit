from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from conftest import FeatureFactory
from dataset_devkit.config import GlobalConfig, ScenarioRuleConfig, ScenariosConfig, SplitConfig
from dataset_devkit.dataset import Dataset, DatasetFormatError
from dataset_devkit.export import ExportEvidence, export_dataset
from dataset_devkit.provenance import SourceFingerprint
from dataset_devkit.scenario_selection import select_scenarios
from dataset_devkit.scenes import build_recording_scenes
from dataset_devkit.split import split_selected_scenes
from test_scenes import _annotation_config, _annotations, _config, _report
from test_split import _scenarios, _selection


def _evidence(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
    *,
    recordings: int = 1,
    tight_timestamps: bool = False,
    colliding_channels: bool = False,
) -> ExportEvidence:
    timestamps = (
        (0, 1, 2_000_000_000, 2_000_000_001)
        if tight_timestamps
        else (0, 1_000_000, 2_000_000_000, 2_001_000_000)
    )
    max_duration = 0.000000001 if tight_timestamps else 0.001
    graphs_list = []
    for index in range(recordings):
        report = _report(
            tmp_path / "inputs" / f"recording-{index}",
            timestamps,
        )
        if colliding_channels:
            names = {"front": "cam-front", "rear": "cam_front"}
            audits = tuple(
                replace(
                    audit,
                    camera_timestamps=tuple(
                        (names[channel], timestamp)
                        for channel, timestamp in audit.camera_timestamps
                    ),
                    samples=tuple(
                        replace(
                            sample,
                            camera_name=names[sample.camera_name],
                            staged_image=replace(
                                sample.staged_image,
                                camera_name=names[sample.camera_name],
                            ),
                        )
                        for sample in audit.samples
                    ),
                )
                for audit in report.final_candidates
            )
            report = replace(report, sample_audits=audits, final_candidates=audits)
        graphs_list.append(
            build_recording_scenes(
                report,
                SourceFingerprint(
                    "https://example.blob.core.windows.net",
                    "recordings",
                    f"mcap-h265/recording-{index}.mcap",
                    f'"recording-{index}"',
                    4,
                ),
                _config(config_factory(), max_duration_s=max_duration),
            )
        )
    graphs = tuple(graphs_list)
    selection = _selection(graphs, feature_factory)
    scenarios = _scenarios(selection)
    split_config = SplitConfig(test_fraction=0.5, seed=77, stratify=True)
    split = split_selected_scenes(
        selection,
        selection.selected_scenes,
        scenarios,
        graphs,
        split_config,
    )
    config = _config(config_factory(), max_duration_s=max_duration).model_copy(
        update={"scenarios": scenarios, "split": split_config}
    )
    return ExportEvidence(
        features_population=selection.selected_scenes,
        scenarios_config=scenarios,
        selection=selection,
        graphs=graphs,
        split_config=split_config,
        split=split,
        resolved_config=config,
        content_manifest={"schema_version": 1, "content_sha256": "a" * 64},
    )


def _annotated_evidence(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> ExportEvidence:
    source = SourceFingerprint(
        "https://example.blob.core.windows.net",
        "recordings",
        "mcap-h265/annotated.mcap",
        '"annotated"',
        2,
    )
    annotation_path = _annotations(
        tmp_path / "annotations.jsonl",
        [
            {
                "blob_path": source.blob_path,
                "timestamp_ns": 0,
                "labels": ["merge"],
            }
        ],
    )
    config = _annotation_config(
        config_factory(),
        mode="annotation_only",
        tolerance_ms=1.0,
        before_s=0.0,
        after_s=0.001,
    )
    config = config.model_copy(
        update={"annotations": config.annotations.model_copy(update={"path": annotation_path})}
    )
    graph = build_recording_scenes(
        _report(tmp_path / "inputs", (0, 1_000_000)), source, config
    )
    assert len(graph.scenes) == 1
    scene = graph.scenes[0]
    population = (
        feature_factory(
            scene_token=scene.token,
            scene_name=scene.name,
            source=source,
            source_blob_path=source.blob_path,
            human_labels=scene.labels,
            computed_tags=("road",),
        ),
    )
    scenarios = ScenariosConfig(
        seed=5,
        rules=[ScenarioRuleConfig(name="road", quota=1, required_all_tags=["road"])],
    )
    selection = select_scenarios(population, scenarios)
    split_config = SplitConfig(test_fraction=0.5, seed=77, stratify=True)
    split = split_selected_scenes(
        selection, population, scenarios, (graph,), split_config
    )
    resolved = config.model_copy(
        update={"scenarios": scenarios, "split": split_config}
    )
    return ExportEvidence(
        population,
        scenarios,
        selection,
        (graph,),
        split_config,
        split,
        resolved,
        {"schema_version": 1},
    )


def test_export_official_tables_extensions_and_sdk_queries(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    evidence = _evidence(tmp_path, config_factory, feature_factory, recordings=2)
    root = tmp_path / "dataset"
    result = export_dataset(root, evidence)

    expected_tables = {
        "category", "attribute", "visibility", "instance", "sensor",
        "calibrated_sensor", "ego_pose", "log", "scene", "sample",
        "sample_data", "sample_annotation", "map",
    }
    version = root / "v1.0-trainval"
    assert {path.stem for path in version.glob("*.json")} == expected_tables
    for name in ("category", "attribute", "visibility", "instance", "sample_annotation"):
        assert json.loads((version / f"{name}.json").read_text()) == []
    maps = json.loads((version / "map.json").read_text())
    logs = json.loads((version / "log.json").read_text())
    assert len(maps) == 1
    assert maps[0]["category"] == "compatibility_scaffold"
    assert maps[0]["log_tokens"] == sorted(item["token"] for item in logs)
    assert maps[0]["filename"] == "maps/dataset-devkit-loader-compatibility.png"
    with Image.open(root / maps[0]["filename"]) as mask:
        assert mask.mode == "L"
        assert mask.size == (1, 1)
        assert mask.getpixel((0, 0)) == 0
    assert result.scene_count == 4
    assert result.sample_data_count == 16
    assert result.content_fingerprint == hashlib.sha256(
        (root / "mz_extensions" / "content_manifest.json").read_bytes()
    ).hexdigest()

    dataset = Dataset(root)
    first_scene = dataset.table("scene")[0]
    samples = dataset.scene_samples(first_scene["token"])
    assert len(samples) == first_scene["nbr_samples"]
    camera = dataset.camera(samples[0]["token"], "CAM_FRONT")
    assert camera["filename"].startswith("samples/CAM_FRONT/")
    assert dataset.ego_pose(camera["token"])["timestamp"] == camera["timestamp"]
    assert dataset.get("sample", samples[0]["token"]) == samples[0]
    assert dataset.field2token("sample", "scene_token", first_scene["token"])
    assert dataset.tags(first_scene["token"])["computed_tags"]
    assert dataset.annotations(first_scene["token"])["scene_token"] == first_scene["token"]
    assert dataset.validity(first_scene["token"])["scene_token"] == first_scene["token"]
    assert dataset.split(first_scene["token"]) in {"train", "test"}
    assert dataset.scenes_in_split(dataset.split(first_scene["token"]))
    assert len(dataset.recordings()) == 2
    assert dataset.validation_report()["state"] == "not_run"
    with Image.open(root / camera["filename"]) as image:
        assert image.size == (camera["width"], camera["height"])


def test_real_timestamps_calibration_pose_and_extension_separation(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    evidence = _evidence(tmp_path, config_factory, feature_factory)
    root = tmp_path / "dataset"
    export_dataset(root, evidence)
    version = root / "v1.0-trainval"
    data = json.loads((version / "sample_data.json").read_text())
    poses = {item["token"]: item for item in json.loads((version / "ego_pose.json").read_text())}
    calibrations = json.loads((version / "calibrated_sensor.json").read_text())
    gnss = json.loads((root / "mz_extensions" / "gnss.json").read_text())
    tags = json.loads((root / "mz_extensions" / "tags.json").read_text())
    annotations = json.loads((root / "mz_extensions" / "annotations.json").read_text())

    assert data[0]["timestamp"] == gnss[0]["timestamp_ns"] // 1000
    assert poses[data[0]["ego_pose_token"]]["timestamp"] == data[0]["timestamp"]
    assert calibrations[0]["camera_intrinsic"] == [
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
    ]
    assert calibrations[0]["rotation"] == [1.0, 0.0, 0.0, 0.0]
    assert all("human_labels" not in item for item in tags)
    assert all("computed_tags" not in item for item in annotations["scenes"])
    assert json.loads(
        (root / "mz_extensions" / "config.json").read_text()
    ) == evidence.resolved_config.model_dump(mode="json")
    assert json.loads((root / "mz_extensions" / "split.json").read_text())["schema_version"] == 1


def test_export_is_deterministic_despite_graph_order(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    evidence = _evidence(tmp_path, config_factory, feature_factory, recordings=2)
    first = tmp_path / "first"
    second = tmp_path / "second"
    export_dataset(first, evidence)
    export_dataset(
        second,
        replace(
            evidence,
            graphs=tuple(reversed(evidence.graphs)),
            features_population=tuple(reversed(evidence.features_population)),
        ),
    )

    def contents(root: Path) -> dict[str, bytes]:
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        }

    assert contents(first) == contents(second)


def test_export_includes_only_selected_scenes(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    base = _evidence(tmp_path, config_factory, feature_factory)
    population = tuple(base.selection.selected_scenes)
    scenarios = ScenariosConfig(
        seed=base.scenarios_config.seed,
        strict_quotas=True,
        rules=[
            ScenarioRuleConfig(name="road", quota=1, required_all_tags=["road"])
        ],
    )
    selection = select_scenarios(population, scenarios)
    split = split_selected_scenes(
        selection,
        population,
        scenarios,
        base.graphs,
        base.split_config,
    )
    config = base.resolved_config.model_copy(
        update={"scenarios": scenarios, "split": base.split_config}
    )
    evidence = replace(
        base,
        features_population=population,
        scenarios_config=scenarios,
        selection=selection,
        split=split,
        resolved_config=config,
    )

    result = export_dataset(tmp_path / "selected", evidence)

    assert result.scene_count == 1
    assert result.sample_count == 2
    assert result.sample_data_count == 4


@pytest.mark.parametrize("kind", ["nonempty", "symlink", "hash", "dimensions", "collision"])
def test_export_refuses_unsafe_destination_or_image_evidence(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
    kind: str,
) -> None:
    evidence = _evidence(tmp_path, config_factory, feature_factory)
    root = tmp_path / "dataset"
    if kind == "nonempty":
        root.mkdir()
        (root / "existing").write_text("x")
    else:
        graph = evidence.graphs[0]
        item = graph.sample_data[0]
        if kind == "symlink":
            link = tmp_path / "linked.jpg"
            link.symlink_to(item.staged_image.path)
            changed = replace(item, staged_image=replace(item.staged_image, path=link))
        elif kind == "hash":
            changed = replace(item, staged_image=replace(item.staged_image, sha256="0" * 64))
        elif kind == "dimensions":
            changed = replace(item, staged_image=replace(item.staged_image, width=99))
        else:
            other = next(value for value in graph.sample_data if value.channel != item.channel)
            changed = replace(item, channel=other.channel, camera_index=other.camera_index)
        changed_graph = replace(
            graph, sample_data=(changed, *graph.sample_data[1:])
        )
        evidence = replace(evidence, graphs=(changed_graph,))
    with pytest.raises((ValueError, FileExistsError)):
        export_dataset(root, evidence)


def test_rejects_microsecond_chain_collision_and_bad_pose(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    with pytest.raises(ValueError, match="microsecond"):
        export_dataset(
            tmp_path / "collision",
            _evidence(tmp_path / "tight", config_factory, feature_factory, tight_timestamps=True),
        )
    with pytest.raises(ValueError, match="normalization collision"):
        export_dataset(
            tmp_path / "channel-collision",
            _evidence(
                tmp_path / "channels",
                config_factory,
                feature_factory,
                colliding_channels=True,
            ),
        )
    evidence = _evidence(tmp_path, config_factory, feature_factory)
    graph = evidence.graphs[0]
    item = graph.sample_data[0]
    bad_pose = replace(item.ego_pose, rotation_wxyz=(0.0, 0.0, 0.0, 0.0))
    changed_items = (replace(item, ego_pose=bad_pose), *graph.sample_data[1:])
    with pytest.raises(ValueError):
        changed_graph = replace(graph, sample_data=changed_items)
        export_dataset(tmp_path / "pose", replace(evidence, graphs=(changed_graph,)))


def test_dataset_rejects_malformed_duplicate_cycle_and_unsafe_filename(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    root = tmp_path / "dataset"
    export_dataset(root, _evidence(tmp_path, config_factory, feature_factory))
    samples_path = root / "v1.0-trainval" / "sample.json"
    samples = json.loads(samples_path.read_text())
    samples.append(dict(samples[0]))
    samples_path.write_text(json.dumps(samples))
    with pytest.raises(DatasetFormatError, match="duplicate"):
        Dataset(root)

    root2 = tmp_path / "dataset2"
    export_dataset(root2, _evidence(tmp_path / "again", config_factory, feature_factory))
    samples_path = root2 / "v1.0-trainval" / "sample.json"
    samples = json.loads(samples_path.read_text())
    samples[0]["next"] = samples[0]["token"]
    samples_path.write_text(json.dumps(samples))
    with pytest.raises(DatasetFormatError, match="cycle"):
        Dataset(root2).scene_samples(samples[0]["scene_token"])
    data_path = root2 / "v1.0-trainval" / "sample_data.json"
    data = json.loads(data_path.read_text())
    data[0]["filename"] = "../escape.jpg"
    data_path.write_text(json.dumps(data))
    with pytest.raises(DatasetFormatError, match="filename"):
        Dataset(root2)


def test_sdk_returned_records_are_defensive_copies(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    root = tmp_path / "dataset"
    export_dataset(root, _evidence(tmp_path, config_factory, feature_factory))
    dataset = Dataset(root)
    scene_token = dataset.table("scene")[0]["token"]

    scene = dataset.get("scene", scene_token)
    scene["name"] = "mutated"
    table = dataset.table("scene")
    table[0]["name"] = "also-mutated"
    traversal = dataset.scene_samples(scene_token)
    traversal[0]["next"] = traversal[0]["token"]
    validity = dataset.validity(scene_token)
    validity["samples"].clear()
    recordings = dataset.recordings()
    recordings[0]["channels"].clear()
    report = dataset.validation_report()
    report["state"] = "succeeded"

    assert dataset.get("scene", scene_token)["name"] != "mutated"
    assert dataset.table("scene")[0]["name"] != "also-mutated"
    assert len(dataset.scene_samples(scene_token)) == 2
    assert dataset.validity(scene_token)["samples"]
    assert dataset.recordings()[0]["channels"]
    assert dataset.validation_report()["state"] == "not_run"


def test_annotation_sdk_full_access_resolution_and_isolation(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    root = tmp_path / "dataset"
    export_dataset(root, _evidence(tmp_path, config_factory, feature_factory))
    dataset = Dataset(root)
    scene_token = dataset.table("scene")[0]["token"]
    path = root / "mz_extensions" / "annotations.json"
    value = json.loads(path.read_text())
    value["scenes"][0].update(
        human_labels=["merge"], annotation_refs=["ann-1"], annotation_window_ref="window-1"
    )
    value["records"] = [
        {"token": "ann-1", "line_number": 1, "labels": ["merge"]}
    ]
    value["matches"] = [
        {"annotation_token": "ann-1", "matched": True, "reason": "matched"}
    ]
    value["windows"] = [
        {"token": "window-1", "annotation_tokens": ["ann-1"]}
    ]
    path.write_text(json.dumps(value), encoding="utf-8")
    dataset = Dataset(root)

    assert dataset.annotation_records()[0]["token"] == "ann-1"
    assert dataset.annotation_matches()[0]["annotation_token"] == "ann-1"
    assert dataset.annotation_windows()[0]["token"] == "window-1"
    assert dataset.annotation_scene_references()[0]["scene_token"] == scene_token
    assert dataset.annotation_record("ann-1")["labels"] == ["merge"]
    assert dataset.annotation_match("ann-1")["matched"] is True
    assert dataset.annotation_window("window-1")["annotation_tokens"] == ["ann-1"]
    resolved = dataset.scene_annotation_evidence(scene_token)
    assert resolved["records"][0]["token"] == "ann-1"
    assert resolved["matches"][0]["annotation_token"] == "ann-1"
    assert resolved["windows"][0]["token"] == "window-1"
    resolved["records"][0]["labels"].append("mutated")
    assert dataset.annotation_record("ann-1")["labels"] == ["merge"]

    value["scenes"][0]["annotation_refs"] = ["missing"]
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DatasetFormatError, match="missing annotation record"):
        Dataset(root).scene_annotation_evidence(scene_token)


def test_exported_annotation_evidence_is_fully_resolvable(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    root = tmp_path / "dataset"
    evidence = _annotated_evidence(tmp_path, config_factory, feature_factory)
    export_dataset(root, evidence)
    dataset = Dataset(root)
    scene_token = dataset.table("scene")[0]["token"]

    resolved = dataset.scene_annotation_evidence(scene_token)

    assert resolved["scene"]["human_labels"] == ["merge"]
    assert resolved["records"][0]["labels"] == ["merge"]
    assert resolved["matches"][0]["matched"] is True
    assert resolved["windows"][0]["annotation_tokens"] == [
        resolved["records"][0]["token"]
    ]


def test_sdk_rejects_ambiguous_camera_rows(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    root = tmp_path / "dataset"
    export_dataset(root, _evidence(tmp_path, config_factory, feature_factory))
    path = root / "v1.0-trainval" / "sample_data.json"
    values = json.loads(path.read_text())
    duplicate = dict(values[0], token="f" * 32)
    values.append(duplicate)
    path.write_text(json.dumps(values))
    dataset = Dataset(root)
    sensor = dataset.get("calibrated_sensor", duplicate["calibrated_sensor_token"])
    channel = dataset.get("sensor", sensor["sensor_token"])["channel"]
    with pytest.raises(DatasetFormatError, match="ambiguous"):
        dataset.camera(duplicate["sample_token"], channel)


def test_official_nuscenes_loader_smoke(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    from nuscenes.nuscenes import NuScenes  # type: ignore[import-untyped]

    root = tmp_path / "dataset"
    export_dataset(root, _evidence(tmp_path, config_factory, feature_factory))
    dataset = NuScenes(version="v1.0-trainval", dataroot=str(root), verbose=False)

    assert len(dataset.scene) == 2
    sample = dataset.get("sample", dataset.scene[0]["first_sample_token"])
    assert set(sample["data"]) == {"CAM_FRONT", "CAM_REAR"}
    sample_data = dataset.get("sample_data", sample["data"]["CAM_FRONT"])
    assert (root / sample_data["filename"]).is_file()
    assert dataset.log[0]["map_token"] == dataset.map[0]["token"]
