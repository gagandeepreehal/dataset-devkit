"""Deterministic joins from INDUS-AD weak labels to selected scene graphs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dataset_devkit.indus_ad_dense import MaterializedDenseAsset
from dataset_devkit.indus_ad_models import (
    CameraAnnotation,
    DenseAssetRef,
    GnssScenarioAnnotation,
    IndusAdAnnotationBundle,
)
from dataset_devkit.provenance import IndusAdSourceFingerprint
from dataset_devkit.scene_models import RecordingSceneResult, SampleDataRecord


@dataclass(frozen=True, slots=True)
class JoinedCameraAnnotation:
    annotation: CameraAnnotation
    source_digest: str
    scene_token: str
    sample_token: str
    sample_data_token: str


@dataclass(frozen=True, slots=True)
class JoinedGnssAnnotation:
    annotation: GnssScenarioAnnotation
    source_digest: str
    scene_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class JoinedDenseAsset:
    ref: DenseAssetRef
    source_digest: str
    scene_token: str
    sample_data_token: str
    materialized_path: Path | None = None
    materialized_sha256: str | None = None
    materialized_size: int | None = None


@dataclass(frozen=True, slots=True)
class AnnotationJoinAudit:
    recording_id: str
    selected_scene_count: int
    camera_available: int
    camera_joined: int
    gnss_available: int
    gnss_joined: int
    dense_available: int
    dense_joined: int
    dense_materialized: int


@dataclass(frozen=True, slots=True)
class JoinedIndusAdAnnotations:
    camera: tuple[JoinedCameraAnnotation, ...] = ()
    gnss: tuple[JoinedGnssAnnotation, ...] = ()
    dense: tuple[JoinedDenseAsset, ...] = ()
    audits: tuple[AnnotationJoinAudit, ...] = ()


def join_indus_ad_annotations(
    *,
    graph: RecordingSceneResult,
    selected_scene_tokens: frozenset[str],
    annotations: IndusAdAnnotationBundle,
    materialized: tuple[MaterializedDenseAsset, ...] = (),
) -> JoinedIndusAdAnnotations:
    """Join by immutable recording identity and exact relative timestamps."""
    if not isinstance(graph.source, IndusAdSourceFingerprint):
        raise ValueError("INDUS-AD annotations require an INDUS-AD source graph")
    recording_id = graph.source.recording_id
    if (
        any(item.recording_id != recording_id for item in annotations.camera)
        or any(item.recording_id != recording_id for item in annotations.gnss)
        or any(item.recording_id != recording_id for item in annotations.dense_refs)
    ):
        raise ValueError("INDUS-AD annotation recording identity differs")

    selected_scenes = {
        scene.token: scene for scene in graph.scenes if scene.token in selected_scene_tokens
    }
    sample_data_index: dict[tuple[str, int], SampleDataRecord] = {}
    for sample_data in graph.sample_data:
        if sample_data.scene_token not in selected_scenes:
            continue
        key = (sample_data.channel, sample_data.timestamp_ns)
        if key in sample_data_index:
            raise ValueError("selected graph has duplicate channel/timestamp identities")
        sample_data_index[key] = sample_data

    camera: list[JoinedCameraAnnotation] = []
    camera_by_frame: dict[str, JoinedCameraAnnotation] = {}
    for annotation in sorted(
        annotations.camera,
        key=lambda value: (
            value.recording_offset_ns,
            value.camera_name,
            value.frame_index,
            value.frame_id,
        ),
    ):
        matched = sample_data_index.get((annotation.camera_name, annotation.recording_offset_ns))
        if matched is None:
            continue
        joined = JoinedCameraAnnotation(
            annotation,
            graph.source.digest,
            matched.scene_token,
            matched.sample_token,
            matched.token,
        )
        if annotation.frame_id in camera_by_frame:
            raise ValueError("camera annotations contain duplicate frame identities")
        camera.append(joined)
        camera_by_frame[annotation.frame_id] = joined

    gnss: list[JoinedGnssAnnotation] = []
    for gnss_annotation in sorted(
        annotations.gnss,
        key=lambda value: (value.window_start_offset_ns, value.window_index),
    ):
        scene_tokens = tuple(
            scene.token
            for scene in sorted(selected_scenes.values(), key=lambda value: value.ordinal)
            if gnss_annotation.window_start_offset_ns <= scene.last_timestamp_ns
            and gnss_annotation.window_end_offset_ns > scene.first_timestamp_ns
        )
        if scene_tokens:
            gnss.append(JoinedGnssAnnotation(gnss_annotation, graph.source.digest, scene_tokens))

    materialized_by_identity = {item.ref.identity: item for item in materialized}
    if len(materialized_by_identity) != len(materialized):
        raise ValueError("materialized dense assets contain duplicate identities")
    dense: list[JoinedDenseAsset] = []
    for ref in sorted(
        annotations.dense_refs,
        key=lambda value: (
            value.recording_offset_ns,
            value.camera_name,
            value.frame_index,
            value.kind,
        ),
    ):
        joined_camera = camera_by_frame.get(ref.frame_id)
        if joined_camera is None:
            continue
        payload = materialized_by_identity.get(ref.identity)
        dense.append(
            JoinedDenseAsset(
                ref,
                graph.source.digest,
                joined_camera.scene_token,
                joined_camera.sample_data_token,
                None if payload is None else payload.path,
                None if payload is None else payload.sha256,
                None if payload is None else payload.size,
            )
        )

    return JoinedIndusAdAnnotations(
        tuple(camera),
        tuple(gnss),
        tuple(dense),
        (
            AnnotationJoinAudit(
                recording_id,
                len(selected_scenes),
                len(annotations.camera),
                len(camera),
                len(annotations.gnss),
                len(gnss),
                len(annotations.dense_refs),
                len(dense),
                sum(item.materialized_path is not None for item in dense),
            ),
        ),
    )
