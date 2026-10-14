"""Self-contained extension payloads for joined INDUS-AD weak labels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path

from dataset_devkit.indus_ad_join import JoinedIndusAdAnnotations


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _record(value: object) -> dict[str, object]:
    result = _jsonable(value)
    if not isinstance(result, dict):
        raise TypeError("extension record must serialize to an object")
    return result


@dataclass(frozen=True, slots=True)
class DenseAssetExport:
    source_path: Path
    relative_path: str
    expected_sha256: str
    expected_size: int


@dataclass(frozen=True, slots=True)
class IndusAdExportPayload:
    camera: dict[str, object]
    gnss_scenarios: dict[str, object]
    dense_assets: dict[str, object]
    join_audit: dict[str, object]
    assets: tuple[DenseAssetExport, ...]


def build_indus_ad_export(
    annotations: JoinedIndusAdAnnotations,
) -> IndusAdExportPayload:
    """Build stable JSON rows and local-copy instructions for selected labels."""
    camera_rows = [
        {
            "source_digest": item.source_digest,
            "scene_token": item.scene_token,
            "sample_token": item.sample_token,
            "sample_data_token": item.sample_data_token,
            **_record(item.annotation),
        }
        for item in annotations.camera
    ]
    gnss_rows = [
        {
            "source_digest": item.source_digest,
            "scene_tokens": list(item.scene_tokens),
            **_record(item.annotation),
        }
        for item in annotations.gnss
    ]
    dense_rows: list[dict[str, object]] = []
    assets: list[DenseAssetExport] = []
    for item in annotations.dense:
        asset_path = None
        if item.materialized_path is not None:
            suffix = item.materialized_path.suffix
            asset_path = (
                f"mz_extensions/indus_ad/assets/{item.ref.kind}/{item.ref.payload_sha256}{suffix}"
            )
            if item.materialized_sha256 is None or item.materialized_size is None:
                raise ValueError("materialized dense asset lacks verified identity")
            assets.append(
                DenseAssetExport(
                    item.materialized_path,
                    asset_path,
                    item.materialized_sha256,
                    item.materialized_size,
                )
            )
        dense_rows.append(
            {
                "source_digest": item.source_digest,
                "scene_token": item.scene_token,
                "sample_data_token": item.sample_data_token,
                "asset_path": asset_path,
                **_record(item.ref),
            }
        )
    if len({item.relative_path for item in assets}) != len(assets):
        raise ValueError("dense export paths are not unique")
    return IndusAdExportPayload(
        {"schema_version": 1, "label_kind": "weak_pseudo_label", "rows": camera_rows},
        {"schema_version": 1, "label_kind": "weak_pseudo_label", "rows": gnss_rows},
        {"schema_version": 1, "label_kind": "weak_pseudo_label", "rows": dense_rows},
        {
            "schema_version": 1,
            "recordings": [_record(item) for item in annotations.audits],
        },
        tuple(assets),
    )
