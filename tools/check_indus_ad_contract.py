"""Verify a pinned INDUS-AD release using only control files."""

from __future__ import annotations

import argparse
import tempfile
from contextlib import nullcontext
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from dataset_devkit.config import IndusAdHfSourceConfig
from dataset_devkit.indus_ad_acquisition import IndusAdHfAcquirer
from dataset_devkit.indus_ad_contract import CAMERA_ANNOTATION_VIEWER_PATH
from dataset_devkit.provenance import canonical_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", type=Path, help="Optional isolated resumable control cache")
    args = parser.parse_args()
    requested: list[str] = []

    def download(**kwargs: object) -> str:
        filename = str(kwargs["filename"])
        allowed = (
            filename == "data/metadata.jsonl"
            or filename.startswith("data/manifests/")
            or filename.startswith("data/viewer/")
            or filename.startswith("data/annotations/camera/dense-index/")
        )
        if not allowed or Path(filename).suffix not in {".json", ".jsonl", ".parquet"}:
            raise ValueError("metadata gate attempted a payload download")
        requested.append(filename)
        return str(hf_hub_download(**kwargs))  # type: ignore[arg-type]

    source = IndusAdHfSourceConfig(
        type="indus_ad_hf",
        contract="indus_ad_integrated_v1",
        repo_id=args.repo_id,
        revision=args.revision,
        splits=["train"],
        annotation_layers=["camera_scalar", "semantic_fractions", "depth_summary", "weather"],
    )
    cache_context = (
        nullcontext(str(args.cache_dir))
        if args.cache_dir is not None
        else tempfile.TemporaryDirectory(prefix="indus-ad-contract-")
    )
    with cache_context as temporary:
        acquirer = IndusAdHfAcquirer(
            source=source, cache_dir=Path(temporary), download_file=download
        )
        plan = acquirer.load_plan()
        annotation_rows = pq.ParquetFile(
            acquirer._control_files[CAMERA_ANNOTATION_VIEWER_PATH]
        ).metadata.num_rows
        print(
            canonical_json(
                {
                    "repo_id": plan.repo_id,
                    "revision": plan.revision,
                    "contract": plan.contract,
                    "recording_count": len(plan.recordings),
                    "video_count": plan.viewer_privacy_counts["face_plate_blurred"],
                    "face_plate_blurred_count": plan.viewer_privacy_counts["face_plate_blurred"],
                    "missing_camera_slot_count": plan.viewer_privacy_counts["missing"],
                    "camera_annotation_rows": annotation_rows,
                    "annotation_schema_sha256": plan.annotation_schema_sha256,
                    "downloaded_control_paths": sorted(set(requested)),
                }
            )
        )


if __name__ == "__main__":
    main()
