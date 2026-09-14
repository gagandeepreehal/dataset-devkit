#!/usr/bin/env python3
"""Build the tiny local decoded-V2 fixture without network access."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Never

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from dataset_devkit.config import DecodedHfSourceConfig, GlobalConfigV2, load_config  # noqa: E402
from dataset_devkit.decoded_acquisition import DecodedHfAcquirer  # noqa: E402
from dataset_devkit.services import BuildRuntime, build_dataset  # noqa: E402
from decoded_v2_fixture import write_decoded_fixture  # noqa: E402


def _raw_fallback(_config: object) -> Never:
    raise AssertionError("decoded fixture attempted an MCAP fallback")


def main(output_root: Path) -> int:
    """Build and print one absolute published dataroot, refusing overwrites."""
    output = output_root.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output root: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="dataset-devkit-fixture-", dir=output.parent) as scratch_text:
        scratch = Path(scratch_text)
        remote = scratch / "source"
        fixture = write_decoded_fixture(remote)
        annotations = scratch / "annotations.jsonl"
        annotations.write_text("", encoding="utf-8")

        data = json.loads((ROOT / "examples/decoded_v2_config.json").read_text(encoding="utf-8"))
        data["source"].update(  # type: ignore[union-attr]
            revision=fixture.revision,
            recording_ids=[fixture.recording_id],
        )
        data["paths"] = {
            "work_dir": str(scratch / "work"),
            "cache_dir": str(scratch / "cache"),
            "output_dir": str(output),
        }
        data["annotations"]["path"] = str(annotations)  # type: ignore[index]
        data["scenes"].update(  # type: ignore[union-attr]
            mode="automatic",
            min_duration_s=0.1,
            max_duration_s=10.0,
            min_samples=2,
            max_sample_gap_ms=1000.0,
        )
        data["gnss"].update(  # type: ignore[union-attr]
            position_sigma_max_m=10.0,
            orientation_variance_max=10.0,
            sync_gap_max_ms=2000.0,
        )
        data["frame_validity"]["camera_timestamp_gap_max_ms"] = 2000.0  # type: ignore[index]
        data["filters"] = {}
        data["scenarios"] = {
            "seed": 7,
            "strict_quotas": True,
            "rules": [{"name": "all", "quota": 1}],
        }
        data["quarantine"]["directory"] = str(scratch / "quarantine")  # type: ignore[index]
        config_path = scratch / "config.json"
        config_path.write_text(json.dumps(data), encoding="utf-8")
        config = load_config(config_path)
        if not isinstance(config, GlobalConfigV2):
            raise AssertionError("fixture configuration did not load as schema 2.0")

        def decoded_factory(value: GlobalConfigV2) -> DecodedHfAcquirer:
            if not isinstance(value.source, DecodedHfSourceConfig):
                raise AssertionError("fixture configuration did not select decoded_hf")

            def download(**kwargs: object) -> str:
                filename = str(kwargs["filename"])
                target = Path(str(kwargs["local_dir"])) / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(remote / filename, target)
                return str(target)

            return DecodedHfAcquirer(
                source=value.source,
                cache_dir=value.paths.cache_dir,
                download_file=download,
            )

        result = build_dataset(
            config,
            runtime=BuildRuntime(
                mcap_acquirer_factory=_raw_fallback,
                decoded_acquirer_factory=decoded_factory,
                official_smoke=False,
            ),
        )

    print(result.dataroot.resolve())
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    arguments = parser.parse_args()
    raise SystemExit(main(arguments.output_root))
