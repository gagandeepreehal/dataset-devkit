from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest

from dataset_devkit.config import (
    DEFAULT_DECODED_CAMERAS,
    DecodedHfSourceConfig,
    FiltersConfig,
    GlobalConfig,
    GlobalConfigV2,
    ScenarioRuleConfig,
    ScenariosConfig,
)
from dataset_devkit.dataset import Dataset
from dataset_devkit.decoded_acquisition import (
    AcquiredDecodedRecording,
    DecodedHfAcquirer,
)
from dataset_devkit.decoded_manifest import (
    DecodedManifestError,
    DecodedRecordingEntry,
    DecodedSelectionPlan,
)
from dataset_devkit.extraction.cache import ExtractionResultCache
from dataset_devkit.provenance import extraction_config_hash
from dataset_devkit.services import BuildOperationalError, BuildRuntime, build_dataset
from dataset_devkit.source_runtime import McapAcquirerProtocol, prepare_source
from dataset_devkit.validation import validate_dataset as validate_output
from decoded_v2_fixture import decoded_source_config, write_decoded_fixture


def _decoded_config(base: GlobalConfig, tmp_path: Path) -> GlobalConfigV2:
    fixture = write_decoded_fixture(tmp_path / "remote")
    common = base.model_dump()
    common.pop("schema_version")
    common.pop("huggingface")
    common.pop("topics")
    common["paths"] = {
        "work_dir": tmp_path / "work",
        "cache_dir": tmp_path / "cache",
        "output_dir": tmp_path / "output",
    }
    return GlobalConfigV2(
        schema_version="2.0",
        source=decoded_source_config(
            fixture.revision, [fixture.recording_id]
        ),
        **common,
    )


def _local_decoded_acquirer(
    config: GlobalConfigV2, remote: Path
) -> DecodedHfAcquirer:
    def download(**kwargs: object) -> str:
        filename = str(kwargs["filename"])
        target = Path(str(kwargs["local_dir"])) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote / filename, target)
        return str(target)

    assert isinstance(config.source, DecodedHfSourceConfig)
    return DecodedHfAcquirer(
        source=config.source, cache_dir=config.paths.cache_dir, download_file=download
    )


def test_prepare_source_dispatches_decoded_backend(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    config = _decoded_config(config_factory(), tmp_path)
    mcap_called = False

    def mcap_factory(_config: object) -> McapAcquirerProtocol:
        nonlocal mcap_called
        mcap_called = True
        raise AssertionError("raw fallback invoked")

    runtime = BuildRuntime(
        mcap_acquirer_factory=mcap_factory,
        decoded_acquirer_factory=lambda value: _local_decoded_acquirer(
            value, tmp_path / "remote"
        ),
        official_smoke=False,
    )

    batch = prepare_source(config, runtime)

    assert batch.source_type == "decoded_hf"
    assert batch.privacy_classification == "privacy_transformed"
    assert batch.pose_frame == "recording_local_enu_v1"
    assert [item.recording_id for item in batch.recordings] == ["recording-code"]
    assert batch.recordings[0].preparation_error is None
    assert batch.recordings[0].extract_owned().result.samples
    assert mcap_called is False


def test_decoded_privacy_failure_never_calls_mcap_factory(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    config = _decoded_config(config_factory(), tmp_path)
    mcap_called = False

    def mcap_factory(_config: object) -> McapAcquirerProtocol:
        nonlocal mcap_called
        mcap_called = True
        raise AssertionError("raw fallback invoked")

    class FailingDecodedAcquirer:
        def load_plan(self) -> DecodedSelectionPlan:
            raise DecodedManifestError("privacy contract differs")

        def acquire(
            self, _entry: DecodedRecordingEntry
        ) -> AcquiredDecodedRecording:
            raise AssertionError("acquire called after privacy failure")

    runtime = BuildRuntime(
        mcap_acquirer_factory=mcap_factory,
        decoded_acquirer_factory=lambda _config: FailingDecodedAcquirer(),
        official_smoke=False,
    )

    with pytest.raises(DecodedManifestError, match="privacy contract"):
        prepare_source(config, runtime)
    with pytest.raises(BuildOperationalError, match="privacy contract"):
        build_dataset(config, runtime=runtime)
    assert mcap_called is False


def test_decoded_backend_runs_through_common_build_orchestration(
    tmp_path: Path, config_factory: Callable[[], GlobalConfig]
) -> None:
    config = _decoded_config(config_factory(), tmp_path)
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text("", encoding="utf-8")
    config = config.model_copy(
        update={
            "annotations": config.annotations.model_copy(update={"path": annotations}),
            "scenes": config.scenes.model_copy(
                update={
                    "mode": "automatic",
                    "min_duration_s": Decimal("0.1"),
                    "max_duration_s": Decimal("10"),
                    "min_samples": 2,
                    "max_sample_gap_ms": Decimal("1000"),
                    "skip_between_scenes_s": Decimal("0"),
                }
            ),
            "gnss": config.gnss.model_copy(
                update={
                    "position_sigma_max_m": 10.0,
                    "orientation_variance_max": 10.0,
                    "sync_gap_max_ms": 2_000.0,
                }
            ),
            "frame_validity": config.frame_validity.model_copy(
                update={
                    "required_cameras": list(DEFAULT_DECODED_CAMERAS),
                    "camera_timestamp_gap_max_ms": 2_000.0,
                }
            ),
            "tags": config.tags.model_copy(
                update={"reference_camera_channel": "cam_front"}
            ),
            "filters": FiltersConfig(),
            "scenarios": ScenariosConfig(
                seed=7,
                strict_quotas=True,
                rules=[ScenarioRuleConfig(name="all", quota=1)],
            ),
            "execution": config.execution.model_copy(
                update={"workers": 2, "allow_partial_export": False}
            ),
            "quarantine": config.quarantine.model_copy(
                update={"directory": tmp_path / "quarantine"}
            ),
        }
    )
    runtime = BuildRuntime(
        decoded_acquirer_factory=lambda value: _local_decoded_acquirer(
            value, tmp_path / "remote"
        ),
        official_smoke=False,
    )
    batch = prepare_source(config, runtime)

    result = build_dataset(config, runtime=runtime)

    assert result.scene_count == 1
    assert result.partial is False
    assert result.source_type == "decoded_hf"
    assert result.privacy_classification == "privacy_transformed"
    assert result.pose_frame == "recording_local_enu_v1"
    dataset = Dataset(result.dataroot, result.version)
    assert dataset.source_metadata() == {
        "global_horizontal_resolution_m": 1.0,
        "pose_frame": "recording_local_enu_v1",
        "privacy_classification": "privacy_transformed",
        "schema_version": 1,
        "source_type": "decoded_hf",
    }
    gnss = json.loads(
        (result.dataroot / "mz_extensions/gnss.json").read_text(encoding="utf-8")
    )[0]
    pose = dataset.ego_pose(gnss["sample_data_token"])
    assert pose["translation"] == pytest.approx(gnss["translation_xyz_m"])
    assert gnss["published_horizontal_resolution_m"] == 1.0
    assert ExtractionResultCache(config.paths.cache_dir).contains(
        batch.recordings[0].fingerprint, extraction_config_hash(config)
    )
    gnss_path = result.dataroot / "mz_extensions/gnss.json"
    gnss["translation_xyz_m"][0] += 1.0
    gnss_path.chmod(0o600)
    gnss_path.write_text(json.dumps(gnss), encoding="utf-8")
    report = validate_output(
        result.dataroot,
        result.version,
        official_smoke=False,
        verify_manifest=False,
    )
    assert not report.succeeded
    assert any(item.code == "extension_value" for item in report.findings)
