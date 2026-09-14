# Decoded V2 Privacy Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a commit-pinned decoded Hugging Face backend that consumes anonymized V2 video and privacy-transformed GNSS, builds trajectories from precise recording-local ENU, and keeps MCAP explicitly classified as restricted raw input.

**Architecture:** Introduce a versioned source union and backend-neutral recording preparation boundary. A decoded control-plane reader verifies the immutable catalog and privacy manifests before a safe selective downloader acquires MP4/Parquet artifacts; a decoded extractor then adapts those artifacts into the existing scene, feature, split, and nuScenes-compatible export pipeline. MCAP parsing remains unchanged, while provenance and validation become source-aware.

**Tech Stack:** Python 3.12+, Pydantic 2, PyArrow 25, PyAV, Pillow, Hugging Face Hub, NumPy, PyProj, pytest, Ruff, strict mypy, JSON Schema 2020-12.

**Spec:** `docs/superpowers/specs/2026-09-14-decoded-v2-privacy-backend-design.md`

## Global Constraints

- Keep `source.type = "mcap_hf"` supported and classify every MCAP build as `restricted_raw`; never claim MCAP video is anonymized.
- `source.type = "decoded_hf"` is the only `privacy_transformed` backend.
- Require a full lowercase 40-character Hugging Face commit SHA; reject branches, tags, repository scans, and implicit latest revisions.
- Validate `data/recordings.parquet` plus the four fixed public manifests before downloading selected payloads.
- Require `global_horizontal_resolution_m == 1.0`, `local_enu_precision == "retained"`, `absolute_dates_removed == true`, and `exact_recording_offsets_retained == true`.
- Treat `recording_offset_ns` as the only ordering and synchronization clock for decoded V2.
- Build ego-pose translation, distance, speed, curvature, scene construction, filtering, and scenario selection from precise recording-local ENU.
- Never reconstruct local ENU from rounded global coordinates; never fall back from decoded V2 to MCAP.
- Default to the six surrounding cameras; make `cam_front_tilted` opt-in.
- Keep scenes within one recording-local coordinate frame.
- Keep production payloads on `mzcloud`; local tests use generated fixtures only.
- Do not hardcode a production privacy revision until the anonymization task has a final immutable verified release.
- Preserve descriptor-relative/no-follow cache and staging protections, immutable source evidence, quarantine, final validation, and no-overwrite atomic publication.

## File Structure

- `src/dataset_devkit/config.py`: version 1 compatibility, version 2 decoded-source configuration, discriminated source union, and normalization.
- `src/dataset_devkit/schema.py`: JSON Schema generation through the versioned configuration adapter.
- `src/dataset_devkit/provenance.py`: MCAP and decoded recording fingerprints plus source/privacy metadata.
- `src/dataset_devkit/decoded_manifest.py`: exact catalog/public-manifest parsing and selection planning.
- `src/dataset_devkit/decoded_acquisition.py`: selective immutable Hugging Face artifact acquisition and verified local bundle creation.
- `src/dataset_devkit/extraction/decoded_gnss.py`: privacy GNSS parsing, local-ENU interpolation, and rounded-global validation.
- `src/dataset_devkit/extraction/decoded.py`: decoded MP4/Parquet recording adapter and safe JPEG staging.
- `src/dataset_devkit/source_runtime.py`: backend-neutral prepared-recording interface and MCAP/decoded dispatch.
- `src/dataset_devkit/services.py`: orchestration over prepared recordings and source-aware result/inspection fields.
- `src/dataset_devkit/export.py`: coordinate-frame and privacy provenance plus decoded GNSS extension fields.
- `src/dataset_devkit/validation.py`: exact source/privacy/pose-frame validation.
- `src/dataset_devkit/dataset.py`: read-only access to source metadata.
- `tests/decoded_v2_fixture.py`: deterministic tiny MP4/Parquet/control-plane fixture writer.
- Focused test modules mirror each new responsibility; existing MCAP tests remain regression coverage.

---

### Task 1: Versioned Source Configuration and Privacy Classification

**Files:**
- Modify: `src/dataset_devkit/config.py`
- Modify: `src/dataset_devkit/schema.py`
- Modify: `schema/dataset_config.schema.json`
- Modify: `tests/test_config.py`
- Modify: `tests/test_schema.py`
- Modify: `tests/test_examples.py`

**Interfaces:**
- Consumes: existing `HuggingFaceConfig`, shared build-policy sections, credential rejection, and relative-path resolution.
- Produces: `McapHfSourceConfig`, `DecodedHfSourceConfig`, `SourceConfig`, `GlobalConfigV1`, `GlobalConfigV2`, `GlobalConfig`, `GLOBAL_CONFIG_ADAPTER`, `normalized_source(config)`, and `privacy_classification(config)`.

- [ ] **Step 1: Prepare the isolated Python 3.12 development environment**

Run:

```bash
.venv/bin/python -m pip install -e '.[dev]'
```

Expected: editable package and development dependencies install into the worktree-local Python 3.12 virtual environment. If package-index access is sandboxed, request network approval and rerun this exact command.

- [ ] **Step 2: Write failing configuration tests**

Add tests that retain the existing example as schema `1.0`, validate a decoded schema `2.0` source, and reject unsafe selections:

```python
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from dataset_devkit.config import (
    DecodedHfSourceConfig,
    load_config,
    privacy_classification,
)


def decoded_source() -> dict[str, object]:
    return {
        "type": "decoded_hf",
        "repo_id": "gagandeepreehal/minuszero-indian-autonomous-driving-dataset-v2",
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "recordings_path": "data/recordings.parquet",
        "recording_ids": ["recording-code"],
        "splits": ["train"],
        "cameras": [
            "cam_front",
            "cam_front_left",
            "cam_front_right",
            "cam_rear",
            "cam_rear_left",
            "cam_rear_right",
        ],
        "modalities": ["video", "gnss", "calibration"],
    }


def test_decoded_source_has_fixed_privacy_classification() -> None:
    source = DecodedHfSourceConfig.model_validate(decoded_source())
    assert source.type == "decoded_hf"
    assert source.privacy_classification == "privacy_transformed"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("revision", "main"),
        ("recordings_path", "../recordings.parquet"),
        ("cameras", ["cam_front", "unknown"]),
        ("modalities", ["video", "gnss"]),
    ],
)
def test_decoded_source_rejects_mutable_or_incomplete_input(
    field: str, value: object
) -> None:
    payload = decoded_source()
    payload[field] = value
    with pytest.raises(ValidationError):
        DecodedHfSourceConfig.model_validate(payload)


def test_legacy_config_normalizes_to_restricted_mcap(example_config: Path) -> None:
    config = load_config(example_config)
    assert config.schema_version == "1.0"
    assert privacy_classification(config) == "restricted_raw"
```

- [ ] **Step 3: Run the focused tests and confirm the new imports fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_config.py tests/test_schema.py tests/test_examples.py -q
```

Expected: FAIL because the versioned source models and adapter do not exist.

- [ ] **Step 4: Implement the exact source models and version adapter**

Keep the existing common sections in a shared base model. Add these exact public types and helpers:

```python
CameraName = Literal[
    "cam_front",
    "cam_front_left",
    "cam_front_right",
    "cam_front_tilted",
    "cam_rear",
    "cam_rear_left",
    "cam_rear_right",
]
DecodedModality = Literal[
    "video", "gnss", "calibration", "camera_labels", "semantic", "depth"
]
PrivacyClassification = Literal["restricted_raw", "privacy_transformed"]

DEFAULT_DECODED_CAMERAS: tuple[CameraName, ...] = (
    "cam_front",
    "cam_front_left",
    "cam_front_right",
    "cam_rear",
    "cam_rear_left",
    "cam_rear_right",
)


class McapHfSourceConfig(HuggingFaceConfig):
    type: Literal["mcap_hf"] = "mcap_hf"
    privacy_classification: Literal["restricted_raw"] = "restricted_raw"


class DecodedHfSourceConfig(StrictModel):
    type: Literal["decoded_hf"]
    repo_id: str
    revision: str
    recordings_path: Literal["data/recordings.parquet"]
    recording_ids: list[SafeSegment] = Field(default_factory=list)
    splits: list[SafeSegment] = Field(default_factory=list)
    cameras: list[CameraName] = Field(
        default_factory=lambda: list(DEFAULT_DECODED_CAMERAS)
    )
    modalities: list[DecodedModality] = Field(
        default_factory=lambda: ["video", "gnss", "calibration"]
    )
    privacy_classification: Literal["privacy_transformed"] = "privacy_transformed"

    @model_validator(mode="after")
    def validate_selection(self) -> DecodedHfSourceConfig:
        required = {"video", "gnss", "calibration"}
        if not required.issubset(self.modalities):
            raise ValueError("decoded source requires video, gnss, and calibration")
        if len(self.cameras) != len(set(self.cameras)):
            raise ValueError("decoded cameras must be unique")
        if len(self.recording_ids) != len(set(self.recording_ids)):
            raise ValueError("decoded recording IDs must be unique")
        if len(self.splits) != len(set(self.splits)):
            raise ValueError("decoded splits must be unique")
        return self


type SourceConfig = McapHfSourceConfig | DecodedHfSourceConfig
type GlobalConfig = GlobalConfigV1 | GlobalConfigV2

GLOBAL_CONFIG_ADAPTER = TypeAdapter(
    Annotated[GlobalConfig, Field(discriminator="schema_version")]
)


def normalized_source(config: GlobalConfig) -> SourceConfig:
    if isinstance(config, GlobalConfigV1):
        return McapHfSourceConfig(**config.huggingface.model_dump())
    return config.source


def privacy_classification(config: GlobalConfig) -> PrivacyClassification:
    return normalized_source(config).privacy_classification
```

Reuse the existing repository ID, revision, credential, uniqueness, and safe-path validators rather than duplicating weaker versions. `GlobalConfigV2` inherits every shared policy section except `topics`; it requires `source: Annotated[SourceConfig, Field(discriminator="type")]`. `load_config` and `validate_config_schema_and_runtime` must use `GLOBAL_CONFIG_ADAPTER` after the existing exact-decimal and relative-path normalization.

- [ ] **Step 5: Render the versioned JSON Schema**

Change `render_schema()` to:

```python
def render_schema() -> str:
    return json.dumps(
        GLOBAL_CONFIG_ADAPTER.json_schema(), indent=2, sort_keys=True
    ) + "\n"
```

Run:

```bash
PYTHONPATH=src .venv/bin/python -m dataset_devkit.schema schema/dataset_config.schema.json
```

Expected: the checked-in schema contains a `schema_version` discriminator with `1.0` and `2.0` branches.

- [ ] **Step 6: Run focused configuration and schema tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_config.py tests/test_schema.py tests/test_examples.py -q
.venv/bin/python -m ruff check src/dataset_devkit/config.py src/dataset_devkit/schema.py tests/test_config.py tests/test_schema.py tests/test_examples.py
.venv/bin/python -m mypy src/dataset_devkit/config.py src/dataset_devkit/schema.py tests/test_config.py tests/test_schema.py
```

Expected: PASS; the existing example remains valid schema `1.0`.

- [ ] **Step 7: Commit the configuration boundary**

```bash
git add src/dataset_devkit/config.py src/dataset_devkit/schema.py schema/dataset_config.schema.json tests/test_config.py tests/test_schema.py tests/test_examples.py
git commit -m "feat: add versioned decoded source config"
```

---

### Task 2: Backend-Neutral Recording Fingerprints

**Files:**
- Modify: `src/dataset_devkit/provenance.py`
- Modify: `src/dataset_devkit/scene_models.py`
- Modify: `src/dataset_devkit/scenes.py`
- Modify: `src/dataset_devkit/features.py`
- Modify: `src/dataset_devkit/filtering.py`
- Modify: `src/dataset_devkit/scenario_selection.py`
- Modify: `src/dataset_devkit/split.py`
- Modify: `tests/test_provenance.py`
- Modify: `tests/test_scenes.py`

**Interfaces:**
- Consumes: existing `SourceFingerprint` MCAP semantics and all downstream uses of `.digest`.
- Produces: `DecodedSourceFingerprint`, `RecordingFingerprint`, `fingerprint_to_dict()`, `fingerprint_from_dict()`, `fingerprint_locator()`, and a stable digest for each decoded recording/artifact set.

- [ ] **Step 1: Write failing decoded-fingerprint tests**

```python
from dataset_devkit.provenance import (
    DecodedSourceFingerprint,
    fingerprint_from_dict,
    fingerprint_to_dict,
)
from dataclasses import replace


def test_decoded_fingerprint_binds_recording_and_artifact_set() -> None:
    fingerprint = DecodedSourceFingerprint(
        repo_id="owner/dataset",
        revision="0123456789abcdef0123456789abcdef01234567",
        recording_id="recording-code",
        catalog_row_sha256="a" * 64,
        artifact_set_sha256="b" * 64,
        total_size=123,
    )
    assert fingerprint.source_type == "decoded_hf"
    assert fingerprint.privacy_classification == "privacy_transformed"
    assert fingerprint_from_dict(fingerprint_to_dict(fingerprint)) == fingerprint


def test_decoded_digest_changes_with_artifact_set() -> None:
    first = DecodedSourceFingerprint(
        "owner/dataset",
        "0123456789abcdef0123456789abcdef01234567",
        "recording-code",
        "a" * 64,
        "b" * 64,
        123,
    )
    second = replace(first, artifact_set_sha256="c" * 64)
    assert first.digest != second.digest
```

- [ ] **Step 2: Run provenance tests and confirm failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_provenance.py tests/test_scenes.py -q
```

Expected: FAIL because decoded fingerprints and the recording union are absent.

- [ ] **Step 3: Implement the source fingerprint union**

Keep `SourceFingerprint` as the MCAP fingerprint for compatibility and add:

```python
@dataclass(frozen=True)
class DecodedSourceFingerprint:
    repo_id: str
    revision: str
    recording_id: str
    catalog_row_sha256: str
    artifact_set_sha256: str
    total_size: int
    source_type: Literal["decoded_hf"] = "decoded_hf"
    privacy_classification: Literal["privacy_transformed"] = "privacy_transformed"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def digest(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def cache_key(self) -> str:
        return self.digest


type RecordingFingerprint = SourceFingerprint | DecodedSourceFingerprint


def fingerprint_to_dict(value: RecordingFingerprint) -> dict[str, object]:
    if isinstance(value, SourceFingerprint):
        return {"source_type": "mcap_hf", "privacy_classification": "restricted_raw", **value.to_dict()}
    return value.to_dict()


def fingerprint_locator(value: RecordingFingerprint) -> str:
    if isinstance(value, SourceFingerprint):
        return value.repo_path
    return value.recording_id
```

Validate exact repo/revision/digest/size shapes and reject unknown keys in `fingerprint_from_dict()`. Change downstream type annotations to `RecordingFingerprint`; replace direct `asdict(source)` with `fingerprint_to_dict(source)` and direct `.repo_path` display with `fingerprint_locator(source)`. Do not change token derivation: it continues to use `.digest`.

- [ ] **Step 4: Run provenance and downstream graph tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_provenance.py tests/test_scenes.py tests/test_features.py tests/test_filtering_selection.py tests/test_split.py -q
.venv/bin/python -m mypy src/dataset_devkit/provenance.py src/dataset_devkit/scene_models.py src/dataset_devkit/scenes.py src/dataset_devkit/features.py
```

Expected: PASS with unchanged MCAP digests and deterministic decoded digests.

- [ ] **Step 5: Commit generalized recording identity**

```bash
git add src/dataset_devkit/provenance.py src/dataset_devkit/scene_models.py src/dataset_devkit/scenes.py src/dataset_devkit/features.py src/dataset_devkit/filtering.py src/dataset_devkit/scenario_selection.py src/dataset_devkit/split.py tests/test_provenance.py tests/test_scenes.py
git commit -m "feat: generalize recording provenance"
```

---

### Task 3: Exact Decoded V2 Control-Plane Parser

**Files:**
- Modify: `pyproject.toml`
- Create: `src/dataset_devkit/decoded_manifest.py`
- Modify: `src/dataset_devkit/repository_paths.py`
- Create: `tests/decoded_v2_fixture.py`
- Create: `tests/test_decoded_manifest.py`
- Modify: `tests/test_repository_paths.py`

**Interfaces:**
- Consumes: `DecodedHfSourceConfig`; exact `recordings.parquet` schema emitted by the privacy rebuild; fixed processing/model/schema manifest assertions.
- Produces: `PublicArtifact`, `DecodedVideo`, `DecodedRecordingEntry`, `PrivacyContract`, `DecodedSelectionPlan`, `parse_decoded_control_plane(root, source)`.

- [ ] **Step 1: Add PyArrow and write the fixture writer**

Add `"pyarrow>=25,<26"` to project dependencies. Create a deterministic fixture helper with no production network access:

```python
@dataclass(frozen=True)
class DecodedFixture:
    root: Path
    revision: str
    recording_id: str


def write_decoded_fixture(root: Path) -> DecodedFixture:
    recording_id = "recording-code"
    revision = "0123456789abcdef0123456789abcdef01234567"
    artifacts = write_tiny_videos_and_tables(root, recording_id)
    write_recordings_parquet(root, recording_id, artifacts)
    write_public_manifests(root, artifacts)
    return DecodedFixture(root, revision, recording_id)
```

Implement `write_tiny_videos_and_tables`, `write_recordings_parquet`, and `write_public_manifests` in the same module using PyAV and `pyarrow.parquet.write_table`. Use two 4x4 RGB frames per camera, offsets `0` and `1_000_000_000`, six default cameras, one calibration row per camera, and two GNSS rows with precise local ENU.

- [ ] **Step 2: Write failing parser and privacy-gate tests**

```python
def test_control_plane_filters_before_payload_selection(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path)
    source = decoded_source_config(fixture.revision, [fixture.recording_id])
    plan = parse_decoded_control_plane(fixture.root, source)
    assert [item.recording_id for item in plan.recordings] == [fixture.recording_id]
    assert plan.privacy == PrivacyContract(
        model_lock_sha256="1" * 64,
        privacy_config_sha256="2" * 64,
        absolute_dates_removed=True,
        exact_recording_offsets_retained=True,
        local_enu_precision="retained",
        global_horizontal_resolution_m=1.0,
    )
    assert {video.camera for video in plan.recordings[0].videos} == set(
        DEFAULT_DECODED_CAMERAS
    )


@pytest.mark.parametrize(
    ("manifest", "key", "value"),
    [
        ("processing-config.json", "global_horizontal_resolution_m", 10.0),
        ("processing-config.json", "local_enu_precision", "rounded"),
        ("schema-audit.parquet", "absolute_dates_removed", False),
        ("schema-audit.parquet", "exact_recording_offsets_retained", False),
        ("model-receipt.json", "model_lock_sha256", "invalid"),
    ],
)
def test_control_plane_rejects_privacy_contract_drift(
    tmp_path: Path, manifest: str, key: str, value: object
) -> None:
    fixture = write_decoded_fixture(tmp_path)
    mutate_manifest(fixture.root, manifest, key, value)
    with pytest.raises(DecodedManifestError):
        parse_decoded_control_plane(
            fixture.root, decoded_source_config(fixture.revision, [fixture.recording_id])
        )
```

- [ ] **Step 3: Run the parser tests and confirm failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_decoded_manifest.py tests/test_repository_paths.py -q
```

Expected: FAIL because the decoded parser and generic safe repository path validator do not exist.

- [ ] **Step 4: Implement exact immutable control-plane models**

Create these frozen contracts:

```python
@dataclass(frozen=True, slots=True)
class PublicArtifact:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class DecodedArtifact(PublicArtifact):
    kind: str


@dataclass(frozen=True, slots=True)
class DecodedVideo(PublicArtifact):
    camera: CameraName
    frame_count: int
    start_offset_ns: int
    end_offset_ns: int


@dataclass(frozen=True, slots=True)
class PrivacyContract:
    model_lock_sha256: str
    privacy_config_sha256: str
    absolute_dates_removed: bool
    exact_recording_offsets_retained: bool
    local_enu_precision: Literal["retained"]
    global_horizontal_resolution_m: Literal[1.0]


@dataclass(frozen=True, slots=True)
class DecodedRecordingEntry:
    recording_id: str
    split: str
    duration_ns: int
    videos: tuple[DecodedVideo, ...]
    artifacts: tuple[DecodedArtifact, ...]
    fingerprint: DecodedSourceFingerprint

    def artifact(self, kind: str) -> DecodedArtifact:
        matches = tuple(item for item in self.artifacts if item.kind == kind)
        if len(matches) != 1:
            raise DecodedManifestError(
                f"recording requires exactly one {kind!r} artifact"
            )
        return matches[0]


@dataclass(frozen=True, slots=True)
class DecodedSelectionPlan:
    repo_id: str
    revision: str
    privacy: PrivacyContract
    recordings: tuple[DecodedRecordingEntry, ...]
```

Use `pyarrow.parquet.ParquetFile(path).read()` so Hive partition inference cannot rewrite column types. Require the exact catalog columns emitted by `anonymized_catalog.py`: recording/split/camera lists, video path/size/hash maps, frame counts, timing offsets, artifact path/size/hash maps, availability flags, and `published_horizontal_resolution_m`. Verify every selected video/artifact appears exactly once in `output-files.parquet` with identical size/hash. Canonically hash the selected catalog row and ordered artifact tuples to construct `DecodedSourceFingerprint`.

Add `validate_repo_file_path(value)` to `repository_paths.py`; it accepts normalized relative `data/...` paths with allowed suffixes `.mp4`, `.parquet`, `.json`, `.jsonl`, and `.tar`, while rejecting traversal, backslashes, percent encoding, query/fragment text, empty components, and platform-reserved segments.

- [ ] **Step 5: Add adversarial manifest tests**

Add concrete mutations for duplicate recording IDs, catalog/manifest digest mismatch, missing default camera, unknown camera, non-integer size, unsafe artifact path, mismatched parallel artifact arrays, unrequested split, and a date token in a selected public path. Each must raise `DecodedManifestError` before any payload acquisition callback is invoked.

Run:

```bash
.venv/bin/python -m pytest tests/test_decoded_manifest.py tests/test_repository_paths.py -q
.venv/bin/python -m ruff check src/dataset_devkit/decoded_manifest.py src/dataset_devkit/repository_paths.py tests/decoded_v2_fixture.py tests/test_decoded_manifest.py
.venv/bin/python -m mypy src/dataset_devkit/decoded_manifest.py src/dataset_devkit/repository_paths.py tests/decoded_v2_fixture.py tests/test_decoded_manifest.py
```

Expected: PASS.

- [ ] **Step 6: Commit the control plane**

```bash
git add pyproject.toml src/dataset_devkit/decoded_manifest.py src/dataset_devkit/repository_paths.py tests/decoded_v2_fixture.py tests/test_decoded_manifest.py tests/test_repository_paths.py
git commit -m "feat: validate decoded V2 control plane"
```

---

### Task 4: Selective Immutable Artifact Acquisition

**Files:**
- Create: `src/dataset_devkit/decoded_acquisition.py`
- Modify: `src/dataset_devkit/huggingface_acquisition.py`
- Create: `tests/test_decoded_acquisition.py`
- Modify: `tests/test_huggingface_acquisition.py`

**Interfaces:**
- Consumes: `DecodedSelectionPlan`, `DecodedRecordingEntry`, and existing safe Hugging Face file verification/cache operations.
- Produces: `AcquiredDecodedRecording`, `DecodedHfAcquirer.load_plan()`, `DecodedHfAcquirer.acquire(entry)`, and reusable `VerifiedHfFileCache.acquire(artifact)`.

- [ ] **Step 1: Write failing selection and integrity tests**

```python
def test_acquirer_downloads_only_selected_artifacts(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path / "remote")
    calls: list[str] = []
    acquirer = make_decoded_acquirer(fixture, tmp_path / "cache", calls)
    plan = acquirer.load_plan()
    bundle = acquirer.acquire(plan.recordings[0])
    assert bundle.entry.recording_id == fixture.recording_id
    assert set(calls) == {
        "data/recordings.parquet",
        "data/manifests/output-files.parquet",
        "data/manifests/processing-config.json",
        "data/manifests/model-receipt.json",
        "data/manifests/schema-audit.parquet",
        *(artifact.path for artifact in bundle.entry.artifacts),
        *(video.path for video in bundle.entry.videos),
    }


def test_acquirer_rejects_downloaded_digest_mismatch(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path / "remote")
    acquirer = make_decoded_acquirer(fixture, tmp_path / "cache", [])
    plan = acquirer.load_plan()
    corrupt_remote_artifact(fixture.root, plan.recordings[0].artifacts[0].path)
    with pytest.raises(IntegrityError):
        acquirer.acquire(plan.recordings[0])
```

- [ ] **Step 2: Run focused acquisition tests and confirm failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_decoded_acquisition.py tests/test_huggingface_acquisition.py -q
```

Expected: FAIL because decoded acquisition and the reusable verified-file cache do not exist.

- [ ] **Step 3: Refactor the safe file cache without weakening MCAP behavior**

Extract the existing download, no-follow verification, single-link checks, per-artifact lock, atomic promotion, and fsync behavior into:

```python
@dataclass(frozen=True, slots=True)
class VerifiedRepositoryFile:
    artifact: PublicArtifact
    path: Path
    cache_hit: bool


class VerifiedHfFileCache:
    def acquire(
        self,
        *,
        repo_id: str,
        revision: str,
        artifact: PublicArtifact,
    ) -> VerifiedRepositoryFile:
        """Download once, verify size/SHA-256, and return an identity-checked cache leaf."""
```

Make the existing `HuggingFaceAcquirer` use the same primitive for MCAP entries. Retain its manifest and extraction-completion behavior byte-for-byte at the public interface.

- [ ] **Step 4: Implement decoded bundle acquisition**

```python
@dataclass(frozen=True, slots=True)
class AcquiredDecodedRecording:
    entry: DecodedRecordingEntry
    files: Mapping[str, Path]

    def path(self, repo_path: str) -> Path:
        try:
            return self.files[repo_path]
        except KeyError as error:
            raise AcquisitionError("decoded bundle lacks an advertised artifact") from error


class DecodedHfAcquirer:
    def load_plan(self) -> DecodedSelectionPlan:
        control_root = self._acquire_control_files()
        return parse_decoded_control_plane(control_root, self.source)

    def acquire(self, entry: DecodedRecordingEntry) -> AcquiredDecodedRecording:
        ordered = (*entry.videos, *entry.artifacts)
        files = {
            artifact.path: self.cache.acquire(
                repo_id=self.source.repo_id,
                revision=self.source.revision,
                artifact=artifact,
            ).path
            for artifact in ordered
        }
        return AcquiredDecodedRecording(entry, MappingProxyType(files))
```

Control files are immutable by commit identity; selected payload files additionally require the public output manifest size/hash. Do not list the repository or infer paths.

- [ ] **Step 5: Run security and acquisition tests**

Add decoded variants of existing symlink, hard-link, inode-swap, returned-path-escape, zero-byte, short-write, concurrent-acquisition, and cache-reuse tests.

Run:

```bash
.venv/bin/python -m pytest tests/test_decoded_acquisition.py tests/test_huggingface_acquisition.py tests/test_repository_paths.py -q
.venv/bin/python -m ruff check src/dataset_devkit/decoded_acquisition.py src/dataset_devkit/huggingface_acquisition.py tests/test_decoded_acquisition.py
.venv/bin/python -m mypy src/dataset_devkit/decoded_acquisition.py src/dataset_devkit/huggingface_acquisition.py tests/test_decoded_acquisition.py
```

Expected: PASS with no MCAP regression.

- [ ] **Step 6: Commit selective acquisition**

```bash
git add src/dataset_devkit/decoded_acquisition.py src/dataset_devkit/huggingface_acquisition.py tests/test_decoded_acquisition.py tests/test_huggingface_acquisition.py
git commit -m "feat: acquire decoded artifacts selectively"
```

---

### Task 5: Privacy-Transformed GNSS and Local-ENU Trajectories

**Files:**
- Create: `src/dataset_devkit/extraction/decoded_gnss.py`
- Modify: `src/dataset_devkit/extraction/models.py`
- Modify: `src/dataset_devkit/extraction/gnss.py`
- Modify: `src/dataset_devkit/extraction/service.py`
- Create: `tests/test_decoded_gnss.py`
- Modify: `tests/test_gnss_extraction.py`
- Modify: `tests/test_features.py`

**Interfaces:**
- Consumes: sanitized GNSS Parquet rows with `recording_offset_ns`, `local_enu`, orientation, rounded global context, and quality mappings.
- Produces: `PrivacyGnssSample`, `parse_privacy_gnss(path)`, `interpolate_privacy_gnss(samples, offset_ns)`, `PoseFrame`, and `GnssInterpolation.translation_xyz_m`.

- [ ] **Step 1: Write failing local-trajectory tests**

```python
def test_privacy_gnss_pose_uses_local_enu_not_global() -> None:
    samples = (
        privacy_sample(0, local=(0.0, 0.0, 0.0), global_xy=(8_000_000.0, 1_500_000.0)),
        privacy_sample(
            2_000_000_000,
            local=(10.0, 4.0, 0.5),
            global_xy=(8_000_010.0, 1_500_004.0),
        ),
    )
    result = interpolate_privacy_gnss(samples, 1_000_000_000)
    assert result.pose_frame == "recording_local_enu_v1"
    assert result.translation_xyz_m == pytest.approx((5.0, 2.0, 0.25))
    assert result.published_horizontal_resolution_m == 1.0


def test_trajectory_features_ignore_rounded_global_context(base_graph) -> None:
    first = graph_with_privacy_poses(base_graph, global_shift=(0.0, 0.0))
    second = graph_with_privacy_poses(base_graph, global_shift=(1_000_000.0, -500_000.0))
    assert compute_recording_features(first, tags_config()) == compute_recording_features(
        second, tags_config()
    )


def test_privacy_gnss_rejects_nonzero_origin_offset(tmp_path: Path) -> None:
    path = write_gnss_parquet(tmp_path, origin_offset_ns=1)
    with pytest.raises(StructuralExtractionError, match="origin offset"):
        parse_privacy_gnss(path)
```

- [ ] **Step 2: Run GNSS and feature tests and confirm failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_decoded_gnss.py tests/test_gnss_extraction.py tests/test_features.py -q
```

Expected: FAIL because the privacy GNSS model and local pose fields do not exist.

- [ ] **Step 3: Extend interpolation with explicit pose-frame output**

Add:

```python
type PoseFrame = Literal["web_mercator_v1", "recording_local_enu_v1"]


@dataclass(frozen=True)
class PrivacyGnssSample:
    recording_offset_ns: int
    is_valid: bool
    local_east_m: float
    local_north_m: float
    local_up_m: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    published_east_m: float
    published_north_m: float
    latitude_deg: float
    longitude_deg: float
    height_m: float
    position_uncertainty: Mapping[str, Any]
    orientation_uncertainty: Mapping[str, Any]
    quality: Mapping[str, Any]
    local_enu_origin_offset_ns: Literal[0]
    published_horizontal_resolution_m: Literal[1.0]
```

Extend `GnssInterpolation` with nullable `translation_xyz_m`, `pose_frame`, `published_east_m`, `published_north_m`, and `published_horizontal_resolution_m`. Existing `interpolate_gnss()` sets `translation_xyz_m=(projected_x, projected_y,height)` and `pose_frame="web_mercator_v1"`; this preserves MCAP behavior. Change `RecordingExtractor.pose_for()` to read `interpolation.translation_xyz_m` rather than rebuilding it from projected fields.

- [ ] **Step 4: Parse and validate the privacy GNSS schema**

`parse_privacy_gnss()` must use `ParquetFile.read()`, require monotonically increasing unique `recording_offset_ns`, finite float64 local ENU, `local_enu_origin_offset_ns == 0`, and `published_horizontal_resolution_m == 1.0`. Validate integer-metre published east/north with `value == float(round_half_away(value))`; inverse-project them and compare to published latitude/longitude within `1e-9` degrees. Reject absolute-time/date field names recursively.

Implement interpolation:

```python
def interpolate_privacy_gnss(
    samples: Sequence[PrivacyGnssSample], recording_offset_ns: int
) -> GnssInterpolation:
    before, after, fraction = bracket_without_extrapolation(
        samples, recording_offset_ns
    )
    local = tuple(
        linear(left, right, fraction)
        for left, right in zip(
            (before.local_east_m, before.local_north_m, before.local_up_m),
            (after.local_east_m, after.local_north_m, after.local_up_m),
            strict=True,
        )
    )
    published_east = float(round_half_away(linear(
        before.published_east_m, after.published_east_m, fraction
    )))
    published_north = float(round_half_away(linear(
        before.published_north_m, after.published_north_m, fraction
    )))
    longitude, latitude = mercator_to_wgs84(published_east, published_north)
    return build_privacy_interpolation(
        recording_offset_ns,
        before,
        after,
        fraction,
        local,
        published_east,
        published_north,
        latitude,
        longitude,
    )
```

`bracket_without_extrapolation`, `linear`, `mercator_to_wgs84`, and `build_privacy_interpolation` are private functions in `decoded_gnss.py` with typed inputs and no fallback path. Reuse the existing quaternion and bounded uncertainty interpolation helpers by promoting them to package-private functions in `gnss.py`.

- [ ] **Step 5: Add boundary and privacy tests**

Cover exact samples, between-sample local interpolation, no leading/trailing extrapolation, midnight-spanning relative offsets, two recordings with identical local paths and distant global context, positive/negative half-away ties, height preservation, quaternion SLERP, HDOP/fix/RTK/satellite retention, non-finite values, duplicate offsets, missing local ENU, invalid resolution, inconsistent inverse latitude/longitude, and forbidden absolute-time fields.

Run:

```bash
.venv/bin/python -m pytest tests/test_decoded_gnss.py tests/test_gnss_extraction.py tests/test_features.py tests/test_validity.py -q
.venv/bin/python -m ruff check src/dataset_devkit/extraction/decoded_gnss.py src/dataset_devkit/extraction/models.py src/dataset_devkit/extraction/gnss.py src/dataset_devkit/extraction/service.py tests/test_decoded_gnss.py tests/test_features.py
.venv/bin/python -m mypy src/dataset_devkit/extraction/decoded_gnss.py src/dataset_devkit/extraction/models.py src/dataset_devkit/extraction/gnss.py src/dataset_devkit/extraction/service.py
```

Expected: PASS; MCAP pose translations remain Web Mercator, decoded translations are local ENU.

- [ ] **Step 6: Commit local-ENU GNSS support**

```bash
git add src/dataset_devkit/extraction/decoded_gnss.py src/dataset_devkit/extraction/models.py src/dataset_devkit/extraction/gnss.py src/dataset_devkit/extraction/service.py tests/test_decoded_gnss.py tests/test_gnss_extraction.py tests/test_features.py
git commit -m "feat: build trajectories from private-safe local ENU"
```

---

### Task 6: Decoded MP4 and Parquet Recording Adapter

**Files:**
- Create: `src/dataset_devkit/extraction/decoded.py`
- Modify: `src/dataset_devkit/extraction/grid.py`
- Modify: `src/dataset_devkit/extraction/staging.py`
- Create: `tests/test_decoded_extraction.py`
- Modify: `tests/test_extraction_grid.py`
- Modify: `tests/test_camera_extraction.py`

**Interfaces:**
- Consumes: `AcquiredDecodedRecording`, privacy GNSS parser/interpolator, catalog frame counts/offsets, camera-frame and calibration Parquet, and existing `stage_jpeg()`.
- Produces: `DecodedRecordingExtractor.extract_owned(bundle) -> OwnedRecordingExtraction` with the existing `RecordingExtractionResult` shape.

- [ ] **Step 1: Write failing end-to-end adapter tests**

```python
def test_decoded_extractor_stages_six_cameras_without_mcap(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path / "remote")
    bundle = acquire_fixture_bundle(fixture, tmp_path / "cache")
    extractor = DecodedRecordingExtractor(
        target_fps=Fraction(1, 1),
        tolerance_ns=1,
        staging_root=tmp_path / "work",
    )
    owned = extractor.extract_owned(bundle)
    assert {sample.camera_name for sample in owned.result.samples} == set(
        DEFAULT_DECODED_CAMERAS
    )
    assert all(
        sample.ego_pose.interpolation.pose_frame == "recording_local_enu_v1"
        for sample in owned.result.samples
    )
    assert not any(path.suffix == ".mcap" for path in tmp_path.rglob("*"))


def test_decoded_extractor_rejects_frame_count_mismatch(tmp_path: Path) -> None:
    fixture = write_decoded_fixture(tmp_path / "remote")
    change_catalog_frame_count(fixture.root, camera="cam_front", frame_count=3)
    bundle = acquire_fixture_bundle(fixture, tmp_path / "cache")
    with pytest.raises(StructuralExtractionError, match="frame count"):
        make_decoded_extractor(tmp_path).extract_owned(bundle)
```

- [ ] **Step 2: Run decoded extraction tests and confirm failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_decoded_extraction.py tests/test_extraction_grid.py tests/test_camera_extraction.py -q
```

Expected: FAIL because `DecodedRecordingExtractor` does not exist.

- [ ] **Step 3: Implement a frame-indexed decoded adapter**

Add:

```python
class DecodedRecordingExtractor:
    def __init__(
        self,
        *,
        target_fps: Fraction,
        tolerance_ns: int,
        staging_root: Path,
    ) -> None:
        self.target_fps = target_fps
        self.tolerance_ns = tolerance_ns
        self.staging_root = staging_root

    def extract_owned(
        self, bundle: AcquiredDecodedRecording
    ) -> OwnedRecordingExtraction:
        camera_rows = read_camera_frames(bundle)
        calibrations = read_calibrations(bundle)
        gnss = parse_privacy_gnss(bundle.path(bundle.entry.artifact("gnss").path))
        selection = select_camera_grid(
            sorted({row.recording_offset_ns for row in camera_rows}),
            self.target_fps,
            self.tolerance_ns,
        )
        return self._decode_selected(
            bundle, camera_rows, calibrations, gnss, selection
        )
```

Use PyAV to decode each selected MP4 in frame order. Match decoded frames to the published camera-frame rows by `(camera_name, frame_index, recording_offset_ns)`. Decode every frame necessary to reach a selected frame, but stage only selected frames. Use the existing `stage_jpeg()` function, whose contract already fixes JPEG quality at 95, through the safe staging invocation and return its `OwnedDirectoryAuthority`.

Build `RawCameraBatch`, `RawCameraFrame`, `ExtractedCameraSample`, `TimestampObservation`, and `RecordingExtractionResult` with relative offsets in the existing timestamp fields. For each selected offset call `interpolate_privacy_gnss()` and set `EgoPose.translation_xyz_m` from its local translation.

- [ ] **Step 4: Add media, join, and cleanup adversarial tests**

Cover six-camera default selection, opt-in tilted camera, variable frame timing, duplicate/missing frame keys, missing calibration, unknown camera, video decode failure, dimension mismatch, out-of-order frame rows, GNSS bracket failure, safe JPEG identity, exceptional cleanup, and camera/video/catalog frame-count disagreement.

Run:

```bash
.venv/bin/python -m pytest tests/test_decoded_extraction.py tests/test_camera_extraction.py tests/test_extraction_grid.py tests/test_extraction_service.py -q
.venv/bin/python -m ruff check src/dataset_devkit/extraction/decoded.py src/dataset_devkit/extraction/grid.py src/dataset_devkit/extraction/staging.py tests/test_decoded_extraction.py
.venv/bin/python -m mypy src/dataset_devkit/extraction/decoded.py src/dataset_devkit/extraction/grid.py src/dataset_devkit/extraction/staging.py tests/test_decoded_extraction.py
```

Expected: PASS and no source MP4 is modified.

- [ ] **Step 5: Commit the decoded extractor**

```bash
git add src/dataset_devkit/extraction/decoded.py src/dataset_devkit/extraction/grid.py src/dataset_devkit/extraction/staging.py tests/test_decoded_extraction.py tests/test_extraction_grid.py tests/test_camera_extraction.py
git commit -m "feat: extract selected decoded V2 frames"
```

---

### Task 7: Backend-Neutral Build Orchestration

**Files:**
- Create: `src/dataset_devkit/source_runtime.py`
- Modify: `src/dataset_devkit/services.py`
- Modify: `src/dataset_devkit/coordinator.py`
- Modify: `src/dataset_devkit/extraction/cache.py`
- Create: `tests/test_source_runtime.py`
- Modify: `tests/test_validation_pipeline.py`
- Modify: `tests/test_quarantine_coordinator.py`

**Interfaces:**
- Consumes: MCAP acquirer/extractor, decoded acquirer/extractor, `RecordingFingerprint`, and existing coordinator/quarantine flow.
- Produces: `PreparedRecording`, `PreparedSourceBatch`, `SourceRuntimeProtocol`, `prepare_source(config, runtime)`, and one common orchestration loop.

- [ ] **Step 1: Write failing dispatch and no-fallback tests**

```python
def test_prepare_source_dispatches_decoded_backend(decoded_config, runtime) -> None:
    batch = prepare_source(decoded_config, runtime)
    assert batch.source_type == "decoded_hf"
    assert batch.privacy_classification == "privacy_transformed"
    assert [item.recording_id for item in batch.recordings] == ["recording-code"]


def test_decoded_privacy_failure_never_calls_mcap_factory(decoded_config) -> None:
    called = False

    def mcap_factory(config):
        nonlocal called
        called = True
        raise AssertionError("raw fallback invoked")

    runtime = BuildRuntime(
        mcap_acquirer_factory=mcap_factory,
        decoded_acquirer_factory=failing_privacy_acquirer,
    )
    with pytest.raises(BuildOperationalError, match="privacy contract"):
        build_dataset(decoded_config, runtime=runtime)
    assert called is False
```

- [ ] **Step 2: Run orchestration tests and confirm failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_source_runtime.py tests/test_validation_pipeline.py tests/test_quarantine_coordinator.py -q
```

Expected: FAIL because the source runtime boundary does not exist.

- [ ] **Step 3: Introduce prepared recording contracts**

```python
@dataclass(frozen=True)
class PreparedRecording:
    recording_id: str
    locator: str
    fingerprint: RecordingFingerprint
    working_path: Path
    extract_owned: Callable[[], OwnedRecordingExtraction]


@dataclass(frozen=True)
class PreparedSourceBatch:
    source_type: Literal["mcap_hf", "decoded_hf"]
    privacy_classification: PrivacyClassification
    pose_frame: PoseFrame
    source_config_hash: str
    recordings: tuple[PreparedRecording, ...]
```

`prepare_source()` creates an MCAP batch with one prepared item per manifest row or a decoded batch with one prepared item per selected catalog recording. Use the catalog `recording_id` for decoded requests; preserve the existing deterministic `recording-000000` IDs for legacy MCAP.

Split the injectable runtime factories explicitly while preserving current defaults:

```python
@dataclass(frozen=True)
class BuildRuntime:
    mcap_acquirer_factory: Callable[[GlobalConfigV1], AcquirerProtocol] = (
        HuggingFaceAcquirer.from_config
    )
    decoded_acquirer_factory: Callable[[GlobalConfigV2], DecodedHfAcquirer] = (
        DecodedHfAcquirer.from_config
    )
    decoder_factory: Callable[[], HevcDecoder] | None = None
    extraction_cache_factory: Callable[[Path], ExtractionResultCache] = (
        ExtractionResultCache
    )
    official_smoke: bool = True
```

`prepare_source()` narrows the versioned config before calling a factory, so neither factory receives a configuration shape it cannot understand.

- [ ] **Step 4: Refactor `_build_evidence_owned()` around prepared recordings**

Replace MCAP-specific entry/acquisition maps with:

```python
batch = prepare_source(config, runtime)
prepared_by_path = {item.working_path.resolve(): item for item in batch.recordings}


def extract_prepared(path: Path) -> RecordingExtractionResult:
    prepared = prepared_by_path[path.resolve()]
    cached = extraction_cache.materialize_recording(
        prepared.fingerprint,
        extraction_hash,
        config.paths.work_dir,
        prepared.recording_id,
    )
    if cached is not None:
        working.register(prepared.recording_id, cached.result, cached.authority)
        return cached.result
    owned = prepared.extract_owned()
    working.register(prepared.recording_id, owned.result, owned.authority)
    extraction_cache.store_recording(
        prepared.fingerprint, extraction_hash, owned.result
    )
    return owned.result
```

Update `ExtractionResultCache` type annotations and serialized fingerprint parsing to accept `RecordingFingerprint`. Keep its safe materialization and cleanup rules unchanged.

Build `RecordingRequest` values from the prepared ID, path, fingerprint digest, and extraction hash. Quarantine messages use `locator`; no decoded failure may expose a removed source date or raw source filename.

- [ ] **Step 5: Run orchestration, quarantine, and cache suites**

Run:

```bash
.venv/bin/python -m pytest tests/test_source_runtime.py tests/test_validation_pipeline.py tests/test_quarantine_coordinator.py tests/test_publication_lease.py -q
.venv/bin/python -m ruff check src/dataset_devkit/source_runtime.py src/dataset_devkit/services.py src/dataset_devkit/coordinator.py src/dataset_devkit/extraction/cache.py tests/test_source_runtime.py
.venv/bin/python -m mypy src/dataset_devkit/source_runtime.py src/dataset_devkit/services.py src/dataset_devkit/coordinator.py src/dataset_devkit/extraction/cache.py
```

Expected: PASS for both backends, including partial-export and cleanup-debt behavior.

- [ ] **Step 6: Commit backend-neutral orchestration**

```bash
git add src/dataset_devkit/source_runtime.py src/dataset_devkit/services.py src/dataset_devkit/coordinator.py src/dataset_devkit/extraction/cache.py tests/test_source_runtime.py tests/test_validation_pipeline.py tests/test_quarantine_coordinator.py
git commit -m "feat: dispatch builds through source runtimes"
```

---

### Task 8: Privacy and Coordinate-Frame Export, Validation, and Inspection

**Files:**
- Modify: `src/dataset_devkit/export.py`
- Modify: `src/dataset_devkit/validation.py`
- Modify: `src/dataset_devkit/dataset.py`
- Modify: `src/dataset_devkit/services.py`
- Modify: `src/dataset_devkit/cli.py`
- Modify: `tests/test_export_dataset.py`
- Modify: `tests/test_validation_pipeline.py`
- Modify: `tests/test_public_api.py`

**Interfaces:**
- Consumes: `PreparedSourceBatch` metadata, source fingerprint union, decoded `GnssInterpolation`, and existing `ExportEvidence`.
- Produces: `mz_extensions/source.json`, source-aware `recordings.json`, local-pose GNSS extension fields, `Dataset.source_metadata()`, and privacy/pose fields in build and inspection JSON.

- [ ] **Step 1: Write failing export and validation tests**

```python
def test_decoded_export_declares_local_pose_and_privacy(tmp_path: Path) -> None:
    result = build_decoded_fixture_dataset(tmp_path)
    source = json.loads((result.dataroot / "mz_extensions/source.json").read_text())
    assert source == {
        "global_horizontal_resolution_m": 1.0,
        "pose_frame": "recording_local_enu_v1",
        "privacy_classification": "privacy_transformed",
        "schema_version": 1,
        "source_type": "decoded_hf",
    }
    pose = load_first_pose(result.dataroot)
    gnss = load_first_gnss_extension(result.dataroot)
    assert pose["translation"] == pytest.approx(gnss["translation_xyz_m"])
    assert gnss["pose_frame"] == "recording_local_enu_v1"
    assert gnss["published_horizontal_resolution_m"] == 1.0


def test_inspect_surfaces_restricted_raw_for_mcap(mcap_dataset: Path) -> None:
    summary = inspect_dataset(mcap_dataset, "v1.0-trainval")
    assert summary.source_type == "mcap_hf"
    assert summary.privacy_classification == "restricted_raw"
    assert summary.pose_frame == "web_mercator_v1"
```

- [ ] **Step 2: Run export/public API tests and confirm failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_export_dataset.py tests/test_validation_pipeline.py tests/test_public_api.py -q
```

Expected: FAIL because source metadata and local-pose GNSS fields are absent.

- [ ] **Step 3: Export exact source metadata and union fingerprints**

Extend `ExportEvidence` with `source_type`, `privacy_classification`, `pose_frame`, and `global_horizontal_resolution_m`. Write:

```python
source_metadata = {
    "schema_version": 1,
    "source_type": evidence.source_type,
    "privacy_classification": evidence.privacy_classification,
    "pose_frame": evidence.pose_frame,
    "global_horizontal_resolution_m": evidence.global_horizontal_resolution_m,
}
_write_json(writer, ("mz_extensions", "source.json"), source_metadata)
```

Serialize each recording with `fingerprint_to_dict()`. MCAP rows use the discriminated `mcap_hf` shape and decoded rows use the exact decoded shape; validation rejects any hybrid or extra-key form.

For decoded GNSS extension rows add `pose_frame`, `translation_xyz_m`, `published_east_m`, `published_north_m`, and `published_horizontal_resolution_m`. Keep public global context separate from the official local `ego_pose.translation`.

- [ ] **Step 4: Validate privacy and coordinate invariants**

Load `source.json` in validation and `Dataset.__post_init__()`. Require:

```python
if source_type == "decoded_hf":
    expected = ("privacy_transformed", "recording_local_enu_v1", 1.0)
else:
    expected = ("restricted_raw", "web_mercator_v1", None)
actual = (
    metadata["privacy_classification"],
    metadata["pose_frame"],
    metadata["global_horizontal_resolution_m"],
)
if actual != expected:
    findings.append(ValidationFinding(
        "error", "privacy_contract", "source", "source privacy or pose frame is inconsistent"
    ))
```

For every decoded sample-data row, verify its official `ego_pose.translation` equals the GNSS extension `translation_xyz_m`, the pose frame is local, the global resolution is exactly `1.0`, and rounded global east/north remain integer metres. Verify scenes never contain more than one decoded recording fingerprint.

- [ ] **Step 5: Surface metadata through SDK and CLI**

Add:

```python
def source_metadata(self) -> JsonRecord:
    return deepcopy(cast(JsonRecord, self._extensions["source"]))
```

Extend `BuildResult` and `InspectionSummary` with `source_type`, `privacy_classification`, and `pose_frame`; emit them in deterministic CLI JSON for `build` and `inspect`. `validate` adds the same fields by reading validated source metadata.

- [ ] **Step 6: Run export, validation, SDK, and CLI tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_export_dataset.py tests/test_validation_pipeline.py tests/test_public_api.py tests/test_sanity.py -q
.venv/bin/python -m ruff check src/dataset_devkit/export.py src/dataset_devkit/validation.py src/dataset_devkit/dataset.py src/dataset_devkit/services.py src/dataset_devkit/cli.py
.venv/bin/python -m mypy src/dataset_devkit/export.py src/dataset_devkit/validation.py src/dataset_devkit/dataset.py src/dataset_devkit/services.py src/dataset_devkit/cli.py
```

Expected: PASS; tampering with source privacy or pose-frame metadata fails validation.

- [ ] **Step 7: Commit privacy-aware output contracts**

```bash
git add src/dataset_devkit/export.py src/dataset_devkit/validation.py src/dataset_devkit/dataset.py src/dataset_devkit/services.py src/dataset_devkit/cli.py tests/test_export_dataset.py tests/test_validation_pipeline.py tests/test_public_api.py
git commit -m "feat: validate privacy-aware dataset outputs"
```

---

### Task 9: Documentation, Examples, Packaging, and Full Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/configuration.md`
- Modify: `docs/extraction.md`
- Modify: `docs/scenes.md`
- Modify: `docs/selection.md`
- Modify: `docs/export.md`
- Modify: `docs/validity.md`
- Create: `examples/decoded_v2_config.json`
- Create: `tools/decoded_fixture_smoke.py`
- Modify: `tests/test_examples.py`
- Modify: `tests/test_schema.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: all implemented public configuration, CLI, SDK, provenance, and validation contracts.
- Produces: copy-paste decoded configuration, explicit MCAP privacy warning, local-ENU trajectory documentation, packaged dependency metadata, and final quality evidence.

- [ ] **Step 1: Write failing documentation/example tests**

```python
def test_decoded_example_is_schema_and_runtime_valid() -> None:
    path = Path(__file__).parents[1] / "examples/decoded_v2_config.json"
    config = validate_config_schema_and_runtime(path)
    assert config.schema_version == "2.0"
    assert privacy_classification(config) == "privacy_transformed"


def test_readme_distinguishes_raw_and_privacy_transformed_sources() -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    assert "restricted raw" in readme.casefold()
    assert "recording-local ENU" in readme
    assert "does not independently rerun the anonymizer" in readme
```

- [ ] **Step 2: Run example tests and confirm failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_examples.py tests/test_schema.py -q
```

Expected: FAIL because the decoded example and required documentation language do not exist.

- [ ] **Step 3: Document both source workflows precisely**

Update README and guides with:

- decoded V2 as the recommended public-consumer workflow;
- MCAP as original non-anonymized restricted raw input;
- the five-file control-plane verification and its limits;
- immutable revision and selective modality configuration;
- six default cameras and optional `cam_front_tilted`;
- exact relative nanosecond synchronization;
- local ENU trajectory/ego-pose calculations;
- one-metre rounded global context and no cross-recording continuity;
- failure behavior with no raw fallback; and
- `mzcloud` for production payload processing.

Use the all-zero immutable fixture revision in `examples/decoded_v2_config.json` and state beside the example that users must replace it with a verified release commit. Do not name `main`, an active candidate branch, or the current in-progress anonymization head.

Create `tools/decoded_fixture_smoke.py` as a local-only verification harness. Its `main(output_root: Path) -> int` writes the same two-frame/six-camera fixture as `tests/decoded_v2_fixture.py`, injects its file-backed downloader through `BuildRuntime`, builds into `output_root`, and prints the absolute published dataroot as its only stdout line. It rejects an existing output root so it cannot overwrite prior artifacts.

- [ ] **Step 4: Verify generated schema and packaged metadata**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m dataset_devkit.schema /tmp/dataset_config.schema.json
cmp schema/dataset_config.schema.json /tmp/dataset_config.schema.json
.venv/bin/python -m build --no-isolation
.venv/bin/python -c 'from zipfile import ZipFile; from pathlib import Path; wheel=sorted(Path("dist").glob("dataset_devkit-*.whl"))[-1]; names=ZipFile(wheel).namelist(); assert any(name.endswith("LICENSE.md") for name in names); assert any(name.endswith("NOTICE") for name in names)'
```

Expected: schema bytes match; wheel build succeeds; legal files remain packaged; wheel metadata includes `pyarrow>=25,<26`.

- [ ] **Step 5: Run the complete local quality gate**

Run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy
git diff --check
```

Expected: all tests pass, Ruff is clean, strict mypy is clean, and `git diff --check` produces no output.

- [ ] **Step 6: Run fresh-install CLI and fixture smoke tests**

Create a temporary virtual environment outside the repository, install the built wheel without editable sources, build the fixture through the explicit local harness, then validate and inspect it using only the installed wheel:

```bash
DEVKIT_SMOKE_ROOT="$(mktemp -d /private/tmp/dataset-devkit-smoke.XXXXXX)"
python3.12 -m venv "$DEVKIT_SMOKE_ROOT/venv"
"$DEVKIT_SMOKE_ROOT/venv/bin/python" -m pip install --no-deps "$(find dist -name 'dataset_devkit-*.whl' -print -quit)"
"$DEVKIT_SMOKE_ROOT/venv/bin/python" -m pip install 'av>=13,<17' 'huggingface-hub>=1.25,<2' 'jsonschema>=4.23,<5' 'numpy>=1.26,<2' 'nuscenes-devkit>=1.1.11,<2' 'pillow>=11,<13' 'protobuf>=5,<7' 'pydantic>=2.10,<3' 'pyarrow>=25,<26' 'pyproj>=3.7,<4' 'scikit-learn>=1.6,<2' 'scipy>=1.15,<2'
"$DEVKIT_SMOKE_ROOT/venv/bin/python" -c 'from dataset_devkit import Dataset; from dataset_devkit.config import GLOBAL_CONFIG_ADAPTER; print("imports-ok")'
"$DEVKIT_SMOKE_ROOT/venv/bin/dataset-devkit" --help
"$DEVKIT_SMOKE_ROOT/venv/bin/python" tools/decoded_fixture_smoke.py "$DEVKIT_SMOKE_ROOT/output"
"$DEVKIT_SMOKE_ROOT/venv/bin/dataset-devkit" validate --dataroot "$DEVKIT_SMOKE_ROOT/output" --version v1.0-trainval
"$DEVKIT_SMOKE_ROOT/venv/bin/dataset-devkit" inspect --dataroot "$DEVKIT_SMOKE_ROOT/output" --version v1.0-trainval
```

Expected: import and help succeed; the local harness, validate, and inspect return exit `0`; JSON reports `decoded_hf`, `privacy_transformed`, and `recording_local_enu_v1`. The smoke fixture remains tiny and local; no production data is downloaded.

- [ ] **Step 7: Commit documentation and final verification assets**

```bash
git add README.md docs/configuration.md docs/extraction.md docs/scenes.md docs/selection.md docs/export.md docs/validity.md examples/decoded_v2_config.json tools/decoded_fixture_smoke.py tests/test_examples.py tests/test_schema.py pyproject.toml
git commit -m "docs: explain privacy-safe decoded V2 workflow"
```

- [ ] **Step 8: Record the final branch evidence**

Run:

```bash
git status --short --branch
git log --oneline --decorate --max-count=12
git diff main...HEAD --check
git diff --stat main...HEAD
```

Expected: clean worktree, only intended commits on the feature branch, no whitespace errors, and changes limited to the approved decoded V2 privacy backend.
