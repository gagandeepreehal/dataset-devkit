# INDUS-AD Integrated Annotations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a schema `2.1` backend that safely consumes the current privacy-video V3 INDUS-AD layout and exposes its integrated camera, semantic, depth, weather, and optional GNSS weak pseudo-labels through deterministic exports and the read-only SDK.

**Architecture:** Add an explicit `indus_ad_hf` adapter rather than auto-detecting layouts inside `decoded_hf`. The adapter validates a pinned repository tree and current Viewer/index schemas, reuses decoded MP4/privacy-GNSS extraction, carries typed annotation bundles beside the legacy scene graph, joins only selected output by exact recording/frame/window identities, and writes independently validated Parquet extensions. Dense semantic/depth archives remain metadata-only unless selected frames explicitly request them.

**Tech Stack:** Python 3.11+, Pydantic v2, PyArrow/Parquet, huggingface_hub, PyAV, pytest, Ruff, strict mypy, standard-library tarfile/hashlib/path safety primitives.

**Spec:** `docs/superpowers/specs/2026-10-05-indus-ad-integrated-annotations-design.md`

## Global Constraints

- Preserve schema `1.0` `mcap_hf` as `restricted_raw`; decoding MCAP is never described as anonymization.
- Preserve schema `2.0` `decoded_hf` behavior and exported bytes.
- Schema `2.1` accepts only `source.type = "indus_ad_hf"`, `contract = "indus_ad_integrated_v1"`, and an exact lowercase 40-character revision.
- The reference revision is `2d46a81376ded440506bd1646c0a5197589ba27b`.
- The pinned Viewer is authoritative for V3 video status: 9,927 non-null `face_plate_blurred` cells and 356 genuine missing tilted-camera cells.
- Require `data/metadata.jsonl`, the pinned videos Viewer, and `data/manifests/video-privacy-release.json` to agree on privacy status and exact control-file digests.
- Build poses, trajectories, speed, distance, curvature, filtering, and scene selection only from precise per-recording `local_enu` and recording-relative nanoseconds.
- Join camera annotations exactly; no basename, date, coordinate, or fuzzy-time matching.
- Keep semantic, depth, weather, and GNSS-derived values visibly classified as machine-generated weak pseudo-labels.
- Keep official nuScenes `sample_annotation.json` empty; write integrated labels only under `mz_extensions/indus_ad/`.
- Dense archives are metadata-only by default and production payload work stays on `mzcloud`.
- Any privacy, schema, identity, path, archive, ownership, or digest mismatch fails closed with no raw-MCAP fallback.

## File Structure

- `src/dataset_devkit/config.py`: schema `2.1` configuration and closed source-selection types.
- `src/dataset_devkit/provenance.py`: integrated-source fingerprint serialization and hashing.
- `src/dataset_devkit/indus_ad_models.py`: immutable repository, annotation, availability, join, and dense-asset records.
- `src/dataset_devkit/indus_ad_contract.py`: pinned tree, Viewer schema, privacy status, catalog, and companion contract validation.
- `src/dataset_devkit/indus_ad_annotations.py`: camera and optional GNSS weak-label parsing and privacy-field rejection.
- `src/dataset_devkit/indus_ad_acquisition.py`: selective metadata/payload acquisition and verified local receipts.
- `src/dataset_devkit/indus_ad_dense.py`: safe selected-member extraction from semantic/depth archives.
- `src/dataset_devkit/indus_ad_join.py`: exact joins from source annotation identities to selected scene/sample-data tokens.
- `src/dataset_devkit/indus_ad_export.py`: deterministic Arrow schemas and extension serialization.
- `src/dataset_devkit/source_runtime.py`: schema `2.1` dispatch without MCAP fallback.
- `src/dataset_devkit/services.py`: carry annotation evidence through selection, dense materialization, export, and inspection.
- `src/dataset_devkit/export.py`: write integrated extensions and source contract metadata.
- `src/dataset_devkit/validation.py`: extension/reference/privacy/content-manifest validation.
- `src/dataset_devkit/dataset.py`: typed read-only annotation indexes and accessors.
- `tests/indus_ad_fixture.py`: tiny local mirror of the integrated tree and exact Arrow schemas.
- `tests/test_indus_ad_*.py`: focused contract, parser, acquisition, dense, join, export, SDK, and live-check tests.
- `tools/check_indus_ad_contract.py`: authenticated metadata-only immutable-revision compatibility check.
- `examples/indus_ad_v21_config.json`, `README.md`, and focused docs: public configuration and usage contract.

---

### Task 1: Add schema 2.1 configuration and provenance types

**Files:**
- Modify: `src/dataset_devkit/config.py:700-885`
- Modify: `src/dataset_devkit/provenance.py:35-190`
- Modify: `tests/test_config.py`
- Modify: `tests/test_schema.py`
- Modify: `tests/test_provenance.py`
- Create: `examples/indus_ad_v21_config.json`
- Modify: `schema/dataset_config.schema.json`

**Interfaces:**
- Produces: `DenseAssetMode`, `IndusAdAnnotationLayer`, `DenseAssetSelection`, `GnssAnnotationSourceConfig`, `IndusAdHfSourceConfig`, `GlobalConfigV21`, and `IndusAdSourceFingerprint`.
- Preserves: `GlobalConfigV1`, `GlobalConfigV2`, `DecodedHfSourceConfig`, and their serialized schemas.

- [ ] **Step 1: Write failing schema 2.1 configuration tests**

Add `minimal_indus_ad_config()` and tests proving immutable revisions, closed literals, unique selectors, required core layers, and metadata-only dense defaults:

```python
def test_indus_ad_v21_config_is_explicit_and_privacy_transformed() -> None:
    config = GLOBAL_CONFIG_ADAPTER.validate_python(minimal_indus_ad_config())

    assert isinstance(config, GlobalConfigV21)
    assert config.source.type == "indus_ad_hf"
    assert config.source.contract == "indus_ad_integrated_v1"
    assert config.source.dense_assets.semantic == "none"
    assert privacy_classification(config) == "privacy_transformed"


@pytest.mark.parametrize("revision", ["main", "FA8CA03" + "0" * 33, "f" * 39])
def test_indus_ad_v21_rejects_mutable_or_noncanonical_revision(revision: str) -> None:
    data = minimal_indus_ad_config()
    data["source"]["revision"] = revision
    with pytest.raises(ValidationError):
        GLOBAL_CONFIG_ADAPTER.validate_python(data)
```

- [ ] **Step 2: Run the focused tests and verify the missing-type failure**

Run:

```bash
.venv/bin/pytest tests/test_config.py tests/test_schema.py tests/test_provenance.py -q
```

Expected: FAIL because schema `2.1` and `IndusAdSourceFingerprint` do not exist.

- [ ] **Step 3: Implement the closed configuration types**

Add these exact public shapes in `config.py`:

```python
DenseAssetMode = Literal["none", "selected_frames"]
IndusAdAnnotationLayer = Literal[
    "camera_scalar",
    "semantic_fractions",
    "depth_summary",
    "weather",
    "gnss_scenarios",
]


class DenseAssetSelection(StrictModel):
    semantic: DenseAssetMode = "none"
    depth: DenseAssetMode = "none"


class GnssAnnotationSourceConfig(HuggingFaceIdentityConfig):
    contract: Literal["indus_ad_gnss_annotations_v1"]
    expected_source_revision: Revision


class IndusAdHfSourceConfig(HuggingFaceIdentityConfig):
    type: Literal["indus_ad_hf"]
    contract: Literal["indus_ad_integrated_v1"]
    recording_ids: list[SafeSegment] = Field(default_factory=list, json_schema_extra={"uniqueItems": True})
    splits: list[SafeSegment] = Field(default_factory=list, json_schema_extra={"uniqueItems": True})
    cameras: list[CameraName] = Field(default_factory=lambda: list(DEFAULT_DECODED_CAMERAS), json_schema_extra={"uniqueItems": True})
    annotation_layers: list[IndusAdAnnotationLayer]
    dense_assets: DenseAssetSelection = Field(default_factory=DenseAssetSelection)
    gnss_annotation_source: GnssAnnotationSourceConfig | None = None
    privacy_classification: Literal["privacy_transformed"] = "privacy_transformed"
```

The model validator must require at least one of `recording_ids` or `splits`, reject duplicates, require `camera_scalar` when any camera-derived layer is selected, require a GNSS companion when `gnss_scenarios` is selected, and reject a configured companion when that layer is absent. Add `GlobalConfigV21`, extend `GlobalConfig`, `SourceConfig`, `normalized_source()`, and `extraction_config_hash()` without changing V1/V2 branches.

- [ ] **Step 4: Implement integrated-source fingerprint serialization**

Add a frozen `IndusAdSourceFingerprint` with these exact fields:

```python
@dataclass(frozen=True)
class IndusAdSourceFingerprint:
    repo_id: str
    revision: str
    contract: Literal["indus_ad_integrated_v1"]
    recording_id: str
    catalog_sha256: str
    selected_artifacts_sha256: str
    annotation_schema_sha256: str
    total_size: int
    source_type: Literal["indus_ad_hf"] = "indus_ad_hf"
    privacy_classification: Literal["privacy_transformed"] = "privacy_transformed"
```

Validate exact repository/revision/digest/size formats, add `to_dict()`, `from_dict()`, `digest`, and `cache_key`, and include it in `RecordingFingerprint`, `fingerprint_from_dict()`, and `fingerprint_locator()`.

- [ ] **Step 5: Add the checked-in example and regenerate JSON Schema**

Copy the spec's schema `2.1` source block into `examples/indus_ad_v21_config.json`, retaining the existing common pipeline sections. Run:

```bash
.venv/bin/python -m dataset_devkit.schema schema/dataset_config.schema.json
.venv/bin/pytest tests/test_config.py tests/test_schema.py tests/test_provenance.py tests/test_examples.py -q
```

Expected: PASS, and `test_checked_in_schema_is_current_and_deterministic` confirms byte equality.

- [ ] **Step 6: Commit the configuration contract**

```bash
git add src/dataset_devkit/config.py src/dataset_devkit/provenance.py tests/test_config.py tests/test_schema.py tests/test_provenance.py tests/test_examples.py examples/indus_ad_v21_config.json schema/dataset_config.schema.json
git commit -m "feat: add INDUS-AD schema 2.1 contract"
```

---

### Task 2: Build the tiny integrated fixture and repository contract adapter

**Files:**
- Create: `src/dataset_devkit/indus_ad_models.py`
- Create: `src/dataset_devkit/indus_ad_contract.py`
- Create: `tests/indus_ad_fixture.py`
- Create: `tests/test_indus_ad_contract.py`

**Interfaces:**
- Consumes: `IndusAdHfSourceConfig` and `IndusAdSourceFingerprint` from Task 1.
- Produces: `RepositoryFile`, `IndusAdArtifact`, `IndusAdRecordingEntry`, `IndusAdSelectionPlan`, and `parse_indus_ad_contract(root, source, tree)`.

- [ ] **Step 1: Create a generated fixture matching the live integrated layout**

Implement `write_indus_ad_fixture(root: Path) -> IndusAdFixture` by extending the existing decoded fixture with:

```text
data/metadata.jsonl
data/viewer/videos/part-00000.parquet
data/viewer/camera-annotations/part-00000.parquet
data/annotations/camera/dense-index/part-shard-fixture.parquet
data/annotations/camera/semantic/shard-fixture.tar
data/annotations/camera/depth/shard-fixture.tar
data/manifests/video-integrity.jsonl
data/manifests/video-privacy-release.json
```

The Viewer video fixture must contain six non-null `face_plate_blurred` video cells, one missing tilted cell, and no embedded video bytes. Its metadata, integrity rows, and privacy-release manifest must match the repaired live control plane and bind exact hashes.

- [ ] **Step 2: Write failing catalog and privacy-authority tests**

Add tests for exact selection and cross-control agreement:

```python
def test_contract_requires_repaired_privacy_controls_to_agree(tmp_path: Path) -> None:
    fixture = write_indus_ad_fixture(tmp_path)
    plan = parse_indus_ad_contract(fixture.root, fixture.source, fixture.tree)

    assert plan.viewer_privacy_counts == {"face_plate_blurred": 6, "missing": 1}
    assert plan.recordings[0].recording_id == fixture.recording_id
    assert all(video.privacy_status == "face_plate_blurred" for video in plan.recordings[0].videos)


def test_contract_rejects_one_nonnull_viewer_cell_not_marked_blurred(tmp_path: Path) -> None:
    fixture = write_indus_ad_fixture(tmp_path)
    rewrite_viewer_status(fixture, camera="cam_front", status="not_face_plate_blurred")
    with pytest.raises(IndusAdContractError, match="Viewer privacy status"):
        parse_indus_ad_contract(fixture.root, fixture.source, fixture.tree)
```

Also cover duplicate recording IDs, missing video paths, unknown path families, tree size disagreement, unexpected Viewer columns/types, unsafe/date-bearing paths, absent selected cameras, unmatched selectors, and manifest/status/digest disagreement.

- [ ] **Step 3: Run the contract tests and verify failure**

```bash
.venv/bin/pytest tests/test_indus_ad_contract.py -q
```

Expected: FAIL because the adapter and model types do not exist.

- [ ] **Step 4: Implement immutable repository and plan models**

Define:

```python
@dataclass(frozen=True, slots=True)
class RepositoryFile:
    path: str
    size: int
    blob_id: str


@dataclass(frozen=True, slots=True)
class IndusAdArtifact:
    path: str
    size: int
    blob_id: str
    kind: Literal["video", "camera_frames", "calibration", "gnss", "camera_annotations", "dense_index", "semantic_archive", "depth_archive"]
    camera: CameraName | None = None


@dataclass(frozen=True, slots=True)
class IndusAdRecordingEntry:
    recording_id: str
    split: str
    videos: tuple[IndusAdArtifact, ...]
    artifacts: tuple[IndusAdArtifact, ...]
    fingerprint: IndusAdSourceFingerprint


@dataclass(frozen=True, slots=True)
class IndusAdSelectionPlan:
    repo_id: str
    revision: str
    contract: Literal["indus_ad_integrated_v1"]
    viewer_privacy_counts: Mapping[str, int]
    annotation_schema_sha256: str
    recordings: tuple[IndusAdRecordingEntry, ...]
```

Freeze mappings with `MappingProxyType` and validate every safe segment, path, size, blob identity, and uniqueness invariant in `__post_init__`.

- [ ] **Step 5: Implement `parse_indus_ad_contract`**

Use exact Arrow schemas for the live video Viewer, camera annotation Viewer, and dense index. Normalize `metadata.jsonl` video paths by prefixing `data/`. Require every non-null Viewer and metadata video cell to have `face_plate_blurred`, require missing tilted cells to have status `missing`, verify the privacy manifest's counts and control-file SHA-256 values, bind every path/size/blob ID to the supplied pinned tree, and derive table paths by exact recording suffix plus one-and-only-one matching path per required family.

Compute `annotation_schema_sha256` from canonical Arrow field/type/nullability descriptions. Build recording fingerprints from canonical catalog rows and the sorted selected artifact identity set. Treat the repaired privacy manifest as an integrity binding, not as permission to overlook Viewer or metadata disagreement.

- [ ] **Step 6: Run tests and commit the contract adapter**

```bash
.venv/bin/pytest tests/test_indus_ad_contract.py tests/test_decoded_manifest.py -q
git add src/dataset_devkit/indus_ad_models.py src/dataset_devkit/indus_ad_contract.py tests/indus_ad_fixture.py tests/test_indus_ad_contract.py
git commit -m "feat: validate integrated INDUS-AD contract"
```

Expected: PASS; schema `2.0` manifest tests remain unchanged.

---

### Task 3: Parse camera and optional GNSS weak pseudo-labels

**Files:**
- Modify: `src/dataset_devkit/indus_ad_models.py`
- Create: `src/dataset_devkit/indus_ad_annotations.py`
- Create: `tests/test_indus_ad_annotations.py`
- Modify: `tests/indus_ad_fixture.py`

**Interfaces:**
- Produces: `SemanticFraction`, `CameraAnnotation`, `GnssScenarioAnnotation`, `DenseAssetRef`, `AnnotationAvailability`, `IndusAdAnnotationBundle`, `parse_camera_annotations()`, `parse_dense_indexes()`, and `parse_gnss_scenarios()`.
- Consumes later: Task 5 join service and Task 6 exporter.

- [ ] **Step 1: Write failing exact-schema and privacy tests**

Cover all current camera Viewer fields, including nested `semantic_fractions`, derived category fractions, depth p10/p50/p90, six weather probabilities, weather label/confidence/margin/uncertainty, luminance/contrast/saturation/sharpness, dark fraction, and overexposed fraction.

```python
def test_camera_parser_preserves_weak_label_values(tmp_path: Path) -> None:
    fixture = write_indus_ad_fixture(tmp_path)
    bundle = parse_camera_annotations(
        fixture.camera_annotations,
        expected_recordings={fixture.recording_id: fixture.duration_ns},
    )

    row = bundle.camera[0]
    assert row.identity == (fixture.recording_id, "frame-fixture")
    assert row.semantic_fractions == (SemanticFraction(class_id=13, fraction=0.25),)
    assert row.weather.label == "clear"
    assert row.provenance == "machine_generated_weak_pseudo_label"
```

Parameterize rejection of NaN/inf, fractions outside `[0, 1]`, sparse semantic sum above `1.0 + 1e-6`, weather sum outside `1.0 ± 1e-5`, negative depth, inconsistent percentile order, duplicate frame IDs, foreign recording IDs, out-of-range offsets, forbidden nested date/epoch/coordinate/OSM fields, and invalid dimensions.

- [ ] **Step 2: Define immutable annotation models**

Use frozen dataclasses and closed literals. `CameraAnnotation` must carry the exact identity fields:

```python
@dataclass(frozen=True, slots=True)
class CameraAnnotation:
    recording_id: str
    camera_name: CameraName
    selected_time_of_day_ns: int | None
    recording_offset_ns: int
    frame_index: int
    frame_id: str
    sample_kind: str
    window_id: str
    decoded_width: int
    decoded_height: int
    semantic_fractions: tuple[SemanticFraction, ...]
    category_fractions: Mapping[str, float | None]
    depth_summary_m: Mapping[str, float | None]
    weather: WeatherWeakLabel
    image_quality: ImageQualitySummary
    provenance: Literal["machine_generated_weak_pseudo_label"] = "machine_generated_weak_pseudo_label"
```

`GnssScenarioAnnotation` must contain `recording_id`, `window_index`, relative window bounds, controlled motion/validity/quality tags, anomaly-candidate boolean, anomaly reasons, confidence, schema version, run ID, and the same weak-label provenance literal. It must contain no coordinates, timestamps, dates, road names, geometry, endpoints, or OSM identifiers.

Define the aggregate with an immutable availability map:

```python
@dataclass(frozen=True, slots=True)
class IndusAdAnnotationBundle:
    camera: tuple[CameraAnnotation, ...]
    gnss: tuple[GnssScenarioAnnotation, ...]
    dense_refs: tuple[DenseAssetRef, ...]
    availability: Mapping[str, AnnotationAvailability]
```

Freeze `availability` with `MappingProxyType`; accepted availability states are exactly `available`, `missing`, `not_selected`, and `unavailable_source` with nonblank stable reason codes.

- [ ] **Step 3: Implement strict streaming parsers**

Read Parquet by row group with bounded batch sizes; validate the exact Arrow schema before reading rows. Implement `scan_forbidden_metadata(value, location)` recursively across field names, schema metadata, nested JSON strings, mappings, lists, and scalar strings. Camera identity is `(recording_id, frame_id)`; source-frame binding identity is `(recording_id, camera_name, frame_index, recording_offset_ns)` and must also be unique.

For GNSS companion input, require exact schema version `indus_ad_gnss_annotations_v1`, exact source revision equality, and exact `(recording_id, window_index)` uniqueness.

- [ ] **Step 4: Parse dense indexes without opening archives**

Require the exact current columns:

```text
artifact_kind, camera_name, dtype, frame_index, height, member_path,
payload_bytes, payload_sha256, recording_id, recording_offset_ns,
shard_path, width
```

Return `DenseAssetRef` keyed by `(recording_id, camera_name, frame_index, recording_offset_ns, artifact_kind)`. Require kinds `semantic` or `depth`, safe archive/member paths, positive shapes and payload sizes, lowercase SHA-256, and a unique camera annotation source-frame binding.

- [ ] **Step 5: Run parser tests and commit**

```bash
.venv/bin/pytest tests/test_indus_ad_annotations.py -q
git add src/dataset_devkit/indus_ad_models.py src/dataset_devkit/indus_ad_annotations.py tests/indus_ad_fixture.py tests/test_indus_ad_annotations.py
git commit -m "feat: parse INDUS-AD weak pseudo-labels"
```

---

### Task 4: Add selective acquisition and safe dense-member materialization

**Files:**
- Modify: `src/dataset_devkit/indus_ad_models.py`
- Create: `src/dataset_devkit/indus_ad_acquisition.py`
- Create: `src/dataset_devkit/indus_ad_dense.py`
- Modify: `src/dataset_devkit/extraction/decoded.py:330-555`
- Create: `tests/test_indus_ad_acquisition.py`
- Create: `tests/test_indus_ad_dense.py`

**Interfaces:**
- Produces: `IndusAdHfAcquirer.load_plan()`, `IndusAdHfAcquirer.acquire(entry)`, `AcquiredIndusAdRecording`, and `materialize_dense_assets(frame_ids)`.
- Reuses: verified cache/path/lock helpers from `huggingface_acquisition.py` and decoded extraction through a structural bundle protocol.

- [ ] **Step 1: Write failing metadata-only acquisition tests**

Inject `list_tree` and `download_file` callables. Assert that `load_plan()` downloads only `data/metadata.jsonl`, the video Viewer, camera annotation Viewer, selected dense indexes, and optional companion control files. Assert that `acquire()` downloads only selected videos/core recording tables and never semantic/depth archives while dense mode is `none`.

```python
assert not any(path.endswith(".tar") for path in calls)
assert bundle.annotations.camera
assert bundle.materialize_dense_assets(frozenset()) == ()
```

Also test immutable revision propagation, returned-path escape, symlink/hard-link rejection, size mismatch, blob-ID mismatch, concurrent cache reuse, and locally recorded SHA-256 receipts.

- [ ] **Step 2: Implement the pinned tree and acquisition boundary**

Add injectable protocols:

```python
ListRepositoryTree = Callable[[str, str], tuple[RepositoryFile, ...]]


@dataclass(frozen=True)
class AcquiredIndusAdRecording:
    entry: IndusAdRecordingEntry
    files: Mapping[str, Path]
    annotations: IndusAdAnnotationBundle
    dense_mode: DenseAssetSelection
    materialize_dense_assets: Callable[[frozenset[str]], tuple[MaterializedDenseAsset, ...]]

    def path(self, repo_path: str) -> Path:
        try:
            return self.files[repo_path]
        except KeyError as error:
            raise AcquisitionError("integrated bundle lacks an advertised artifact") from error
```

Default tree listing must use authenticated `HfApi.list_repo_tree(..., recursive=True, expand=True)` at the configured revision and convert results to stable `RepositoryFile` records. Downloads remain pinned to the same revision. Every downloaded file is identity-checked, size-checked, hashed locally, and recorded in a canonical receipt under the verified cache.

- [ ] **Step 3: Generalize decoded extraction structurally**

In `extraction/decoded.py`, replace the concrete `AcquiredDecodedRecording` argument with private protocols exposing `entry.recording_id`, `entry.videos`, `entry.artifact(s)`, and `path(repo_path)`. Keep decoded frame/GNSS behavior unchanged. Add regression tests proving the existing `AcquiredDecodedRecording` and new integrated bundle produce identical extraction results for identical tiny media/tables.

- [ ] **Step 4: Write adversarial dense-archive tests**

Create archives containing traversal paths, absolute paths, symlinks, hard links, duplicate names, unexpected members, wrong member sizes/digests, oversized compressed members, and decompression over-budget payloads. Assert rejection before any unverified output becomes visible. Also assert that requesting one frame reads and materializes only its indexed semantic/depth members.

- [ ] **Step 5: Implement selected-member extraction**

Use descriptor-relative, no-follow cache paths. Pre-index tar headers, require each selected member to be a unique regular file with normalized metadata, stream exactly `payload_bytes` through SHA-256, reject trailing/short content, decode enough to validate dtype/shape, and atomically publish the verified member. Use explicit budgets:

```python
MAX_DENSE_MEMBER_BYTES = 256 * 1024 * 1024
MAX_DENSE_MEMBERS_PER_REQUEST = 250_000
MAX_DENSE_TOTAL_BYTES_PER_REQUEST = 64 * 1024 * 1024 * 1024
```

Return frozen `MaterializedDenseAsset` records containing source ref, verified local path, size, and SHA-256.

```python
@dataclass(frozen=True, slots=True)
class MaterializedDenseAsset:
    ref: DenseAssetRef
    path: Path
    size: int
    sha256: str
```

- [ ] **Step 6: Run acquisition/dense tests and commit**

```bash
.venv/bin/pytest tests/test_indus_ad_acquisition.py tests/test_indus_ad_dense.py tests/test_decoded_acquisition.py tests/test_decoded_extraction.py -q
git add src/dataset_devkit/indus_ad_models.py src/dataset_devkit/indus_ad_acquisition.py src/dataset_devkit/indus_ad_dense.py src/dataset_devkit/extraction/decoded.py tests/test_indus_ad_acquisition.py tests/test_indus_ad_dense.py tests/test_decoded_extraction.py
git commit -m "feat: acquire selected INDUS-AD annotation assets"
```

---

### Task 5: Dispatch schema 2.1 and join annotations to selected graph records

**Files:**
- Create: `src/dataset_devkit/indus_ad_join.py`
- Modify: `src/dataset_devkit/source_runtime.py`
- Modify: `src/dataset_devkit/services.py`
- Modify: `src/dataset_devkit/extraction/models.py`
- Modify: `tests/test_source_runtime.py`
- Create: `tests/test_indus_ad_join.py`
- Create: `tests/test_indus_ad_pipeline.py`

**Interfaces:**
- Produces: `IndusAdAcquirerProtocol`, `join_indus_ad_annotations(graph, bundle) -> JoinedIndusAdAnnotations`, and schema `2.1` build orchestration.
- Preserves: scene graph/token derivation and legacy output when no integrated bundle exists.

- [ ] **Step 1: Write failing dispatch/no-fallback tests**

Add a `BuildRuntime.indus_ad_acquirer_factory` injection and prove only it is called for `GlobalConfigV21`. Make both MCAP and schema `2.0` factories raise if invoked. Assert batch metadata is:

```python
assert batch.source_type == "indus_ad_hf"
assert batch.privacy_classification == "privacy_transformed"
assert batch.pose_frame == "recording_local_enu_v1"
assert batch.global_horizontal_resolution_m == 1.0
```

Add a failing-contract test proving neither legacy factory is called after a schema/privacy failure.

- [ ] **Step 2: Carry source-frame identity without altering legacy graph tokens**

Do not add nullable fields to `SampleDataRecord`, because its dataclass serialization participates in legacy graph validation. Instead, retain each integrated `IndusAdAnnotationBundle` and dense callback on `PreparedRecording`:

```python
@dataclass(frozen=True)
class PreparedRecording:
    # existing fields unchanged
    indus_ad_annotations: IndusAdAnnotationBundle | None = None
    materialize_dense_assets: Callable[[frozenset[str]], tuple[MaterializedDenseAsset, ...]] | None = None
```

The scalar join uses exact `(recording_id, camera_name, recording_offset_ns)` from the validated annotation source-frame index and exact `SampleDataRecord.channel/timestamp_ns`; the parser has already proven the associated `frame_index` and `frame_id` are unique. This is exact equality, not tolerance matching.

- [ ] **Step 3: Write failing join tests**

Cover one-to-one camera joins, selected-output filtering, unmatched/missing availability, duplicate sample-data bindings, camera mismatch, cross-recording rows, GNSS window overlap, and deterministic ordering.

```python
joined = join_indus_ad_annotations(graph, fixture.annotations)
assert joined.camera[0].sample_data_token == graph.sample_data[0].token
assert joined.camera[0].annotation.frame_id == "frame-fixture"
assert joined.audit.excluded_camera_rows == 1
```

GNSS window joins must use recording-relative inclusive bounds and list exact matching scene/sample tokens. No global coordinates or inferred timestamps may enter the function.

- [ ] **Step 4: Implement join models and service**

Define `JoinedCameraAnnotation`, `JoinedGnssAnnotation`, `JoinedDenseAsset`, `AnnotationJoinAudit`, and `JoinedIndusAdAnnotations`. Index graph sample data by exact `(channel, timestamp_ns)`, require at most one source annotation per key, and emit only selected records. Keep absence states explicit. Sort camera rows by `(sample_data_token, frame_id)` and GNSS rows by `(recording_id, window_index)`.

- [ ] **Step 5: Integrate joins after scenario selection and dense acquisition after joins**

In `_build_evidence_owned`, retain bundles by `source.digest`, build scenes/features/splits unchanged, then join only `selected_graphs`. Collect selected joined frame IDs per source and call dense materializers only after this set is final. A dense acquisition failure becomes a recording failure at stage `dense_annotation_acquisition` and obeys the existing partial-publication/quarantine gate.

Add `indus_ad_annotations: Sequence[JoinedIndusAdAnnotations] = ()` to `ExportEvidence`; legacy builds pass the default empty tuple.

- [ ] **Step 6: Run end-to-end fixture tests and commit**

```bash
.venv/bin/pytest tests/test_source_runtime.py tests/test_indus_ad_join.py tests/test_indus_ad_pipeline.py -q
git add src/dataset_devkit/indus_ad_join.py src/dataset_devkit/source_runtime.py src/dataset_devkit/services.py src/dataset_devkit/extraction/models.py tests/test_source_runtime.py tests/test_indus_ad_join.py tests/test_indus_ad_pipeline.py
git commit -m "feat: join INDUS-AD annotations to selected scenes"
```

---

### Task 6: Export deterministic INDUS-AD extension tables

**Files:**
- Create: `src/dataset_devkit/indus_ad_export.py`
- Modify: `src/dataset_devkit/export.py:320-365,1150-1280`
- Modify: `tests/test_export_dataset.py`
- Create: `tests/test_indus_ad_export.py`

**Interfaces:**
- Produces: `camera_annotation_schema()`, `gnss_annotation_schema()`, `dense_asset_schema()`, `serialize_indus_ad_extensions()`, and four files under `mz_extensions/indus_ad/`.
- Consumes: `ExportEvidence.indus_ad_annotations` from Task 5.

- [ ] **Step 1: Write failing Arrow-schema and deterministic-byte tests**

Assert exact non-null join keys/provenance, nullable weak-label values, no forbidden location/date fields, and byte-identical serialization for the same sorted input. Assert camera extension rows reference real exported `sample_data_token` values and GNSS arrays reference selected scene/sample tokens.

```python
assert set(camera_annotation_schema().names) >= {
    "recording_id", "frame_id", "sample_data_token", "recording_offset_ns",
    "semantic_fractions", "weather_label", "weak_label_provenance",
}
assert "date" not in camera_annotation_schema().names
assert first_bytes == second_bytes
```

- [ ] **Step 2: Implement exact extension schemas**

Use fixed PyArrow schemas. Dictionary-encode no fields during writing; use plain strings for stable cross-version output. Encode semantic fractions as `list<struct<class_id:int16,fraction:float32>>`, relative times as `int64`, dimensions/indexes as signed integers, fractions/probabilities/statistics as `float32`, and token/digest/provenance values as non-null strings.

Write Parquet through `pa.BufferOutputStream` with:

```python
pq.write_table(
    table,
    sink,
    compression="zstd",
    use_dictionary=False,
    write_statistics=False,
    version="2.6",
)
```

Sort all rows before table construction and reject non-finite values before serialization.

- [ ] **Step 3: Write `contract.json` and audit counts**

Emit exact keys:

```json
{
  "schema_version": 1,
  "contract": "indus_ad_integrated_v1",
  "source_repo_id": "gagandeepreehal/indus-ad",
  "source_revision": "2d46a81376ded440506bd1646c0a5197589ba27b",
  "annotation_schema_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "selected_layers": [],
  "camera_rows": 0,
  "gnss_rows": 0,
  "dense_rows": 0,
  "join_audit": {},
  "weak_label_provenance": "machine_generated_weak_pseudo_label"
}
```

Populate values from evidence; do not write timestamps, local paths, credentials, dates, or mutable refs.

- [ ] **Step 4: Integrate extension writes without touching official annotations**

Add safe writer calls for:

```text
mz_extensions/indus_ad/contract.json
mz_extensions/indus_ad/camera_annotations.parquet
mz_extensions/indus_ad/gnss_annotations.parquet
mz_extensions/indus_ad/dense_assets.parquet
```

For schema `2.1`, set `source.json.source_type` to `indus_ad_hf`. Keep official `sample_annotation.json` and human `mz_extensions/annotations.json` behavior unchanged. All files must be included automatically by final content-manifest generation.

- [ ] **Step 5: Run export tests and commit**

```bash
.venv/bin/pytest tests/test_indus_ad_export.py tests/test_export_dataset.py -q
git add src/dataset_devkit/indus_ad_export.py src/dataset_devkit/export.py tests/test_indus_ad_export.py tests/test_export_dataset.py
git commit -m "feat: export INDUS-AD annotation extensions"
```

---

### Task 7: Validate extensions and expose typed SDK accessors

**Files:**
- Modify: `src/dataset_devkit/validation.py:1100-2200`
- Modify: `src/dataset_devkit/dataset.py`
- Modify: `tests/test_validation_pipeline.py`
- Create: `tests/test_indus_ad_dataset.py`

**Interfaces:**
- Produces: complete published-dataroot validation plus `Dataset.camera_annotation()`, `camera_annotations()`, `gnss_annotation()`, `gnss_annotations()`, and `dense_asset()`.
- Consumes: exact schemas and immutable models from Tasks 3 and 6.

- [ ] **Step 1: Write failing validation-mutation tests**

Start from one valid schema `2.1` fixture export, then independently mutate contract revision, schema digest, weak-label provenance, source recording, frame ID, sample-data token, GNSS window identity, scene/sample references, dense digest, Parquet schema/type, row count, and content-manifest entry. Each mutation must yield a stable `extension_reference`, `extension_schema`, `extension_value`, or `content_manifest` finding.

Also assert any `indus_ad_hf` output missing the extension namespace fails, while V1/V2 outputs remain valid without it.

- [ ] **Step 2: Extend source metadata validation**

Update `_validated_source_metadata()` and final validation to accept exactly:

```python
("indus_ad_hf", "privacy_transformed", "recording_local_enu_v1", 1.0)
```

Load `contract.json`, compare its revision/contract/schema digest with every integrated fingerprint, validate all Arrow schemas before reading rows, require unique identities, and verify camera/GNSS/dense references against official tables and selected recordings.

- [ ] **Step 3: Write failing SDK lookup tests**

```python
dataset = Dataset(exported_root)
camera = dataset.camera_annotation("frame-fixture")
assert camera.sample_data_token == sample_data_token
assert dataset.camera_annotations(recording_id=recording_id, camera_name="cam_front") == (camera,)
assert dataset.gnss_annotation(recording_id, 0).window_index == 0
assert dataset.dense_asset("frame-fixture", "semantic").kind == "semantic"
```

Cover unknown filters, missing identities, duplicates, malformed Parquet, source-type mismatch, and caller immutability.

- [ ] **Step 4: Implement typed read-only indexes**

Load integrated Parquet only when source type is `indus_ad_hf`. Convert each row back to immutable Task 3/5 models, construct indexes by `frame_id`, `(recording_id, window_index)`, and `(frame_id, kind)`, and retain deterministic file order for collection methods. Singular accessors require exactly one result and raise `DatasetFormatError` otherwise.

Do not return mutable dictionaries for these new APIs. Existing table/extension methods continue returning defensive deep copies.

- [ ] **Step 5: Run validation/SDK tests and commit**

```bash
.venv/bin/pytest tests/test_validation_pipeline.py tests/test_indus_ad_dataset.py -q
git add src/dataset_devkit/validation.py src/dataset_devkit/dataset.py tests/test_validation_pipeline.py tests/test_indus_ad_dataset.py
git commit -m "feat: validate and query INDUS-AD extensions"
```

---

### Task 8: Add inspection, documentation, and the immutable live compatibility gate

**Files:**
- Modify: `src/dataset_devkit/services.py`
- Modify: `src/dataset_devkit/cli.py`
- Create: `tools/check_indus_ad_contract.py`
- Create: `tests/test_indus_ad_live_check.py`
- Modify: `tests/test_public_api.py`
- Modify: `README.md`
- Modify: `docs/configuration.md`
- Modify: `docs/extraction.md`
- Modify: `docs/export.md`
- Modify: `docs/scenes.md`

**Interfaces:**
- Produces: annotation counts in `InspectionSummary`, public CLI JSON, exact copy-paste documentation, and a metadata-only live gate.

- [ ] **Step 1: Write failing inspect/CLI tests**

Extend `InspectionSummary` and CLI JSON with:

```python
camera_annotation_count: int
gnss_annotation_count: int
dense_asset_count: int
annotation_contract: str | None
annotation_source_revision: str | None
```

Legacy datasets must report zeros and `None`; integrated fixtures must report exact extension counts and revision.

- [ ] **Step 2: Implement inspection fields**

Read only validated SDK values after `validate_dataset()` succeeds. Do not summarize malformed extension files directly. Preserve every existing JSON key and append the five new keys so current consumers remain compatible.

- [ ] **Step 3: Implement the metadata-only live checker**

`tools/check_indus_ad_contract.py` must accept `--repo-id` and mandatory `--revision`, instantiate the real adapter with an isolated temporary cache, call only `load_plan()`, and print canonical JSON containing:

```text
repo_id, revision, contract, recording_count, video_count,
face_plate_blurred_count, missing_camera_slot_count,
camera_annotation_rows, annotation_schema_sha256,
downloaded_control_paths
```

It must fail if any `.mp4`, `.tar`, semantic payload, depth payload, or per-recording sensor table is requested. Its reference invocation is:

```bash
.venv/bin/python tools/check_indus_ad_contract.py \
  --repo-id gagandeepreehal/indus-ad \
  --revision 2d46a81376ded440506bd1646c0a5197589ba27b
```

Expected counts: 1,469 recordings, 9,927 videos, 9,927 `face_plate_blurred`, 356 missing tilted-camera slots, and 34,306 camera annotation rows.

- [ ] **Step 4: Document exact usage and caveats**

Document the schema `2.1` example, pinned revision requirement, repaired cross-control privacy agreement, six default/one optional cameras, exact join keys, metadata-only dense default, optional GNSS companion, SDK calls, and extension paths. State explicitly that model-generated annotations are weak pseudo-labels, all current non-null V3 videos are marked blurred, residual privacy risk remains, and raw MCAP is restricted/non-anonymized.

- [ ] **Step 5: Run docs/API tests and authenticated live gate**

```bash
.venv/bin/pytest tests/test_public_api.py tests/test_examples.py tests/test_indus_ad_live_check.py -q
.venv/bin/python tools/check_indus_ad_contract.py --repo-id gagandeepreehal/indus-ad --revision 2d46a81376ded440506bd1646c0a5197589ba27b
```

Expected: tests PASS; live checker reports the exact reference counts and downloads no production payload.

- [ ] **Step 6: Commit docs and live gate**

```bash
git add src/dataset_devkit/services.py src/dataset_devkit/cli.py tools/check_indus_ad_contract.py tests/test_indus_ad_live_check.py tests/test_public_api.py README.md docs/configuration.md docs/extraction.md docs/export.md docs/scenes.md
git commit -m "docs: document integrated INDUS-AD annotations"
```

---

### Task 9: Run regression, packaging, and fresh-install acceptance gates

**Files:**
- Modify only files needed to fix failures directly caused by Tasks 1-8.
- Verify: `pyproject.toml`, `schema/dataset_config.schema.json`, `dist/`, and all source/tests/docs.

**Interfaces:**
- Produces: final evidence that schemas `1.0`, `2.0`, and `2.1` coexist without regression.

- [ ] **Step 1: Run focused integrated and legacy suites**

```bash
.venv/bin/pytest \
  tests/test_indus_ad_contract.py \
  tests/test_indus_ad_annotations.py \
  tests/test_indus_ad_acquisition.py \
  tests/test_indus_ad_dense.py \
  tests/test_indus_ad_join.py \
  tests/test_indus_ad_pipeline.py \
  tests/test_indus_ad_export.py \
  tests/test_indus_ad_dataset.py \
  tests/test_source_runtime.py \
  tests/test_decoded_manifest.py \
  tests/test_decoded_extraction.py \
  tests/test_export_dataset.py \
  tests/test_validation_pipeline.py -q
```

Expected: PASS with no network access.

- [ ] **Step 2: Prove generated schema and legacy fixture stability**

```bash
.venv/bin/python -m dataset_devkit.schema /tmp/dataset_config.schema.json
cmp /tmp/dataset_config.schema.json schema/dataset_config.schema.json
.venv/bin/pytest tests/test_schema.py tests/test_config.py tests/test_examples.py tests/test_source_runtime.py -q
```

Expected: schema files are byte-identical; all legacy tests PASS. Add golden assertions that schema `1.0` and `2.0` source metadata, official tables, human annotation extension, and graph tokens are unchanged.

- [ ] **Step 3: Run the full static and test gate**

```bash
.venv/bin/ruff check .
.venv/bin/mypy --strict src/dataset_devkit
.venv/bin/pytest -q
git diff --check
```

Expected: every command exits `0`. Record the exact pass count; do not report older remembered counts.

- [ ] **Step 4: Build distributions and smoke-test a fresh environment**

```bash
INDUS_AD_DIST_DIR=$(mktemp -d /tmp/dataset-devkit-dist.XXXXXX)
.venv/bin/python -m build --outdir "$INDUS_AD_DIST_DIR"
python3 -m venv /tmp/dataset-devkit-indus-ad-smoke
/tmp/dataset-devkit-indus-ad-smoke/bin/pip install --no-deps "$INDUS_AD_DIST_DIR"/*.whl
/tmp/dataset-devkit-indus-ad-smoke/bin/dataset-devkit --help
/tmp/dataset-devkit-indus-ad-smoke/bin/python -c 'from dataset_devkit import Dataset; from dataset_devkit.config import GLOBAL_CONFIG_ADAPTER; print(Dataset, GLOBAL_CONFIG_ADAPTER)'
```

Expected: wheel and sdist build; CLI help and imports succeed. Use a fresh temporary path and remove it only after recording results.

- [ ] **Step 5: Re-run the authenticated immutable live gate**

```bash
.venv/bin/python tools/check_indus_ad_contract.py \
  --repo-id gagandeepreehal/indus-ad \
  --revision 2d46a81376ded440506bd1646c0a5197589ba27b
```

Expected: exact reference counts, current schema digest, no payload downloads, and exit `0`. This establishes metadata compatibility only; it does not re-run anonymization or download the corpus.

- [ ] **Step 6: Review the branch diff and commit acceptance fixes**

```bash
git status --short
git diff --stat 7453f02844ee53265dbb11a6548c91a70c3d1a7f..HEAD
git diff --check
git add -u
git diff --cached --name-only
git commit -m "test: verify integrated INDUS-AD backend"
```

If no acceptance fixes were required, do not create an empty commit. Confirm that no cache, downloaded dataset payload, credential, temporary artifact, or unrelated worktree file is staged.

---

## Final Review Checklist

- [ ] Every selected source revision and optional companion revision is immutable.
- [ ] All 9,927 current non-null Viewer video cells validate as `face_plate_blurred`; 356 tilted cells validate as missing.
- [ ] Metadata, Viewer, integrity rows, and privacy-manifest counts/digests agree exactly.
- [ ] Annotation rows join exactly to selected source/output identities.
- [ ] Dense archives remain untouched unless `selected_frames` is configured.
- [ ] Dense extraction passes adversarial archive tests and content verification.
- [ ] Trajectory and scene computation remain local-ENU-only.
- [ ] Weak pseudo-labels remain separate from human labels and official nuScenes object annotations.
- [ ] Schema `1.0` and `2.0` regression and byte-stability assertions pass.
- [ ] Every integrated extension is validated and included in the final content manifest.
- [ ] Full pytest, Ruff, strict mypy, schema equality, package build, fresh-install smoke, `git diff --check`, and immutable live gate pass with fresh evidence.
