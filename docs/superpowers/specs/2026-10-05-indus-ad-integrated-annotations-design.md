# INDUS-AD Integrated Annotations Design

**Status:** Approved design

**Date:** 2026-10-05

**Live contract retarget approved:** 2026-10-15

**Target repository:** `gagandeepreehal/dataset-devkit`

**Reference dataset:** `gagandeepreehal/indus-ad`
at immutable revision `2d46a81376ded440506bd1646c0a5197589ba27b`

## Purpose

Add a first-class, fail-closed dataset-devkit backend for the current integrated
INDUS-AD repository layout. The backend must consume privacy-transformed decoded
sensor data and its machine-generated annotation layers without reconstructing or
falling back to raw MCAP. It must expose typed camera, semantic, depth, weather,
and optional GNSS weak-label data through the Python SDK and deterministic export
extensions.

The implementation adds schema `2.1` and leaves the existing schema `1.0`
`mcap_hf` and schema `2.0` `decoded_hf` contracts unchanged. Original MCAP is
always `restricted_raw`; only the decoded integrated source is
`privacy_transformed`.

## Verified reference state

The reference dataset revision was re-resolved after the repository rename on
2026-10-15. Its public metadata
declares four Viewer configurations: `videos`, `camera_annotations`, `gnss`, and
`semantic_depth`. Its immutable tree contains:

- 1,607 files below `data/annotations/camera/`;
- one 34,306-row camera-annotation Viewer Parquet file;
- 16 GNSS Viewer Parquet files;
- 68 semantic/depth Viewer Parquet files;
- 1,469 privacy-transformed recording tables for each core sensor-table family;
- 8,814 camera videos across the six default cameras plus 1,113 optional
  `cam_front_tilted` videos, for 9,927 videos total.

Fresh authenticated reconciliation at the reference revision confirmed that the
current Viewer marks all 9,927 non-null video cells `face_plate_blurred`; the 356
remaining tilted-camera cells are genuinely missing. The immutable history shows
the final receipt-bound V3 payload promotions before commit
`d3cc48b81b5d26f9b295e76fb99a23dcffd8ad3b` updated the Viewer statuses.

Commit `0229c5d2837dbc2d701e0df3a7485ba13e8365a9` repaired the release control
plane after the repository became `gagandeepreehal/indus-ad`. At the pinned
reference revision, `data/metadata.jsonl`, the videos Viewer, and
`data/manifests/video-privacy-release.json` now agree on 9,927 blurred videos and
356 missing tilted-camera slots. The manifest binds exact byte counts and
SHA-256 values for metadata, video integrity, and the Viewer. The adapter must
validate this agreement and every advertised digest; no file is permitted to
override a disagreement in another control.

The reference camera annotation Viewer schema contains exact recording/frame
identity, relative timing, semantic fractions, derived semantic category
fractions, depth percentiles, weather probabilities and labels, and image-quality
statistics. These values are machine-generated weak pseudo-labels or model
estimates, not human ground truth.

The integrated tree does not provide the five control files required by schema
`2.0`: `data/recordings.parquet`, `data/manifests/output-files.parquet`,
`data/manifests/processing-config.json`, `data/manifests/model-receipt.json`, and
`data/manifests/schema-audit.parquet`. Schema `2.1` therefore validates the actual
integrated tree and Parquet contracts directly instead of fabricating the old
control plane.

## Goals

1. Load the current integrated anonymized dataset at an immutable revision.
2. Preserve exact recording/frame/window joins without filename inference.
3. Expose camera scalar annotations, semantic fractions, depth summaries,
   weather estimates, and optional GNSS scenario annotations as typed records.
4. Keep large dense semantic/depth assets metadata-only by default and acquire
   only explicitly selected frame members.
5. Export joined annotations as INDUS-AD extensions without fabricating nuScenes
   object annotations.
6. Preserve privacy-safe local-ENU trajectory processing and one-metre rounded
   global context.
7. Fail closed on contract, privacy, identity, schema, or digest drift.
8. Preserve schemas `1.0` and `2.0` without output changes.

## Non-goals

- Anonymizing imagery inside dataset-devkit.
- Treating MCAP decoding as anonymization.
- Publishing or consuming raw MCAP through schema `2.1`.
- Converting semantic pixels into object counts, movement claims, 3D boxes, or
  nuScenes `sample_annotation` records.
- Calling model-generated semantic, depth, weather, or GNSS outputs human ground
  truth.
- Downloading the production corpus or dense asset archives to the Mac during
  development or verification.
- Mutating the Hugging Face dataset or requiring a newly generated compatibility
  manifest in that repository.
- Joining through source filenames, dates, raw global coordinates, or fuzzy
  timestamp matching.

## Source configuration

Schema `2.1` adds one source discriminator:

```json
{
  "schema_version": "2.1",
  "source": {
    "type": "indus_ad_hf",
    "contract": "indus_ad_integrated_v1",
    "repo_id": "gagandeepreehal/indus-ad",
    "revision": "2d46a81376ded440506bd1646c0a5197589ba27b",
    "recording_ids": [],
    "splits": ["train"],
    "cameras": [
      "cam_front",
      "cam_front_left",
      "cam_front_right",
      "cam_rear",
      "cam_rear_left",
      "cam_rear_right"
    ],
    "annotation_layers": [
      "camera_scalar",
      "semantic_fractions",
      "depth_summary",
      "weather"
    ],
    "dense_assets": {
      "semantic": "none",
      "depth": "none"
    },
    "gnss_annotation_source": null
  }
}
```

`revision` must be an exact lowercase 40-character commit SHA. Mutable refs are
invalid. `contract` is a closed literal and is part of all source fingerprints.
At least one of `recording_ids` or `splits` must select data. The six untilted
cameras remain the defaults and `cam_front_tilted` remains opt-in.

Dense policies are exactly `none` or `selected_frames`. `none` acquires only
indexes and scalar metadata. `selected_frames` permits acquisition only after
scene selection and only for dense members whose `frame_id` is present in the
final selected sample-data graph.

The main integrated repository contains privacy-transformed GNSS sensor tables.
GNSS scenario annotations remain a distinct optional weak-label source. When
configured, `gnss_annotation_source` contains its own repository ID, immutable
revision, contract literal, and expected source-dataset revision. A missing
optional companion yields an explicit `unavailable` state; it never yields empty
or invented scenario labels.

## Architecture

### Contract adapter

`IndusAdHfSourceConfig` selects an `IndusAdContractAdapter`. The adapter validates
the immutable repository tree and reads only lightweight indexes needed to build
an `IndusAdSelectionPlan`. It does not share schema detection with `decoded_hf`:
the explicit source discriminator prevents remote tree shape from silently
changing runtime behavior.

The adapter verifies required paths, allowed path families, exact Parquet column
names and Arrow types, and the dataset privacy contract. It accepts future
revisions only when they satisfy `indus_ad_integrated_v1` exactly. Unknown
required columns, removed columns, incompatible types, forbidden metadata, or
unrecognized path families fail before payload acquisition.

### Normalized recording catalog

The adapter derives a normalized in-memory catalog from the pinned tree,
`data/metadata.jsonl`, Viewer indexes, recording tables, and integrity/privacy
manifests that actually exist at the revision. Each recording entry contains:

- opaque `recording_id` and split;
- available cameras and immutable video identities;
- recording-relative duration and offsets;
- privacy GNSS, calibration, and camera-frame table identities;
- camera annotation availability and index identities;
- dense semantic/depth member references where available;
- optional GNSS annotation availability;
- a canonical catalog digest and selected-artifact digest.

The catalog never restores dates or private source filenames. Hugging Face commit,
path, size, and service metadata bind discovery. Every acquired file is hashed
locally; its SHA-256 is recorded in the build receipt and later export provenance.

### Annotation parsing

New immutable models separate four concerns:

- `CameraAnnotation`: frame identity, selection metadata, dimensions, semantic
  fractions, derived category fractions, depth statistics, weather estimates,
  and image-quality statistics;
- `GnssScenarioAnnotation`: recording/window identity, weak scenario labels,
  anomaly-candidate fields, and source provenance;
- `DenseAssetRef`: frame identity, semantic/depth kind, archive/member identity,
  expected decoded media properties, and verified local digest when materialized;
- `AnnotationAvailability`: `available`, `missing`, `not_selected`, or
  `unavailable_source`, with a stable reason code.

Semantic, depth, weather, and GNSS-derived values remain visibly labeled as weak
pseudo-labels. Parsers retain `unknown`, `mixed`, `uncertain`, and unmatched
states rather than normalizing them away.

### Join service

Camera annotations join by exact `(recording_id, frame_id)`. The service rejects
duplicate frame IDs, foreign recordings, inconsistent camera names, inconsistent
dimensions, and any annotation that attempts to bind to more than one source
frame.

GNSS scenario annotations join by exact `(recording_id, window_index)`. Their
window bounds use recording-relative offsets. Export references are derived from
overlap with selected recording-local samples; global coordinates are never used
for joins or scene construction.

There is no basename matching, filename parsing, date reconstruction, fuzzy
recording matching, or fallback to absolute timestamps.

### Dense asset service

Dense semantic and depth archives remain remote unless `selected_frames` is
configured. After the final scene graph exists, the dense service computes the
exact requested `(recording_id, frame_id, kind)` set, resolves members through
the validated dense index, and materializes only those members into the verified
cache.

Archive reads reject absolute or traversal paths, duplicate members, symlinks,
hard links, non-regular files, unexpected members, oversized compressed or
decoded content, and decompression-budget violations. Extracted bytes are hashed
and decoded/inspected according to their declared media type before use.

## Data flow

1. Load and strictly validate schema `2.1` configuration.
2. Resolve the already supplied immutable revision; never resolve `main` at
   runtime.
3. List the pinned repository tree once and validate the integrated contract.
4. Acquire lightweight Viewer/index/manifest inputs and construct the normalized
   recording catalog.
5. Apply recording, split, camera, and annotation-layer selection before payload
   acquisition.
6. Acquire selected video, calibration, privacy GNSS, camera-frame, and scalar
   annotation files through the existing verified cache.
7. Parse each recording independently; validate row ownership before
   interpolation or joining.
8. Build poses, trajectories, validity, scenes, distance, speed, curvature, tags,
   and filtering from precise per-recording `local_enu` only.
9. Join selected camera and GNSS annotation records to the final graph.
10. If enabled, materialize dense assets for final selected frames only.
11. Export official nuScenes tables plus deterministic INDUS-AD extensions.
12. Validate the complete output, write the content manifest last, and publish
    atomically through the existing pipeline.

## Export contract

Existing `mz_extensions/annotations.json` remains reserved for user-supplied
human scene-label input. Integrated weak pseudo-labels are written under:

```text
mz_extensions/indus_ad/contract.json
mz_extensions/indus_ad/camera_annotations.parquet
mz_extensions/indus_ad/gnss_annotations.parquet
mz_extensions/indus_ad/dense_assets.parquet
```

`contract.json` records schema version, source/companion revisions, contract
names, selected layers, exact extension schemas, accepted/excluded/rejected
counts, and canonical schema/catalog digests. It contains no local paths,
credentials, wall-clock timestamps, or original dates.

Camera export rows contain the exact source `recording_id` and `frame_id`, plus
the joined official `sample_data_token`. Only annotations for exported sample
data are written. GNSS scenario rows retain `(recording_id, window_index)` and
contain explicit matching scene/sample token arrays. Rows with no selected
output reference are excluded and counted in pipeline audit reason codes.

Official nuScenes `sample_annotation.json` remains empty. The new extension files
are covered by `content_manifest.json`, read-only publication permissions, and
the same no-follow/identity validation as every other exported file.

## Python SDK

The read-only `Dataset` SDK validates `contract.json`, Parquet schemas, unique
identities, export-token references, and extension content-manifest entries at
load time. It exposes immutable records through:

```python
dataset.camera_annotation(frame_id)
dataset.camera_annotations(recording_id=None, camera_name=None)
dataset.gnss_annotation(recording_id, window_index)
dataset.gnss_annotations(recording_id=None)
dataset.dense_asset(frame_id, kind)
```

Singular lookups require exactly one result and raise `DatasetFormatError` for a
missing or ambiguous identity. Collection methods preserve deterministic export
order and return immutable typed records. Caller mutation cannot alter cached
indexes or subsequent lookups.

## Privacy and validation rules

All selected annotation rows must satisfy these rules before scene construction:

- `recording_id` equals the catalog recording being processed;
- `recording_offset_ns` and all relative bounds are within recording duration;
- `frame_id` and `(recording_id, window_index)` identities are unique in their
  respective layers;
- fractions, probabilities, confidence values, dimensions, luminance metrics,
  and depth statistics are finite and within their declared domains;
- sparse semantic fractions do not exceed `1.0` beyond the documented floating
  tolerance;
- weather probability sums satisfy the documented tolerance when probabilities
  are present;
- dimensions agree with selected camera-frame evidence;
- no field or nested metadata contains absolute dates, GPS week/day, epochs, raw
  latitude/longitude, geographic cells, road names, geometry, endpoints, or OSM
  feature identifiers;
- companion source-dataset revision and recording ownership match exactly.

Annotation absence is represented through `AnnotationAvailability`; it is not
converted to zeros, empty labels, or ground-truth negatives. Any privacy,
identity, schema, path, or digest violation rejects that recording with a stable
sanitized reason. There is no raw-MCAP fallback.

## Failure handling and resumability

Tree and schema failures abort before payload acquisition. Per-recording parsing
or join failures preserve sanitized audit evidence and cannot enter scene
construction. Verified cached files remain reusable under their immutable
revision/path/content identity. Partial acquisition may resume, but final export
requires every selected layer and dense member to be complete and verified.

Reports provide exact accepted, excluded, missing, and rejected counts per layer
and recording. They never expose access tokens, cache paths, private cloud paths,
or rejected raw metadata values.

## Testing strategy

Tests use generated tiny fixtures that mirror the reference Arrow schemas and
repository layout without production payloads.

Required coverage includes:

1. Schema `2.1` configuration, JSON Schema generation, immutable revisions,
   closed literals, selectors, and dense policies.
2. Integrated tree validation, allowed/required path families, schema signatures,
   catalog determinism, and contract drift rejection.
3. Complete camera/GNSS annotation parsing, nullable fields, numeric bounds,
   availability states, and weak-label provenance.
4. Exact frame/window joins, cross-recording rejection, duplicate identity
   rejection, and deterministic ordering.
5. Metadata-only acquisition and post-selection dense-member acquisition.
6. Adversarial archive traversal, links, duplicate members, type confusion,
   oversize content, and decompression limits.
7. Deterministic Parquet/JSON extension export, official-token references,
   content-manifest coverage, and empty official object annotations.
8. SDK singular/collection accessors, filtering, missing/ambiguous errors,
   immutability, and malformed-extension rejection.
9. Full regression coverage proving schema `1.0` and schema `2.0` configuration,
   builds, validation, inspection, and exported bytes remain unchanged.

The final quality gate is the full pytest suite, Ruff, strict mypy, generated
schema equality, `git diff --check`, wheel/sdist build, and fresh-environment
CLI/import smoke. A separate authenticated metadata-only check against immutable
revision `2d46a81376ded440506bd1646c0a5197589ba27b` must validate the remote contract,
confirm 9,927 `face_plate_blurred` Viewer video cells plus 356 genuine missing
tilted-camera cells, and build a selection plan without downloading video or
dense payloads.

## Documentation

Update README and focused guides to:

- recommend schema `2.1` for the current integrated INDUS-AD release;
- preserve the explicit `restricted_raw` versus `privacy_transformed` boundary;
- explain exact recording/frame/window joins;
- distinguish sensor GNSS from optional GNSS scenario weak labels;
- document metadata-only defaults and opt-in dense acquisition;
- identify all model-generated outputs as weak pseudo-labels;
- show immutable revision configuration and extension SDK examples;
- state that production payload work belongs on `mzcloud`.

## Acceptance criteria

The update is complete only when:

1. Schema `2.1` validates the immutable reference tree and builds a selection plan
   without the obsolete five-file schema `2.0` control plane.
2. A tiny fixture builds end to end with camera scalar, semantic fraction, depth
   summary, and weather annotation joins.
3. Optional GNSS scenario rows are typed, source-bound, and joined only by
   `(recording_id, window_index)`.
4. Precise local ENU remains the sole trajectory/pose/scene computation source.
5. Dense assets are absent by default and only selected frame members are
   acquired when explicitly enabled.
6. Exported camera rows reference exact official `sample_data_token` values and
   GNSS rows reference exact selected scene/sample tokens.
7. Official nuScenes object annotations remain empty and no weak label is called
   human ground truth.
8. All extension files are independently validated and covered by the final
   content manifest.
9. Privacy, cross-recording, schema, path, archive, and digest violations fail
   closed with no MCAP fallback.
10. Schemas `1.0` and `2.0` retain their existing behavior and outputs.
11. The complete quality gate and authenticated metadata-only immutable-revision
   compatibility check pass with recorded evidence.
