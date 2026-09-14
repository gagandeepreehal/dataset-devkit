# Decoded V2 Privacy Backend Design

**Date:** 2026-09-14

**Status:** Approved

## Goal

Add a decoded Hugging Face backend to `dataset-devkit` for the privacy-rebuilt
Minus Zero V2 dataset. The backend consumes anonymized MP4 video and the new
date-free GNSS representation directly, while retaining the existing MCAP
backend as an explicitly restricted raw-source workflow.

The decoded backend must build trajectories and all motion-derived features
from precise recording-local ENU coordinates. Metre-rounded global coordinates
are public context only and must never be used to reconstruct the trajectory.

## Context

The existing backend acquires immutable MCAP recordings, decodes camera frames,
parses GNSS protobuf messages, and exports a camera-only nuScenes-compatible
dataset. Those MCAPs contain original, non-anonymized video. An MCAP-derived
output therefore has no privacy-safe publication claim.

The privacy-rebuilt V2 dataset has a different source contract:

- face- and number-plate-anonymized MP4 videos;
- date-free `recording_offset_ns` timing;
- precise `local_enu` computed before global quantization;
- Web Mercator east/north rounded to a 1 metre grid;
- latitude/longitude inverse-derived from the rounded Web Mercator point;
- preserved height, orientation, uncertainty, and GNSS quality fields;
- public content/config/model/schema manifests; and
- optional camera pseudo-label, semantic, and depth artifacts.

Video anonymization is still being finalized. Implementation may use synthetic
fixtures immediately, but documentation and examples must not name a production
revision until the complete release has a frozen immutable commit and final
verification evidence.

## Decisions

### Additive backend

The decoded backend is additive. It must not route through MCAP reconstruction
and must not change the MCAP parser, protobuf compatibility contract, or HEVC
extraction behavior.

The two source privacy classifications are fixed:

| Source type | Classification | Meaning |
| --- | --- | --- |
| `mcap_hf` | `restricted_raw` | Original video may contain identifiable faces and plates. No anonymization claim is made. |
| `decoded_hf` | `privacy_transformed` | Consumer-facing source transformed under the V2 privacy contract and validated against its public manifests. |

The CLI, inspection output, build audit, and exported provenance must surface
the classification. A decoded-source validation failure must never fall back to
MCAP or another raw source.

### Versioned configuration

Schema version `1.0` remains accepted for existing MCAP configurations.
Schema version `2.0` introduces a discriminated `source` object. A version 1
configuration is normalized internally to `source.type = "mcap_hf"` and retains
its build behavior, while inspection and provenance identify it as
`restricted_raw`.

A decoded V2 configuration uses JSON:

```json
{
  "schema_version": "2.0",
  "source": {
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
      "cam_rear_right"
    ],
    "modalities": ["video", "gnss", "calibration"]
  }
}
```

`revision` must be a full lowercase 40-character commit SHA. Repository scans,
branches, tags, implicit latest revisions, and source paths not advertised by
the catalog are rejected.

When `cameras` is omitted, the backend selects the six surrounding cameras:

- `cam_front`;
- `cam_front_left`;
- `cam_front_right`;
- `cam_rear`;
- `cam_rear_left`; and
- `cam_rear_right`.

`cam_front_tilted` remains an explicit optional seventh camera. Unknown camera
names are invalid. `video`, `gnss`, and `calibration` are required for dataset
generation. `camera_labels`, `semantic`, and `depth` are optional modalities
that are acquired only when requested.

## Public release contract

The decoded backend first fetches only these control files at the configured
immutable revision:

```text
data/recordings.parquet
data/manifests/output-files.parquet
data/manifests/processing-config.json
data/manifests/model-receipt.json
data/manifests/schema-audit.parquet
```

The backend validates the complete control plane before downloading selected
videos or table shards.

Required processing assertions are:

```json
{
  "global_horizontal_resolution_m": 1.0,
  "local_enu_precision": "retained"
}
```

The model receipt must contain nonempty lowercase SHA-256 values for
`model_lock_sha256` and `privacy_config_sha256`. The schema audit must assert:

```json
{
  "absolute_dates_removed": true,
  "exact_recording_offsets_retained": true,
  "published_horizontal_resolution_m": 1.0
}
```

`output-files.parquet` is authoritative for public artifact path, byte size,
and SHA-256. Each selected path must appear exactly once. Downloaded bytes are
accepted only after independent size and SHA-256 verification.

These checks verify the public release contract and immutable bytes. They do
not independently rerun the detector or expose the private frame-level receipt
ledger. Documentation must distinguish release-declared anonymization plus
manifest verification from an independent visual or detector audit.

## Components

### Source models and normalization

Introduce a source union with `McapHfSourceConfig` and
`DecodedHfSourceConfig`. Legacy version 1 configuration is normalized into the
MCAP variant before build orchestration. Downstream services consume the source
union rather than reading a backend-specific top-level configuration object.

The normalized source exposes:

- immutable repository identity;
- privacy classification;
- selected recording IDs, splits, cameras, and modalities;
- a deterministic source-configuration digest; and
- backend-specific acquisition and extraction services.

### Decoded control-plane reader

The control-plane reader parses the five fixed control files, rejects extra or
missing required columns, validates normalized relative POSIX paths, and builds
an immutable selection plan. Filtering occurs before payload download.

The plan contains only artifacts reachable from selected catalog rows and
requested modalities. Dense TAR artifacts are selected through their published
index; the backend must not download a complete dense archive merely to discover
whether a requested member exists.

### Selective Hugging Face acquisition

Reuse the existing immutable Hugging Face downloader and safe cache principles:

- commit-pinned download;
- descriptor-relative, no-follow cache operations;
- regular single-link file validation;
- size and SHA-256 verification;
- atomic promotion and fsync;
- source/config-keyed cache generations; and
- identity-safe cleanup.

The acquisition interface accepts the already validated artifact plan instead
of an MCAP manifest row. Large production runs execute on `mzcloud`; tests and
developer smoke checks use small generated fixtures and must not download the
production corpus to the Mac.

### Decoded recording adapter

The adapter reads anonymized MP4 video and the selected Parquet tables directly
and produces the same stable recording/extraction interfaces used by scene
construction, validity checks, filtering, splitting, and export.

Camera alignment uses the canonical join:

```text
(recording_id, camera_name, frame_index, recording_offset_ns)
```

Video decoding preserves camera identity, frame order, dimensions, and timing.
The adapter never reconstructs an MCAP and never searches for raw alternatives.

## Privacy-transformed GNSS contract

### Authoritative time domain

`recording_offset_ns` is the authoritative GNSS and camera timeline. It is a
signed integer nanosecond offset from the synchronized video start. It remains
continuous across midnight. Absolute epoch timestamps, dates, GPS week/day,
day-of-year, or reversible absolute instants are forbidden.

`time_of_day_ns`, when present, is display/context metadata only. It is not used
for ordering, synchronization, interpolation, scene boundaries, or identifiers.

### Required coordinates

Each decoded GNSS row must contain finite float64 values:

```text
local_enu.local_east_m
local_enu.local_north_m
local_enu.local_up_m
```

It must also contain:

```text
local_enu_origin_offset_ns = 0
published_horizontal_resolution_m = 1.0
```

The adapter treats published local ENU as authoritative. It must not regenerate
local ENU from rounded latitude/longitude or global Web Mercator coordinates.

### Trajectory and ego poses

All trajectory calculations use precise local ENU:

```text
translation_xyz_m = (
    local_east_m,
    local_north_m,
    local_up_m,
)
```

This translation drives:

- per-frame ego poses;
- distance and speed;
- acceleration derived from the relative timeline;
- curvature and turn direction;
- stationary/stopping classification;
- scene construction and gap checks;
- trajectory filters; and
- scenario selection.

Each recording has its own `(0, 0, 0)` tangent-frame origin. A trajectory is
precise and continuous within one recording but intentionally has no global
continuity across recording boundaries. Scenes may not cross a recording
boundary.

Exported provenance records:

```json
{
  "pose_frame": "recording_local_enu_v1",
  "pose_origin_offset_ns": 0,
  "global_horizontal_resolution_m": 1.0
}
```

### Interpolation

Frame-time translation is linearly interpolated independently in local east,
north, and up using bracketing `recording_offset_ns` rows. Orientation retains
the existing shortest-path quaternion SLERP and source validity rules.

Height, position uncertainty, orientation uncertainty, HDOP, fix type, RTK
state, satellite fields, and other supported quality diagnostics retain their
existing conservative interpolation or endpoint rules.

A frame timestamp outside the GNSS range is invalid; the adapter must not clamp,
extrapolate, or invent an origin.

### Rounded global context

Published global context remains separate from trajectory state. GNSS source
rows may expose:

- Web Mercator east/north already rounded to the 1 metre grid;
- inverse-derived public `latitude_deg` and `longitude_deg`;
- full-precision source height and vertical values; and
- `published_horizontal_resolution_m = 1.0`.

The adapter verifies that Web Mercator east/north are integer-metre values under
the release's half-away-from-zero rule and that public latitude/longitude are
consistent with their inverse projection within the schema tolerance.

If a frame-time global value is exported, east/north are interpolated only from
the sanitized endpoints, rounded again to the 1 metre grid using half-away-from-
zero ties, and then inverse-transformed to latitude/longitude. Global context is
never fed into trajectory, feature, filtering, or splitting code.

### GNSS export

nuScenes-compatible `ego_pose.translation` uses recording-local ENU. The
`mz_extensions` provenance declares the coordinate frame so consumers do not
mistake different recording origins for one global map frame.

The GNSS extension retains the sanitized public coordinate and quality fields.
It must not contain the original recording ID with date prefix, a source file
name, a source commit, a source video hash, or an absolute timestamp.

## Privacy behavior for MCAP

The MCAP backend remains useful for authorized internal work, but its source and
output provenance are always marked `restricted_raw`. Documentation must state
that MCAP video is not anonymized and that devkit decoding does not add face or
plate anonymization.

The package does not upload generated datasets, so no synthetic "public publish"
switch is added. Instead, `inspect` and the build result expose the privacy
classification prominently, and generated audit/provenance files retain it.

## Error handling

The decoded backend fails closed for:

- mutable or malformed revisions;
- a missing, duplicated, unsafe, or unadvertised artifact path;
- byte-size or SHA-256 mismatch;
- absent model/config digests;
- an unsuccessful date-removal assertion;
- missing exact recording offsets;
- a global horizontal resolution other than exactly `1.0`;
- missing, non-finite, or wrong-typed local ENU;
- a nonzero or missing `local_enu_origin_offset_ns`;
- inconsistent rounded Web Mercator and latitude/longitude values;
- nonmonotonic or duplicate GNSS offsets where uniqueness is required;
- a frame outside the GNSS interpolation range;
- an unexpected camera or missing selected camera video;
- video/table frame-count or join mismatch;
- an absolute-date finding in selected public metadata; or
- any attempted fallback to an MCAP/raw source.

Authentication and retryable Hub transport failures use bounded retries and
preserve verified cache state. Credentials and absolute source timestamps are
redacted from errors. A failed decoded recording follows the existing
quarantine and partial-publication policy; the privacy contract itself is never
downgraded to a warning.

## Validation and inspection

`dataset-devkit validate` additionally verifies:

- privacy classification and source backend identity;
- coordinate-frame provenance;
- local-ENU-driven ego-pose translation;
- per-recording origin separation;
- exact relative timestamp domain;
- one-metre global-context invariants; and
- absence of forbidden absolute-time and source-linkage fields.

`dataset-devkit inspect` reports source type, privacy classification, immutable
revision, selected cameras/modalities, pose frame, global resolution, and
whether all decoded control-plane checks passed. It must not report private
receipt details or source identifiers removed by the privacy transform.

## Testing strategy

Implementation follows test-driven development.

### Configuration and source tests

- Legacy schema `1.0` normalizes to `mcap_hf` and `restricted_raw`.
- Schema `2.0` accepts the fixed decoded source and rejects mutable revisions,
  unknown cameras, unsafe paths, missing required modalities, and embedded
  credential-shaped values.
- The generated JSON Schema is byte-identical to the runtime model.

### Control-plane and acquisition tests

- Synthetic Parquet/JSON fixtures cover the exact five control files.
- Missing assertions, malformed digests, duplicate paths, traversal, size/hash
  mismatch, and catalog/manifest disagreement fail before payload acquisition.
- Selection downloads only requested recording/camera/modality artifacts.
- Cache symlink, hard-link, inode-swap, partial-write, and cleanup-debt tests
  retain the existing security envelope.

### GNSS tests

- Local ENU is used for pose translation even when rounded global motion differs.
- Exact zero origin and interpolated frame-time local ENU are verified.
- Two recordings with identical local trajectories but distant rounded global
  coordinates produce identical trajectory features and remain separate frames.
- Trajectories never cross recording boundaries.
- Midnight-spanning offsets remain monotonic without an absolute date.
- Missing brackets, non-finite values, duplicate offsets, missing local ENU,
  and nonzero origin offsets fail.
- Positive and negative half-metre global ties use half-away-from-zero rounding.
- Inverse WGS84 consistency, precise height, orientation, uncertainties, HDOP,
  fix type, RTK state, and satellite fields are preserved.
- Feature, filter, and scenario tests prove that rounded global coordinates are
  not read by trajectory calculations.

### Video and join tests

- Anonymized MP4 fixtures preserve dimensions, frame count, frame order, and
  relative timing.
- Six cameras are selected by default; `cam_front_tilted` is opt-in.
- Missing videos, frame-count mismatches, orphan table rows, and invalid joins
  fail the recording.
- Optional camera labels, semantic masks, and depth maps join only through the
  canonical frame key.

### End-to-end and regression tests

- One fixture recording builds and validates from decoded MP4/Parquet without
  constructing an MCAP.
- Exported ego poses and motion tags are based on local ENU.
- Inspection and provenance carry `privacy_transformed` and
  `recording_local_enu_v1`.
- A decoded privacy failure cannot select the MCAP backend.
- Existing MCAP tests remain green and outputs are marked `restricted_raw`.
- Ruff, strict mypy, generated-schema equality, full pytest, wheel build, and a
  fresh-environment CLI/import smoke test form the final local quality gate.

## Documentation

Update the README and focused guides to:

- make decoded V2 the recommended public-consumer workflow;
- identify MCAP as restricted raw source material with no anonymization;
- explain immutable control-plane verification and its limits;
- document six-default/seven-optional cameras;
- explain per-recording local ENU trajectory semantics;
- distinguish local trajectory coordinates from rounded global context;
- show selective modality configuration; and
- state that production payload processing belongs on `mzcloud`.

Until anonymization is complete, examples use generated fixture identities and
state that no production privacy revision is bundled. After the release passes
final verification, documentation may add its immutable commit in a separately
reviewed change; it must never use `main` as a reproducibility revision.

## Non-goals

- Anonymizing MCAP video inside `dataset-devkit`.
- Describing MCAP-derived output as privacy-safe.
- Training or fine-tuning a face/plate detector.
- Recomputing precise local ENU from rounded public coordinates.
- Providing globally continuous trajectories across recordings.
- Publishing original source dates, filenames, hashes, or private receipts.
- Downloading the production corpus to the Mac.
- Replacing the nuScenes-compatible export or fabricating lidar, radar, boxes,
  maps, or benchmark annotations.

## Acceptance criteria

The update is complete when:

1. Existing MCAP builds remain supported and are visibly classified
   `restricted_raw`.
2. A commit-pinned decoded V2 fixture is selected and verified through the five
   public control files before payload download.
3. Selected anonymized videos and privacy-transformed GNSS build without MCAP
   reconstruction.
4. All ego poses, trajectories, and motion-derived features use precise local
   ENU and exact relative nanosecond timing.
5. Rounded global context remains 1 metre resolution and is never used for
   trajectory computation.
6. Privacy-contract violations fail closed without a raw fallback.
7. Six cameras are selected by default and the tilted camera remains optional.
8. Output provenance and inspection clearly declare source privacy and pose
   frame semantics.
9. Documentation makes no completion claim about the production anonymized
   release before its immutable verified revision exists.
10. Focused and full tests, Ruff, strict mypy, schema regeneration, wheel
    packaging, and fresh-install smoke checks pass.
