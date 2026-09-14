# Configuration

`dataset_config.json` is validated by the strict, versioned Pydantic `GlobalConfig` model. Schema
`"1.0"` is the legacy MCAP form and schema `"2.0"` adds an explicit discriminated source. Unknown
keys are rejected and relative local paths resolve from the directory containing the configuration
file.

Public consumers should start with
[`examples/decoded_v2_config.json`](../examples/decoded_v2_config.json); the all-zero revision is a
fixture placeholder and must be replaced with a verified release commit. The legacy MCAP example
is [`examples/dataset_config.json`](../examples/dataset_config.json). The generated
[`schema/dataset_config.schema.json`](../schema/dataset_config.schema.json) supports editor and CI
validation, while `load_config` remains authoritative for runtime and cross-field checks.

## Hugging Face sources

Both inputs are Hugging Face dataset repositories pinned to immutable commits:

```json
{
  "schema_version": "2.0",
  "source": {
    "type": "decoded_hf",
    "repo_id": "owner/privacy-transformed-dataset",
    "revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "recordings_path": "data/recordings.parquet",
    "recording_ids": [],
    "splits": ["train"],
    "cameras": [
      "cam_front", "cam_front_left", "cam_front_right",
      "cam_rear", "cam_rear_left", "cam_rear_right"
    ],
    "modalities": ["video", "gnss", "calibration"]
  }
}
```

`decoded_hf` is the recommended public-consumer backend and is fixed to
`privacy_classification: "privacy_transformed"`. `recording_ids` and `splits` are intersected; at
least one must select data. The six untilted cameras are the default. The seventh supported camera,
`cam_front_tilted`, is opt-in. Video, GNSS, and calibration are required modalities; camera labels,
semantic masks, and depth can be selected for acquisition without changing the core scene builder.

The equivalent explicit MCAP source is:

```json
{
  "schema_version": "2.0",
  "source": {
    "type": "mcap_hf",
    "repo_id": "owner/dataset",
    "revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "manifest_path": "manifest.jsonl"
  }
}
```

- `repo_id` is exactly `owner/name`.
- `revision` is a full lowercase 40-character commit SHA. Branches and tags are rejected.
- `manifest_path` is a normalized relative POSIX path without traversal, backslashes, query
  strings, fragments, or percent encoding.

`mcap_hf` and schema `"1.0"` MCAP configurations are always classified `restricted_raw`. Gating or
private repository access does not make their camera data anonymous. Private repositories use the
standard `huggingface_hub` login or environment. Do not add a token to the configuration.
Credential-shaped keys and values are rejected before model validation.

## Decoded V2 privacy control plane

Before selecting payloads, the decoded backend downloads and verifies exactly five public control
files at the pinned commit:

- `data/recordings.parquet`;
- `data/manifests/output-files.parquet`;
- `data/manifests/processing-config.json`;
- `data/manifests/model-receipt.json`; and
- `data/manifests/schema-audit.parquet`.

The catalog must declare the selected recording, split, cameras, modality artifacts, sizes, and
SHA-256 values. The public manifests must agree, declare one-metre global horizontal resolution,
retain precise local ENU, and record removal of absolute dates. Only catalog-selected artifacts are
downloaded and independently hashed. These are structural privacy-release checks; the devkit does
not rerun face/plate detection or independently prove the anonymizer's model behavior.

A decoded privacy, catalog, schema, size, or hash failure stops the build. There is no raw MCAP
fallback. Production decoded payloads, work directories, and caches belong on `mzcloud`; local
execution is intended for configuration checks and tiny fixtures.

## MCAP compatibility

Strict source-schema validation is the default. A release containing one of the observed legacy
schema variants can enable only the required compatibility paths:

```json
{
  "mcap_compatibility": {
    "compatible_gnss_numeric_types": true,
    "allow_gnss_rec_timestamp_log_time_fallback": true,
    "allow_native_camera_calibration_resolution": true,
    "allow_camera_timestamp_batch_fallback": true
  }
}
```

- `compatible_gnss_numeric_types` accepts required GNSS numeric fields encoded as protobuf
  `float` or `double` instead of requiring `double` exactly.
- `allow_gnss_rec_timestamp_log_time_fallback` uses the MCAP record `log_time` only when a GNSS
  message omits `rec_timestamp`, and records the substitution in `raw_identifiers`.
- `allow_native_camera_calibration_resolution` accepts positive, same-aspect-ratio intrinsic
  dimensions and scales the intrinsic matrix into the camera batch coordinate system.
- `allow_camera_timestamp_batch_fallback` uses the camera batch `timestamp` only when the
  descriptor has no per-camera `camera_timestamp` field, and records the timestamp source on each
  raw frame.

These values are part of the extraction cache identity. Changing any of them forces extraction
under a new cache key.

## MCAP repository manifest

The repository manifest is UTF-8 JSONL. Blank and comment-only lines are ignored. Every recording
row must provide these fields:

```json
{"repo_path":"data/2025-04-11/run.mcap","source_size":12,"sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}
```

`repo_path` must be an exact normalized `data/...mcap` path, `source_size` must be a positive JSON
integer, and `sha256` must be lowercase hexadecimal. Duplicate paths, malformed JSON, unsafe paths,
and invalid sizes or hashes fail the build. Extra provenance fields are ignored. Manifest order is
the build order; version 1 does not provide source selection or subsetting.

## Cache and provenance

The manifest and every MCAP are downloaded with the configured repository ID, dataset repository
type, and exact commit. A downloaded MCAP is accepted only after its local size and independently
computed SHA-256 match the repository manifest. The same verified inode is atomically promoted
under `paths.cache_dir/huggingface/{source_digest}/`.

Acquisition and extraction-completion manifests use versioned, source-discriminated fingerprints.
MCAP identity contains `repo_id`, `revision`, `repo_path`, `sha256`, and `size`. Decoded identity
contains `repo_id`, `revision`, opaque `recording_id`, a selected-artifact digest, and fixed
`privacy_transformed` classification. Artifact evidence contains cache-relative paths, hashes, and
sizes; source and artifact evidence must agree.

Cache leaves must be single-link regular files. Symbolic links and hard links fail closed.
Acquisitions for one source are serialized with a POSIX file lock, manifests are written
atomically and fsynced, and cache reuse re-verifies the complete local file. This backend is
POSIX-only.

Extraction results use a separate source-and-config-keyed immutable generation. Cache hits are
materialized into fresh build-owned working directories, and identity-safe cleanup is required
before publication.

## Other sections

- `paths`: isolated work, cache, and output directories; overlaps are rejected.
- `topics`: schema `"1.0"` MCAP-only camera and GNSS topic names.
- `mcap_compatibility`: opt-in handling for the observed legacy source-schema variants.
- `downsampling`: positive target FPS and nonnegative timestamp tolerance.
- `image`: the version 1 JPEG quality contract.
- `gnss`, `frame_validity`, and `sanity_checks`: extraction validity and sanity policy.
- `scenes` and `annotations`: automatic, annotation-only, or hybrid scene construction.
- `tags`, `filters`, and `scenarios`: feature derivation, filtering, and exact-quota selection.
- `split`: deterministic scene-level train/test assignment.
- `execution`: worker count and partial-export policy.
- `quarantine`: mandatory isolated rejection output.
- `publication`: fixed version and no-overwrite publication contract.

Detailed scene, validity, selection, extraction, and publication behavior is documented in the
other files in this directory.

## Validation and schema regeneration

```bash
PYTHONPATH=src python -c 'from pathlib import Path; from dataset_devkit.config import load_config; load_config(Path("dataset_config.json"))'
PYTHONPATH=src python -m dataset_devkit.schema
pytest tests/test_schema.py
```
