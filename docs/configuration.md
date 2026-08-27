# Configuration

`dataset_config.json` is validated by the strict, versioned Pydantic `GlobalConfig` model. The
current `schema_version` is `"1.0"`. Unknown keys are rejected and relative local paths resolve
from the directory containing the configuration file.

Start with [`examples/dataset_config.json`](../examples/dataset_config.json). The generated
[`schema/dataset_config.schema.json`](../schema/dataset_config.schema.json) supports editor and CI
validation, while `load_config` remains authoritative for runtime and cross-field checks.

## Hugging Face source

The only supported input is a Hugging Face dataset repository:

```json
{
  "huggingface": {
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

Private repositories use the standard `huggingface_hub` login or environment. Do not add a token
to the configuration. Credential-shaped keys and values are rejected before model validation.

## Repository manifest

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

Acquisition and extraction-completion manifests use version 1. The source identity contains
`repo_id`, `revision`, `repo_path`, `sha256`, and `size`; artifact evidence contains the local
cache-relative path, SHA-256, and size. Source and artifact hashes and sizes must agree.

Cache leaves must be single-link regular files. Symbolic links and hard links fail closed.
Acquisitions for one source are serialized with a POSIX file lock, manifests are written
atomically and fsynced, and cache reuse re-verifies the complete local file. This backend is
POSIX-only.

Extraction results use a separate source-and-config-keyed immutable generation. Cache hits are
materialized into fresh build-owned working directories, and identity-safe cleanup is required
before publication.

## Other sections

- `paths`: isolated work, cache, and output directories; overlaps are rejected.
- `topics`: nonblank camera and GNSS topic names.
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
