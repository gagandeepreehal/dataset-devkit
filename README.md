# dataset-devkit

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
![Status: Alpha](https://img.shields.io/badge/status-alpha-orange)
![Platform: POSIX](https://img.shields.io/badge/platform-POSIX-lightgrey)

Build deterministic, nuScenes-compatible robotics datasets directly from MCAP recordings stored
in a Hugging Face dataset repository.

`dataset-devkit` pins every build to an immutable repository commit, verifies every recording by
size and SHA-256, extracts synchronized camera and GNSS data, constructs scenes, and publishes a
validated dataset through an atomic, no-overwrite workflow.

> [!IMPORTANT]
> This project is in alpha. Its configuration and output contracts are strict by design, but the
> public API may still evolve before a stable release.

## Highlights

- **Reproducible inputs** — one Hugging Face dataset repository, one full commit SHA, and one
  ordered manifest define the complete source corpus.
- **Verified acquisition** — downloaded MCAPs are checked against their declared byte size and
  SHA-256 before entering the trusted local cache.
- **Native extraction** — protobuf camera and GNSS messages are read directly from MCAP, with
  persistent HEVC decoding, deterministic frame selection, and GNSS interpolation.
- **Auditable quality controls** — validity rules, sanity policies, quarantine reports, filtering,
  scenario quotas, and train/test assignment produce explicit evidence.
- **Safe publication** — the nuScenes-compatible output is validated before one atomic rename;
  existing datasets are never overwritten.
- **Read-only SDK** — inspect published tables, scenes, samples, camera records, and ego poses from
  Python.

## Pipeline

```text
Hugging Face dataset repository
        │
        ▼
Pinned manifest → verified MCAP cache → camera/GNSS extraction → validity and scenes
        │
        ▼
Features and scenarios → scene-level split → nuScenes export → validation → atomic publication
```

## Quickstart

### Requirements

- Python 3.12 or newer
- A POSIX environment such as Linux or macOS
- Access to a Hugging Face dataset repository containing the expected MCAP schema

Windows is not supported because the cache and publication safety model relies on POSIX file
locks and no-follow, descriptor-relative filesystem operations.

### Install from source

```bash
git clone https://github.com/gagandeepreehal/dataset-devkit.git
cd dataset-devkit

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For a private Hugging Face repository, authenticate through the standard client before building:

```bash
hf auth login
```

Tokens are read by `huggingface_hub`; they do not belong in the dataset configuration.

### Build the example

```bash
cp examples/dataset_config.json dataset_config.json
cp examples/annotations.jsonl annotations.jsonl

dataset-devkit build --config dataset_config.json
```

The example configuration is pinned to a real immutable commit of
`gagandeepreehal/minuszero-indian-autonomous-driving-monocam`. Adjust its source, topics, camera
requirements, scene rules, and output paths for your dataset.

## Source contract

The configuration identifies exactly one Hugging Face dataset repository:

```json
{
  "huggingface": {
    "repo_id": "owner/dataset",
    "revision": "b13c3bd3a049c73b560910ef5dbc60cbd28c441b",
    "manifest_path": "manifest.jsonl"
  }
}
```

The revision must be a full lowercase 40-character commit SHA. Branches and tags are rejected so
the same configuration always identifies the same repository snapshot.

The repository manifest is UTF-8 JSONL with one MCAP per row:

```json
{"repo_path":"data/2025-04-11/run.mcap","source_size":30883381,"sha256":"4af1b3aaa2db2f146c0ace8d1d339678640852181307980e7c918b107491ea96"}
```

Each row requires:

| Field | Contract |
| --- | --- |
| `repo_path` | Normalized repository path below `data/`, ending in `.mcap` |
| `source_size` | Positive byte size as a JSON integer |
| `sha256` | Lowercase 64-character SHA-256 digest |

Manifest order is build order. Duplicate or unsafe paths, malformed hashes, invalid sizes, and an
empty manifest fail before recording processing begins.

See [Configuration](docs/configuration.md) for the complete typed configuration contract.

## Command-line interface

```bash
# Build and atomically publish a dataset
dataset-devkit build --config dataset_config.json

# Validate an existing publication
dataset-devkit validate --dataroot DATASET --version v1.0-trainval

# Print a compact dataset summary
dataset-devkit inspect --dataroot DATASET --version v1.0-trainval
```

Every command prints one deterministic JSON object to standard output. Configuration and usage
errors exit with status `2`; operational and validation failures exit with status `1` and a safe,
concise diagnostic.

## Published dataset

A successful build publishes the following structure below `paths.output_dir`:

```text
v1.0-trainval/
├── maps/
├── mz_extensions/
├── samples/
└── v1.0-trainval/
    ├── calibrated_sensor.json
    ├── ego_pose.json
    ├── log.json
    ├── sample.json
    ├── sample_data.json
    ├── scene.json
    └── ...
```

The official nuScenes tables and image assets are accompanied by deterministic extension files
containing source fingerprints, validity evidence, annotations, tags, split decisions, pipeline
audit data, and a content manifest.

Published datasets are read-only artifacts. Rebuild into an absent destination instead of editing
an existing publication.

## Python SDK

```python
from pathlib import Path

from dataset_devkit import Dataset

dataset = Dataset(dataroot=Path("DATASET"), version="v1.0-trainval")

scene = dataset.table("scene")[0]
samples = dataset.scene_samples(scene["token"])
front_camera = dataset.camera(samples[0]["token"], "CAM_FRONT")
ego_pose = dataset.ego_pose(front_camera["token"])
```

The SDK is intentionally read-only: published evidence is inspected, never mutated in place.

## Reliability and failure handling

- Each recording is acquired, extracted, validated, and preflighted independently.
- Recording failures receive durable quarantine reports and block publication by default.
- Optional partial publication is allowed only when every failure is quarantined and all owned
  working data is safely cleaned up.
- Cache reuse re-verifies source and extraction evidence.
- Symbolic links, hard-linked cache files, unsafe repository paths, and cache identity changes fail
  closed.
- Final output is validated, content-manifested, and atomically renamed into place.

For the exact guarantees and threat boundaries, read
[Export and publication](docs/export.md#publication-threat-model-and-read-only-contract).

## Documentation

| Guide | Contents |
| --- | --- |
| [Configuration](docs/configuration.md) | Hugging Face source, manifest, cache, and all policy sections |
| [Extraction](docs/extraction.md) | MCAP schema, timestamps, HEVC decoding, GNSS interpolation, and staging |
| [Validity](docs/validity.md) | Invalidity codes, sanity checks, quarantine, and partial publication |
| [Scenes](docs/scenes.md) | Automatic, annotation-only, and hybrid scene construction |
| [Selection](docs/selection.md) | Features, filters, scenario quotas, and deterministic splitting |
| [Export](docs/export.md) | nuScenes tables, extension evidence, validation, SDK, and publication |

## Current scope

- Camera and GNSS data only
- MCAP recordings using the expected protobuf and HEVC source schema
- Hugging Face dataset repositories only
- Commit-pinned, manifest-driven full-corpus builds
- POSIX environments only
- nuScenes `v1.0-trainval` publication contract

LiDAR ingestion, symbolic revisions, ad hoc repository scanning, source subsetting, mutable output,
and Windows support are intentionally outside the first release.

## Development

Install the development dependencies:

```bash
python -m pip install -e '.[dev]'
```

Run the complete local quality gate:

```bash
pytest -q
ruff check .
mypy
PYTHONPATH=src python -m dataset_devkit.schema
git diff --exit-code schema/dataset_config.schema.json
python -m build --wheel --no-isolation
```

## License

This repository does not yet include an open-source license. Until one is added, the source is
publicly visible but remains all rights reserved.
