# Minus Zero Dataset DevKit

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
![Status: Alpha](https://img.shields.io/badge/status-alpha-orange)
![Platform: POSIX](https://img.shields.io/badge/platform-POSIX-lightgrey)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue)](LICENSE.md)

Welcome to the development kit for creating compact, reproducible, nuScenes-compatible datasets
from Minus Zero autonomous-driving releases. The recommended public-consumer input is the decoded
V2 Hugging Face release after its privacy transformation. Original MCAP is also supported, but it
is classified as **restricted raw** input and carries no anonymization claim.

Neither source is a nuScenes dataset. This devkit selects useful driving scenes, subsamples the
sensor streams, and publishes a derived dataset using the nuScenes table and directory format.

> [!IMPORTANT]
> This project is independent of the official nuScenes project and is not affiliated with or
> endorsed by Motional. Compatibility refers to the published data structure and supported loader
> behavior, not to the sensor suite, annotations, tasks, or benchmark content of the nuScenes
> dataset.

## Overview

- [What this devkit does](#what-this-devkit-does)
- [Devkit setup](#devkit-setup)
- [Source dataset setup](#source-dataset-setup)
- [Getting started](#getting-started)
- [Subsampling and selection](#subsampling-and-selection)
- [Output format](#output-format)
- [Python SDK](#python-sdk)
- [Documentation](#documentation)
- [Known limitations](#known-limitations)
- [Citation](#citation)
- [License](#license)

## What this devkit does

`dataset-devkit` turns a pinned Minus Zero dataset release into a smaller dataset for research,
experimentation, and model development:

```text
Pinned decoded V2 or restricted raw MCAP release
                         │
                         ▼
        verified acquisition and extraction
                    │
                    ▼
 camera/GNSS subsampling, validation, and scene construction
                    │
                    ▼
       filtering and scenario-based selection
                    │
                    ▼
      train/test split and nuScenes-compatible export
```

The pipeline provides:

- immutable, commit-pinned decoded V2 or MCAP acquisition from Hugging Face;
- selective decoded artifact downloads with public privacy-manifest verification;
- deterministic camera-frame downsampling and GNSS interpolation;
- automatic, annotation-driven, or hybrid scene construction;
- scene tagging, quality filtering, and deterministic scenario quotas;
- deterministic scene-level train/test splitting;
- validated nuScenes-compatible tables and camera assets; and
- provenance, audit, quarantine, and content-manifest extensions.

The result is a **derived subset** of a Minus Zero release. Source artifacts are not modified. A
decoded build never falls back to raw MCAP if its catalog, manifest, or privacy checks fail.

## Devkit setup

The devkit requires Python 3.12 or newer and a POSIX environment such as Linux or macOS.

```bash
git clone https://github.com/gagandeepreehal/dataset-devkit.git
cd dataset-devkit

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Windows is not currently supported because safe caching and publication rely on POSIX file locks
and descriptor-relative, no-follow filesystem operations.

## Source dataset setup

Minus Zero dataset releases are hosted as Hugging Face dataset repositories. Every source uses an
exact repository name and full 40-character commit SHA; branch and tag names are rejected.

For public-consumer work, start with
[`examples/decoded_v2_config.json`](examples/decoded_v2_config.json). Its all-zero revision is an
intentional non-release placeholder: replace it with the full commit SHA of a verified privacy
release before running a build. `decoded_hf` is fixed to `privacy_transformed`; it verifies
`data/recordings.parquet` plus the four public files under `data/manifests/`: `output-files.parquet`,
`processing-config.json`, `model-receipt.json`, and `schema-audit.parquet`. It then downloads only
the selected recordings, cameras, and modalities.

The decoded default is the six untilted cameras. Add `cam_front_tilted` explicitly when required.
`video`, `gnss`, and `calibration` are mandatory; `camera_labels`, `semantic`, and `depth` are
optional selections. This verification checks the published privacy contract, schemas, sizes, and
hashes. It **does not independently rerun the anonymizer** or claim that raw MCAP is anonymous.

The legacy [`examples/dataset_config.json`](examples/dataset_config.json) targets an original MCAP
release. MCAP inputs are always `restricted_raw`, even when access-controlled, and should be
handled under the source release's privacy and access policy.

For a private or gated release, authenticate with the standard Hugging Face client:

```bash
hf auth login
```

Authentication tokens are read by `huggingface_hub` and must not be stored in the configuration.

For the MCAP backend, each manifest row identifies one recording and its expected content:

```json
{"repo_path":"data/2025-04-11/run.mcap","source_size":30883381,"sha256":"4af1b3aaa2db2f146c0ace8d1d339678640852181307980e7c918b107491ea96"}
```

| Field | Description |
| --- | --- |
| `repo_path` | Path to an `.mcap` recording below the release's `data/` directory |
| `source_size` | Expected file size in bytes |
| `sha256` | Expected lowercase SHA-256 digest |

The commit and manifest make the input corpus reproducible. Repository scans and unverified
recordings are not accepted as build inputs.

## Getting started

Copy the recommended decoded configuration and annotations into the working directory:

```bash
cp examples/decoded_v2_config.json dataset_config.json
cp examples/annotations.jsonl annotations.jsonl
```

Replace the all-zero revision with a verified privacy-release commit. Review recording/split,
camera and modality selection, sampling rate, scene rules, scenario quotas, and output paths, then
build the derived dataset:

```bash
dataset-devkit build --config dataset_config.json
```

The command downloads only the selected decoded artifacts, processes each recording independently,
and publishes the result only after final validation succeeds. Keep production payload processing
and caches on `mzcloud`; the included fixture smoke test is intentionally tiny and local.

To validate or inspect an existing output:

```bash
dataset-devkit validate --dataroot DATASET --version v1.0-trainval
dataset-devkit inspect --dataroot DATASET --version v1.0-trainval
```

Every command prints one deterministic JSON object to standard output. Configuration and usage
errors exit with status `2`; operational and validation failures exit with status `1`.

Contributors can exercise the complete decoded path without network or production data:

```bash
python tools/decoded_fixture_smoke.py /private/tmp/dataset-devkit-fixture
```

The script refuses an existing output root and prints the absolute published dataroot as its only
standard-output line. Pass that printed path—not its parent—to `validate` or `inspect`.

## Subsampling and selection

Subsampling is explicit and reproducible. It happens in several stages:

1. **Temporal sampling** selects camera frames at `downsampling.target_fps` within a configured
   timestamp tolerance.
2. **Validity checks** evaluate camera availability, timestamp continuity, GNSS quality, and sensor
   synchronization.
3. **Scene construction** groups valid samples into bounded driving scenes.
4. **Feature and tag generation** describes motion such as straight driving, curvature, turns,
   stopping, and stationary behavior.
5. **Scenario rules** use seeded ranking to select the requested quota. With the default
   `strict_quotas: true`, a deficit fails the build; non-strict rules select up to the quota.
6. **Scene-level splitting** assigns every selected scene to one train or test partition without
   splitting its camera chains.

For example, a scenario rule can request a deterministic subset of left-turn scenes:

```json
{
  "name": "Left Turn",
  "quota": 100,
  "required_all_tags": ["left_turn"],
  "excluded_tags": ["stationary"]
}
```

The selected result records its source identities, filtering decisions, scenario assignments, and
split evidence so it can be audited and reproduced.

For decoded V2, trajectory features and official ego poses are calculated from the precise
per-recording **recording-local ENU** stream created by the privacy transformation. Published global
east/north and inverse-derived latitude/longitude are one-metre-rounded context only. Each
recording starts its own local frame; the devkit never creates cross-recording continuity.

## Output format

A successful build publishes a nuScenes-compatible dataroot below `paths.output_dir`:

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

The core tables and camera assets follow the supported nuScenes layout. `mz_extensions/` preserves
information that does not belong in the standard tables, including the fixed source/privacy/pose
contract, source fingerprints, validity evidence, scenario assignments, split decisions, pipeline
audit data, and the final content manifest.

Published outputs are read-only artifacts. The devkit validates the complete staging dataset and
then performs one atomic, no-overwrite publication. To change a dataset, rebuild it into a new,
absent destination.

## Python SDK

The included read-only SDK provides convenient access to a published dataset:

```python
from pathlib import Path

from dataset_devkit import Dataset

dataset = Dataset(dataroot=Path("DATASET"), version="v1.0-trainval")

scene = dataset.table("scene")[0]
samples = dataset.scene_samples(scene["token"])
front_camera = dataset.camera(samples[0]["token"], "CAM_FRONT")
ego_pose = dataset.ego_pose(front_camera["token"])
```

The official `nuscenes-devkit` is also smoke-tested against the exported table structure during
publication. Minus Zero extension files remain specific to this project.

## Documentation

| Guide | Contents |
| --- | --- |
| [Configuration](docs/configuration.md) | Source identity, paths, sensors, policies, and publication settings |
| [Extraction](docs/extraction.md) | MCAP and decoded V2 extraction, timestamps, local ENU poses, and staging |
| [Validity](docs/validity.md) | Quality rules, sanity checks, quarantine, and partial publication |
| [Scenes](docs/scenes.md) | Automatic, annotation-only, and hybrid scene construction |
| [Selection](docs/selection.md) | Features, filters, scenario quotas, and deterministic splitting |
| [Export](docs/export.md) | Tables, extensions, validation, SDK behavior, and publication guarantees |

## Known limitations

- Dataset generation currently uses camera, calibration, and GNSS data; optional decoded semantic,
  depth, and camera-label artifacts are selectable but are not converted to object ground truth.
- Source MCAPs must use the expected Minus Zero protobuf and HEVC schema and remain restricted raw.
- Outputs contain selected Minus Zero scenes; they do not reproduce the official nuScenes sensor
  suite, annotations, maps, evaluation tasks, or benchmark splits.
- `v1.0-trainval` is the only publication version currently supported.
- Input releases must be hosted on Hugging Face and pinned by a full commit SHA and the applicable
  manifest/control-plane contract.
- LiDAR ingestion, arbitrary source repositories, symbolic revisions, and Windows are not
  supported.

## Citation

When publishing work based on a generated dataset, cite the specific Minus Zero source release
using the citation information on its Hugging Face dataset card. Also identify this devkit by its
repository URL and the release tag or commit used for generation. This keeps the original data and
the derived dataset-building software independently traceable.

## Development

Install the development dependencies and run the local quality gate:

```bash
python -m pip install -e '.[dev]'

pytest -q
ruff check .
mypy
PYTHONPATH=src python -m dataset_devkit.schema
git diff --exit-code schema/dataset_config.schema.json
python -m build --wheel --no-isolation
```

## License

The software is source-available under the
[PolyForm Noncommercial License 1.0.0](LICENSE.md). It may be used, copied, modified, and
distributed for noncommercial purposes under those terms. Commercial use requires a separate
license from the licensor. See the project [copyright notice](NOTICE).

Because commercial use is restricted, this is not an OSI-approved open-source license. Each Minus
Zero dataset release remains governed separately by the license stated on its dataset card.
