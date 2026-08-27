# Minus Zero Dataset DevKit

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
![Status: Alpha](https://img.shields.io/badge/status-alpha-orange)
![Platform: POSIX](https://img.shields.io/badge/platform-POSIX-lightgrey)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue)](LICENSE.md)

Welcome to the development kit for creating compact, reproducible, nuScenes-compatible datasets
from the openly released Minus Zero autonomous-driving datasets.

The source releases contain Minus Zero sensor recordings in MCAP format. They are **not**
nuScenes datasets. This devkit reads those recordings, selects useful driving scenes, subsamples
the sensor streams, and publishes a derived dataset using the nuScenes table and directory format.

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
Minus Zero MCAP release on Hugging Face
                    │
                    ▼
       verified download and extraction
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

- commit-pinned and checksum-verified MCAP acquisition from Hugging Face;
- deterministic camera-frame downsampling and GNSS interpolation;
- automatic, annotation-driven, or hybrid scene construction;
- scene tagging, quality filtering, and deterministic scenario quotas;
- deterministic scene-level train/test splitting;
- validated nuScenes-compatible tables and camera assets; and
- provenance, audit, quarantine, and content-manifest extensions.

The result is a **derived subset** of a Minus Zero release. The original MCAP files remain the
source of truth and are not modified.

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

Minus Zero dataset releases are hosted as Hugging Face dataset repositories. A build uses three
pieces of source identity:

1. the Hugging Face repository name;
2. a full 40-character commit SHA; and
3. the release's ordered `manifest.jsonl`.

The included example targets the
[`gagandeepreehal/minuszero-indian-autonomous-driving-monocam`](https://huggingface.co/datasets/gagandeepreehal/minuszero-indian-autonomous-driving-monocam)
release. Other supported Minus Zero releases can be selected by changing the source configuration
and the sensor policies that describe that release.

For a private or gated release, authenticate with the standard Hugging Face client:

```bash
hf auth login
```

Authentication tokens are read by `huggingface_hub` and must not be stored in the configuration.

Each manifest row identifies one recording and its expected content:

```json
{"repo_path":"data/2025-04-11/run.mcap","source_size":30883381,"sha256":"4af1b3aaa2db2f146c0ace8d1d339678640852181307980e7c918b107491ea96"}
```

| Field | Description |
| --- | --- |
| `repo_path` | Path to an `.mcap` recording below the release's `data/` directory |
| `source_size` | Expected file size in bytes |
| `sha256` | Expected lowercase SHA-256 digest |

The commit and manifest make the input corpus reproducible. Branch names, tags, repository scans,
and unverified recordings are not accepted as build inputs.

## Getting started

Copy the example configuration and annotations into the working directory:

```bash
cp examples/dataset_config.json dataset_config.json
cp examples/annotations.jsonl annotations.jsonl
```

Review the source revision, topics, required cameras, sampling rate, scene rules, scenario quotas,
and output paths, then build the derived dataset:

```bash
dataset-devkit build --config dataset_config.json
```

The command downloads and verifies the source MCAPs, processes each recording independently, and
publishes the result only after final validation succeeds.

To validate or inspect an existing output:

```bash
dataset-devkit validate --dataroot DATASET --version v1.0-trainval
dataset-devkit inspect --dataroot DATASET --version v1.0-trainval
```

Every command prints one deterministic JSON object to standard output. Configuration and usage
errors exit with status `2`; operational and validation failures exit with status `1`.

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
information that does not belong in the standard tables, including source fingerprints, validity
evidence, scenario assignments, split decisions, pipeline audit data, and the final content
manifest.

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
| [Extraction](docs/extraction.md) | MCAP schema, HEVC decoding, timestamps, GNSS interpolation, and staging |
| [Validity](docs/validity.md) | Quality rules, sanity checks, quarantine, and partial publication |
| [Scenes](docs/scenes.md) | Automatic, annotation-only, and hybrid scene construction |
| [Selection](docs/selection.md) | Features, filters, scenario quotas, and deterministic splitting |
| [Export](docs/export.md) | Tables, extensions, validation, SDK behavior, and publication guarantees |

## Known limitations

- The first release supports camera and GNSS data only.
- Source MCAPs must use the expected Minus Zero protobuf and HEVC schema.
- Outputs contain selected Minus Zero scenes; they do not reproduce the official nuScenes sensor
  suite, annotations, maps, evaluation tasks, or benchmark splits.
- `v1.0-trainval` is the only publication version currently supported.
- Input releases must be hosted on Hugging Face and pinned by commit and manifest.
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
license from the licensor.

Because commercial use is restricted, this is not an OSI-approved open-source license. Each Minus
Zero dataset release remains governed separately by the license stated on its dataset card.
