# Minus Zero Dataset DevKit

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
![Status: Alpha](https://img.shields.io/badge/status-alpha-orange)
![Platform: POSIX](https://img.shields.io/badge/platform-POSIX-lightgrey)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue)](LICENSE.md)

Build reproducible, nuScenes-compatible subsets from Minus Zero driving datasets.
The devkit selects camera frames and driving scenes, interpolates GNSS poses,
applies scenario quotas, and exports validated train/test splits with provenance.

Integrated INDUS-AD support includes camera, semantic, depth, weather, and optional
GNSS weak pseudo-labels. These model outputs remain separate from human labels.

> Compatibility refers to the nuScenes file structure and supported loader behavior.
> This project is independent of Motional and does not reproduce the nuScenes
> sensor suite, annotations, maps, or benchmarks.

## Setup

Requires Python 3.12+ and Linux or macOS.

```bash
git clone https://github.com/gagandeepreehal/dataset-devkit.git
cd dataset-devkit
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For gated or private datasets, obtain access and authenticate with `hf auth login`.
Keep credentials out of configuration files.

## Choose a source

All sources require an exact repository ID and a full 40-character commit SHA.

| Source | Schema | Example |
| --- | --- | --- |
| Integrated INDUS-AD with annotations | `2.1`, `indus_ad_hf` | [indus_ad_v21_config.json](examples/indus_ad_v21_config.json) |
| Older decoded V2 layout | `2.0`, `decoded_hf` | [decoded_v2_config.json](examples/decoded_v2_config.json) |
| Original MCAP | `1.0`, `mcap_hf` | [dataset_config.json](examples/dataset_config.json) |

INDUS-AD and decoded V2 are classified as `privacy_transformed`; original MCAP is
restricted raw (`restricted_raw`) and carries no anonymization claim. Manifest verification
checks published evidence and integrity; it does not independently rerun the anonymizer.

## Build an INDUS-AD subset

```bash
cp examples/indus_ad_v21_config.json dataset_config.json
cp examples/annotations.jsonl annotations.jsonl
```

The configuration pins a verified release and selects the full training split.
Set `source.recording_ids` for a smaller build. Review camera and annotation layers,
sampling rate, scene rules, quotas, and work/cache/output paths before running:

```bash
dataset-devkit build --config dataset_config.json
```

The human annotation file is optional input content but must exist at its configured
path. Remove the template's MCAP example row if you are not supplying human labels.
The older decoded V2 example uses an all-zero revision placeholder that must be replaced.

The pipeline verifies selected artifacts, samples frames, builds scenes, applies
filters and seeded scenario quotas, and assigns train/test splits. Strict quota
shortfalls fail the build. For transformed sources, trajectories and motion features
use precise recording-local ENU; rounded global coordinates provide context only.

Six untilted cameras are selected by default; `cam_front_tilted` is optional. Dense
semantic/depth payloads are metadata-only by default. `selected_frames` modes require
`camera_scalar` and verify dense members after scene selection.

## Output and validation

A successful build publishes `paths.output_dir/v1.0-trainval`:

```text
v1.0-trainval/
├── maps/
├── samples/
├── mz_extensions/
└── v1.0-trainval/       # official JSON tables
```

`mz_extensions/` contains source fingerprints, validity, labels, selection/split
audits, and a content manifest. Integrated weak labels are Parquet tables under
`mz_extensions/indus_ad/`. Official object annotation tables remain empty.
Dense payloads stay in the verified cache; exported rows retain source references
and materialization state. Dense acquisition failures block publication.

```bash
dataset-devkit validate --dataroot DATASET --version v1.0-trainval
dataset-devkit inspect --dataroot DATASET --version v1.0-trainval
```

Use the published dataroot as `DATASET`. Outputs are read-only and existing
destinations are never overwritten. Rebuild into a new destination to change content.
Commands emit JSON; configuration errors exit `2`, and operational/validation errors exit `1`.

## Python SDK

```python
from dataset_devkit import Dataset

dataset = Dataset("DATASET")
scene = dataset.table("scene")[0]
sample = dataset.scene_samples(scene["token"])[0]
front = dataset.camera(sample["token"], "CAM_FRONT")
pose = dataset.ego_pose(front["token"])

# Integrated INDUS-AD output
labels = dataset.camera_annotations(camera_name="cam_front")
if labels:
    weather = labels[0].annotation.weather
```

See the [SDK guide](docs/export.md) for annotation lookups, splits, and audit access.

## Documentation

| Guide | Contents |
| --- | --- |
| [Integrated INDUS-AD](docs/indus-ad.md) | Annotation layers, joins, dense references, and companion GNSS |
| [Configuration](docs/configuration.md) | Sources, paths, sensors, and policies |
| [Extraction](docs/extraction.md) | Camera decoding, timestamps, GNSS, and staging |
| [Validity](docs/validity.md) | Quality rules, quarantine, and partial publication |
| [Scenes](docs/scenes.md) | Automatic, annotation-driven, and hybrid scenes |
| [Selection](docs/selection.md) | Features, filters, quotas, and splitting |
| [Export and SDK](docs/export.md) | Tables, extensions, validation, and publication |

## Limitations

- Camera, calibration, and GNSS inputs are supported; LiDAR ingestion is not.
- Inputs must follow a supported source contract. Arbitrary repositories and symbolic revisions are rejected.
- `v1.0-trainval` is the only publication version. Windows is unsupported.
- Production sensor extraction and a production GNSS companion release have not been
  verified for the integrated backend; acceptance includes local sensor fixtures and
  the pinned release's live metadata contract.

## Development

```bash
python -m pip install -e '.[dev]'
pytest -q
ruff check src tests
mypy
python -m dataset_devkit.schema
git diff --exit-code schema/dataset_config.schema.json
python -m build --no-isolation
```

For a tiny offline decoded build:

```bash
python tools/decoded_fixture_smoke.py /tmp/dataset-devkit-fixture
```

The fixture destination must be absent; the script prints the published dataroot.

## Citation and license

Cite the source dataset release using its Hugging Face dataset card, and identify
this devkit by repository URL and the release tag or commit used.

The software uses the [PolyForm Noncommercial License 1.0.0](LICENSE.md).
Commercial use requires a separate license. See [NOTICE](NOTICE) for copyright information.
