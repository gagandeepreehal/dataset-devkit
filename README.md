# dataset-devkit

`dataset-devkit` is a Python 3.12 package for building deterministic robotics datasets
from MCAP recordings in Azure Blob Storage. This foundation release defines the public
configuration, CLI, and Python API boundaries. Azure acquisition is implemented as a focused,
injectable service. Native MCAP/protobuf extraction, persistent HEVC decode, deterministic camera
selection, GNSS interpolation, and verified JPEG staging are a separate focused service. Typed
validity/sanity policy, safe quarantine reports, independent-recording partial-export gating,
deterministic automatic/annotation/hybrid scene graphs, real-timestamp features and tags,
exact-quota scenario selection, and auditable scene-level train/test splitting are implemented.
Deterministic nuScenes export, comprehensive validation, an indexed read-only Dataset SDK,
content manifests, and no-overwrite atomic publication are implemented as one standalone pipeline.

## Install for development

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Copy the example files, then validate their paths and policies in code:

```bash
cp examples/dataset_config.json dataset_config.json
cp examples/mcap_blobs.txt mcap_blobs.txt
cp examples/annotations.jsonl annotations.jsonl
python -c 'from pathlib import Path; from dataset_devkit.config import load_config; print(load_config(Path("dataset_config.json")))'
```

The stable command contracts are:

```bash
dataset-devkit build --config dataset_config.json
dataset-devkit validate --dataroot DATASET --version v1.0-trainval
dataset-devkit inspect --dataroot DATASET --version v1.0-trainval
```

Each command prints one concise deterministic JSON object to stdout. Configuration/usage failures
exit 2; operational or validation failures exit 1 with one safe stderr diagnostic and no normal
traceback. A successful build publishes at `paths.output_dir/v1.0-trainval`; the dataroot itself
contains the `v1.0-trainval/`, `samples/`, `maps/`, and `mz_extensions/` children.

Builds run in a uniquely named sibling `.v1.0-trainval.staging-*` directory. Tables, assets,
extensions, official-SDK smoke loading, and the final manifest are validated before one atomic
rename. Existing final dataroots are never overwritten. An invocation-owned staging directory is
removed after an ordinary failure; if identity-safe cleanup cannot be proven, it is deliberately
left for operator recovery instead of deleting an ambiguous path.

The secure cache backend is supported on POSIX platforms only. It requires POSIX file locks and
descriptor-relative, no-follow filesystem operations; Windows is not a supported runtime.

## Azure acquisition

Blob-list entries must be exact container-relative paths under `mcap-h265/` ending in `.mcap`.
The acquisition service uses `DefaultAzureCredential`; it never accepts embedded keys, SAS
tokens, or connection strings. Verified downloads are cached by the exact account, container,
blob path, ETag, and size. Compatible `.partial` downloads can resume, while incompatible
partials restart. Each finalized recording has a canonical provenance manifest recording its
source fingerprint, local SHA-256 and size, Azure MD5 verification when available (or the
size-and-stable-ETag fallback), result status, and requested extraction-config hash. That request
hash is not proof that extraction ran. Extraction reuse becomes valid only when the extraction
stage calls `AzureBlobAcquirer.record_extraction_complete(source, config_hash)`. Reuse decisions
must use `AzureBlobAcquirer.extraction_cache_reusable(source, config_hash)`. These source-keyed
methods record and read the separate completion manifest under the same trusted cache traversal
and per-recording lock as acquisition; they do not accept caller-supplied manifest paths.
Acquisition cache hits never create that proof.

See [configuration.md](docs/configuration.md#managed-identity-smoke-check) for authentication and
an optional one-blob smoke check.

See [extraction.md](docs/extraction.md) for the exact source schema, timestamp, interpolation,
staging, and structural-failure contract.

See [validity.md](docs/validity.md) for invalidity codes and thresholds, audit/drop behavior,
sanity modes, quarantine reports, and the partial-export authorization gate.

See [scenes.md](docs/scenes.md) for exact greedy segmentation, strict annotation JSONL matching,
hybrid exclusion, UUIDv5 identities, per-camera chains, and structural graph validation.

See [selection.md](docs/selection.md) for real-timestamp trajectory features, complete-evidence
filtering, and deterministic exact-quota scenario rules.

See [export.md](docs/export.md) for the official nuScenes table layout, exact timestamp and image
copy rules, loader-required non-semantic map compatibility scaffold, extension tables, exporter
evidence boundary, validation/manifest semantics, atomic publication, and Dataset SDK methods.

All recordings are attempted independently. Any acquisition, extraction, validity, or sanity
failure is quarantined and blocks publication by default. Setting
`execution.allow_partial_export` to `true` permits successful recordings only when every failure
report was durably quarantined; the resulting build JSON explicitly reports `partial: true` and
the failed blob names. Managed identity is always supplied by `DefaultAzureCredential`; secrets,
SAS URLs, account keys, and connection strings are prohibited in configuration. The optional
one-blob Azure smoke procedure is documented in
[configuration.md](docs/configuration.md#managed-identity-smoke-check).

## Scene-level train/test split

`split_selected_scenes` validates the complete Task 6 feature population and scenario-selection
result before assigning every selected scene exactly once. The exact test target is
`floor(scene_count * test_fraction + 0.5)`. When configured, primary-scenario stratification uses
deterministic SHA-256 ranking and largest-remainder apportionment; singleton strata and global
targets that cannot preserve both train and test are recorded as explicit fallbacks. A scene's
complete sample and per-camera sample-data chains always stay in its assigned split.

`write_split_extension` revalidates all upstream evidence before atomically writing canonical,
wall-clock-free `mz_extensions/split.json`. The extension records source-disambiguated
assignments, exact counts, per-stratum audits, fingerprints, and chronological adjacent-scene
cross-split evidence with a neighboring-context leakage warning. Published validation uses the
compact complete Task 5 scene chronology in `pipeline_audit.json`, rather than treating filtered
or unselected gaps as adjacency. Use `validate_scene_split` to
recompute and validate an in-memory result; these three functions are the public Task 7 APIs.

The stable Python import is:

```python
from pathlib import Path

from dataset_devkit import Dataset

dataset = Dataset(dataroot=Path("DATASET"), version="v1.0-trainval")
scene = dataset.table("scene")[0]
samples = dataset.scene_samples(scene["token"])
front = dataset.camera(samples[0]["token"], "CAM_FRONT")
pose = dataset.ego_pose(front["token"])
```

See [configuration.md](docs/configuration.md) for the complete configuration contract and
authentication rules. The generated schema is checked in at
`schema/dataset_config.schema.json`.

## Quality checks

```bash
pytest
ruff check .
mypy
PYTHONPATH=src python -m dataset_devkit.schema
git diff --exit-code schema/dataset_config.schema.json
python -m build --wheel --no-isolation
python -m venv --system-site-packages /tmp/dataset-devkit-smoke
/tmp/dataset-devkit-smoke/bin/pip install --no-deps dist/dataset_devkit-0.1.0-py3-none-any.whl
/tmp/dataset-devkit-smoke/bin/dataset-devkit --help
```
