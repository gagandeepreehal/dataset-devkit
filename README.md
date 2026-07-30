# dataset-devkit

`dataset-devkit` is a Python 3.12 package for building deterministic robotics datasets
from MCAP recordings in Azure Blob Storage. This foundation release defines the public
configuration, CLI, and Python API boundaries. Azure acquisition is implemented as a focused,
injectable service. Native MCAP/protobuf extraction, persistent HEVC decode, deterministic camera
selection, GNSS interpolation, and verified JPEG staging are a separate focused service; validity
policy, scene construction, export, and publication remain later stages.

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
python -c 'from pathlib import Path; from dataset_devkit.config import load_config; print(load_config(Path("dataset_config.json")))'
```

The stable command contracts are:

```bash
dataset-devkit build --config dataset_config.json
dataset-devkit validate --dataroot DATASET --version v1.0-trainval
dataset-devkit inspect --dataroot DATASET --version v1.0-trainval
```

The command parsers and explicit service boundaries exist now. Build, validation, and
inspection services intentionally report that they are not implemented until their pipeline
tasks land.

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

The stable Python import is:

```python
from pathlib import Path

from dataset_devkit import Dataset

dataset = Dataset(dataroot=Path("DATASET"), version="v1.0-trainval")
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
