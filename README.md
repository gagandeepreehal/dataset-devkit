# dataset-devkit

`dataset-devkit` is a Python 3.12 package for building deterministic robotics datasets
from MCAP recordings in Azure Blob Storage. This foundation release defines the public
configuration, CLI, and Python API boundaries; pipeline stages are implemented separately.

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
```
