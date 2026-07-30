# Configuration

`dataset_config.json` is validated by the versioned Pydantic `GlobalConfig` model. The current
`schema_version` is `"1.0"`. Unknown keys are rejected at every object level. Relative paths
are resolved from the directory containing the JSON file, not from the process working
directory. Runtime primitive types are strict: JSON strings, numbers, and booleans are not
silently converted into one another.

Start with [`examples/dataset_config.json`](../examples/dataset_config.json) and use
[`schema/dataset_config.schema.json`](../schema/dataset_config.schema.json) for editor or CI
validation.

## Authentication

The `azure` section contains only a Blob service `account_url`, container name, and blob-list
path. Runtime Azure access must use `azure.identity.DefaultAzureCredential`. In production,
prefer a managed identity. For local development, authenticate with `az login`; the same
credential chain can then use the Azure CLI session.

Never place account keys, SAS tokens, connection strings, client secrets, passwords, bearer
tokens, private keys, JWTs, or other credentials in JSON. The loader rejects explicit
credential-bearing field names and structured secret values before configuration is accepted.
`account_url` must not contain URL userinfo or credential-bearing query parameters. Nonempty
signature fields such as `sig` make a SAS URL credential-bearing; SAS metadata such as `sp`,
`st`, `se`, `sr`, and `sv` is not secret by itself. The same query checks apply to other URL
values, while ordinary query parameters remain valid. Opaque bearer tokens are recognized only
when the complete value uses the bearer scheme followed by a conservative RFC 6750 token-shaped
payload. Values assigned to configured path fields are exempt from bearer-token classification;
path-shaped text in a non-path field is not. Ordinary URLs, paths, source identifiers, and prose
are not treated as credentials merely because their text contains words such as `bearer` or
`secret`.

## Sections

- `azure`: HTTPS Azure Blob service URL, Azure-compliant container name, and line-oriented MCAP
  blob list. Public, sovereign-cloud, and private-link Blob hostnames are accepted.
- `paths`: isolated work, cache, and output directories. Equality or ancestor/descendant
  overlap between any pair is rejected to prevent accidental corruption.
- `topics`: nonblank logical camera and GNSS topic/channel names.
- `downsampling`: positive target FPS and non-negative timestamp tolerance.
- `image`: JPEG quality from 1 through 100.
- `gnss`: non-negative position, orientation-variance, and synchronization thresholds.
- `frame_validity`: audit/drop policy and typed invalidation rules for missing camera data,
  invalid GNSS, and excessive synchronization gaps.
- `sanity_checks`: timestamp violation action plus positive speed and position-jump limits.
- `scenes`: segmentation mode, duration bounds, sample minimum, gap limit, and scene spacing.
- `annotations`: relative JSONL path, match tolerance, and before/after windows.
- `tags`: stationary-speed and turn-angle thresholds.
- `filters`: valid-sample fraction and tags required for export.
- `scenarios`: deterministic seed and uniquely named typed rules. Tag lists contain nonblank,
  unique values; required and excluded tags cannot overlap. Each rule has a sampling fraction in
  `(0, 1]` plus an optional positive scene cap.
- `split`: deterministic test fraction in `(0, 1)`, seed, and stratification switch.
- `execution`: positive worker count and partial-export policy.
- `quarantine`: enablement, isolated output directory, and basename-only rejection-manifest
  filename. When enabled, its directory cannot overlap work, cache, or output.
- `publication`: safe single-segment public dataset version and overwrite refusal.

## Authoritative validation

`load_config` is the authoritative validator. JSON Schema is useful for editors and structural
CI checks, but it cannot express every credential scan, resolved-path overlap rule, or other
cross-field policy. CI should run the loader as well as the schema-drift test, for example:

```bash
PYTHONPATH=src python -c 'from pathlib import Path; from dataset_devkit.config import load_config; load_config(Path("dataset_config.json"))'
pytest tests/test_schema.py
```

## Blob list

The blob-list file contains one container-relative `.mcap` blob name per line. It is not a URL
list and must not contain credentials. See [`examples/mcap_blobs.txt`](../examples/mcap_blobs.txt).

## Regenerating the schema

After changing any configuration model, regenerate and test the artifact:

```bash
PYTHONPATH=src python -m dataset_devkit.schema
pytest tests/test_schema.py
```
