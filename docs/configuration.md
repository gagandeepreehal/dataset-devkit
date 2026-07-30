# Configuration

`dataset_config.json` is validated by the versioned Pydantic `GlobalConfig` model. The current
`schema_version` is `"1.0"`. Unknown keys are rejected at every object level. Relative paths
are resolved from the directory containing the JSON file, not from the process working
directory. Runtime primitive types are strict: JSON strings, numbers, and booleans are not
silently converted into one another.

Start with [`examples/dataset_config.json`](../examples/dataset_config.json) and use
[`schema/dataset_config.schema.json`](../schema/dataset_config.schema.json) for editor or CI
validation.

JSON Schema 2020-12 directly enforces Task 6 constraints that it can represent, including the rule
that `reference_camera_policy: "require"` needs a non-null `reference_camera_channel`, each
per-channel coverage value is a finite ratio in `[0, 1]`, and predicate/blacklist items and scenario
rule names are nonblank with no leading or trailing whitespace (internal spaces are allowed). Standard
JSON Schema cannot compare arbitrary numeric sibling values, compare matching values in two maps,
prove arbitrary predicate arrays disjoint, or require unique nested rule names. Each affected
definition therefore carries deterministic `x-dataset-devkit-runtime-constraints` entries with the
exact constraint code, message, and fields. Schema-only validation is not full configuration
validation: `load_config` is authoritative.

For CI that explicitly runs both layers, use the public combined validator:

```bash
PYTHONPATH=src python -c 'from pathlib import Path; from dataset_devkit import validate_config_schema_and_runtime; validate_config_schema_and_runtime(Path("dataset_config.json"))'
```

## Authentication

The `azure` section contains only a Blob service `account_url`, container name, and blob-list
path. Runtime Azure access must use `azure.identity.DefaultAzureCredential`. In production,
prefer a managed identity. For local development, authenticate with `az login`; the same
credential chain can then use the Azure CLI session.

For a VM managed identity, grant the identity the narrowest suitable data-plane role (normally
`Storage Blob Data Reader`) on the required storage scope. No account key, SAS token, connection
string, client secret, or other credential belongs in the configuration or blob-list file.

Never place account keys, SAS tokens, connection strings, client secrets, passwords, bearer
tokens, private keys, JWTs, or other credentials in JSON. The loader rejects explicit
credential-bearing field names and structured secret values before configuration is accepted.
`account_url` must identify the Blob service root over HTTPS; container/blob paths, fragments,
malformed ports, URL userinfo, and credential-bearing query parameters are rejected. Nonempty
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
- `image`: exact JPEG quality `95`, matching the native staging encoder's v1 contract.
- `gnss`: non-negative position, orientation-variance, and synchronization thresholds.
- `frame_validity`: `retain_for_audit`/`drop`, exact required camera identities, a positive
  per-camera timestamp-gap limit, and eight independent typed invalidator toggles. Unknown reason
  names are rejected.
- `sanity_checks`: an explicit `error`, `warn`, or `off` policy for each of
  `empty_selected_grid`, `empty_final_candidates`, `all_gnss_sources_invalid`, and
  `zero_required_camera_coverage`. Unknown check names and other policy words are rejected.
- `scenes`: exact `automatic`, `annotation_only`, or default `hybrid` mode; stable dataset UUID
  namespace; duration bounds; sample minimum; gap limit; and scene spacing. Decimal time values
  are parsed from exact JSON numeric text and must resolve exactly to integer nanoseconds. Direct
  model-validation callers use `Decimal`; `load_config` remains the authoritative JSON boundary.
- `annotations`: safely resolved relative JSONL path, nearest-match tolerance, and before/after
  windows. See [scenes.md](scenes.md) for the strict line format and matching algorithm.
- `tags`: exact reference-camera/fallback policy, stationary speed in m/s, minimum movement in
  meters, and ordered straight/curvature/turn net-heading thresholds in degrees. The required
  relationship is `straight_max_heading_change_deg < curvature_min_heading_change_deg <
  turn_min_heading_change_deg`.
- `filters`: optional duration seconds, scene-valid, source-GNSS-valid, and camera-coverage ratios, per-channel
  coverage, maximum sync error milliseconds, distance meters, computed-tag and human-label
  predicates, and exact scene/source/blob blacklists. Empty means accept all.
- `scenarios`: deterministic integer seed, strict exact-quota policy, and uniquely named ordered
  rules. Rules keep computed tags and human labels separate and may add the same metric filters.
  Required and excluded predicates cannot overlap; quotas are nonnegative integers.
- `split`: deterministic test fraction in `(0, 1)`, seed, and stratification switch.
- `execution`: positive worker count and partial-export policy.
- `quarantine`: mandatory `enabled: true`, an isolated output directory, and a basename-only
  rejection-manifest filename. The directory cannot overlap work, cache, or output; disabling
  quarantine is rejected because every failed recording must receive a report.
- `publication`: exact version `v1.0-trainval` and mandatory `refuse_overwrite: true`. The v1
  publisher never replaces an existing dataset, symlink, file, or directory.

## Authoritative validation

`load_config` is the authoritative validator. JSON Schema is useful for editors and structural
CI checks, but it cannot express every credential scan, resolved-path overlap rule, or other
cross-field policy. CI should run the loader as well as the schema-drift test, for example:

```bash
PYTHONPATH=src python -c 'from pathlib import Path; from dataset_devkit.config import load_config; load_config(Path("dataset_config.json"))'
pytest tests/test_schema.py
```

## Blob list

The blob-list file contains one exact container-relative blob name per line. Blank lines and
comment-only lines (including comments preceded by whitespace) are ignored. Every accepted path
must begin with `mcap-h265/` and end with `.mcap`. Duplicate paths, absolute or traversal paths,
backslashes, percent-encoded/normalization-ambiguous paths, query strings, fragments, directories,
and paths outside the prefix are rejected. Non-comment path text is used exactly as written; it
is not stripped or normalized. See [`examples/mcap_blobs.txt`](../examples/mcap_blobs.txt).

## Cache and provenance

Acquisition uses a hash-derived layout below `paths.cache_dir`; blob names never become local
path components. A cache object is reusable only when the account URL, container, exact blob
path, ETag, and size match and the local file re-verifies. Downloads first land in a same-directory
`.partial` file. Resume is allowed only when its sidecar proves that exact source fingerprint;
otherwise the partial is discarded and restarted. Azure properties are checked before and after
download, exact size is mandatory, and a supplied content MD5 must match before atomic rename.
When Azure provides no content MD5, the manifest explicitly records `size_etag`, meaning exact
size plus stable ETag was the available integrity check.
Integrity metadata is accepted only when `verified` is true. `content_md5` requires a canonical
base64-encoded 16-byte MD5 value, while `size_etag` must not include an MD5 claim.

Each recording manifest is canonical JSON and contains the source fingerprint, download status
(`downloaded`, `resumed`, or `cache_hit`), cache-relative artifact path, local size and SHA-256,
integrity method/result, and `requested_extraction_config_hash`. This acquisition field describes
the current request; it does not claim that extraction completed. After producing actual output,
the extraction stage must explicitly call
`AzureBlobAcquirer.record_extraction_complete(source, config_hash)` to atomically write the
separate extraction-completion manifest. Reuse must be decided through
`AzureBlobAcquirer.extraction_cache_reusable(source, config_hash)`. Neither method accepts a
caller-supplied path; both derive the manifest leaf from the source fingerprint, traverse the
trusted cache with directory descriptors, and hold the recording lock for the complete write or
read decision. Only that file proves a source fingerprint and extraction-config hash pair.
Acquisition cache hits can update acquisition request/status provenance but never write or replace
extraction-completion provenance. Missing, malformed, linked, or inconsistent completion manifests
are cache misses.

Cache artifacts, partials, sidecars, and manifests are required to be single-link regular files
within the configured cache. Unsafe symbolic or hard links are never followed, modified, or
promoted into final cache objects. Download verification is bound to the partial file's device
and inode; finalization checks that same inode before and after rename and then repeats full-file
integrity verification before writing the acquisition manifest. Empty source blobs follow the
same checks using an exclusively created zero-length partial. A non-empty partial is resumable
only when Azure supplies a whole-blob content MD5; otherwise acquisition restarts from byte zero
because the existing prefix cannot be verified. Acquisitions for the same recording are serialized
with an operating-system file lock. Cache directories are opened component by component without
following symbolic links, and leaf operations remain relative to those trusted directory handles.
This secure cache backend is POSIX-only because it requires `flock`, directory-relative file
operations, and no-follow opens. Windows is not a supported runtime.

## Managed-identity smoke check

After assigning the VM identity Blob Data Reader access and placing one valid path in the
configured blob list, this optional command downloads and verifies only the first blob:

```bash
PYTHONPATH=src python -c 'from pathlib import Path; from dataset_devkit.acquisition import AzureBlobAcquirer; from dataset_devkit.blob_list import parse_blob_list; from dataset_devkit.config import load_config; c=load_config(Path("dataset_config.json")); p=parse_blob_list(c.azure.blob_list)[0]; r=AzureBlobAcquirer.from_config(c).acquire(p); print(r.manifest.status, r.artifact_path)'
```

On a developer workstation, run `az login` first and use the same command. The credential chain
selects the managed identity on Azure or the Azure CLI login locally; the command and config are
identical and contain no credentials.

## Regenerating the schema

After changing any configuration model, regenerate and test the artifact:

```bash
PYTHONPATH=src python -m dataset_devkit.schema
pytest tests/test_schema.py
```
