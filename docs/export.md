# nuScenes export and Dataset SDK

`export_dataset(staging_dataroot, evidence)` consumes immutable Task 5 through Task 7 evidence.
It recomputes scenario selection, scene splits, and every recording scene graph before writing.
The resolved global scenario/split configuration and dataset namespace must match that evidence.
The destination must be absent or an empty real directory; existing content is never
overwritten. Every destination write uses pinned no-follow directory descriptors and exclusive
leaf creation. A failed export may leave a partial staging root for audit; direct callers must
remove that root before retrying. Task 9 owns sibling-staging cleanup, validation, and atomic
publication.

## Official layout

The exporter writes all tables expected by the official nuScenes loader under
`v1.0-trainval`: `category`, `attribute`, `visibility`, `instance`, `sensor`,
`calibrated_sensor`, `ego_pose`, `log`, `scene`, `sample`, `sample_data`,
`sample_annotation`, and `map`. Camera-backed tables contain real selected evidence. Unsupported
object, lidar, and radar tables are valid empty JSON arrays; human labels are never converted into
fabricated object annotations.

nuScenes-devkit 1.2.0 unconditionally dereferences `map[0]` while constructing its reverse index,
even for camera-only datasets. The exporter therefore writes exactly one prominently identified
`compatibility_scaffold` map record, associates every real log token with it, and writes a
deterministic 1-by-1 black grayscale PNG under `maps/`. This is only an upstream-loader
compatibility scaffold: it contains and claims no semantic map, object, lidar, or radar data.

Source camera names are normalized to safe deterministic `CAM_*` channels. Distinct source names
that normalize to the same channel are rejected, and the original-to-normalized mapping is kept
in `mz_extensions/recordings.json`. Verified JPEG bytes are copied to
`samples/CAM_*/<sample_data_token>.jpg`. The source must be a regular non-symlink whose inode,
size, SHA-256, JPEG dimensions, and staged evidence agree. The exclusive destination copy is
hashed and decoded again.

Official timestamps use integer microseconds computed as `timestamp_ns // 1000`. Original
nanoseconds are retained in extension evidence. Export rejects any sample or per-camera chain
whose strict order would collide or reverse after conversion. Camera rotation vectors are
converted to normalized `wxyz` quaternions; real translation, intrinsic matrices, image sizes,
and camera-time ego poses are required and must be finite.

`mz_extensions` contains canonical `recordings.json`, `gnss.json`, `validity.json`,
`validation.json`, `tags.json`, `annotations.json`, `split.json`, `config.json`, and
`content_manifest.json`, plus `pipeline_audit.json` for filtering, rejection, and ordered
scenario-selection evidence. The pipeline audit also retains a compact, complete Task 5 scene
chronology so validation distinguishes genuinely adjacent scenes from selected scenes separated
by an unselected scene. Computed tags and human labels stay separate. Before Task 9 runs,
`validation.json` truthfully records `state: "not_run"` and never claims success.
`validity.json` uses schema version 2 with `recordings` for complete per-source observations and
grid/sample audits, plus `scenes` for selected-scene aggregates and official sample references.

## Final validation, manifest, and publication

The build service first validates token uniqueness, all official foreign keys, scene/sample and
per-camera chain symmetry/acyclicity/endpoints/timestamps, required-camera coverage, JPEG
existence/decodability/dimensions, finite pose/calibration values, normalized quaternions,
compatibility map/log links, extension references, and full disjoint split assignment. It then
must instantiate the official `NuScenes` loader, query the first selected scene/sample/camera,
and open the resolved image. Empty selected datasets cannot be published.

Only after those checks does finalization replace `validation.json` with deterministic success
evidence. `content_manifest.json` is written last and lists every published single-link regular
file by sorted relative POSIX path, byte size, and SHA-256. The manifest excludes only itself,
records that exclusion explicitly, and hashes the canonical entry list as `root_sha256`. It has
no timestamp or staging/final absolute path, so the same inputs produce the same content hash.
Revalidation rejects missing, extra, changed, linked, or identity-raced content.

The final dataroot is `paths.output_dir/v1.0-trainval`. A successful staging tree is flushed and
renamed atomically from its sibling directory. `publication.version` must be
`v1.0-trainval`, `publication.refuse_overwrite` must be `true`, and `image.jpeg_quality` must be
`95` in the v1 contract. Existing destinations, symlinks, and unsafe directory replacements are
rejected rather than overwritten.

### Publication threat model and read-only contract

A published dataroot is a read-only artifact. Before final verification and rename, the publisher
sets every regular file to owner-read-only mode `0400` and every directory to owner-read/execute
mode `0500`. Supported consumers use `validate`, `inspect`, or the read-only `Dataset` SDK; there
is no supported in-place update operation. Changing a published file makes its content manifest
invalid. To change dataset content, run a new build into an absent final destination rather than
editing or replacing the existing tree.

Publication is designed to fail closed against accidental mutation and cooperative concurrent
writers. It pins directory and file identities with no-follow descriptors, rejects path, inode,
hard-link, and symlink substitution, and hashes the complete tree with descriptor use bounded by
tree depth while comparing each opened file before and after its read. It also rechecks the
authorized manifest around flushing and immediately after the final rename. A content or metadata
change observed during those checks prevents publication. If the post-rename check fails, the
publisher moves the same identity-bound tree out of the final name.
An existing final path is never overwritten.

These checks and read-only permissions are integrity safeguards, not a security boundary against
a non-cooperating same-UID actor. Such an actor can reverse the publisher's `chmod`, retain
writable descriptors, or ignore advisory `flock` coordination, and is outside the supported threat
model. If writers are not trusted to cooperate, run the pipeline under a dedicated UID with private
directories, or publish through filesystem snapshots or storage that enforces immutability
independently of the producing process.

Run validation and deterministic read-only inspection with:

```bash
dataset-devkit validate --dataroot DATASET --version v1.0-trainval
dataset-devkit inspect --dataroot DATASET --version v1.0-trainval
```

Inspection reports official table and image counts, normalized camera channels, recording and
train/test counts, final validation state, and the root content hash. It validates first and does
not manufacture summaries from malformed content.

## Read-only SDK

`Dataset(dataroot, version="v1.0-trainval")` accepts `str` or `Path`, loads every official table
and extension, and rejects malformed JSON, unsafe versions or filenames, missing tokens, and
duplicate tokens. Its stable methods are:

- `table(table_name)`, `get(table_name, token)`, and
  `field2token(table_name, field_name, query)`;
- `scene_samples(scene_token)` with cycle, endpoint, scene, and count checks;
- `camera(sample_token, "CAM_*")` and `ego_pose(sample_data_token)`;
- `validity(scene_token)`, `recording_validity(source_digest)`, `tags(scene_token)`, and
  `annotations(scene_token)`;
- `annotation_records()`, `annotation_matches()`, `annotation_windows()`,
  `annotation_scene_references()`, individual annotation resolvers, and
  `scene_annotation_evidence(scene_token)`;
- `split(scene_token)` and `scenes_in_split("train" | "test")`;
- `recordings()`, `validation_report()`, and `pipeline_audit()`.

Camera lookup uses a load-time validated `(sample_token, channel)` index, requires exactly one row,
and rejects missing, malformed, or ambiguous references. Every returned table, record, traversal,
or extension value is a defensive deep copy, so caller mutation cannot alter cached indexes or
subsequent queries.
