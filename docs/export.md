# nuScenes export and Dataset SDK

`export_dataset(staging_dataroot, evidence)` consumes immutable Task 5 through Task 7 evidence.
It recomputes scenario selection, scene splits, and every recording scene graph before writing.
The resolved global scenario/split configuration and dataset namespace must match that evidence.
The destination must be absent or an empty real directory; existing content is never
overwritten. Task 9 owns sibling staging, validation, and atomic publication.

## Official layout

The exporter writes all tables expected by the official nuScenes loader under
`v1.0-trainval`: `category`, `attribute`, `visibility`, `instance`, `sensor`,
`calibrated_sensor`, `ego_pose`, `log`, `scene`, `sample`, `sample_data`,
`sample_annotation`, and `map`. Camera-backed tables contain real selected evidence. Unsupported
object, lidar, radar, and map tables are valid empty JSON arrays; human labels are never converted
into fabricated object annotations.

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
`content_manifest.json`. Computed tags and human labels stay separate. Before Task 9 runs,
`validation.json` truthfully records `state: "not_run"` and never claims success.

## Read-only SDK

`Dataset(dataroot, version="v1.0-trainval")` accepts `str` or `Path`, loads every official table
and extension, and rejects malformed JSON, unsafe versions or filenames, missing tokens, and
duplicate tokens. Its stable methods are:

- `table(table_name)`, `get(table_name, token)`, and
  `field2token(table_name, field_name, query)`;
- `scene_samples(scene_token)` with cycle, endpoint, scene, and count checks;
- `camera(sample_token, "CAM_*")` and `ego_pose(sample_data_token)`;
- `validity(scene_token)`, `tags(scene_token)`, and `annotations(scene_token)`;
- `split(scene_token)` and `scenes_in_split("train" | "test")`;
- `recordings()` and `validation_report()`.

Camera lookup requires exactly one row for the requested sample/channel and rejects missing or
ambiguous evidence.
