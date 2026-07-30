# Deterministic scenes and annotation windows

Task 5 consumes one successful recording policy result at a time. Its only sample input is
`ValidityReport.final_candidates`. Invalid retained audit samples, dropped invalid samples, and
grid misses remain in the Task 4 report and can never become scenes. The builder also takes the
exact Azure `SourceFingerprint`; local MCAP and staging paths are preserved as references but are
never token inputs.

## Configuration and integer time

`scenes.mode` is exactly `automatic`, `annotation_only`, or `hybrid` (the default). The scene
section also requires a stable `dataset_namespace` UUID, positive minimum/maximum durations,
positive minimum sample count, nonnegative maximum sample gap, and nonnegative inter-scene skip.
`min_duration_s` cannot exceed `max_duration_s`. The annotation section contains the safely
resolved JSONL `path`, nonnegative nearest-match tolerance, and nonnegative before/after windows.
`load_config` is authoritative: it parses Task 5 time literals from their exact decimal JSON text
and converts them to exact integer nanoseconds. Scientific notation is supported. Values just
below or above an integer nanosecond are rejected rather than rounded through a binary float.
Other configuration float fields retain their existing strict-float behavior. Direct
`GlobalConfig.model_validate` callers must supply `Decimal` instances for Task 5 time fields;
JSON numeric strings are not accepted, and the generated schema exposes numbers only.

## Valid runs and automatic segmentation

Final valid logical samples must already be strictly ordered and unique by grid timestamp. A
consecutive gap strictly greater than `max_sample_gap_ms` starts a new valid run; equality remains
inside the run. Within each run the automatic algorithm is deliberately greedy:

1. Start at the first eligible sample.
2. Append samples while `timestamp - start <= max_duration`; equality is included.
3. Finalize immediately before the first sample that would exceed the maximum.
4. Keep the candidate only when both its duration and sample count meet their inclusive minima.
   Every sample in a rejected candidate is audited as `candidate_too_short`; it is consumed and
   never rebalanced into a neighbor.
5. After a kept scene, consume samples as `inter_scene_skip` until
   `next_timestamp - kept_end >= skip`; equality is eligible. Then repeat.

This also defines leftover behavior: a final short candidate is rejected intact. No balancing,
overlap, or sample reuse occurs.

## Annotation JSONL

The file is UTF-8 JSONL. Blank lines and lines whose first non-whitespace character is `#` are
ignored. Every other line is one object with exactly these keys:

```json
{"blob_path":"mcap-h265/fleet/run.mcap","timestamp_ns":1720000000000000000,"labels":["turn","rain"]}
```

`blob_path` must be an exact validated `mcap-h265/...mcap` container-relative path; basename and
prefix matching are never used. `timestamp_ns` is a nonnegative JSON integer (not a boolean,
float, or string). `labels` is a nonempty array of unique, trimmed, nonblank strings. Unknown or
duplicate JSON keys and duplicate `(blob_path, timestamp_ns, labels)` records are line-numbered
errors.

Parsing is binary streaming and never loads the complete file. Stable inclusive defaults bound
the file to 64 MiB, each physical line payload to 256 KiB, records to 250,000, labels per record
to 256, each label to 256 Unicode characters/1,024 UTF-8 bytes, and each blob path to 2,048
characters/4,096 bytes. The line limit excludes either an LF or CRLF terminator consistently; the
total-file limit includes every terminator byte. Exceeding any bound is a line-numbered format
error.

Every parsed record receives an audit. Records for another exact blob are
`different_recording`. For the current blob, matching uses the nearest final valid logical
sample. An exact distance tie chooses the earlier sample. Tolerance equality matches; a larger
distance is `outside_tolerance`. Audits preserve the nearest sample timestamp, signed
`sample - annotation` error, and absolute error only for a match. Every unmatched reason has null
sample/error fields, so it cannot accidentally claim an out-of-contract sample reference.

A matched logical sample is the stable window anchor. The requested before/after interval is
clipped to that sample's valid run, so it cannot cross a Task 4 invalid span, grid gap, configured
maximum gap, or recording boundary. Windows whose time boundaries overlap or touch are merged.
The merged window retains every source annotation token in line order and unions labels in first
appearance order. Annotation scenes may be shorter than automatic duration/sample minima.

## Modes

- `annotation_only` emits only merged annotation windows. Other final valid samples are
  `annotation_mode_excluded`.
- `automatic` parses and audits configured annotations, but annotation windows do not construct
  scenes and human labels are not attached to automatic scenes.
- `hybrid` constructs annotation scenes first. Their complete sample ranges are hard boundaries;
  automatic segmentation runs separately on each remaining range. Automatic scenes cannot
  overlap or bridge an annotation range.

## UUIDv5 graph and validation

Scene, sample, per-camera sample-data, source annotation, and merged-window tokens use UUIDv5
under `scenes.dataset_namespace`. Canonical identities include the exact source fingerprint/blob
path and the relevant kind, configuration, timestamps, annotation/window identity, camera
channel, and real camera timestamp/ordinal. They never include wall time, Python `hash()`, random
UUIDs, local directories, or mapping iteration order.

The result persists the complete immutable build contract: mode, minimum/maximum duration in
nanoseconds, minimum samples, valid-run gap, inter-scene skip, annotation tolerance/before/after,
namespace, and the versioned same-run overlap-or-touch merge rule. The build-config UUID and every
scene UUID bind all of those values. Validation re-runs the shared pure automatic partitioner over
the source timestamps (and hybrid annotation exclusions), then requires the exact scene order,
kind, partition, skip/short leftovers, and annotation-only exclusions.

Each scene records endpoints/count, timestamps, source, kind, human labels, annotation references,
and (for annotation scenes) its merged-window reference. Logical samples have symmetric
`prev`/`next` links within their one scene. Each
camera channel has a separate within-scene sample-data chain; its timestamp is the camera's real
timestamp, not the logical grid time. Sample data preserves the deterministic export filename,
staged image, calibration, and ego-pose reference. Each record also carries its camera index and
within-channel ordinal. Its UUID and exact `samples/<source-digest>/<channel>/<uuid>.jpg` POSIX
filename are recomputed during validation, and the staged camera name/index/timestamp must match
the record exactly. Filenames are globally unique. Chains never cross scenes. Human labels remain
separate; Task 6 computed tags are not present.

Every graph embeds immutable `source_samples` records derived from Task 4 final candidates. Each
record contains one logical timestamp, its canonical sorted nonempty expected camera-channel set,
the selected batch timestamp, and its valid-run identity. Sample data carries the complete immutable
staged-image evidence rather than only a local path. Validation binds each sample's batch timestamp
back to this source evidence, recomputes annotation UUIDs, and requires every scene's exact source
blob path.
`validate_scene_graph` checks globally unique tokens, foreign references, endpoint/count/order
consistency, symmetric acyclic chains, within-scene channel chains and real pose timestamps,
exactly one sample-data record per expected channel, no missing/extra/duplicate channel, unique
unassigned timestamps, complete disjoint assigned/unassigned source coverage, and
annotation match decisions, merged-window derivation, mode/config identity, safe POSIX export
filenames, and staged asset content/root identity. The builder runs it before returning and raises
`StructuralExtractionError` on any malformed graph.
