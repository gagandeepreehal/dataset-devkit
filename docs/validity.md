# Validity, sanity, quarantine, and publication gating

Task 4 consumes a real immutable `RecordingExtractionResult`; it does not reconstruct camera or
GNSS state from a toy policy input. `evaluate_validity(result, config)` returns deterministic grid
and logical-sample audits keyed by target timestamp, selected batch timestamp, and exact source
camera names/timestamps. One selected batch is one logical sample containing all per-camera image
candidates and poses. Extra cameras are preserved. A reason attached to any camera or pose
invalidates the complete logical sample when that reason's toggle is enabled. A grid miss has an
audit record but no extracted or final sample.

## Invalidity observations

Every condition that is observed is emitted, even when its invalidator toggle is disabled and even
when another condition already invalidated the sample. Each immutable observation has a stable
code and scope, measured values/raw details, the applicable threshold, target/batch/camera
timestamps, exact camera identity, and `enabled_as_invalidator`. A logical sample is valid exactly
when none of its observations is enabled. Recording-level report validity also accounts for grid
misses. Threshold equality is accepted; `*_exceeded` means strict `measured > threshold`.

| Code | Measurement and semantics |
| --- | --- |
| `gnss_source_invalid` | Interpolation is unavailable or either preserved endpoint/source has `is_valid == false`; availability, endpoint booleans/timestamps, and interpolation fraction are retained. |
| `position_sigma_exceeded` | Any interpolated `east_sigma_m`, `north_sigma_m`, or `up_sigma_m` strictly exceeds `gnss.position_sigma_max_m`; all available axes are reported. |
| `orientation_variance_exceeded` | The conservative maximum across every finite numeric `orientation_error` field whose name contains `variance` strictly exceeds `gnss.orientation_variance_max`; interpolated and both raw endpoint mappings remain in details. |
| `gnss_sync_gap_exceeded` | The maximum of the preserved before/after bracket gaps strictly exceeds `gnss.sync_gap_max_ms`; each gap and the maximum are reported in nanoseconds. |
| `camera_timestamp_non_monotonic` | For one exact camera identity across the recording, current minus previous real camera timestamp is zero or negative. Previous/current/delta are reported. Batch/grid times are not substituted. |
| `camera_timestamp_gap_exceeded` | That per-camera real timestamp delta strictly exceeds `frame_validity.camera_timestamp_gap_max_ms`. |
| `missing_required_camera` | A configured exact `required_cameras` identity is absent from the selected logical sample. Present, required, and missing identities are reported; extras stay in the sample. |
| `grid_miss` | A target grid timestamp has no batch inside extraction tolerance. The target remains auditable although no sample exists. |

All eight names are explicit fields below `frame_validity.invalidate_on`; unknown names are config
errors. GNSS thresholds are finite and nonnegative, the camera gap threshold is finite and
positive, and required camera names are nonblank, unique, traversal-safe single segments. Camera
matching is exact and case-sensitive.

## Retain and drop

`invalid_sample_policy` is exactly `retain_for_audit` (the default) or `drop`. Valid logical
samples are final candidates in both modes. With `retain_for_audit`, invalid logical samples and
their staged images remain only in `audit_only_samples`; they never enter `final_candidates`.
With `drop`, invalid logical sample objects are omitted from that collection while all reason/grid
records and counts remain. Each owned staged JPEG is removed only after its direct invocation
parent, regular-file type, single-link count, and stored device/inode identity re-verify. Changed,
linked, prior, external, cache, source-MCAP, and other-invocation files are never deleted.

## Nonstructural sanity

Each typed check has an independent `error`, `warn`, or `off` policy. `off` skips evaluation and
emission; `warn` returns a structured code/message/details/scope observation and continues;
`error` evaluates the configured checks then raises `NonstructuralSanityError` for coordinator
quarantine. The implemented checks are:

- `empty_selected_grid`: extraction selected no camera batches;
- `empty_final_candidates`: validity produced no publishable logical samples;
- `all_gnss_sources_invalid`: every source GNSS message is marked invalid;
- `zero_required_camera_coverage`: no configured required camera appears in any selected sample.

These checks are nonstructural warnings about usable coverage. They do not downgrade malformed
MCAP/protobuf/descriptors, impossible arrays/calibration/schema, corrupt or undecodable video,
decoder frame-count mismatch, unsafe/unverifiable JPEG staging, invalid timestamps/nonfinite poses,
or inconsistent internal/final references. Those always raise `StructuralExtractionError`.

## Quarantine report contract

Every coordinator failure is classified as `structural`, `sanity`, or `unexpected` and gets a
canonical JSON report with `schema_version: "1.0"`, recording/source identity, quarantined status,
category, exception type/message, stage, deterministic source details, validity/audit context,
available source/extraction config hashes, and `artifact_handling`. Deterministic content has no
wall-clock field. Runtime logging may add time separately, but it is not part of this report.

Reports use collision-resistant exclusive leaf creation below an absolute no-follow directory
chain, never overwrite, and are reopened to verify their same single-link inode and exact bytes.
Concurrent identical failures therefore have separate paths with identical canonical content.
Failed-invocation artifacts, when available, are reported as preserved in place; no acquisition
cache object or source MCAP is moved or modified. A report is still written when no artifacts
exist.

## Independent recordings and the partial-export gate

`RecordingCoordinator` rejects an empty request set, unsafe identities, and duplicate recording
identities before work begins. It processes every accepted recording independently and returns
outcomes, successes, failures, policy reports, and quarantine artifacts in input order. This task
does not publish.

After all recordings finish, any failure with `allow_partial_export=false` raises
`PublicationBlockedError`. The exception carries every success/failure but explicitly authorizes
zero recording identities. With `allow_partial_export=true`, the returned result authorizes only
successful recording identities; failed recordings remain quarantined and unauthorized. With no
failures, every successful input is authorized. Scene/tag/split construction and nuScenes export
or publication are outside this boundary.
