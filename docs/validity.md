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
| `position_sigma_exceeded` | Any interpolated `east_sigma_m`, `north_sigma_m`, or `up_sigma_m` strictly exceeds `gnss.position_sigma_max_m`; all interpolated axes plus both raw endpoint uncertainty mappings, fraction, timestamps, and endpoint gaps are reported. |
| `orientation_variance_exceeded` | The conservative maximum across all finite numeric leaves in interpolated `orientation_error` strictly exceeds `gnss.orientation_variance_max`; stable flattened paths, the deterministic maximum path, incompatible/uninterpolated paths, both raw endpoint mappings, fraction, timestamps, and endpoint gaps remain in details. |
| `gnss_sync_gap_exceeded` | The maximum of the preserved before/after bracket gaps strictly exceeds `gnss.sync_gap_max_ms`; each gap and the maximum are reported in nanoseconds. |
| `camera_timestamp_non_monotonic` | For one exact camera identity across the recording, current minus previous real camera timestamp is zero or negative. Previous/current/delta are reported for selected and unselected batches. Batch/grid times are not substituted. |
| `camera_timestamp_gap_exceeded` | That per-camera real timestamp delta strictly exceeds `frame_validity.camera_timestamp_gap_max_ms`. |
| `missing_required_camera` | A configured exact `required_cameras` identity is absent from the selected logical sample. Present, required, and missing identities are reported; extras stay in the sample. |
| `grid_miss` | A target grid timestamp has no batch inside extraction tolerance. The target remains auditable although no sample exists. |

All eight names are explicit fields below `frame_validity.invalidate_on`; unknown names are config
errors. GNSS thresholds are finite and nonnegative, the camera gap threshold is finite and
positive, and required camera names are nonblank, unique, traversal-safe single segments. Camera
matching is exact and case-sensitive.

Orientation numeric leaves are traversed recursively through mappings and repeated sequences.
Canonical protobuf-JSON `int64`/`uint64` integer strings count as numeric; booleans and arbitrary
prose/decimal strings do not. The measured mapping uses sorted dotted mapping paths and bracketed
sequence indexes. If multiple leaves share the maximum, the lexicographically first flattened path
is reported. Equality with the configured threshold remains accepted, and disabling the invalidator
changes only sample validity—the observation is still emitted when the threshold is exceeded.

## Retain and drop

`invalid_sample_policy` is exactly `retain_for_audit` (the default) or `drop`. Valid logical
samples are final candidates in both modes. With `retain_for_audit`, invalid logical samples and
their staged images remain only in `audit_only_samples`; they never enter `final_candidates`.
With `drop`, invalid logical sample objects are omitted from that collection while all reason/grid
records and counts remain. Each owned staged JPEG is removed only after its direct invocation
parent, regular-file type, single-link count, and stored device/inode identity re-verify. Changed,
linked, prior, external, cache, source-MCAP, and other-invocation files are never deleted.
Cached JPEGs are not working images: cache reuse first materializes a unique owned invocation, so
`drop` cannot mutate the immutable cache generation or race another build's validity pass.

Multi-image drop is transactional. After one complete preflight, each owned image is hard-linked
to an exclusive UUID tombstone in the same trusted directory, verified as the same two-link inode,
then unlinked at its original name and reverified as a single-link tombstone. A racing destination
therefore causes no overwrite. A link, source-unlink, identity, or commit-directory-fsync failure
rolls every prepared image back before a structural error is raised, so original sample names are
not partially deleted. Only after all tombstone names are durably committed are they unlinked.

A post-commit cleanup failure raises a dedicated retryable structural error carrying immutable
records for every remaining tombstone: trusted invocation root and full ancestor identity chain,
relative tombstone/original names, device/inode, and expected regular single-link state. The retry
helper component-wise reopens that chain without following symlinks, verifies both `lstat` and
opened-file identity, deletes only exact matches, fsyncs changed directories, and returns explicit
cleaned/remaining/mismatched state. Replaced, hard-linked, symlinked, or ancestor-moved entries are
never deleted. Coordinator artifact detection uses the same record verification and reports the
records rather than trusting path existence.

## Nonstructural sanity

Each typed check has an independent `error`, `warn`, or `off` policy. `off` skips evaluation and
emission; `warn` returns a structured code/message/details/scope observation and continues;
`error` evaluates the configured checks then raises `NonstructuralSanityError` for coordinator
quarantine. The implemented checks are:

- `empty_selected_grid`: extraction selected no camera batches;
- `empty_final_candidates`: validity produced no publishable logical samples;
- `all_gnss_sources_invalid`: every source GNSS message is marked invalid;
- `zero_required_camera_coverage`: no configured required camera appears in any selected sample.

Required-camera coverage uses the immutable extraction/source sample presence, before retain/drop
filtering, so changing audit retention cannot create a false zero-coverage finding.

These checks are nonstructural warnings about usable coverage. They do not downgrade malformed
MCAP/protobuf/descriptors, impossible arrays/calibration/schema, corrupt or undecodable video,
decoder frame-count mismatch, unsafe/unverifiable JPEG staging, invalid timestamps/nonfinite poses,
or inconsistent internal/final references. All present scalar, repeated, and nested
`orientation_error` numeric values must also be finite, and nonnumeric uncertainty descriptor
fields are rejected. These conditions always raise `StructuralExtractionError`.

## Quarantine report contract

Every coordinator failure is classified as `structural`, `sanity`, or `unexpected` and gets a
canonical JSON report with `schema_version: "1.0"`, recording/source identity, quarantined status,
category, exception type/message, stage, deterministic source details, complete validity/audit
context (measurements, thresholds, raw details, timestamps, camera, and enabled flag),
available source/extraction config hashes, and `artifact_handling`. Deterministic content has no
wall-clock field. Runtime logging may add time separately, but it is not part of this report.

Reports use a collision-resistant, exclusively created private temporary inode below an absolute
no-follow directory chain. Canonical bytes are written, file-fsynced, reread, and verified before a
no-replace hard link atomically exposes the final name; the private name is then unlinked and the
directory is fsynced. Every newly created directory component fsyncs its parent immediately. Final
names therefore never expose partial bytes or overwrite collisions. Concurrent identical failures
have separate paths with identical canonical content.
Failed-invocation artifacts, when available, are reported as preserved in place; no acquisition
cache object or source MCAP is moved or modified. A report is still written when no artifacts
exist.

## Independent recordings and the partial-export gate

`RecordingCoordinator` rejects an empty request set, unsafe identities, and duplicate recording
identities before work begins. It processes every accepted recording independently and returns
outcomes, successes, failures, policy reports, and quarantine artifacts in input order. This task
does not publish.

Quarantine persistence is isolated from the original recording exception. A persistence failure
cannot abort later recordings or replace the original category/type/message. `RecordingFailure`
records whether quarantine persisted, its optional final report path, and an explicit persistence
error type/message/details. Any incomplete quarantine blocks all publication authorization—even
when partial export was requested—and the aggregate blocked result retains successes plus both
failure layers. Nested report details are defensively deep-frozen before canonical serialization.
Exception messages are formatted defensively; an exception with a broken `__str__` retains its
original type and receives a deterministic unprintable-message marker without aborting later work.

After all recordings finish, any failure with `allow_partial_export=false` raises
`PublicationBlockedError`. The exception carries every success/failure but explicitly authorizes
zero recording identities. With `allow_partial_export=true`, the returned result authorizes only
successful recording identities when every failure report persisted and `cleanup_complete` is
`true`; failed recordings remain quarantined and unauthorized. Incomplete quarantine persistence
or any cleanup debt forces the blocked exception and zero authorization regardless of that flag.
With no failures and `cleanup_complete=true`, every successful input is authorized. Task 5 scene
construction consumes only the successful recording's
`final_candidates`; tag/split construction and nuScenes export or publication remain outside this
boundary. The production boundary immediately validates that every enabled observation
was enforced: invalid logical samples and grid misses cannot enter final candidates, enabled
recording-scope observations quarantine the source, and `retain_for_audit` versus `drop` changes
only invalid-sample evidence/image retention rather than quarantining a recording whose invalid
frames were correctly excluded.

Independent failure handling continues after scene construction. Feature computation and the
per-recording export preflight (owned JPEG identity/hash/dimensions, calibration, pose, timestamp,
and Task 8 structure) quarantine only the attributable source. The default/partial authorization
gate is rerun after those stages; global selection, relational validation, and publication failures
remain whole-build failures.

Published `mz_extensions/validity.json` is a versioned object with `recordings` and `scenes`.
Every exported source retains its complete observation list, grid audits, sample audits,
audit-only/final-candidate identities, configured policy, and report validity. Scene rows retain the
per-scene aggregates and official sample references. No observed invalidity reason is discarded.
