# Native MCAP extraction contract

`dataset_devkit.extraction.RecordingExtractor` is the one-recording boundary between verified
acquisition and validity/scene/export policy. It reads a local MCAP, stages selected camera images,
and returns immutable typed records. It does not decide whether timestamp gaps, GNSS quality, or
individual samples are acceptable; `evaluate_validity` applies those Task 4 policies afterward.

## Required source streams

The configured camera and GNSS topic names are exact. Both channels must use `protobuf` message
encoding and MCAP schemas whose `encoding` is `protobuf`; schema data must be a valid serialized
`google.protobuf.FileDescriptorSet`. Classes are built dynamically, with descriptor dependencies
resolved before dependants and `google.protobuf.Timestamp` supplied as a well-known dependency.
No generated source files are required.

The camera descriptor is checked before message decoding for exact field numbers, scalar/message
types, repeated-versus-singular cardinality, Timestamp types, and the real nested
`CompressedVideos.CameraIntrinsic` / `CompressedVideos.CameraExtrinsic` shapes. Intrinsic scalar
and dimension fields and extrinsic vectors are protobuf `float`; distortion coefficients are
`double`. A same-named but wire-incompatible lookalike schema is a structural failure.

The camera schema name is exactly `autonome.CompressedVideos`. The extractor requires exact
`format == "h265"`, positive shared dimensions, a positive `number_of_cameras`, and index-aligned
`data`, `name`, `camera_timestamp`, `camera_intrinsic`, and `camera_extrinsic` arrays. Names are
unique and nonblank; dimensions, index identity, and calibration remain stable for the recording.
Calibration stability uses exact decoded protobuf-value equality; extraction applies no numeric
tolerance to intrinsic or extrinsic changes. Each payload is one Annex-B HEVC access unit with no
bytes before its first start code, no empty NAL units, valid two-byte HEVC headers, and at least one
VCL NAL unit. The indexed
`camera_timestamp` is mandatory: it is never replaced by the batch or grid timestamp.

The GNSS protobuf type name may vary. Its descriptor must expose `timestamp`, `rec_timestamp`,
`is_valid`, `lat_lon_ht`, `orientation`, `position_error`, and `orientation_error` with the required
field shape. GNSS is indexed by protobuf `timestamp`; duplicate timestamps are structurally
ambiguous. Unknown top-level identifiers and dynamically shaped numeric orientation uncertainty
remain serializable in the result. GNSS scalar and nested descriptor types are validated before
payload decoding. Orientation-uncertainty descriptors may contain scalar, repeated, or nested
numeric fields; nonnumeric/recursive shapes and any present nonfinite numeric value are structural
failures. Every message must contain both timestamps and all four required nested messages; a
nested numeric value is also required when its protobuf descriptor exposes field presence.

All uncertainty descriptor, protobuf-value, interpolation, immutable-copy, and policy traversals
share one deterministic bound: maximum container depth 32, 10,000 visited nodes, 8,000 scalar
leaves, and 200,000 path/work units. Python container cycles and exceeded limits are structural
failures with the stable offending flattened path and reason; they cannot escape as a
`RecursionError` or unbounded walk. Boundary-depth structures remain accepted.

## Selection, decode, and pose output

The target grid is anchored at the first camera batch `rec_timestamp`. Its period is rational,
derived without accumulated floating-point drift. Periods below one nanosecond (`target_fps >
1e9`) are rejected. Rounded targets must increase strictly, and extraction refuses to materialize
more than 10,000,000 targets; lower `target_fps` or split the recording if that safety limit is
reached. Candidates are sorted/indexed once. The nearest unused batch within tolerance wins; an
exact distance tie chooses the earlier batch. Misses, signed/absolute error, and unused batches are
returned.

Extraction is two-pass and bounded with respect to compressed camera data. Pass one incrementally
indexes schemas, compact camera metadata, timestamps, and GNSS while discarding each compressed
payload after structural validation. Pass two reopens the local MCAP and streams access units; the
source device, inode, size, modification time, and change time must remain identical before,
during, and after the pass. Neither the index nor `RecordingExtractionResult` contains H.265 bytes.

All camera access units are fed in MCAP log-time order through exactly one persistent PyAV HEVC
decoder context per camera index, including unselected batches needed as inter-frame references.
Every submitted packet receives a unique deterministic PTS and nanosecond time base. Decode calls
may return zero, one, or multiple frames; output PTS associates each frame with its originating
access unit even when the codec delays or reorders output. Pending association state contains only
compact camera/batch/grid metadata; it never retains a `CameraAccessUnit`, payload bytes, or a view
of those bytes. Every decoder is flushed at EOF, and the recording fails for an unknown, duplicate,
ambiguous, or missing final output. Only outputs whose originating batches were selected are staged.

Each extraction invocation exclusively creates a UUID-suffixed subdirectory below the trusted
staging root. Deterministic leaf names include batch ordinal, camera index, sanitized camera name,
and actual camera timestamp, so duplicate timestamps cannot collide. Selected frames are converted
to RGB, atomically linked without clobber as JPEG quality 95, and independently reopened with
Pillow to verify JPEG format, RGB mode, complete decode, and dimensions. The POSIX writer rejects
unsafe recording identifiers, symlink directories/targets, and hard-linked overwrite targets.
Every staging ancestor is traversed relative to no-follow directory descriptors and its device/
inode identity is rechecked around publication and verification. Verification reads through the
trusted directory descriptor, requires the exact encoded byte sequence and digest, then decodes
those same bytes with Pillow. Rollback removes only files whose stored device/inode identities are
still owned by that invocation and then removes only the still-identical empty invocation directory;
prior, concurrent, or substituted paths are never unlinked by name alone. Successful results expose
the invocation root and staged paths for later publication.

Every staged candidate keeps three distinct times: grid target, chosen batch `rec_timestamp`, and
its real camera timestamp. Ego poses are keyed by the real camera timestamp. GNSS values are
bracketed without extrapolation; outside-range results are explicit and unavailable. Available
poses linearly interpolate geodetic position/height and numeric uncertainty. Uncertainty
interpolation recursively preserves compatible mappings and equal-length sequences. Finite numeric
scalars and canonical protobuf-JSON `int64`/`uint64` integer strings are interpolated; booleans,
decimal/prose strings, and other nonnumeric leaves are not treated as numbers. An incompatible
mapping branch or sequence length is endpoint-only: its numeric leaves are omitted from the
interpolated mapping, listed by deterministic flattened path, and remain available in both raw
endpoint mappings. GNSS roll, pitch, and yaw use a right-handed Cartesian frame and active
body-to-world rotation. They are applied as
fixed-axis roll about +X, then pitch about +Y, then yaw about +Z, so composition is
`qz * qy * qx`; the quaternion is stored in `(w, x, y, z)` order. Endpoint attitudes use
shortest-path quaternion SLERP. Longitude/latitude are projected from EPSG:4326 to EPSG:3857 with
`always_xy=True`. Raw geodetic endpoints, source validity, uncertainties, interpolation fraction,
uninterpolated numeric paths, and both synchronization gaps remain available for audit and later
policy. Result mappings are defensive, read-only copies; nested mapping/list/set values are
recursively frozen.

Before validity policy runs, selected entries and misses must form one auditable target partition:
target timestamps are unique, disjoint, and ordered within their streams; selected batches are
unique source batches; and the unique unused-batch tuple exactly equals every source batch not
selected. Every selected entry's signed error must equal `batch_timestamp_ns -
target_timestamp_ns`, and its absolute error must equal the magnitude of that signed error.
Contradictions are structural failures. Combined selected/miss audit output is always sorted by
target timestamp.

## Structural failures

`StructuralExtractionError` stops the recording for malformed MCAP/protobuf data, unresolved
descriptors, absent required streams, wrong schema/format, invalid protobuf timestamps, impossible
or changing camera arrays/calibration, corrupt HEVC, decoder frame-count violations, unsafe JPEG
staging, duplicate grid batch times, or duplicate GNSS times. Decoder contexts are closed on both
success and failure.
If decoding fails after earlier frames were staged, this extraction invocation rolls back only its
inode-bound files and directory before propagating the failure.

Backward or gapped but otherwise valid camera timestamps are not a structural failure. The result
contains per-stream timestamp deltas so the validity stage can apply the configured policy.
