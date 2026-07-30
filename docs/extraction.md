# Native MCAP extraction contract

`dataset_devkit.extraction.RecordingExtractor` is the one-recording boundary between verified
acquisition and later validity/scene/export policy. It reads a local MCAP, stages selected camera
images, and returns immutable typed records. It does not decide whether timestamp gaps, GNSS
quality, or individual samples are acceptable.

## Required source streams

The configured camera and GNSS topic names are exact. Both channels must use `protobuf` message
encoding and MCAP schemas whose `encoding` is `protobuf`; schema data must be a valid serialized
`google.protobuf.FileDescriptorSet`. Classes are built dynamically, with descriptor dependencies
resolved before dependants and `google.protobuf.Timestamp` supplied as a well-known dependency.
No generated source files are required.

The camera schema name is exactly `autonome.CompressedVideos`. The extractor requires exact
`format == "h265"`, positive shared dimensions, a positive `number_of_cameras`, and index-aligned
`data`, `name`, `camera_timestamp`, `camera_intrinsic`, and `camera_extrinsic` arrays. Names are
unique and nonblank; dimensions, index identity, and calibration remain stable for the recording.
Each payload is one Annex-B HEVC access unit containing a VCL NAL unit. The indexed
`camera_timestamp` is mandatory: it is never replaced by the batch or grid timestamp.

The GNSS protobuf type name may vary. Its descriptor must expose `timestamp`, `rec_timestamp`,
`is_valid`, `lat_lon_ht`, `orientation`, `position_error`, and `orientation_error` with the required
field shape. GNSS is indexed by protobuf `timestamp`; duplicate timestamps are structurally
ambiguous. Unknown top-level identifiers and dynamically shaped orientation uncertainty remain
serializable in the result.

## Selection, decode, and pose output

The target grid is anchored at the first camera batch `rec_timestamp`. Its period is rational,
derived without accumulated floating-point drift. The nearest unused batch within tolerance wins;
an exact distance tie chooses the earlier batch. Misses, signed/absolute error, and unused batches
are returned.

All camera access units are fed in MCAP log-time order through exactly one persistent PyAV HEVC
decoder context per camera index, including unselected batches needed as inter-frame references.
Every access unit must produce exactly one frame. Selected frames are converted to RGB, atomically
written as JPEG quality 95 under the recording staging directory, and independently reopened with
Pillow to verify JPEG format, RGB mode, complete decode, and dimensions. The POSIX writer rejects
unsafe recording identifiers, symlink directories/targets, and hard-linked overwrite targets.

Every staged candidate keeps three distinct times: grid target, chosen batch `rec_timestamp`, and
its real camera timestamp. Ego poses are keyed by the real camera timestamp. GNSS values are
bracketed without extrapolation; outside-range results are explicit and unavailable. Available
poses linearly interpolate geodetic position/height and numeric uncertainty, use shortest-path
quaternion SLERP in documented `(w, x, y, z)` order, and project longitude/latitude from EPSG:4326
to EPSG:3857 with `always_xy=True`. Raw geodetic endpoints, source validity, uncertainties,
fraction, and both synchronization gaps remain available for audit and later policy.

## Structural failures

`StructuralExtractionError` stops the recording for malformed MCAP/protobuf data, unresolved
descriptors, absent required streams, wrong schema/format, invalid protobuf timestamps, impossible
or changing camera arrays/calibration, corrupt HEVC, decoder frame-count violations, unsafe JPEG
staging, duplicate grid batch times, or duplicate GNSS times. Decoder contexts are closed on both
success and failure.

Backward or gapped but otherwise valid camera timestamps are not a structural failure. The result
contains per-stream timestamp deltas so the later validity stage can apply the configured policy.
