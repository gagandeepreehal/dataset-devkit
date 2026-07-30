# Features, filtering, and scenario selection

Task 6 consumes only a validated `RecordingSceneResult`. It does not read extraction candidates,
change human annotations, split a dataset, or export nuScenes. Human labels remain in
`human_labels`; derived values remain in the sorted immutable `computed_tags` tuple.

## Real-time trajectory features

Each logical sample contributes exactly one trajectory observation. `reference_camera_channel`
with policy `require` demands that exact channel in every sample. Policy
`lexicographic_fallback` chooses the lexicographically first `(channel, camera_index)` and records
the channels used plus the fallback count. The observation uses that camera's integer nanosecond
timestamp and Web-Mercator ego-pose `(x, y)` coordinates. Logical-grid intervals and configured
FPS are never used as elapsed trajectory time.

Segment duration is the positive difference between consecutive real timestamps in seconds.
Distance is planar Euclidean Web-Mercator distance. `mean_speed_mps` is the arithmetic mean of the
segment speeds; median and maximum are calculated over the same segment population.
`time_weighted_speed_mps` is separately named and equals total distance divided by total real
duration. A one-sample scene has zero finite duration, distance, and speed. Zero-distance segments
have zero speed and curvature.

Quaternion input is normalized `(w, x, y, z)`. Yaw follows a right-handed ENU frame, so positive
shortest signed yaw change is left and negative change is right. Heading deltas are unwrapped into
`[-pi, pi]`; net change is the signed sum and total change is the sum of absolute deltas. Signed
curvature is heading delta divided by segment distance.

`stationary` means no segment exceeds `stationary_speed_mps` with the configured minimum movement;
otherwise the scene is `moving`. `stopping` and `starting` record moving-to-stationary and
stationary-to-moving segment transitions. Direction tags use the absolute net heading change:
straight is inclusive at its maximum, turn is inclusive at its minimum, and curvature occupies the
configured middle band. Straight, turn, and curvature tags are mutually exclusive. Coverage tags
are `camera_coverage_complete`/`camera_coverage_partial`; GNSS state tags are `gnss_valid`,
`gnss_partial`, or `gnss_invalid`.

Source-GNSS ratio, expected/present camera coverage, and synchronization error are computed from
the sealed Task 5 evidence. The sync population contains one grid error per source sample plus one
camera error per present camera record; it is independent of the trajectory reference channel.
Task 6 raises `StructuralExtractionError` for missing,
nonfinite, inconsistent, or non-increasing reference evidence; such failures are not filter
rejections.

## Declarative filters

An empty `filters` object accepts everything. Duration, scene-valid and source-GNSS-valid ratios,
overall/per-channel camera coverage, maximum synchronization error (milliseconds), distance, tags, human labels, and
exact scene-token/source-digest/blob-path blacklists are independent criteria. Every configured
criterion is evaluated, so one rejected scene may carry multiple stable `RejectionReason` records.
Each reason contains its code, measured value, operator, threshold or expected set, scene token,
source digest, and exact blob path. Accepted and rejected outputs preserve input order.

## Ordered exact-quota rules

Rules match computed tags and human labels separately with `required_any_*`, `required_all_*`, and
`excluded_*`; optional `filters` add metric constraints. Rules run in configuration order. Eligible,
currently unassigned scenes are ranked by SHA-256 over canonical JSON containing the integer seed,
rule index/name, scene token, and source fingerprint. Ranking does not use Python hashing, random
state, input order, filesystem paths, or time.

Each rule selects exactly `quota`. Under the default `strict_quotas: true`, a deficit raises
`ScenarioQuotaError` with eligible, selected, deficit, and partial deterministic rule audit. A zero
quota selects nothing. The result contains selected assignments in rule-order then rank-order,
candidate/rank/prior-rule audits, and an audit reason for every accepted scene not selected.
Reordering identical inputs produces the same canonical result.

Copy [`examples/scenario_templates.json`](../examples/scenario_templates.json) for the canonical
Straight, Stopping, Left Turn, Right Turn, Left Curvature, Right Curvature, and annotation-label
rules. Quotas are explicit zero placeholders and must be replaced for a selecting configuration.
