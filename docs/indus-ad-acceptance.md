# INDUS-AD implementation acceptance, 2026-10-15

The feature branch `feat/indus-ad-integrated-annotations` implements the additive
schema 2.1 integrated backend. It includes pinned contract verification, strict
weak-label parsing, selective acquisition, exact camera-frame checks, selected
scene joins, dense member verification, Parquet extensions, immutable SDK records,
extension validation, inspection counts, and a metadata compatibility checker.

The source contract used for acceptance is `gagandeepreehal/indus-ad` at
`2d46a81376ded440506bd1646c0a5197589ba27b`.

The authenticated metadata gate passed after fetching the pinned repository tree
and independently verifying the cached controls against its file identities:

| Evidence | Count |
| --- | ---: |
| Recordings | 1,469 |
| Videos marked face_plate_blurred | 9,927 |
| Missing camera slots | 356 |
| Camera annotation rows | 34,306 |

Annotation schema SHA-256:
`86e15f0cac34681647496d726bcd984163b0219e12d70b1c63c2ebbee4861025`.
No production videos, semantic/depth archives, or sensor tables were downloaded by
this gate. The final run reused verified controls, so its downloaded-path list was
empty. Source tree identities were fetched from Hugging Face during that run.

Local verification includes the complete regression suite (994 passing tests),
Ruff, strict source mypy, wheel/source builds, and whitespace checks. A tiny local
integrated recording exercises acquisition through atomic publication and manifest
validation. Mutation tests cover provenance, revision, missing token references,
and forbidden schema additions. Published weather label vocabulary was checked
against the live camera annotation table.

The wheel was installed in a separate temporary virtual environment with declared
dependencies, and the installed command's help entry point succeeded. Package
version remains the existing 0.1.0; this work does not publish a package release.

Production sensor extraction has not been run in this acceptance session. The
optional pinned GNSS companion path is covered by synthetic fixtures; no production
companion repository is configured in the example. Dense acquisition failures
block publication, including builds configured to allow other partial recording
failures. Dense export rows retain source references and materialization state;
dense payload files remain in the verified cache.

The checked-in design and implementation plan are historical planning documents.
Use this acceptance note and `docs/indus-ad.md` for the implemented behavior and its
verification limits. Source repositories and the user's main checkout were not
published or replaced by this work.
