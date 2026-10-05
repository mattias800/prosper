# `progress` — static HLE coverage census

Static, no-dump counterpart to `nid_census` (which needs game modules) and `hle_calls`
(which needs a live process). Answers: "of the handlers prosper declares, how many are
real implementations vs deliberately-overridable placeholders?"

- `progress.py` — scans `prosper/src/hle/**/*.cpp`, classifies each `HLE(name)` body,
  groups by `hle/<area>/`, writes `progress.json`, prints text, diffs two JSON files.
- `README.md` — what done/todo mean and what this deliberately does not cover.
- `test_progress.py` — pytest self-tests with hand-built fixtures plus a positive control.

Deliberate non-goals for v1 (see README): corpus-unregistered NIDs (use `nid_census`),
live traffic (use `hle_calls`), shader ISA coverage, SVG badges. Each is a separate PR.
