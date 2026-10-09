# tools/saves

Local save fixtures: a saved copy of both prosper save roots (`PROSPER_SAVEDATA_DIR` and
`PROSPER_SAVE0`) taken after a title's forced first-boot steps (EULA, initial settings, language and
brightness prompts), so a test run can start from that state instead of re-navigating it.

`save_fixture.py` is the only tool: `capture`, `seed`, `list`, `verify`, `pin`. Its docstring has
the usage.

What belongs here: the tool, its tests, and `manifest.json` (optional, committed) which pins a
fixture's sha256 and content version so a route can name the fixture it needs without shipping it.

What never belongs here: the fixtures themselves. A save is derived from game content and this
repository is public. They live under `~/prosper-saves/<TITLE_ID>/<state>/`, or
`$PROSPER_SAVE_FIXTURES`. `PROSPER_SAVE_FIXTURES` and `PROSPER_SAVE_MANIFEST` are read by this tool
only; prosper itself never sees them.

Things a newcomer learns by getting them wrong:

- `seed` copies into FRESH per-run directories and refuses a non-empty one. Never point a run at the
  fixture directory itself, and never reuse a seeded run directory: a stale or mutated save has
  turned a whole run black and read as a render defect.
- Capture from a directory the run has finished writing, and record the route (`--route`) that
  produced the state, so the fixture can be rebuilt when the content version changes.
- A content-version mismatch warns loudly on stderr; the dump, not the fixture, is authoritative, so
  recapture rather than ignore it (`--strict-version` makes it an error).
- `tools/snapshot/snapshot.py` seeds from a snapshot entry's `save_fixture` field and SKIPS the
  guard when the fixture is absent locally. Do not make it fall back to a fresh save.
