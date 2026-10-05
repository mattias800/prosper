# `progress` — static HLE coverage census

Static, no-dump counterpart to `nid_census` (which needs game modules) and `hle_calls` (which needs
a live process). It answers: "of the Sony NIDs prosper registers a handler for, how many get a real
handler and how many only a placeholder?"

- `progress.py` builds on the registration parser in `../re/hle_handler_map.py`; it does not keep
  regexes of its own for registration shapes. It counts per NID, groups by `src/hle/<area>/`, writes
  `progress.json`, prints text, and diffs two JSON files.
- `README.md` says what done/todo mean, how to cross-check the total, and what is not covered.
- `test_progress.py` holds hand-built fixtures in the shapes a regex census got wrong, plus a
  real-tree check against `hle_handler_map.py`. ctest runs it as `progress_census`.

If `hle_handler_map.py` learns a new registration shape, this census picks it up with no change.
A new non-literal NID form (one that is neither a table element nor an index_sequence fold) shows
up as an unresolved expression and exit 3. Handle that here, not by weakening the exit code.
