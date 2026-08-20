# `diagnostics` — observation only

**Nothing in this folder is on a rendering path.** That is the folder's defining property: code here
may be removed, gated off or rate-limited without changing a single rendered pixel. If something here
starts affecting output, it belongs somewhere else.

- `diagnostic_selectors` — choosing what to observe.
- `diag_ratelimit` — rate limiting. **Check a diagnostic's rate limit before quoting its volume as a
  frequency**; several phantom findings came from reading a capped count as a real one.
- `watch_list`, `compute_tree_watch`, `compute_parent_walk` — watching addresses and walking compute
  parentage.

Two standing cautions, both learned expensively:

- **Prefer instruments that detect their own invalidity.** A diagnostic that can fail silently will,
  and its silence reads as evidence.
- **A count is only as good as its population.** Know whether a diagnostic fires per dispatch, once
  per program, or only on some sub-population — several published figures were off by two orders of
  magnitude because the answer was assumed rather than read.
