# Performance capture analysis

Offline readers and regression checks for structured F8 captures. These tools analyze recorded
observations; they do not change guest execution or run games. Runtime collection and schema live
in `frontends/shared/perf/`.

`perf_caller_report.py` separately reads strict native `perf script --no-inline` exports. It reports
libc family weights, named recorded application ancestors, leaf-only/unresolved/suspicious stacks,
and optional raw-record loss markers. Application and libc DSO selectors must be exact and disjoint;
unobserved selectors are not proof of absent work. Event/thread denominators remain separate.
See `PERF_CALLER_REPORT.md` for commands and limits. A valid parse does not establish build identity,
complete unwinding, a complete recording, worker coverage or low profiler overhead.

Keep missing measurements distinct from measured zeros, and retain signed timing residuals.
Nested CPU timers and GPU intervals cannot be added as independent work. Buffer comparison byte
counts are requested spans, not physical memory traffic or the bytes an early-exiting comparison
actually reads. Guest flips and repeated host presentations are not counts of newly rendered frames.

The optional buffer-write-watch population is separate from the older residency fields. Missing
watch fields must not erase valid older residency observations. Watch time is already included in
resident time, and watch-proven bytes are already included in reused bytes.

`compute_buffer_cache_report.py <runtime.log>` summarizes the opt-in full compute buffer-cache
census, including owners not bound by selected dispatches. It requires complete snapshots and
matching charge totals; malformed, truncated, duplicate or reset-context evidence fails visibly.
Full materialization keys distinguish owners. Observed last-use changes and access-clock ages
are neither elapsed time nor future reuse predictions. A resident owner with no observed accesses
is a candidate for investigation, not proof that evicting it is profitable.
