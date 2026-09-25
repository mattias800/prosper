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

`perf_f8_gap.py` joins a process-scoped monotonic `cpu-clock:u` export to complete F8
renderer-submit spans. It refuses partial interval overlap and mixed process IDs; perf loss and
peer contention remain external validity checks. Samples in a wall-time gap are CPU observations,
not an allocation of that gap's elapsed time. See `PERF_F8_GAP.md`.

`schedstat_probe.py` is an opt-in Linux `/proc` sampler for a selected process; `schedstat_f8.py`
joins one recorded TID to F8 spans using complete read brackets. It separates scheduler runtime,
runnable wait, and a signed sleep/unknown residual, with explicit coverage. The probe is a
measurement aid, not a game FPS benchmark; see `PERF_F8_GAP.md`.

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

`compare_capture_pair.py` checks routed F8 A/B run identities and balanced environment keys before
computing the mean of recurring-shape renderer medians. Its weights and matching coverage stay visible; they are
local recorded work, not equal-input proof or newly rendered FPS. See `COMPARE_CAPTURE_PAIR.md`.
