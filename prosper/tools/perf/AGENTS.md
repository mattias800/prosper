# Performance capture analysis

Offline readers and regression checks for structured F8 captures. These tools analyze recorded
observations; they do not change guest execution or run games. Runtime collection and schema live
in `frontends/shared/perf/`.

Keep missing measurements distinct from measured zeros, and retain signed timing residuals.
Nested CPU timers and GPU intervals cannot be added as independent work. Buffer comparison byte
counts are requested spans, not physical memory traffic or the bytes an early-exiting comparison
actually reads. Guest flips and repeated host presentations are not counts of newly rendered frames.

The optional buffer-write-watch population is separate from the older residency fields. Missing
watch fields must not erase valid older residency observations. Watch time is already included in
resident time, and watch-proven bytes are already included in reused bytes.
