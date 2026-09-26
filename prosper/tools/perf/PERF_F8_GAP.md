# Join perf CPU samples to F8 renderer spans

`perf_f8_gap.py` answers which **CPU samples** occurred inside completed renderer
semantic-submit spans and which occurred between them. It does not assign wall time
to a symbol. Parallel worker samples add CPU time, not critical-path latency, and a
between-span sample can be guest work, scheduling, or frontend work. Use it to pick a
source trace to investigate, then measure a change in a matched game route.

Record `cpu-clock:u` for **one game process** with `-k CLOCK_MONOTONIC` so perf and
F8 use the same timestamp basis. Trigger F8 while recording; the perf recording must
cover the entire five-second F8 detail window. Export the leaf IP even when stack
unwinding fails:

```sh
perf script -i perf.data -G --no-inline -F comm,pid,tid,time,period,event,ip,sym,dso > perf-flat.txt
python3 tools/perf/perf_f8_gap.py capture.prperf perf-flat.txt > gap.json
```

The parser refuses incomplete or dropped F8 detail, missing/overlapping renderer
spans, multiple recorded processes, perf samples that do not cover both ends of
the F8 span interval (within 0.5 s), a zone pair where either comparison zone has no
samples at all, and a recording in which **no** sample carries a named frame from the
selected binary. It keeps samples outside the interval separate.

Application frames are selected by DSO **basename**, `prosper-app` by default. A recording
made under another frontend needs `--binary screenshot` or `--binary boot_trace`; without it
every sample would land in `no_named_app_frame` and all three `top_app_frames` lists would be
empty, which reads as "no application work" rather than "wrong binary name" — hence the
refusal above. The name actually used is echoed back as `app_binary`. Renderer spans that
**touch** (one submit span ending exactly where the next begins) are accepted: that is an
observation about the capture, and it leaves the gap zone with zero expected wall time.
Inspect **perf recording loss**, the exact game binary and route, and concurrent
build/game processes independently; neither this reader nor a successful `perf
script` export establishes those conditions. A low or zero sample count on a
particular thread is not proof that thread was idle. `sampled_cpu_clock_ms` is the
sum of sample periods in that zone and must not be compared directly with the F8
wall-time gap.

On the 2026-09-25 Outer Wilds investigation, the first recording lost a perf chunk,
the second ended before the F8 window, and the third overlapped a Kena build. All
three were discarded for source attribution; these are the failure modes this
reader's explicit clock, coverage check, and external peer audit are meant to catch.

For a Linux-only scheduler question, start `python3 tools/perf/schedstat_probe.py --pid PID
--seconds 20 --hz 200 --out sched.json` beside the monotonic perf recording. It samples threads
named `prosper-app` (`--name` for another frontend) at bounded intervals and records each
`/proc` read bracket. After identifying the primary TID in the process-scoped perf export, run
`python3 tools/perf/schedstat_f8.py capture.prperf sched.json --pid PID --tid TID`. The join
counts only adjacent observations whose full read brackets lie inside the same renderer
span or gap. Inspect coverage, sampler overruns, thread identity, process/build
identity, peer audit, and perf/F8 overlap before interpreting the result. The
signed sleep/unknown remainder includes all time neither recorded as execution
nor runnable wait; it is not a cause attribution or a whole-game FPS estimate.

Every zone in that join always carries `segments`, `wall_ns`, `runtime_ns`, `runnable_ns`,
`sleep_or_unknown_ns` and `max_bracket_ns`, seeded at zero, so a zone with nothing attributable
reports a measured zero rather than an absent key — the expected shape when renderer spans
cover the sampled window. `coverage` is the one field that can be **null**: a gap zone of zero
expected wall time (touching spans, or a single span) is unmeasurable rather than zero.
