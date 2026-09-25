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
spans, multiple recorded processes, and perf samples that do not cover both ends of
the F8 span interval (within 0.5 s). It keeps samples outside the interval separate.
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
