# Native copy and allocation caller recovery

Use `perf_caller_report.py` before assigning hot libc samples to renderer owners. A named
`memcpy` leaf with no caller identifies the operation but leaves its owner unknown. This reader
reports that gap instead of silently charging the sample to a guessed subsystem.

Export one complete recording, checking both commands' exit status:

```sh
perf script --no-inline -i "$run_dir/cpu.data" \
    -F comm,pid,tid,time,period,event,ip,sym,dso > "$run_dir/stacks.txt"
perf script -D -i "$run_dir/cpu.data" > "$run_dir/records.txt"
python3 prosper/tools/perf/perf_caller_report.py "$run_dir/stacks.txt" \
    --app-dso "$recorded_app_dso" --libc-dso "$recorded_libc_dso" \
    --records "$run_dir/records.txt" > "$run_dir/callers.json"
```

Selectors are exact paths as recorded, not the current working directory or a basename guess.
Repeat a selector to include multiple known mappings. The report explicitly lists unobserved
selectors: a wrong path and missing frames can both produce that state. Overlapping application
and libc selectors are refused. Confirm executable/library build IDs against `perf buildid-list`
and the actual binaries separately. Hashes identify the supplied exports; matching record counts
alone cannot prove two exports came from the same recording.

Each observed user/kernel event has separate aggregate and PID/TID populations. Copy, comparison,
fill, allocation and free families classify only known leaf spellings in the selected libc DSOs.
Other symbols remain in the event denominator but outside those families. Periods weight samples;
they are neither invocation counts, byte traffic nor elapsed CPU time. The per-thread percentage
denominator is that thread's event population, not the whole process.

For each family, inspect:

- `leaf_only`: no recorded ancestor at all.
- `named_app_ancestor` and `no_named_app_ancestor`: partition the family by whether a named,
  matching application frame occurs after the leaf.
- `unresolved_before_app`: an unknown symbol or low instruction address lies before that named
  ancestor. The report does not pretend those frames are contiguous callers.
- `suspicious_low_ip`: at least one recorded address is below 4096. This is a limited heuristic,
  not general stack validation; even a named recovered chain can be wrong.

The quality categories overlap. Do not add them as disjoint costs. The caller table names the
nearest **recorded matching ancestor**, not necessarily a direct caller. Missing families have
zero observations; undefined weighted shares are JSON null. An entirely zero-weight event,
empty input, unsupported event, malformed line or raw/exported sample-count disagreement fails.

With `--records`, the report counts observed `PERF_RECORD_LOST`/`PERF_RECORD_LOST_SAMPLES` markers.
Those are record markers, not lost sample counts. Without it, loss is unobserved (JSON null).
Also inspect recorder/exporter stderr: a mapping timeout can destroy attribution while leaving
zero lost records. No-loss markers and matching sample counts do not certify recording completeness.

## Improving truncated native stacks

Default frame-pointer stacks can truncate in library code. Try a separate bounded, low-frequency
DWARF recording when that happens, and verify recovery with a known-owner control. On one
mapping-heavy Linux workload, increasing the ring buffer and map-synthesis timeout and delaying
sampling avoided the initial recorder's timeout/loss:

```sh
perf record -e cpu-cycles:u -F 49 --no-inherit --call-graph dwarf,16384 \
    -m 8M --proc-map-timeout 10000 -D 1000 -t "$selected_tid" \
    -o "$run_dir/cpu.data" -- sleep 20
```

These are experiment settings, not portable defaults or permission guarantees. Recording buffer
limits differ by host. Mapping synthesis and the delay consume recorder wall time; the command's
sleep interval does not establish the actual sampling window. Check the recorded events and
target identity, and measure perturbation before making performance claims. Do not silently
change unwind methods or diagnostics between throughput comparison arms. Selected-thread
recordings exclude other workers; use verified worker identities when those costs matter.
