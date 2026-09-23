# Routed F8 A/B preflight

`compare_capture_pair.py` reads two run directories from the routed profile
harness. Each directory must have `run.json`, `child-result.json`,
`peer-samples.json`, and exactly one complete `.prperf` capture.

```bash
python3 prosper/tools/perf/compare_capture_pair.py \
  <control-run-dir> <candidate-run-dir> \
  --delta PROSPER_FRAME_RESOURCE_RESERVE > pair.json
```

Both arms must contain the named environment variable. Its **value** must
differ but have the same byte length; every other non-isolation environment
value and the complete key order must match. This catches a control-only flag that changes the length of the
process environment and can itself change frequently called `getenv` work.
The tool also requires matching executable, route, driver and capture-driver
hashes, F8 anchor/settings, clean peer samples, and uncapped renderer detail.
Run-specific save, cache, capture and temporary paths may differ; the route
file's content hash must still match. It refuses missing measurements instead
of treating them as zeros.

The JSON reports medians grouped by exact `(draws, callbacks, texture_bytes)`
renderer shape. A shape needs at least 20 records in each arm by default;
`--min-records` changes this threshold. Each shape's median is weighted by
the smaller arm's record count, then divided by the total matched count to
form a mean of shape medians in milliseconds per matched record. Extra records
in one arm do not silently become more work in the estimate. Coverage and every matched shape are kept
in the output. The result describes **local recorded renderer work**, not
equal guest inputs, newly rendered frames, simulation FPS, or a whole-game
speedup. A clean preflight is necessary for a comparison but cannot prove
the machine was otherwise quiet or that the selected scene was identical.
