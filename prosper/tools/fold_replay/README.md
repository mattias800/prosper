# Exact-input resource-fold replay

Build `fold_replay` with the normal Prosper CMake build. It executes
`resolve_dynamic_fetch` against an ordered transcript of consumed resource reads.
It compares every `DynFetch` and `SrtUse` field, their order, and execution metadata.
It does not use the F9 materializer, which bypasses this evaluator.

Capture is intrusive and off by default. In a native game launch, set:

```sh
PROSPER_FOLD_CAPTURE_DIR=<BUILD>/private-folds
PROSPER_FOLD_CAPTURE_SKIP=0
PROSPER_FOLD_CAPTURE_LIMIT=16
```

Export these variables in the launch environment. Each process reserves its own
session subdirectory. SKIP counts fold invocations from zero; LIMIT selects at most
256 subsequent calls. Refused calls count against that budget. An existing nonempty
SRT output vector, diagnostic whole-buffer tracing/coherence, oversized inputs or an
unreadable declared code span cause a visible refusal and the ordinary fold runs once.
Keep title-derived files private in an ignored build directory.

The selected fold owns its bounded code and initial register arrays before evaluating
them once. Code/entry arrays must remain immutable during that initial copy. The owned
code includes the declared tail used by PC-relative dispatch; copied dispatch metadata
retains its original meaning. Resource pointers remain live during capture, and each
consumed word/probe/prefix copy is recorded in order, including repeated reads of the
same address. Only the existing partially OOB scalar-buffer path takes a prefix snapshot.
The caller receives the actual evaluation outputs even if recording or file writing fails.
The copy changes decode-cache identity and capture adds allocations and I/O: capture
timings are not ordinary gameplay timings. No synchronization callback is introduced.

```sh
<BUILD>/fold_replay <FILE.prfold> --iterations 10000
<BUILD>/fold_replay <FILE.prfold> --iterations 100 --cold-each
<BUILD>/fold_replay <FILE.prfold> --iterations 0 --mutate-word 1=0x12345678
```

Cold replay starts after clearing the actual shader decode cache. Warm iterations retain
that cache; actual control-plan construction and evaluated-instruction counters accompany
wall timings. Cache clearing and result destruction are outside the timed interval.
File parsing is outside it; transcript validation, evaluator work and output comparison
are inside it. Report these as **offline replay costs**, not isolated evaluator or FPS
measurements. They omit live mapping probes, guest scheduling and GPU dependencies.
Input mutation disables expected-output comparison explicitly; it still refuses requests
without matching backing events. The report counts changed outputs against the recording.
Source revisions are printed, allowing captures to be replayed against candidate code.

Use `heaptrack --record-only <BUILD>/fold_replay ...` and `heaptrack_print` for allocation counts and
peak retained memory, separately for cold-only (`--iterations 0`), repeated cold and warm
runs. These process-wide reports include parsing and tool startup; use allocation stacks
to attribute evaluator/cache/validation costs. Do not call file size or retained payload
size total process memory. Time uninstrumented replays separately from heap profiling.

Missing, reordered, extra, malformed, oversized and incomplete input fails visibly.
Limits are 65,536 code dwords/events, 16,384 outputs per kind, 4 MiB event payload and
16 MiB per file. Files use explicit little-endian fields, version 1, a code checksum and
a whole-file checksum; these are corruption guards, not cryptographic authentication.
Captured addresses are logical identities and are never mapped or dereferenced offline.
Agreement proves reproduction of Prosper's current evaluator, not correct guest semantics.

For a direct-reader control, run `<BUILD>/test_fold_control_plan benchmark` and
`<BUILD>/test_fold_control_plan benchmark-long` (the latter retains 1,500 additional
scalar instructions). Both warm the selected stream and check descriptor results on each
invocation. These synthetic timings include live probes, validation and result destruction;
they are separate from the replay CLI's timing contract. Compare identical workloads on
both revisions, alternate execution order, and exclude concurrent builds/games.

The reusable plan also stores instruction-only EXEC-write and mip-operand classification.
It never stores the runtime zero-mip result: current register values, intervening EXEC writes
and block boundaries still determine that proof on every invocation. These facts occupy the
existing 16-byte step's padding. Construction performs the classification once; warm folds
avoid three repeated helper calls per instruction. Cold/reconstructed plans still pay for it.
