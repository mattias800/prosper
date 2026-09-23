# `new_probe`: find C++ allocation callers when libc unwind stops early

This Linux/glibc-only `LD_PRELOAD` shim intercepts ordinary scalar and array
`operator new`, then delegates to the real libstdc++ operators. It records each
successful allocation's direct caller return address, OS TID, call count, and
requested bytes in a fixed 32,768-slot table. No dynamic allocation occurs in
the hot accounting path. A bounded collision search reports unmatched calls and
bytes as `overflow_*`; any nonzero overflow makes the caller distribution
incomplete. Failed allocations are counted separately and rethrow the original
exception. During `dlsym` bootstrap, a recursion-safe glibc allocator fallback
obeys throwing-new/new-handler behavior and is counted separately.

The reporter writes one file per PID after one second, every ten seconds, and
at normal exit. Periodic reports survive `timeout`'s SIGTERM; their totals and
site rows are sampled at different moments, so they need not sum exactly. The
exit report is the preferred closed count. Each report is atomically renamed
from a temporary file so readers never see a half-written table. Failed file
operations print `[new-probe] REPORT FAILED` to stderr and increment
`report_failures` if a later report succeeds; a missing or stale file is never
silently treated as zero allocations.

Rows are tab-separated with exactly eight columns. In DSO and symbol fields,
the probe escapes a literal tab, newline, carriage return, backslash, or other
control byte as `\t`, `\n`, `\r`, `\\`, or `\xHH`. Spaces remain part of the
field. Parse on **tabs**, then decode those escapes; whitespace splitting can
misassign columns when a DSO path contains spaces.

If the target forks without exec, its child inherits the parent's counters
but loses the reporter thread. An `atfork` hook therefore refuses child
attribution with an async-safe stderr message and a refusal-only
`new-sites-<child-pid>.tsv` marker. It does not publish inherited rows. An
exec starts a fresh preload instance in the new image.

```bash
cd prosper/tools/new_probe
c++ -std=c++17 -O2 -fPIC -shared -fno-optimize-sibling-calls \
    new_probe.cpp -ldl -pthread -o new_probe.so
PROSPER_NEW_PROBE_DIR=<OUTPUT_DIR> LD_PRELOAD="$PWD/new_probe.so" \
    <PROSPER_APP> <DUMP_ROOT>/<TITLE_ID>-app0
./test_probe.sh
```

The native control compiles a tiny host program and checks all four cells of a
two-callsite by two-OS-thread matrix: scalar and array calls, 500 each per TID,
with exact requested bytes. It proves both `ET_EXEC` raw-address and `ET_DYN`
load-relative address mapping, paths with spaces and literal tabs, the complete
periodic table before SIGTERM, `std::bad_alloc` propagation, fork refusal,
report-write failure diagnostics, and bounded overflow reporting. It uses no
GPU or title dump. Run it inside the same distrobox used for the app to confirm
the compiler and runtime ABI.

For source mapping, the report includes the raw return address, DSO, load-
relative `dso_offset`, and nearest exported symbol. Check the DSO with
`readelf -hW <dso>`. For an `ET_EXEC` executable, give
`raw_return_address - 1` to `addr2line -e <dso> -f -C -i`; the load-relative
offset is not the executable's link-time address. For PIE/shared `ET_DYN`, give
`dso_offset - 1` for ordinary zero-based segments. If the object's loadable
segments have nonzero virtual addresses, derive its load bias and give
`raw_return_address - load_bias - 1`. The subtraction points to the call
instruction rather than its return point. Keep the exact binary used for the
run, including its build ID and line information.

DSO identity is resolved by `dladdr` **when a report is written**, not at
allocation time. If a module unloads before that report, its row may say `?`;
if another module reuses the same address, the row can even name the wrong
module. Confirm suspicious dynamic-library rows against the live process's
module map. This probe does not prove the lifetime of a DSO between allocation
and report.

This is **attribution**, not an allocation-time, frame-time, or FPS run. The
interposer adds a hash probe and atomic counters per allocation; the reporter
also resolves symbols periodically. Requested bytes are not retained bytes,
and the table does not cover aligned `operator new`, C `malloc`, custom
allocators, or allocations optimized away by the compiler. Do not infer a
speedup from a large row: first establish whether those allocations are
necessary and measure an actual candidate separately.
