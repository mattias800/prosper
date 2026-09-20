# memcpy_probe — attribute `memcpy`/`memmove`/`memcmp` to call sites, by cycles

An `LD_PRELOAD` shim that interposes the three libc block routines, times each call with `rdtsc`,
and aggregates by the caller's return address. Writes a report every 10 s and again at exit.

## Why this exists: `perf` cannot attribute these on this machine

Measured on a routed *Grand Theft Auto V* window, 2026-09-17, twice:

- `perf record --call-graph dwarf` resolves a callchain for **1 of 807**
  `__memmove_avx512_unaligned_erms` samples; 42.9% of the render comm's samples (4,465 of 10,417)
  carry no callchain at all. Raising the user-stack dump from the default 8 KiB to 32 KiB and
  restricting to `cycles:u` gives **0 of 964**.

  Count the samples. Do **not** cite `perf report -g`'s "children == self" for this, which an
  earlier revision of this file did: that is a **tautology for any leaf function**, true whether or
  not unwinding worked, so it establishes nothing about unwinding.
- `--call-graph fp` against a `-fno-omit-frame-pointer` build *does* produce chains, and they are
  **wrong at the first hop**. glibc's AVX-512 copy never pushes `%rbp`, so the unwinder reads the
  caller's frame base and silently skips the direct caller. Stacks like
  `__memmove_avx512 → __clone3` are the visible symptom.
- `backtrace()` called at the *sample* site fails the same way, for the same reason. It does **not**
  fail when called from this probe's own frame after the real routine has returned — which is why
  the large-copy feature below can use it.

So the symbol is 8–10% of the busiest thread group and nothing in `perf` says who calls it.
`__builtin_return_address(0)` inside an interposer sidesteps unwinding entirely.

## Build and use

```bash
gcc -shared -fPIC -O2 -o probe.so memcpy_probe.c -ldl -lpthread

LD_PRELOAD=$PWD/probe.so PROSPER_MEMCPY_PROBE_DIR=$PWD \
    PROSPER_RENDER=1 PROSPER_GUEST_ARGS=-force-gfx-direct \
    ./prosper-app <DUMP_ROOT>/<TITLE_ID>-app0

sort -k2 -rn memcpy-sites-<pid>.txt | head -20
```

## Attribute one OS thread (opt-in)

The normal report deliberately keeps its existing process-wide schema. For a serial-path question,
set exactly one of these before launch:

```bash
# Only when the target TID is already known before launch.
PROSPER_MEMCPY_PROBE_TID=<TID>

# Normal routed case: leave this path absent before launch. The parent atomically publishes one
# decimal TID after its CPU-delta selection, then verifies that TID's identity/liveness at the end
# of the run.
PROSPER_MEMCPY_PROBE_TID_FILE=<RUN>/selected-tid
```

File mode is polled by the reporter once per second only while it is waiting, never by the copy
hot path. It latches the first valid positive decimal TID forever; later file changes cannot
redirect a run. Before that latch, calls are deliberately skipped; the report calls that period
`preselection_skipped_calls=unobserved` because counting it would add contention to every copy. A
filtered report carries `probe_mode=tid-filter`, `target_tid`, `observed_tid`, and a `tid` column on every
caller row. `observed_tid != target_tid`, a nonzero `overflow_calls`, an invalid-file count, or a
preselection period covering the window of interest makes full attribution unavailable.

The filter checks a TLS-cached `gettid`; it does not make a syscall, open a file or inspect `/proc`
per copy. Non-target threads call the real libc function without the timestamp/table work. Counts,
bytes and return-address attribution in this mode are for that selected TID only. The cycle column
is still elapsed TSC and **not latency, FPS, or process CPU time**.

Resolve the recorded address according to the DSO's ELF type, not by applying one formula to every
binary. Check with `readelf -hW <dso>`. For the current `prosper-app`, an `ET_EXEC` image with an
image base of `0x200000`, pass the **raw `return_address - 1`** from the report to
`addr2line -e <dso> -f -C -i`; do not substitute `dso_offset`. For an ordinary PIE or shared
`ET_DYN` object, use the load-relative `dso_offset - 1`. If its loadable segments have nonzero
virtual addresses, first establish the actual load bias and use `raw_return_address - load_bias -
1` instead. The subtraction identifies the call instruction rather than its return address.

Run the native controls (shim and CPU-only test program only) with:

```bash
tools/memcpy_probe/test_tid_filter.sh
```

They prove the default report remains process-wide, two workers executing the same call site can be
selected separately by their actual OS TID, each selected arm reports exactly 50,000 calls with no
overflow, and an invalid target refuses rather than silently falling back to process-wide output.
That separation is the point of a serial-path investigation: an older process-wide copy result
from parallel workers cannot establish that the selected serial render thread has the same cost or
the same callers. It is attribution evidence only; it does not by itself show which current cost
is avoidable.

`PROSPER_MEMCPY_PROBE_BIG_BYTES` (default 1 MiB) also captures a short `backtrace()` for copies at
or above that size — useful when the return address lands in a runtime thunk rather than in
prosper. It unwinds from this probe's frame, so glibc's missing frame pointer does not apply to it;
what does limit it is (a) a tail-called copy, whose caller is already gone from the stack, (b) the
`BIG_FRAMES` bound of 6, and (c) `dladdr` resolving only *dynamic* symbols, so a static function
reports the nearest exported one. The observed GTA shape — three frames ending in libc — is (a).

## The cycle column is calibrated, not assumed

Counts and bytes are exact. **Cycles are not**, and cycles is the column this README tells you to
rank by, so the control measures the bias rather than leaving you to trust it. Run the control
**without** the preload (it refuses to calibrate under one, because both arms would be interposed
and the comparison would measure the instrument with itself):

```
  an rdtsc pair alone costs ~24-29 cycles
       bytes    bracketed    amortised  inflation
          64        23.72         3.75     532.3%
        1024        23.70        14.63      62.0%
       65536      2459.09      2575.27      -4.5%
     1048576     32585.19     32998.57      -1.3%
     3145728    140741.90    140622.39       0.1%
```

So: **rank large sites by the cycle column and treat sub-KiB rows as mostly instrument.** A row of
90 M calls at 64 bytes will show a few percent of total cycles that is very nearly all bracketing
overhead. Figures vary a few tens of percent run to run at the small end because the absolute
numbers are single-digit cycles; the conclusion does not.

**The delta is ELAPSED tsc.** Anything that deschedules the thread mid-copy — another lane on the
box, an interrupt — is counted as copy time. Measured: identical work with one competing spinner
pinned to the same core read 1.98x higher. Run attribution on a quiet machine for the same reason a
timing run needs one.

## A periodic report does not sum exactly, and it under-counts

`total_calls - sum(per-site calls) - overflow_calls` is the natural self-check, and on a **periodic**
report it comes out slightly **negative**. Measured at ~15.5 M calls/s across 12 threads: `0`, `-154`,
`-180`, `-489`.

The mechanism is the dump itself, not the counters. `dump()` reads the three totals once and then
walks 8,192 slots doing a `dladdr` per live row, so the rows are sampled tens of microseconds later
than the totals and have grown in between. The skew therefore scales with the dump's duration times
the call rate, and a real prosper run has more live rows than that control did, so expect a larger
gap than those figures, not a smaller one.

The **exit** report is written after the work stops and was exactly summable (`diff = 0`) in the
same test. So: use the final report when you want the sum to close, and read a small negative
residual on a periodic one as this, not as a defect. A LARGE or positive residual is not this.

## Verify it before believing it

`probe_control.c` is the positive control: a known 200,000 + 100,000 + 50,000 copies and 70,000
comparisons at known sizes. It prints what to expect and the report must match exactly. An empty or
mis-bucketed census is otherwise indistinguishable from a working one.

The control also pins a caveat worth knowing: its first version used **constant** sizes and gcc
expanded the 64-byte copy inline, so 300,000 calls never reached libc at all. A small
constant-size copy is invisible to this probe *and* absent from perf's `memmove` samples, because
no call happens. Only a copy that really calls libc is in scope for either instrument.

## What it does NOT tell you

- **Wall-clock cost.** The cycle column is summed across threads. prosper splits large uploads
  across workers (`parallel_render_memcpy`), so a bucket can be most of the summed cycles while
  costing a fraction of that in elapsed time. Divide by the worker count before comparing against a
  `perf` share, or do not compare at all.
- **The call site, where the compiler tail-calls the routine.** A function whose last act is
  `memcpy(...)` compiles to `jmp memcpy`, and the return address is then the *grandparent's*.
  Implausible attributions — a comparison charged to a destructor, a copy charged to an unrelated
  lambda — are this, not a bug in the counter. The totals and per-size distributions are unaffected;
  treat the site column as a lead to confirm in the source, never as proof.
- **Time.** Two `rdtsc` and a hash probe per call is small but real, and a run under this preload is
  an **attribution** run. Never quote a frame rate from one, and never combine it with a
  probe-free timing run.
- **Copies made before the constructor runs** — by the dynamic loader itself — are counted nowhere.
  `bootstrap_copy` serves them correctly and silently.
- **`__memcpy_chk` / `__memmove_chk` are not interposed.** A caller built with
  `_FORTIFY_SOURCE` and a known destination size reaches those instead, and they are invisible here.
  Checked on this workload before it mattered: zero `_chk` samples in the profile, so nothing was
  being missed — but a different binary could differ, and `perf` is where to check that.
