# memcpy_probe — attribute `memcpy`/`memmove`/`memcmp` to call sites, by cycles

An `LD_PRELOAD` shim that interposes the three libc block routines, times each call with `rdtsc`,
and aggregates by the caller's return address. Writes a report every 10 s and again at exit.

## Why this exists: `perf` cannot attribute these on this machine

Measured on a routed *Grand Theft Auto V* window, 2026-09-17, twice:

- `perf record --call-graph dwarf` unwinds **0 of 807** `__memmove_avx512_unaligned_erms` samples.
  `perf report -g` reports **children == self** for that symbol — no callers at all — while 42.9% of
  the render comm's samples carry no callchain whatsoever. Raising the user-stack dump from the
  default 8 KiB to 32 KiB and restricting to `cycles:u` changed nothing: **0 of 964**.
- `--call-graph fp` against a `-fno-omit-frame-pointer` build *does* produce chains, and they are
  **wrong at the first hop**. glibc's AVX-512 copy never pushes `%rbp`, so the unwinder reads the
  caller's frame base and silently skips the direct caller. Stacks like
  `__memmove_avx512 → __clone3` are the visible symptom.
- `backtrace()` fails the same way, for the same reason.

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

`PROSPER_MEMCPY_PROBE_BIG_BYTES` (default 1 MiB) also captures a short `backtrace()` for copies at
or above that size — useful when the return address lands in a runtime thunk rather than in
prosper, though it inherits the unwinder's limits above.

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
