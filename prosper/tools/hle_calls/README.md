# hle_calls

**"Which Sony functions is the guest calling *right now*?"** — a per-function call histogram over
**every** HLE handler in the prosper binary, taken from a live process over a bounded window.

Reach for it when a title goes quiet — the frame loop is alive, nothing crashes, and you need to
know what the guest is polling (or prove that it is polling nothing).

## Why the existing instruments do not answer this

| instrument | what it covers | the gap |
|---|---|---|
| `PROSPER_SVCLOG=1` | only handlers that call `svc_log` | opt-in per handler. Measured on the revision this tool was written against: **~1,035 registration sites (287 direct `register_fn` calls plus 748 through the `R("name", fn)` macro) against 94 `svc_log` call sites.** Most of the implemented Sony surface never appears in the log. |
| `PROSPER_PROGRESS_UNIMPL=1` | per-NID counts for **unimplemented** NIDs | the exact complement — it cannot see a handler that exists. |
| `guest_bt` | where a thread is *parked* | a thread that is running, not blocked, has no wait frame to report. |

So "the stall shows only `sceUserServiceGetEvent` in the svc log" bounds the *instrumented* surface,
not the guest's traffic. `hle_calls` closes that gap without an emulator rebuild or a gating env var:
every handler shares the six-`unsigned long` HLE signature, so the tool enumerates them out of the
binary with `nm` and counts them with non-stopping gdb Python breakpoints.

## Usage

```bash
python3 tools/hle_calls/hle_calls.py --pid $(pgrep -x boot_trace) --ticks 400

# only the libSce service layer (skip kernel / file / graphics handlers)
python3 tools/hle_calls/hle_calls.py --pid <pid> --filter '^s_'

# cover INIT, and record what each handler answered rather than only that it ran
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct \
python3 tools/hle_calls/hle_calls.py --ticks 30 --values --filter '^s_' \
    --launch build-linux/boot_trace <DUMP_ROOT>/PPSA05325-app0
```

Flags: `--binary` (defaults to `/proc/<pid>/exe`, or the `--launch` program), `--ticks N` (window
length), `--clock SYMBOL` (what advances the window, default `prosper::k_usleep` — one or more
entries per frame on every title observed so far), `--filter REGEX`, `--order N`, `--gdb PATH`,
`--timeout SECONDS`.

Output:

```
clock=prosper::k_usleep entries=400 armed=151 mode=attach calls=72 finish-failures=0 exited=0
first-calls: 1:pad_read_state  2:s_user_getevent  3:s_syss_getstatus  …
      36  s_user_getevent
      36  s_syss_getstatus
distinct=2 total=72
```

## `--launch`: the init window

`--pid` **cannot see init.** The process is past it before anything can attach, and arming hundreds
of breakpoints takes longer still. Measured on Sonic Origins (`PPSA05325`): an arm that started
`boot_trace` and attached at t=1 s with `--filter '^s_'` produced a *boot* pass and a *settled* pass
with the identical two-handler histogram — the window was real, it simply never covered init.

`--launch` runs the program under gdb itself: every breakpoint is armed while the inferior is still
loaded-but-not-started, so the window opens at the first instruction. It passes the invoking
environment straight through, so put the `PROSPER_*` switches on the `hle_calls` command line.

**Do not take init coverage on faith — the result carries its own check.** `first-calls` lists the
leading calls *in call order*. A window that covered init leads with allocator / module-setup
handlers; one that opened late leads with the frame-loop pollers (`pad_read_state`,
`s_user_getevent`, `s_syss_getstatus`). A histogram cannot distinguish those two, which is exactly
how the attach arm above looked like a negative when it was void.

A launch run ends when the window closes (the inferior is killed) or when the guest exits first;
`exited=1` in the header says which. On a timeout the tool kills the whole process group, so a
launched emulator is never orphaned holding the GPU.

## `--values`: what prosper answered, not just what ran

A count says *which* Sony function ran, never whether prosper answered it **correctly**. The
remaining class of bring-up blocker after every absence check comes back clean is a
*registered-but-mismodelled* call whose wrong value was consumed once, early — invisible to
`PROSPER_PROGRESS_UNIMPL` (which only sees handlers that do **not** exist) and to a histogram alike.

`--values` adds a per-handler histogram of return values:

```
      36  s_user_getevent   ret 0x80960007 x35, 0x0 x1
```

**That line is also this feature's positive control, and it is worth using every time.**
`s_user_getevent` delivers the initial LOGIN event exactly once (returning `0`) and reports
`SCE_USER_SERVICE_ERROR_NO_EVENT` (`0x80960007`) forever after, so a correct capture shows exactly
one `0x0` against many `0x80960007`. If it does not, distrust every other value in that run rather
than believing the surprising one.

**Completeness is per row, not per run.** `finish-failures` in the header counts captures that could
not be *armed or decoded*, and it does not see the more common loss: a handler that leaves its frame
without returning normally (longjmp, thread teardown) loses its value silently. Each row therefore
prints `(captured N/M)` whenever fewer values were recorded than calls counted, and that is the field
to read before calling a value set complete.

Two limits before believing a number:

- `--values` records the return **register**, never an out-struct. A handler that returns 0 while
  writing wrong bytes through a pointer argument is invisible — and on this codebase that is a common
  bug shape.
- On a handler that longjmps back into its own caller, gdb's finish breakpoint sits at an address the
  landing pad reuses, so it can capture the *landing* value as if it were a return (measured on
  gdb 17.2). A `(captured N/M)` row's other values are therefore suspect, not merely fewer.

`window=complete|SHORT` in the header is the matching check for the run as a whole: `run`/`continue`
also return on a signal gdb stops for (`SIGABRT` is not passed through, and these titles do abort),
which would otherwise print a truncated histogram that reads as a finished window.

Cost: a second trap per call, so pair `--values` with `--filter`.

One behaviour change that also reaches `--pid`: the gdb side now passes SIGSEGV/SIGBUS/SIGILL/SIGFPE
straight through without stopping. prosper uses SIGSEGV as a working mechanism (write watches, the
fault handler) and gdb's default is to stop, so an attach window previously truncated at the first
guest fault.

## Reading a zero — and the trap this tool was built around

An empty histogram with a **non-zero `entries`** is a real measurement: the guest entered no HLE
handler while the clock advanced. An empty histogram with `entries=0` means the window never
opened; that run is **void, not negative**.

Always include a handler you independently know fires as a positive control before believing a
surprising zero. `sceUserServiceGetEvent` (`s_user_getevent`) is a good default: engines poll it
every frame.

The first version of this tool read `gdb.Breakpoint.hit_count` for each handler. For a breakpoint
whose Python `stop()` returns `False`, that reported **zero for all 151 handlers** — which reads
exactly like "the guest called nothing at all" and was wrong. The counting now lives in the script's
own state, and the positive control is what caught it. The same shape of mistake is why the header
line always prints `entries` and `armed`: a result you cannot sanity-check against a known-live
number is not a result.

## Requirements

Linux, `gdb` with Python support, `nm`/`c++filt`, and ptrace attach permitted
(`kernel.yama.ptrace_scope=0`). The prosper binary must not be stripped.

## Cost

One trap per handler entry. On an idle/stalled guest this is free. On a title in full flight the
trap rate can be thousands per second, which perturbs timing badly — use `--filter` to narrow the
set, and never quote a frame rate from a run under this tool.
