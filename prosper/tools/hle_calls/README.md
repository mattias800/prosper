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
```

Flags: `--binary` (defaults to `/proc/<pid>/exe`), `--ticks N` (window length), `--clock SYMBOL`
(what advances the window, default `prosper::k_usleep` — one or more entries per frame on every
title observed so far), `--filter REGEX`, `--gdb PATH`, `--timeout SECONDS`.

Output:

```
clock=prosper::k_usleep entries=400 armed=151
      36  s_user_getevent
      36  s_syss_getstatus
distinct=2 total=72
```

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
