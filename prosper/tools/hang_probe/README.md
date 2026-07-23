# hang_probe

Estimate how often a title's main thread ends up **stuck**, and show **what it's stuck on** — for
intermittent boot/loading hangs that a single run may or may not reproduce.

The motivating case is the undelivered-GPU-EOP deadlock family (#1113 / #1195 / #987): `UnityEOPThread`
parks in `k_eq_wait` waiting for an EOP completion that never arrives, and the main + loader threads
park in `k_sema_wait` on the semaphore it would post. It's a timing race — some runs hang, most don't —
so you need a **rate** and a **backtrace of a stuck run**, not one launch. This tool automates the
`run → wait → attach → classify → repeat` loop that is otherwise hand-orchestrated.

## What it does

For each run it launches the title headless via `boot_trace` (with the gated `PROSPER_GUEST_FS` /
`-force-gfx-direct` / `PROSPER_RENDER` switches and a fresh per-run savedata), waits, then attaches
`guest_bt` to the main thread (Thread 1) and classifies it:

- **BLOCKED** — the top HLE frame is a kernel wait (`k_sema_wait` / `k_eq_wait` / `k_cond_wait` /
  `k_ef_wait` / `k_usleep`). On a BLOCKED run it prints Thread 1's stack as evidence.
- **RUNNING** — an active render/submit frame (`execute_ordered` / `agc_driver_submit` /
  `run_command_buffer` / `present` / `SubmitDcb`).

Plain `gdb` **cannot** unwind the guest main thread through prosper's HLE stub boundary (its backtrace
is all `??? `), so `guest_bt` (which re-sections the flattened module ELFs) is required.

Exit status is `1` if any run was BLOCKED, `0` otherwise — so it doubles as a bisect/gate predicate
(e.g. "does commit X still hang evergate at boot?").

## Requirements

Linux, `gdb`, and ptrace attach permitted (`kernel.yama.ptrace_scope=0`). Build `boot_trace` first
(`cmake --build build-linux --target boot_trace`).

## Usage

```bash
# 10 evergate boot runs, 35s each, report the hang rate + any stuck stack
python3 tools/hang_probe/hang_probe.py --dump /path/PPSA01885-app0 --runs 10 --wait 35

# A/B a delivery-path switch under the same probe (each --extra-env is K=V, repeatable)
python3 tools/hang_probe/hang_probe.py --dump <dump> --runs 12 --extra-env PROSPER_EOP_SYNC=1
```

Key flags: `--runs N`, `--wait SECONDS` (time to reach the hang window), `--thread N` (default 1 =
main), `--extra-env K=V`, `--boot-trace PATH`, `--guest-bt PATH`, `--keep-logs`, `--no-stack`.

## Caveats

- Single-sample classification: a main thread transiently in a frame-pacing wait at the sample instant
  reads as BLOCKED. For a definitive verdict on a flagged run, re-check with `--keep-logs` + `guest_bt
  --all` (a genuinely stuck run stays stuck across samples; a paced one moves).
- Logging/`PROSPER_GFXLOG` perturb fine timing races (they can hide a Heisenbug) — this tool keeps the
  run log-free and samples only via the gdb attach, which happens after the hang window.
- The hang can be rare and condition-sensitive (system load, scheduling); a `0%` result over N runs
  bounds but does not prove absence. See #1113 for the reproduction history.
