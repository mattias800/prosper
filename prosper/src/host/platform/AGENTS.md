# `prosper/src/host/platform/` — process-wide host primitives with no OS-integration deps

The small pieces of *host* behaviour that the core needs but that must not drag a window system, a
graphics API, or a frontend into `prosper_core`: cooperative lifecycle signals, a precise sleep, the
POSIX shims Windows lacks, a raw-syscall escape hatch, and `immortal.hpp` for objects that must
outlive static destruction.

The unifying constraint is the dependency arrow `FRONTEND_APP.md` describes: everything here is
callable from `prosper_core`, from the offscreen backend, from `boot_trace`, from `tools/screenshot`
and from `prosper-app` alike, so **nothing here may include Vulkan, SDL, or anything under
`frontends/`**. Signals are declared here and *observed* by whoever cares; the frontend sets them.
That is why the files are deliberately tiny and header-plus-one-source: a dependency added here is
one every consumer inherits.

Two of them are shutdown-related and are easy to confuse:

- **`lifecycle.hpp`** — boolean cooperative signals (`prosper_request_stop`, pause/resume). "Please
  wind down at your next boundary." Nothing is forced; a signal nobody polls does nothing, which is
  exactly the situation `run_entry` is still in.
- **`gpu_submit_gate.hpp`** — a *counted region* plus a bounded drain, for the case where asking
  nicely is not enough. It exists because the frontend cannot join the guest thread and so exits with
  `_Exit` while that thread may be inside a GPU submission; closing the gate refuses new submissions
  so the drain can reach zero (#3225). It bounds that window rather than closing it, and its own
  header says so — do not let it be described as the fix for the missing join.

One of them is half something else, and worth knowing before you go looking:
**`precise_sleep.hpp` also holds this family's PURE ARITHMETIC** — deadline conversion, poll
cadence, saturating guest-interval conversion — extracted from the callers in `src/hle/kernel/` and
`src/hle/sync/` that use it. That is not scope creep. The arithmetic those callers need is mostly guarded by
`#ifdef _WIN32` or buried inside an `HLE()` body, so in its original home it could not be compiled
on a POSIX host at all, let alone driven through its saturating and out-of-range cases; here it is
`constexpr`, platform-neutral, and exercised by `tests/host/platform/test_precise_sleep.cpp` with a
fake clock on every platform. Put the next one here too rather than inline at a call site.

New code belongs here only if it is genuinely process-wide, genuinely host-level, and genuinely free
of OS-integration dependencies. Per-platform *image* mapping and execution live in `src/host/image/`;
memory and TLS have their own sibling folders.
