# `src/host/fault/` — what runs, and what may be reasoned about, inside a signal handler

This folder holds the pieces of prosper's fault and trap handling that are **pure enough to be
extracted, tested and reused**. The handler itself is not here: it lives in
`src/host/image/exec_image_linux.cpp` (and its Windows counterpart), because it is inseparable from
image mapping, guest `%fs` switching and the recovery `sigsetjmp`. What lands here is the part of
that machinery a second consumer needs, or that a unit test can drive without raising a real signal.

The boundary against `src/host/image/` is therefore *reuse and testability*, not subject matter. A
routine belongs here once a second call site wants it, or once "is this logic right?" becomes a
question you would rather answer with a ctest case than by provoking a SIGSEGV.

## The constraint every file here is written under

Everything in this folder may execute **inside a signal handler, on a guest thread that owns `%fs`**.
That rules out more than the usual async-signal-safety list:

- no allocation, no locks, no `errno`, no C++ `thread_local` — glibc TLS is unreachable when the
  guest owns `%fs`, which is why the handler uses process-globals and raw syscalls rather than
  `write()`;
- **no unguarded dereference of a guest pointer.** A fault handler that faults produces no report at
  all. Anything here that reads guest memory takes the readability predicate from its caller
  (`walk_rbp_chain`) or is fed a snapshot the caller already validated — it never decides for itself
  that an address is probably fine.

Header-only for the same reason: these are small enough to inline, and keeping them out of a
translation unit that links platform state makes them trivially unit-testable.

## What is here

- **`fault_context.hpp`** — one fault's registers, snapshotted from *its own* `ucontext` into a
  stack local. The handler's process-globals are not a per-fault record: a second thread faulting
  mid-report overwrites them, and the first report then continues with the second thread's registers
  while reading as one complete fault (#2018). Guest faults are correlated, so this is the common
  case rather than a corner one.
- **`trap_arbitration.hpp`** — decides which instrument owns an incoming SIGTRAP. `PROSPER_HWBP`
  (hardware, traps *before* the instruction) and `PROSPER_BP` (software int3, traps *after* it) have
  incompatible recovery sequences, and letting the wrong state machine consume a hit fabricates a
  plausible-looking record while skipping a guest instruction (#1932).
- **`rbp_chain.hpp`** — the frame-pointer walk that answers "who called this?", shared by the fault
  reporter and `PROSPER_HWBP_STACK`. Extracted rather than copied: two hand-written loops that chase
  guest-controlled pointers inside a handler are how the two quietly diverge.

## Reading anything this folder produces

A frame-pointer chain is a **lead, not an oracle**, and the same caveat covers `gdb`'s guest
backtraces (instrument traps 114 and 217): PS5 guest code has no CFI here, a function that omitted
the prologue is skipped silently, and a chain can therefore be missing a level rather than being
wrong. Every link is checkable, though, and cheaply — a return address must point immediately after
a `call` to the frame below it, so one `objdump` at the named site confirms or kills a link. Confirm
the links you are going to reason from; do not publish a chain you have only read.

## The two recoverers, and why there are two

`rbp_chain.hpp` and `guest_stack_scan.hpp` answer the same question — *which guest code called this?*
— by opposite methods, and both are here because **which one works is a property of how the TITLE was
compiled, not something the host can know.**

- `rbp_chain.hpp` follows saved-`rbp` links. Precise, and **often empty** on an optimised title:
  FINAL FANTASY TACTICS' fatal raise recovers exactly one frame, the module entry point, because
  every guest frame between omitted its prologue.
- `guest_stack_scan.hpp` scans the stack instead, so it does not depend on frame pointers at all. A
  return address pushed by `call` is on the stack either way. It pays for that with false positives,
  which it removes by **confirming a `call` instruction ends immediately before each candidate**.

A caller that needs an answer should ask both and print both, rather than choosing — a short chain
and a shallow stack look identical, so "the chain returned one frame" is not evidence about the
stack.

`guest_caller.hpp` is the other half of the same story: the `PROSPER_CAPTURE_RBP` / `_RSP` macros
that snapshot an HLE handler's entry registers. They are macros rather than functions on purpose —
a function would capture its OWN frame, one level too deep, and a build that inlined it would change
the answer silently.

**Both recoverers take the caller's `readable` probe as a parameter and dereference nothing without
it.** That is not defensive style here: these run on paths that are already terminating the process,
where a nested SIGSEGV would replace the diagnostic with a confusing fault report about the
diagnostic itself.
