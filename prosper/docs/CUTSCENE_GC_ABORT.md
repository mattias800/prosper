# Cutscene blocker — IL2CPP GC "Unexpected state" abort on the level1 (intro) load (2026-07)

After the title screen (`level0`) renders correctly, the game **always reaches the intro
cutscene's scene load** — it opens `/app0/Media/level1`, `sharedassets1.assets`, and streams
`resources.resource` — then **crashes in the IL2CPP garbage collector** before the cutscene
can render. This is the next boot wall after "first real frames". This doc records exactly
what the crash is, the evidence, what was ruled out, and the open leads.

## TL;DR

The crash is a **Boehm GC (bdwgc, IL2CPP's collector) `ABORT("Unexpected state")`** during a
stop-the-world collection. The title allocates too little to ever trigger a GC; the cutscene's
asset load allocates enough to trigger the **first big collection**, which aborts. It is
**timing-sensitive** (a race): it disappears entirely under `PROSPER_SYNCLOG` (the extra
logging/serialization perturbs the timing) and manifests on either the main thread (caught as
`RUN ENDED kind=2`) or a worker thread (`WORKER-THREAD FAULT`, which `_exit(90)`s the process).

## How to reproduce

```
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_PAD_PRESS=1 \
  ./boot_trace <dump>
```

`PROSPER_PAD_PRESS=1` holds CROSS so the title advances into the level1 load. ~6/6 runs reach
`level1` and crash. (No renderer needed — the crash is on the CPU asset-load path; with
`PROSPER_RENDER=1` the title renders many times first, then the same crash fires at level1.)

## Evidence chain (all reproducible)

1. **The fault site is a deliberate GC abort trap.** The main-thread crash is
   `rip = libc.prx+0x4a20`, whose bytes are `cd 45 90 0f 0b` = `int 0x45 ; nop ; ud2` — a
   trap, not a normal fault. The caller is `Il2cpp+0x5423: call Il2cpp+0x1e3000 ; ud2` — the
   classic `call noreturn_abort ; ud2` shape.

2. **The abort function's argument is the message "Unexpected state".** Breaking at the abort
   function (`PROSPER_HWBP=0x401e3000 PROSPER_HWBP_STRDUMP=1`) shows `rdi -> "Unexpected state"`.
   The rodata around it (dumped via `PROSPER_CRASHPEEK`) is
   `"Unexpected state\0No space for mark stack\0suspendCond\0Cannot mark..."` — bdwgc's
   mark/suspend error-string table. So IL2CPP's bdwgc uses a **condition variable** for
   suspend/resume ("suspendCond"), and this is the collector's stop-the-world state machine.

3. **The whole call stack is guest code** (`libc.prx` ← `Il2cpp` ← the game's compiled C# at
   `eboot+0xa8c989` ← the Unity runtime init `eboot+0x147b494 / 0x1485851`). Nothing of ours is
   on the stack: the collector itself detected a thread in an unexpected suspend state and
   aborted. Registers are deterministic across runs (`rax`/`rdi`/`rsi=0`/`rdx=8`; only the
   stack pointer varies with ASLR).

4. **The GC stop-the-world handshake works for many cycles, then fails.** With
   `PROSPER_SYNCLOG=1`, `[exc]` tracing shows the collector raising exceptions and threads
   ack/resume correctly for several cycles (16 raises / 16 handler-enters / 15 exits — the last
   thread simply still-suspended at the kill). Under SYNCLOG the crash does **not** happen at
   all (it idles on the render watchdog instead). So suspension is not categorically broken —
   there is a specific race that only the first level1 collection hits.

## Ruled out (with evidence)

- **Semaphore lost-wakeup** (`k_sema_wait/signal`, hle_kernel.cpp): counting sem, `count += n`
  under the mutex + broadcast; a signal before a wait is retained. Correct.
- **Condition-variable HLE** (`k_cond_*`): thin pass-throughs to host `pthread_cond_*` with the
  guest owning the predicate loop. Correct.
- **sigaltstack `thread_local` aliasing**: `install_sigaltstack` runs in `thread_trampoline`
  **before** `guest_tls_activate_thread()`, i.e. on the host `%fs`, so its `thread_local` guard
  is not aliased. Not the bug.
- **TSD-destructor `%fs` at thread exit** (hypothesis: exited workers don't run bdwgc's cleanup
  key destructor because `thread_trampoline` returns on the guest `%fs`, so glibc's
  `__nptl_deallocate_tsd` reads the destructor array through the wrong TCB): **tried** (restore
  host `%fs` before `thread_trampoline` returns) — **did not remove the abort**. Reverted.
- **GC handler running on the wrong `%fs`** (hypothesis: the suspend RT-signal interrupts a
  thread mid-HLE-call on the host `%fs`, so `exc_delivery_handler` runs the guest suspend
  handler with the wrong TCB — the code even comments that it "must KEEP the guest %fs"):
  **tried** (ensure guest `%fs` for the handler via a per-thread guest-TP `thread_local`) — the
  abort **still fired**, and the `%fs`-swap inside the signal handler additionally tripped the
  host stack-protector ("stack smashing detected"). Reverted. The `%fs`-mid-HLE case is a real
  latent hazard worth fixing properly, but it is **not** the (sole) cause of this abort.

## Attempted bypass: disable/defer the GC via its env knobs — BLOCKED (dead end)

bdwgc reads `GC_DONT_GC` / `GC_INITIAL_HEAP_SIZE` / `GC_PRINT_STATS` from the environment; since
the abort only fires *during* a collection, disabling or deferring collection would sidestep it
entirely (a bounded leak is fine for a short cutscene — a standard emulator GC-bug workaround).
Both delivery paths are blocked:

1. **Stack `envp`** — the loader now builds a SysV `envp` from `PROSPER_GUEST_ENV`, but the PS5
   crt0 does not consume it (`GC_PRINT_STATS=1` produced no GC output). The guest's `environ`
   stays empty. (PS5 process env likely comes from the Sony process-param / `sceLibcParam`
   structure, not the SysV stack vector.)
2. **`getenv` HLE override** — registering our own `getenv` does not intercept the call: the
   loader resolves imports "cross-module export beats a stub slot" (linker.cpp), and `libc.prx`
   **exports** `getenv`, so the guest binds to the real libc.prx `getenv` (which reads the empty
   `environ`). `PROSPER_GETENVLOG` showed zero calls routed to our handler.

To make env-based GC control work, a future change must either fix the crt0 `envp` consumption
(populate `libc.prx`'s `environ`) or patch the Sony process-param env — and even then it is
unproven that `GC_DONT_GC` helps, since a Unity scene-transition `GC.Collect()` is explicit and
`GC_DONT_GC` only disables *automatic* collection. Both env-plumbing changes were reverted (they
did not function); only the diagnostics below remain.

## Open leads (ranked, for the next investigator)

1. **A thread created/exited DURING a stop-the-world.** Unity's job system spins worker threads
   up and down during asset load. bdwgc serializes create/exit with the collector via the GC
   lock + a thread-registration handshake; the codebase already fought one instance of this
   ("Bad stack base … fast-starting worker ran before the parent's registration",
   hle_kernel.cpp:183). "Unexpected state" is the collector finding a thread whose
   `stop_info`/registration state doesn't match the current stop-count — exactly what a
   create/exit racing the stop produces. **Decisive capture:** log `scePthreadCreate` /
   thread-exit against the `[exc]` raise/ack timeline (gate on the level1 load) and look for a
   thread whose lifetime brackets a stop-the-world.
2. **Disassemble the guest suspend handler** (installed for exception type `0x1e`; the log shows
   `handler=0x4c0000210`) to learn the exact primitive it acks/blocks on (sem vs the
   `suspendCond` condvar) and the state field it sets — then verify our delivery preserves that
   state's invariants. This is the ground truth that ends the hypothesis-guessing.
3. **RT-signal double-delivery.** RT signals queue (no coalescing). If the collector raises the
   same thread twice within one stop (or a stale queued signal from a prior cycle is delivered),
   the guest handler runs twice → double-ack → state mismatch. Instrument `k_raise_exception` to
   count raises per (thread, stop-count).

## Tooling added this session (gated, no default behavior change)

- **`PROSPER_CRASHPEEK`** (boot_trace.cpp): after a recovered main-thread fault, dump guest
  memory at the fault registers + 24 instruction bytes at the fault rip and each backtrace
  frame's call site (for offline `objdump -b binary -m i386:x86-64`). This is what decoded the
  abort trap and the GC rodata table.
- **`PROSPER_RENDER_EVERY=K`** (hle_agc.cpp): render only every Kth draw-carrying submit. Under
  llvmpipe a 1080p render is ~10–20 s and `execute_and_present` is synchronous in the submit
  call, throttling the whole frame loop; sampling lets the game run near full speed toward a
  distant scene while still producing periodic frames.
- **`PROSPER_PEEK_CODE=0xADDR[,0xADDR…]`** (boot_trace.cpp, under `PROSPER_CRASHPEEK`): dump 512
  code bytes at each guest address after the recovered fault (guest memory still mapped) for
  offline `objdump -b binary -m i386:x86-64`. Used to disassemble the guest GC suspend-handler
  thunk at `0x4c0000210` (confirmed it extracts `ctx[0xf8]` as the thread sp — which our exception
  delivery already populates correctly — and tail-calls the real handler behind a global pointer).
