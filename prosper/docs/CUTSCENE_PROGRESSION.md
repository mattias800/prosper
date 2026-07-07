# Title/loading screen → cutscene: progression investigation (2026-07-07)

**Goal:** advance the game past the loading screen (the "Werewolf" title art) into the first cutscene
(mostly-black background + text, 8-bit graphics).

## What was fixed — `sceSystemServiceParamGetString` (committed)

The single biggest blocker was a one-line HLE gap. `sceSystemServiceParamGetString` (NID `SsC-m-S9JTA`)
was **not registered**, so it fell through to the default `unimplemented → returning 0` stub. That returns
`0` (Sony SUCCESS) **without writing the caller's string buffer**, so the game read the uninitialized
buffer as a "valid" system string and dereferenced into it → a null-object crash in managed (Il2cpp) code
(`SIGSEGV addr=0x20` at `Il2cpp+0x1637697`), reached right after the call.

Implemented it to match this HLE's policy (never leave an output uninitialized): write a valid empty
NUL-terminated string and report success. **Effect: the game goes from crashing at `SubmitDcb #46` to
running thousands of frames (`#6530`+) and actively streaming the cutscene scene.** How it was found: the
NID resolves to `sceSystemServiceParamGetString` (computed via `nid_hash`), and it was the last unimplemented
call logged before the crash; the faulting instruction (`cmpb $0x0, 0x20(%rbx)`, `rbx=null`) confirmed a
null-object deref.

## Where the game is now — actively loading the cutscene, then a per-frame Il2cpp crash

With the fix, the game:
1. Renders and fades in the loading screen (time-driven animation works — verified via frame dumps).
2. **Streams the cutscene scene's assets**: the loader thread `pread`s progressively through
   `resources.assets` (77 MB) to ~block 977 (≈82%), plus `resources.resource` / `sharedassets1.assets`
   (verified with `PROSPER_PREADLOG`; offsets increase monotonically — it is loading, not spinning).
3. **Crashes on the main thread inside a per-frame Il2cpp runtime call**, before the load finishes and the
   scene can activate.

### The crash (the remaining blocker)

Every crash shares one call chain from Unity's frame loop:
```
eboot+0x147b494 (frame loop) → eboot+0xa8c989 → Il2cpp+0x49d1 → …null-deref
```
`Il2cpp+0x49d1` is the return address of a `call 0x5df0` inside a **locked critical section** (acquire
`[0x267d8a8]` → check a flag → call) — an Il2cpp runtime per-frame pump (GC/finalizer-class). The actual
fault rip **varies across runs** (`Il2cpp+0x5980`, `+0x1637697`, and a `memcpy(dst, NULL, 8)` at
`libc.prx+0x4a20`), always a **null / `+0x20`-of-null** dereference. One reachable path leads to an Il2cpp
abort (`ud2` after loading an error string; the string table there holds `"Unexpected state."`,
`"No space for mark stack"` — GC messages).

**Interpretation (CONFIDENCE: MED):** the non-deterministic null derefs of managed objects, correlated with
the cutscene-scene load, point to **deserialization producing corrupt objects** (a managed reference left
null) that then crash when a per-frame update/GC/finalizer touches them — i.e. the same Unity
`CachedReader`/`FileCacher` deserialization frontier documented in `DESER_STALE_CACHE_BLOCK.md`
(loader callers `eboot+0x93a810` / `0xb1767e` match). GC-heap corruption under the scene's heavy allocation
is the alternative/compounding cause. It is **not** a worker-thread fault (0 observed) and **not** a missing
HLE call (no unimplemented call fires during the steady-state load).

## Also fixed — spurious `%fs` stack-smash abort in `fault_handler`

A second, intermittent fatal crash independent of the GC one: `*** stack smashing detected ***` (SIGABRT).
Caught with an `LD_PRELOAD` `__stack_chk_fail` interposer — the "smash" was inside our own SIGSEGV/SIGBUS/
SIGILL `fault_handler`, with `%fs:0x28` already equal to the host canary at abort time (no real overflow).
The handler is entered while the faulting thread runs on the **guest `%fs`**, then switches to the **host
`%fs`** mid-function (`guest_fs_enter_host_for_signal`); a `-fstack-protector` prologue reads its canary from
the guest TCB and the epilogue re-reads it from the host TCB, so any divergence trips `__stack_chk_fail`.
Fixed by marking exactly this `%fs`-switching handler `__attribute__((no_stack_protector))`
(`exec_image_linux.cpp`) — removes the false positive at its source. Eliminated one of the two intermittent
long-run crash modes (`smashing=0` across repeated runs); the incremental-GC crash below remains.

**The per-frame pump is Boehm's INCREMENTAL GC:** `Il2cpp+0x5df0` is a time-sliced work loop, and `0x5ee0`
is a 6-state jump-table mark/sweep machine driven by `[Il2cpp+0x267dee8]`. The remaining crash is a **race**
(point varies `SubmitDcb #20..#6530`) inside that collector — stop-the-world likely not covering a
scene-load thread. The scene never activates (draw shader/`color0_base` unchanged; reads plateau ~block 977),
because the GC crash kills the game during scene integration.

## KEY finding — the GC is collecting LIVE objects (root/marking bug), not just tripping over deser garbage

Added a bring-up diagnostic `PROSPER_PATCH_RET=addr[,addr...]` (boot_trace) that writes `0xC3` (ret) at a
guest address to neutralize a function. Patching the per-frame incremental-GC pump
(`PROSPER_PATCH_RET=0x440005df0`, = `Il2cpp+0x5df0`) **changes the render output**: `color0_base` goes from
always `0x0` to real render-target addresses (`0x…df790000`). So with the incremental GC stopped, the
scene's render targets **survive** — i.e. the incremental GC was **collecting live objects** (a missed
root / missed write during incremental marking), not merely dereferencing a corrupt deserialized object.
This redirects the root-cause hunt from "deser produces garbage" toward "our incremental-GC support drops a
live reference." **Decisive evidence (not a timing confound):** across a full unpatched run `color0_base` is
`0x0` for *every* draw — a real render target is *never* bound; with `0x5df0` patched, real render-target
addresses appear from the *first* draw. So the collector running is precisely what prevents the scene's
render targets from surviving. This dovetails with the parallel agent's data point that all 13/13 GC
stop-the-world suspends caught threads **parked in a futex** (host code), never an actively-mutating guest
thread — i.e. an active mutator may escape suspension, so the incremental mark proceeds over a heap that is
still changing and reclaims a still-referenced object. (Root capture in `exc_delivery_handler` itself looks
complete — all 15 GP regs + rsp + `ctx[0xf8]`; suspect the incremental write-barrier / active-mutator
suspension, or static-field root registration.)

****Cleanly separating the two root causes (via `PROSPER_PATCH_BYTE=0x440005dfe=0xeb`, which forces the pump's
`je` to an unconditional `jmp` so incremental marking never runs — no half-collection, unlike ret'ing the
whole pump):** real render targets survive from draw 1 AND the game deterministically advances to a *new*
crash at `SubmitDcb #28` — `Il2cpp+0x1637697`, same backtrace every run. Its caller (`Il2cpp+0x175c991`)
does `mov 0x40(%rbx),%rdi; call 0x16375e0`, i.e. it passes **`[callerObj+0x40]`, which is null**. That field
stays null with the GC OFF, so this second crash is an **uninitialized / mis-deserialized object field**, NOT
a GC-collected reference — i.e. the deser frontier the parallel agent owns. So reaching the cutscene needs
BOTH: (1) the incremental-GC live-object drop, and (2) the deser/init null at `[obj+0x40]`.

**Crash-free-but-stuck isolation (strongest handle on the wall):** disabling incremental marking
(`0x440005dfe=0xeb`) *and* stubbing the null-`this` getter to return 0
(`0x4416375e0=0x31,0x4416375e1=0xc0,0x4416375e2=0xc3` = `xor eax,eax; ret`) makes the game **run crash-free
for the full run** — but it stays on the loading screen (`color0_base=0x0`, the single blit). So the getter
`0x16375e0` is on the **scene-activation path**: returning 0 (because `[callerObj+0x40]` is null) makes the
game not advance; with the GC on but the getter crashing, real render targets briefly appear (the game *was*
starting to render the scene). Net: the crashes are fully avoidable via workarounds, and the pure remaining
blocker to the cutscene is **populating the null managed field `[callerObj+0x40]`** (an uninitialized / mis-
deserialized object) so the getter returns its real value and activation proceeds. That single field is the
next thing to fix; identifying `callerObj`'s type + what writes `+0x40` (via il2cpp metadata / the deser
path) is the concrete remaining task.

A second, INDEPENDENT null remains** even with the GC pump patched: the crash at `Il2cpp+0x1637697` is in
a function whose start (`0x16375e0`) sets `rbx = rdi` (its argument), then derefs `[rbx+0x20]` — i.e. the
**caller passed a null object**. This one is not GC-collected (it persists with GC off), so there are at
least two distinct corruption sources feeding the null-deref crashes; both must be resolved to reach the
cutscene. `PROSPER_NULL_PAGE` does not rescue these (the surviving faults include null writes / non-low
addresses → uncaught SIGSEGV).

## Next steps (for reaching the cutscene)

1. **Resolve the scene-load deserialization corruption** (the `FileCacher` stale-block issue in
   `DESER_STALE_CACHE_BLOCK.md`): a corrupt deserialized object is the most likely source of the null a
   per-frame update dereferences. Verify by dumping the object graph around the block-977 asset.
2. **GC stability under load**: confirm our stop-the-world root scanning covers all roots during the
   cutscene's allocation storm (the `"No space for mark stack"` string suggests GC mark-stack pressure).
3. **Rendering** (once the game survives to the cutscene): the cutscene's draws currently resolve with
   `color_write_mask==0` / `color0_base==0` and are skipped by the single-draw executor — the per-draw
   state / multi-draw executor work addresses this. The guest scanout buffer is black because our renderer
   targets Vulkan, not guest memory (`PROSPER_DUMP_SCANOUT` confirms).

## Reproduce
```sh
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_PREADLOG=1 \
  ./build/boot_trace /path/to/PPSA24651-app0
# watch resources.assets block offsets climb, then a main-thread SIGSEGV in the Il2cpp+0x49d1 chain.
```

## Follow-up (2026-07-07, second pass) — the wall is the GC stop-the-world race; it fires BEFORE scene-activation

Confirmed the "remaining Il2cpp crash" is the same **IL2CPP Boehm-GC `ABORT("Unexpected state")`**
documented in `CUTSCENE_GC_ABORT.md` (fault `rip=libc.prx+0x4a20` = `int 0x45 ; ud2`; abort arg
`rdi -> "Unexpected state"`; call chain `Il2cpp+0x5428 → +0x4a6f → +0x49d1`, identical registers
every run). It is a **timing-sensitive stop-the-world race** during the cutscene's asset-load
allocation storm, and it fires **before the scene activates** — so the cutscene is never drawn,
regardless of how deep the asset stream gets.

Measured behavior (this pass):
- **Deep survival is timing-dependent, not progress-dependent.** Plain run: crashes after ~2
  `resources.assets` reads. With `PROSPER_PREADLOG=1` (heavy per-read logging = serialization):
  **144 reads, block 2916** before the *same* GC abort. Adding/removing `PROSPER_PAD_PRESS` does
  not change the early crash. The extra serialization just shifts the race window — it does not
  eliminate it, and the scene still never activates.
- **The synchronous 1080p render makes it worse.** `execute_and_present` blocks the guest submit
  thread ~15 s inside host Vulkan; while blocked there the GC's stop-the-world can't get that
  thread's ack → earlier crash (~350 submits at full res). `PROSPER_RENDER_SCALE=6` (~0.5 s
  render) pushes that to ~2640 submits — but every captured frame is still the **title composite**
  (the scene never activated), then the same GC abort.
- **The guest scanout buffer is empty** (`PROSPER_DUMP_SCANOUT`, all-zero to flip #2700): our
  renderer targets a Vulkan image, not guest memory, so there is no cheap non-render capture path.

Attempted and ruled out this pass (details in `CUTSCENE_GC_ABORT.md`): TSD-exit-`%fs`, GC-handler
guest-`%fs`, and env-based GC-disable (both env delivery paths blocked). A **short-read fix**
(`read_full`, this branch) is a real correctness bug on the asset-streaming path but does not
remove the GC abort.

**Bottom line:** reaching the cutscene visually requires fixing the GC stop-the-world race
(top open lead: a Unity job-worker created/exiting *during* a stop-the-world — see
`CUTSCENE_GC_ABORT.md`), AND then the per-draw/multi-draw executor (PR #31) to render the
cutscene's `mask=0`/`color0_base=0` draws. Neither is a small change.

## 2026-07-07 (third pass) — CORRECTION: the async scene load COMPLETES; the crash is AT activation, not a load stall

Independent re-verification on the current tip (`334dc7f`), `PROSPER_GUEST_FS=1
PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_PREADLOG=1`. Earlier passes said the load "plateaus ~block 977
(≈82%)" and "crashes before the load finishes and the scene can activate". That framing is **wrong** — the
async load runs to completion:
```
[preadlog] pread  resources.assets … off=0x49f0000 (blk 1183, ≈97% of the 77.6 MB file)
[preadlog] close  sharedassets1.assets
[preadlog] close  level1
[preadlog] close  resources.assets      <-- Unity closes every scene file: the async load reached 100%
=== RUN ENDED: SIGSEGV at addr=0x20  rip=Il2cpp+0x1637697 ===   <-- crash fires immediately after
```
So the crash is **at scene ACTIVATION** (the main-thread integrate/activate step that runs right after the
async load closes its files), not a mid-load stall and not an `allowSceneActivation`/progress gate. Every
one of the preceding ~3000–7500 submits is the level0 1-draw loading composite; the cutscene's multi-draw
geometry is never submitted because the process dies the instant activation begins. Also re-confirmed this
pass: no `unimplemented -> returning 0` HLE call fires at activation (last calls are all implemented AGC
direct-mode), and the `Il2cpp+0x1637697` null (`[callerObj+0x40]`) persists with the incremental-GC pump
patched to `ret` (`PROSPER_PATCH_RET=0x440005df0`) — so this null is the deser/init one, independent of the
GC. Which specific terminal fault wins (`Il2cpp+0x1637697` activation-NRE vs. the `libc.prx+0x4a20` GC
"Unexpected state" abort) is timing-dependent, but both fire at/just-before activation. **Net: the single
remaining blocker is unchanged — populate the null managed field `[callerObj+0x40]` on the activation path
(and/or fix the GC stop-the-world race) — but the scene now provably loads to 100% first.**

## UPDATE (post-#42) — crash is now DETERMINISTIC and identified: `WorkerThread.field_0x40` is null

After #42 (static-mutex/GC-lock fix) landed, the level1 crash changed character completely:
- **Before #42:** random crash signatures (`0x500004a20` GC abort, `0x44000b03b`, `0x44017fc23`, …) —
  the fingerprint of heap/free-list corruption from the never-locked GC allocation lock.
- **After #42:** 4/4 repros crash **identically** at `Il2cpp+0x1637697`, `kind=2`, after exactly 2
  `resources.assets` reads. The corruption randomness is gone; this is now a single deterministic bug.

**Identified via `PROSPER_HWBP_KLASS` (new, this branch)** — a bp at the caller (`Il2cpp+0x175c991`,
`mov rdi,[rbx+0x40]; call getter`) dumps the il2cpp class name of the receiver:
```
[hwbp-klass] rbx obj=… klass=… name="WorkerThread"    [obj+0x40]=0x0     <- the null field
[hwbp-klass] r14 obj=… klass=… name="AsyncRequest`1"  [obj+0x40]=<valid> <- the node being iterated
```
So a **`WorkerThread`** managed object has a **null field at +0x40**, and the caller loop (iterating a
list of `AsyncRequest\`1`) calls a property getter on that null → `cmpb 0x20(%rbx)` with `rbx=null`
(`Il2cpp+0x1637694`). This is Unity's **async asset-loading job system**; `WorkerThread.field_0x40` (a
managed object the getter reads `[+0x20]` from) is never initialized before the main thread accesses it.

**Next (the pure remaining blocker):** find what sets `WorkerThread.field_0x40` and why it's null here —
candidates: the worker's C# init/ctor didn't run, or a native icall backing it returns wrong. It is now a
clean, deterministic, single-field bug (no longer corruption), so a data write-watch on that field or the
`WorkerThread` ctor is the decisive next probe.

### Stub test + full backtrace (confirming the field must be REAL, not bypassed)

Stubbing the getter to `xor eax,eax; ret` (`0x4416375e0=0x31,c0,c3`) — the workaround the parallel agent
tried pre-#42 — was RE-TESTED on merged master (with #42's corruption fixes): the game now runs
**crash-free for 386+ frames** (no crash), BUT stays **stuck** — only 3 `resources.assets` reads, every
frame a plain blue clear, the cutscene scene never activates. So the getter's real return value drives
async-load progression; returning 0 stalls it. **The field must be genuinely populated, not bypassed.**

Full crash backtrace (main frame loop → Unity PreloadManager async-scene integration → managed getter):
```
Il2cpp+0x1637697 (cmpb 0x20(%rbx), rbx=WorkerThread.field_0x40=null)
Il2cpp+0x175c99c → +0x618556 → +0x617fff → +0x16ba41 → +0x16b981
eboot +0xcc8e5f → 0xd36ebb → 0xce8671 → 0xce8eb3 → 0xce8593  (SerializedFile / PreloadManager integrate)
eboot +0xd4f607 → 0xd4f3c7 → 0xb03e32 → 0xb03c24 → 0xb05e7b → 0xb06d3a → 0xae47e1 → 0xae4836
eboot +0x147b483 (Unity main frame loop) → 0x1485851 → 0xaf
```
So the remaining, precisely-scoped blocker: **the managed `WorkerThread` object's `+0x40` field is never
initialized before Unity's PreloadManager integrates the async scene load on the main thread.** Next probe:
a write-watch on `[WorkerThread+0x40]` from the object's allocation to find its intended writer (the
WorkerThread ctor / a native icall), or resolve the getter's declaring class to name the expected type.

### Write-watch: `WorkerThread.field_0x40` is NEVER written (confirmed missing init)

Armed a HW write-watch on `[WorkerThread+0x40]` (`PROSPER_HWWATCH=0x40 PROSPER_HWWATCH_REG=rbx` at the
caller bp) with the getter stubbed so the game runs freely. Result: the watch armed, the game ran, and
**no writer of that field was ever caught on the main thread** — the field is simply never initialized.
Also observed: with the getter bypassed the async loader **stalls after 3 `resources.assets` reads**
(386 blue frames, no progress) — integration and the loader are coupled, so the missing field both
crashes integration AND leaves the scene unloaded.

Conclusion: the blocker is a **missing initialization of the managed `WorkerThread.field_0x40`** in Unity's
async-scene-load path (PreloadManager). It is deterministic (post-#42), never written by the main thread,
and cannot be bypassed (stub → scene never activates). The fix requires identifying the field's intended
initializer — either the `WorkerThread` ctor path (if main-thread) or a worker/preload thread's setup
(a HW watch is per-thread, so a worker-thread writer would not appear in the main-thread watch above).
This is Unity async-load engine internals; handing off with the exact field + backtrace + repro.

### RESOLVED the type: `WorkerThread.field_0x40` is a null `System.Diagnostics.Stopwatch`

`PROSPER_HWBP_GLOBAL=0x442302518` resolves the getter's declaring class from its class-global:
**`name="Stopwatch"`**. So `WorkerThread.field_0x40` is a `System.Diagnostics.Stopwatch` instance that
Unity's async loader uses to time each `AsyncRequest`, and it is null. Our timer HLE
(`sceKernelGetProcessTimeCounter`, `clock_gettime`, `sceKernelReadTsc`) is implemented, so the Stopwatch
*class* works — the *instance* is simply never created for this WorkerThread.

`PROSPER_HWBP_FIELDS=1` dumps the WorkerThread object — it is **partially initialized**, NOT a raw
uninitialized object:
```
WorkerThread @…: +0x10=0 +0x18=0 +0x20=<obj> +0x28=<obj> +0x30=<obj> +0x38=0 +0x40=0(Stopwatch) +0x48=<obj> +0x50=0 +0x58=0 +0x60=<obj>
```
Several fields hold real objects (its ctor ran); only the Stopwatch at +0x40 (and a few others) is null.
So the Stopwatch is created **later than the ctor** — most likely lazily on first use, or in the worker
thread's `Run()` (to time its own work). The main thread's PreloadManager integration reaches the timing
loop and reads the Stopwatch **before** it is created → deterministic null deref.

**Precise remaining fix (for whoever owns Unity async-load):** ensure `WorkerThread`'s Stopwatch (+0x40)
is created before PreloadManager's main-thread integration times its AsyncRequests — i.e. the worker's
`Run()`/lazy Stopwatch setup must complete first (a start-ordering / worker-not-yet-run condition), OR the
timing path must tolerate a not-yet-started worker. New diagnostics on this branch: `PROSPER_HWBP_KLASS`,
`PROSPER_HWBP_GLOBAL`, `PROSPER_HWBP_FIELDS`.

### Null-guard experiment: Stopwatch is crash #1, but a SEPARATE async-load stall (#2) sits behind it

Added `PROSPER_NULLGUARD=addr,len` (boot_trace): a trampoline that returns early (0, or an rdtsc counter)
when a guest method's receiver (`rdi`) is null. Installed on the Stopwatch getter, it **eliminates the
crash** — the game runs 700+ frames with no fault. But the cutscene still does NOT render:
- The async loader **stalls at exactly 3 `resources.assets` reads** and every frame is a blue clear.
- Tested BOTH guard returns — `0` (unstarted-timer) and `rdtsc` (large monotonic). **Both stall at the
  same 3 reads**, so the stall is NOT a Stopwatch-timing-magnitude artifact — it is a genuine SECOND
  blocker in the async-scene-load path, sitting directly behind the Stopwatch crash.

So reaching the cutscene needs (in order): #42 (done) → the `WorkerThread` Stopwatch init (crash #1) →
whatever stalls `resources.assets` streaming at 3 reads (#2 — the loader stops issuing reads; likely
waiting on a main-thread integration step that the missing-Stopwatch path never reaches on real HW).
The layers keep peeling — each is Unity async-load engine internals. `PROSPER_NULLGUARD` is kept as a
gated diagnostic (default no-op) for bisecting null-receiver crashes.

## 2026-07-08 — GC-suspend guest-%fs landmine FIXED; post-#43 blocker is an async-load *completion stall*

Built on merged #43 (the null `System.Diagnostics.Stopwatch` at `WorkerThread+0x40`). Findings:

**1. Fixed a real GC-suspend bug (this branch).** The IL2CPP GC stop-the-world handler (exception
type `0x1e`) keeps the guest `%fs` to run the guest's suspend handler, but logged with glibc `write()`
— a cancellation point whose prologue reads `THREAD_SELF` at `%fs:0x10` (0 on our guest TCB) and
dereferences `self->cancelhandling` at `+0x308`. That is the `SIGSEGV addr=0x308 rip=<libc>` seen under
`PROSPER_SYNCLOG` (faulting insn `mov %fs:0x10,%rcx; mov 0x308(%rcx),%edx`). Switched to raw
`syscall(SYS_write)`. A latent guest-`%fs` landmine in the suspend path; the fix also re-enables
`SYNCLOG`-instrumented GC tracing (which previously self-crashed).

**2. `GC_DONT_GC=1` is a NO-OP here.** Injecting it via `PROSPER_GUEST_ENV` does **not** stop the world —
type-`0x1e` suspend cycles still fire with it set. IL2CPP's bundled Boehm does not honor that env knob in
this build. (An earlier "determinism" reading was actually just the absence of `SYNCLOG` timing overhead,
not GC being disabled. Corrected.)

**3. `PROSPER_ONE_CPU` (new, gated) rules out core-count as the worker driver.** Reporting a 1-core
affinity mask (`scePthreadAttrGetaffinity` → `0x01`) does **not** reduce the ~47 `Loading.PreloadManager`
threads — they are cumulative async-load churn, not a core-sized pool. So single-threading the job system
via affinity is not the lever.

**4. The true post-#43 blocker is an async-load COMPLETION STALL, not a crash.** With the persistent
`PROSPER_NULLGUARD` bypass of the #43 Stopwatch crash, the game does **not** crash — it runs 270+ rendered
frames. It fades from the title to black to load the cutscene (`level1`), and the loader **does stream the
cutscene's assets**: it opens `level1`, `resources.assets`, `resources.assets.resS`, and reads ~28
`resources.resource` chunks (the real texture/audio blob). Then the load **stalls**: the worker thread
pool keeps churning on its per-thread futexes but the load never *completes/integrates*, so the scene
never activates — every frame stays a static black clear. Reaching the cutscene now means resolving why
async-load integration never finishes after the assets stream in (an engine-level PreloadManager step),
not fixing a fault. (Note: a one-shot fault-handler unwind of the Stopwatch getter — `PROSPER_SKIP_WT_GETTER`
— instead diverges to a `%fs:-0x80` thread-local artifact at `eboot+0x9f1ba0`; the persistent `NULLGUARD`
is the correct, non-diverging bypass and gets strictly further.)

**5. The stall is INDEPENDENT of the Stopwatch return value, and of APR.** Added `PROSPER_NULLGUARD_RET`
(`tsc`|`zero`|`ms`) to select what the guarded null-receiver returns. All three **stall at the same 28
`resources.resource` reads** — so the time-budget/unit theory is wrong; the guard's return value does not
gate the stall. (`ms` holds on a dark-blue/purple screen, `tsc` fades to black — cosmetic difference only,
neither reaches the cutscene text card, which has white text; the stalled frames have 0% white.) Also: the
"not considered suitable for apr reads flags:0x0" line is the GAME's own `LocalFileSystemPS5` decision and
prints for **every** file (including ones that load fine during boot), so APR-rejection is not the blocker.
Net: the wall is specifically Unity's async **load-completion / integration** after the scene's bytes have
streamed in — the main thread stays in its frame loop (renders 90+ frames) but the operation never reports
complete. That main-thread integration step is the precise next target.
## Hypotheses TESTED AND ELIMINATED for crash #1 (`WorkerThread.+0x40` null Stopwatch)

A parallel session (95a32e9e) working on top of merged #42/#43 tested and ruled out the following as the
cause of the never-initialized Stopwatch field — recorded so they are not re-tried:

- **Thread-start ordering race** — added a `scePthreadCreate` start-handshake (creator waits for the new
  thread to begin running + a slice budget for its `Run()` init) gated by `PROSPER_TSTART`. Crash unchanged
  (`Il2cpp+0x1637697` still). Workers start and park normally, so it is **not** a "worker hasn't started" race.
- **Thread-creation failure** — instrumented `k_pthread_create` to log any `pthread_create` that returns
  nonzero: **0 failures** the whole run. The worker threads are created successfully.
- **Stopwatch timestamp/frequency backing** — `sceKernelReadTsc`/`GetTscFrequency` return a sane 1 GHz and
  `ns_now()` is monotonic, so `Stopwatch.StartNew()` cannot divide-by-zero / throw on the timer path.
- **CPU-count / .NET ThreadPool starvation** — the game imports **no** `sysconf`/`sched_getaffinity`/
  `GetCpumaskInfo` (only `UnityEngine.SystemInfo::GetProcessorCount`, backed by `scePthreadGetAffinity`
  which we already return as 8 cores). So `Environment.ProcessorCount` is not low; the managed pool is not
  CPU-count-starved.
- **Null-bypass with a fake value** — `PROSPER_NULLGUARD` returning `rdtsc` did NOT stall at 3 reads here; it
  **looped** re-reading `unity default resources` to **4096 preads → OOM**. Returning 0 stalls (per #43).
  Confirms both: the field must be *genuinely* populated, and crash #2 (loader mis-behaving) is real.
- **Missing HLE** — all `unimplemented → 0` NIDs fire once at startup; none during the steady-state load.

**Architecture traced:** the one managed thread created has `Run()` at `Il2cpp+0x1cac50` — a **.NET
ThreadPool worker** (atomic work-count on `[r15+0x10]`, then `call *%r12` to dispatch a queued delegate).
So the Stopwatch-init runs as a **dispatched managed work item that never executes** in our env; the scene
still loads (block 1183) because the *native* Unity job workers do the streaming, but the managed work item
that would init `WorkerThread.+0x40` is the gap.

**Decisive next probe (untried — needs process-wide tooling):** a hardware WRITE-watch on
`[WorkerThread+0x40]` armed from the object's *allocation* across *all* threads (not the caller bp, which is
after the write should have happened — that main-thread-only watch caught nothing). It answers definitively:
does some worker/pool thread write the field (a dispatch/visibility race) or does no thread ever run the
init (a never-dispatched work item)? That points straight at the fix.
