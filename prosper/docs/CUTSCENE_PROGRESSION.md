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
