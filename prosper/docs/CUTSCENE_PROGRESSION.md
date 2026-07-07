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
