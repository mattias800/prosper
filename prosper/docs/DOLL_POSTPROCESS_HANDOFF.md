# DOLL (DQ VII Reimagined, PPSA17942) post-process recompiler frontier — handoff

**Status as of 2026-07-19.** This document hands off the DOLL post-process compute-shader
recompiler work **and** the development-environment change (WSL-on-Windows → native Linux). Read
the environment section first if you are setting up the new machine.

---

## TL;DR

- The DQ VII title screen is **black after the boot splashes**. Root cause is the shared **#319
  composite**: the game renders the scene into a buffer, then a UE4 post-process chain multiplies it
  by color-grading / exposure / volume surfaces that **compute shaders produce**. Several of those
  compute shaders **failed to recompile** (RDNA2→SPIR-V), so their surfaces were never produced and
  the composite collapsed to ~0 (near-black, `max` pixel = 1/255).
- This session drove the count of DOLL post-process shaders that fail to recompile from **4 → 1**,
  via four merged PRs (see below). **The title is still black** — the last shader plus a
  re-verification of the composite remain.
- **The one remaining shader (`exec_cs_201a5a0000`, run-local address) is blocked by NESTED loops.**
  Everything else in DOLL's post-process opcode surface is now covered.
- **You are moving to native Linux with a real GPU.** This is a big deal: WSL only had `llvmpipe`
  (software Vulkan), so every render capture took 3–7 minutes. On real hardware Vulkan the same
  captures take **seconds**, which makes the remaining loop-structurizer work far more iterable.

---

## What landed this session (all merged to `master`)

| PR | What | Reject count |
|----|------|--------------|
| **#1028** | Compute storage-image **formats**: `Uint8/Sint8/Uint16/Sint16/Unorm2_10_10_10` pack/unpack in `frontends/shared/live_compute.cpp`. DOLL's 3D color-grading LUT / exposure / volume dispatches were *skipped* (`storage format has no channel pack/unpack yet`). | (skips) 5 → 0 |
| **#1041** | Recompiler: integer sub-dword **`buffer_load_format`** at a runtime byte address (`dyn_int` path in `src/gpu/rdna2_to_spirv.cpp`). A stride-1 `Uint8` table load rejected as `[mubuf-unaligned]`. | 4 → 3 |
| **#1044** | Recompiler: the full **MUBUF 32-bit atomic RMW family** (`swap/add/sub/smin/umin/smax/umax/and/or/xor` → `OpAtomic*`). Only `umax` was implemented before; DOLL uses `buffer_atomic_add` (op 0x32). | 3 → 2 |
| **#1053** | Recompiler: integer sub-dword **`buffer_store_format`** via race-free `atomicAnd(clear field)` + `atomicOr(set field)` (`dyn_int_store` path). A stride-2 `Uint16` store rejected as `[mubuf-unaligned]`. | 2 → 1 |

Also: a **CLAUDE.md policy** was added — *an unsupported shader/GPU op is a fatal gap to implement,
not a silent skip.* And the split "opcode coverage vs CFG structurizing" is now explicit:

- **Opcode coverage** (instruction encodings/semantics from the AMD RDNA2 ISA) — now **complete for
  DOLL's post-process path**.
- **CFG structurizing** (reconstructing SPIR-V *structured* control flow from RDNA2's arbitrary
  branches) — this is the recompiler's own algorithmic problem, **not** dictated by the ISA doc, and
  it is the one remaining gap.

Each recompiler PR shipped with a real-Vulkan **execution** test in
`tests/test_rdna2_to_spirv.cpp` (e.g. the atomic accumulates 128×3=384; the sub-dword store lands two
disjoint 16-bit fields into a shared dword). Each was independently code-reviewed.

---

## The remaining work

### 1. The last post-process shader: NESTED loops (the deep piece)

`exec_cs_201a5a0000` (run-local addr; also seen as `201a520000`). 224 instructions. Its control flow:

```
pc 34: s_cbranch_execz -> 78     (guard skipping the whole nested-loop region)
pc 42: OUTER loop header
  pc 43: s_cbranch_execz -> 75   (outer exit)
  pc 50: INNER loop header
    pc 51: s_cbranch_execz -> 70 (inner exit)
    ... inner body ...
    pc 69: s_branch -> 50        (INNER back-edge)     <-- inside the outer body
  pc 70: (inner exit target)
    ... outer body tail ...
  pc 74: s_branch -> 42          (OUTER back-edge)
pc 75: (outer exit target)
...
pc 83..190: s_barrier (SOPP 0x0a) x11 + more code (this part is fine)
```

The inner loop `[50,69]` is **entirely inside** the outer loop `[42,74]` → genuinely nested.

**Where the structurizer is** (all in `src/gpu/rdna2_to_spirv.cpp`):
- `detect_divergent_loops` (~line 2667) — collects `DivLoop`s. It **already handles multiple
  *sequential* (disjoint) loops** (line ~2693) and top-level `s_barrier`. It **explicitly rejects
  nested loops** at line ~2706 (`if (in.simm16 < 0) return {}; // second back-edge inside -> nested`).
- The single-loop SPIR-V **emission driver** is ~lines 7250–7450 (`emit_loopmerge`, the check block,
  the condition region `[header, exit_branch)`, the body `[exit_branch+1, backedge)`, phi collection
  for loop-carried registers via `loop_written_regs`, exec save/restore). It is written for a flat
  list of sequential loops.

**What implementing nested loops requires:**
1. Let `detect_divergent_loops` accept one level of nesting: detect that an inner back-edge lies
   inside an outer loop's body and record the parent/child relationship (rather than `return {}`).
2. Make the emission **recursive**: when emitting the outer loop's body range, if an inner loop's
   header/back-edge falls within it, emit a nested `OpLoopMerge` (inner) inside the outer loop body,
   with the inner loop's phis/exec nested inside the outer's.
3. Preserve the existing **per-invocation exec model** (each lane iterates while its EXEC bit holds;
   see the long comment at ~line 2459). The inner loop's exec state must nest correctly under the
   outer's.

This is correctness-critical: a wrong structurizer silently emits wrong pixels. Gate it with
`spirv-val` + an execution kernel (a nested-loop compute kernel with a known numeric result) **and**
the live boot A/B (does the shader recompile + does the title composite change). Do **not** ship it
on recompile-success alone.

### 2. Re-verify the composite once all four LUT generators recompile

Even with the last shader fixed, confirm the title actually renders (it may expose a *further*
composite issue — this is the #319 family). Capture frames with `PROSPER_FRAME_DIR` (below) and check
that frames past the splashes have real content (`max` pixel ≫ 1). As of this handoff, with 3 of 4
shaders recompiling, the post-splash title is still `max=1` (near-black). Splash logos (frames ~28–35)
render fine at `max=255`.

### 3. Tracking

The umbrella issue is **#590**. It has the full reject map and the 4→1 progression as comments
(2026-07-19). File the nested-loop structurizer as its own focused issue if you want a clean
`Fixes #NN`.

---

## Environment migration: WSL-on-Windows → native Linux

This is the biggest practical change. The old setup drove **git from PowerShell** (Windows git) and
**built/ran in WSL Ubuntu-24.04**, because worktrees created on Windows stored a Windows-path gitdir
link that WSL's git couldn't resolve. On native Linux, **all of that goes away** — it is just a
normal Linux checkout.

### What changes for the better

- **Real GPU Vulkan instead of `llvmpipe`.** WSL exposed only `llvmpipe` (software rasterizer), so
  live renders were minutes-long. On a native box with an NVIDIA/AMD GPU and its Vulkan ICD, renders
  run on hardware — **seconds, not minutes.** This transforms the iteration loop; the render captures
  in this doc that took ~400s will be quick. Verify with `vulkaninfo | grep -i "deviceName"` — you
  want your real GPU, not `llvmpipe`. Install the vendor Vulkan driver + `vulkan-tools`.
- **One shell.** No more `wsl -d Ubuntu-24.04 -e bash -lc '…'` wrappers and no more
  PowerShell-for-git / WSL-for-build split. `git`, `cmake`, `ctest`, and the tools all run in the
  same native shell.
- **No `/mnt/c`.** Paths are just normal Linux paths.

### Things to redo / watch on the new machine

- **Line endings.** The repo's tracked files are **CRLF** (a Windows artifact). On Linux the editors
  and tools produce **LF**, and a careless whole-file save will show up as a massive EOL diff. Two
  options: (a) add a `.gitattributes` that normalizes text to LF and do a one-time renormalize commit
  (cleanest long-term now that Windows is out of the picture), or (b) keep matching CRLF per-file as
  before. Whatever you choose, **check `git diff --stat` before committing** — a change that touches
  60 lines should show ~60 changed lines, not ~1700. (The old workflow re-CRLF'd edited files with a
  small Python pass; on an all-Linux repo you can retire that and normalize to LF.)
- **The game dump.** `PPSA17942-app0` was copied to `/root/PPSA17942-app0` (WSL ext4) for fast IO,
  separate from the gitignored Windows copy. On native Linux, put the dump on a fast local disk and
  point `-DGAME_DUMP=<path>` and the runtime at it. It is **gitignored — never commit it.**
- **Worktrees.** They work normally now (`git worktree add …`). The multi-agent worktree discipline
  from CLAUDE.md still applies (work in your own worktree; the main checkout is shared).
- **Build dirs.** `prosper/build-linux` (primary). `cmake -S prosper -B prosper/build-linux
  -DGAME_DUMP=<dump>` then `cmake --build prosper/build-linux -j$(nproc) && (cd prosper/build-linux &&
  ctest)`. Expect the 3 Messenger-dump-specific tests (`module_loads_eboot`,
  `boot_reaches_first_syscall`, `real_shader_render`) to fail **only** if the build is configured with
  a non-Messenger dump (e.g. the DOLL dump); they are environmental, not real regressions. Configure a
  Messenger-dump build to get a fully green `ctest`.
- **`llvm-mc` / SPIR-V tools.** The recompiler tests assemble RDNA2 with `llvm-mc` (gfx1010 target)
  and validate with `spirv-val`/`spirv-dis`. Install LLVM + `spirv-tools` so those are on `PATH`.

### Reproduction: boot DOLL and see the reject state

```bash
# from prosper/ , with a DOLL-dump build of boot_trace:
PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_GUEST_ARGS= PROSPER_RENDER=1 PROSPER_DBG=1 \
  PROSPER_SHADER_DUMP=/tmp/doll_dump \
  ./build-linux/boot_trace <path-to-PPSA17942-app0> 2> boot.log

grep -c "skip unsupported program" boot.log      # expect 1 (the nested-loop shader)
grep "skip unsupported program" boot.log | sort -u
# reject census (which op fails, straight-line vs cfg path):
grep -hoE "\[.*recompile-reject\] pc=[0-9]+ .*op=0x[0-9a-f]+" boot.log \
  | sed -E "s/pc=[0-9]+/pc=N/" | sort | uniq -c | sort -rn
```

`PROSPER_GUEST_ARGS=` is **empty** for DOLL (UE4) — do NOT pass `-force-gfx-direct` (that's the
Messenger/IL2CPP switch). On real-GPU Linux you can drop the old `llvmpipe` slowness workarounds
(`PROSPER_RENDER_SCALE`, `PROSPER_RENDER_EVERY`, timed sampling) entirely — keep the faithful
defaults.

---

## Diagnostic tooling reference (the force multipliers)

All gated, off by default. Names are `PROSPER_*` env vars unless noted.

- **`PROSPER_DBG=1`** — the recompiler prints why it rejects an instruction:
  - `[recompile-reject] pc=… words=… fmt=… op=0x…` — straight-line emit reject (the CFG structurizer
    did **not** accept the shape; `fmt=4`=SOPP branch, `fmt=12`=MUBUF).
  - `[cfg-recompile-reject] pc=… fmt=… op=0x…` — the CFG **did** structurize but an instruction in it
    couldn't be emitted.
  - `[mubuf-unaligned] pc=… fmt=… idxen=… stride=…` — a packed/sub-dword buffer op whose element base
    isn't provably aligned. `fmt` here is the `DataFormat` enum (7=Uint16, 11=Uint8, 21=Unorm2_10_10_10).
  - `[mubuf-unresolved] … srsrc=sN …` — a buffer op whose V# descriptor couldn't be resolved.
- **`PROSPER_DYNTRACE_FAIL=1`** — for a compute dispatch that fails to recompile, dumps the **user
  SGPRs** and replays `resolve_dynamic_fetch`, printing each recovered descriptor (`[dynfail] …`,
  `[dyntrace] MUBUF fetch pc=… SRSRC=sN … have_descr=…`). This is how you read a shader's actual V#
  format/stride/base for a failing dispatch (e.g. it revealed the stride-2 Uint16 store). Add
  **`PROSPER_DYNTRACE_FAIL_ADDR=<hex code address>`** to replay only that exact failed program; a BVH
  trace prints all four live descriptor words even when their provenance is insufficient to bind it.
- **`PROSPER_SHADER_DUMP=<dir>`** — writes each failing shader's raw bytes to
  `<dir>/exec_cs_<addr>.bin` (also `exec_vs_/exec_ps_`). Decode offline with **`shader_inspect
  <file>`** (built tool): per-dword PCs, decoded fmt/op, branch targets. This is how you read a
  shader's CFG (back-edges, execz/vccz guards, barriers) without re-booting.
- **`PROSPER_SHADER_DUMP_SUCCESS=<dir>`** — dumps successfully-recompiled RDNA2/SPIR-V pairs for
  inspection.
- **`PROSPER_FRAME_DIR=<dir>`** (boot_trace) — dumps every rendered frame as a BMP, bypassing the
  screenshot tool's distinct-frame gate. Check brightness to see if/when the picture goes black. The
  `[render] frame N rendered … nz=…` log line reports the render-buffer non-zero count; the BMP is the
  presented front buffer (the #319 gap is: render buffer has content, BMP is black).
- **`PROSPER_RESOURCE_HASH_DIM=WxH` / `PROSPER_PROVENANCE_DIM=WxH`** — at a target extent, correlate a
  surface's raw vs sampled hash and *who wrote it* (color/compute/DMA), to prove "scene content exists
  but is multiplied by a zero mask" (the #319 signature).
- **`PROSPER_RTTLOG=1`** — per-pass flip state + `px_nonzero`/`rgb_nonblack` + sample HIT/miss.
- **`PROSPER_GFXLOG=1`** — general graphics diagnostics; **`PROSPER_COMPUTELOG=1`** — compute dispatch
  trace.

### Method that worked (reuse it)

1. Boot with `PROSPER_DBG=1 PROSPER_SHADER_DUMP=…`; take the reject census.
2. For each failing compute shader, `PROSPER_DYNTRACE_FAIL=1` to read its V#s, and `shader_inspect`
   the dump to read its CFG.
3. Separate **opcode gaps** (implement the op + an execution test) from **CFG-shape gaps** (the
   structurizer). Fix opcode gaps first — they're bounded.
4. After each fix, re-boot and confirm the `skip unsupported program` count drops; when all are
   handled, capture frames with `PROSPER_FRAME_DIR` and verify the composite.

---

## Key source locations

- `prosper/frontends/shared/live_compute.cpp` — the live Vulkan compute backend. Storage-image
  format pack/unpack (`storage_(un)pack_texel`, `storage_(un)pack_supported`), upload/writeback,
  tiling, DCC.
- `prosper/src/gpu/rdna2_to_spirv.cpp` — the RDNA2→SPIR-V recompiler.
  - MUBUF/MTBUF buffer ops (loads, stores, atomics, `dyn_int`/`dyn_int_store`): the big `case
    Rdna2Format::MUBUF/MTBUF` block (~line 4960+).
  - `cbuf_atomic_rtn` (~975) — the generic atomic emit (any `OpAtomic*`).
  - `detect_divergent_loops` (~2667), the emission driver (~7250–7450) — **the nested-loop work.**
  - `detect_forward_if` / multi-if (~2763+), `safe_execz_branches` (~2126).
- `prosper/src/gpu/gpu_executor.cpp` — `resolve_dynamic_fetch` (~1262), `add_compute_buffer_resources`
  (~1964), the compute dispatch path + the `PROSPER_DYNTRACE_FAIL` dump (~2942).
- `prosper/src/gpu/agc_shader_layout.cpp` — `decode_buffer_descriptor` (~116), `rdna2_buffer_format`
  (the V# format-field → `DataFormat` table), `gen5_image_format`.
- `prosper/tests/test_rdna2_to_spirv.cpp` — recompiler execution-differential tests (the atomic +
  sub-dword store cases added this session are near the "kernel 24" block).
- `prosper/tools/shader_inspect/` — the offline RDNA2 decoder.

## AMD reference

RDNA2 ISA reference (document 70648) — the authoritative opcode/encoding/semantics source (linked in
CLAUDE.md). PS5-specific AGC/Gen5 behavior still needs live title evidence + focused tests. The PS5
3.20 firmware NID↔name database is the sibling `../PS5-3.20_Libs/` (gitignored).
