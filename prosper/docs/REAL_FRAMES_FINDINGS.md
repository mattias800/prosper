# Real game frames — findings & fixes (2026-07-07 "nuclear" run)

Goal: get *The Messenger* rendering real frames in prosper. This maps the **complete path** to a real title
frame, discovered this session via a multi-agent investigation. It lists what's fixed, what remains, and the
exact root cause + next step for each remaining item.

## TL;DR — the game runs; the title composite is one recompiler bug + one register-apply bug away

After the merged flip/event fixes, the game **no longer stalls** — it runs a healthy frame loop (~23 fps
native; 6435 GPU submits in 280 s, no input polling, no per-frame file I/O). Each frame it draws a fullscreen
**composite that samples the loaded 1920×1080 title/splash texture**. That composite renders the **actual
title art** (proven: **1992 distinct colors** via a reference fullscreen VS + the game's real PS + a detiled
texture). Getting the game's *own* shaders to render it needs four things — two fixed here, two identified
with exact root causes:

| # | Item | Status | Root cause / fix |
|---|------|--------|------------------|
| 1 | Tiled texture not de-swizzled | ✅ fixed here | T# is `tile_mode=5` (GFX10 SW_4KB_S). Threaded `tile_mode` into `ShaderResource`; renderer auto-detiles. |
| 2 | Color writes disabled (`CB_TARGET_MASK=0`) | ⚠️ worked-around here; real fix identified | The game DOES write `0x8e`: **`0xF` for the composite** template and **`0x0` for a depth-clear pass** in the same indirect block. Our **flat last-wins apply** (`command_processor.cpp`) lets the clear's `0x0` clobber the composite's `0xF`, and `execute_gpustate` renders with the submit-final register file. Real fix = **per-draw state capture + AGC indirect-log segmentation** (each draw consumes only its own `0x8e`-terminated template). Workaround in this branch: default `CB_TARGET_MASK` `0→0xF`. |
| 3 | Real VS outputs degenerate UV | 🔗 fixed in a sibling PR | VS and PS both numbered descriptor bindings from a shared **set 0** → invalid descriptor layout → **corrupted VS reads → degenerate geometry/UV** → the PS samples one texel (uniform). Fix (concurrent agent): VS = set 0, PS = set 1. |
| 4 | Title *scene* never loads (deeper) | ❌ open | See "Deeper blocker" below. |

The modulate color is a **title fade-in** (`ff000000 → … → ffffffff`, black→white over ~24 frames) — *not*
a bug; it completes to full white. (This killed the earlier "black composite" red herring.)

Note on determinism: llvmpipe is ~400× slower than the native loop, so a "good" captured frame (title
texture loaded, before the level0→level1 transition) is timing-dependent under the live renderer — the same
code yields 1992 colors on a lucky run and 1 color on an unlucky one. The fix set is correct; capture is flaky.

## Deeper blocker: the title SCENE crashes on load (loader-thread SIGSEGV)

The composite is Unity's near-empty **level0** boot scene. After ~80 s at level0 the game organically starts
the async **level1 (title scene)** load — streams `/app0/Media/level1/resources.assets` (77 MB) in 64 KB
preads on a dedicated loader thread. **~60 MB in (block ~966), the loader thread SIGSEGVs**: it dereferences
the ASCII string `"Rewired_"` (`0x5f64657269776552`, the Rewired input plugin) **as a pointer** during scene
deserialization — a corrupted/shifted object layout planted by an earlier wrong HLE return on the load path.
Deterministic across runs; crash RIP varies (downstream corruption). So the real title/gameplay scene never
instantiates. Prime suspects (first-exercised on the load path, all return 0): libSceAgc trace-only stubs
`H7uZqCoNuWk`/`-KRzWekV120`/`tSBxhAPyytQ`, and the `libSceAmpr` trio (PS5 async streaming). Repro under gdb:
`PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_FILELOG=1 PROSPER_PREADLOG=1 gdb ./boot_trace`.

## What's in THIS branch (`frames/mask-autodetile`)

- `shader_resources.hpp` + `agc_shader_layout.cpp`: thread the T# `tile_mode` into `ShaderResource` (item 1).
- `boot_trace.cpp`: **auto-detile** tiled sampled surfaces by `r.tile_mode` (was the `PROSPER_DETILE` gate);
  `PROSPER_NODETILE` disables it; widened frame-dump gating (`n<60 || n%10`) to capture the fade-in.
- `agc_reg_defaults.cpp`: `CB_TARGET_MASK` default `0→0xF` — a **workaround** for item 2 (see the table;
  the principled fix is per-draw state capture + indirect-log segmentation, not the default).
- `videoout_present.cpp`: `PROSPER_DUMP_SCANOUT=<dir>` dumps the raw guest display buffer the game flips —
  proved the game flips *black* buffers (it composites but wrote nothing before the mask workaround).
- `docs/REAL_FRAMES_FINDINGS.md`: this file.

Validated: builds green, ctest 48/48. With items 1+2 (this branch) and item 3 (sibling PR), a reference-VS
render of the composite produces the title (1992 colors) in the normal pipeline path.

## Path to a real title frame, in priority order

1. **Land item 3** (VS descriptor-set fix, sibling PR) → the game's *own* VS renders the composite.
2. **Do item 2 properly**: per-draw `RenderState` capture in `run_command_buffer` + segment the AGC indirect
   register log at each `0x8e`-terminated template, so the composite draw consumes only its own mask=0xF
   (not the clear pass's 0x0). Then drop the `agc_reg_defaults` workaround. (Disassemble libSceAgc
   `DcbSetCxRegistersIndirect`/`PatchAddRegisters` from the dump — Kyty's flat pair-apply is NOT ground truth.)
3. **Determinism**: capture on demand at a chosen post-texture-load frame (the live renderer is too slow to
   rely on the dump cadence hitting the good window).
4. **Item 4 (loader SIGSEGV)** for the real title/gameplay scene: watchpoint the slot that receives the
   `"Rewired_"` bytes where a pointer belongs; implement the suspect libSceAgc/libSceAmpr NIDs (Kyty cross-check).
