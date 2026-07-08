# Cutscene render frontier — reaching the intro cutscene's pixels

> **See also `CUTSCENE_PROGRESSION.md`** for the boot/load history (getting *through* IL2CPP + the async
> loader to where the cutscene's draws are submitted). This doc is the **render** frontier that follows it:
> those draws reach the GPU translator but don't yet produce pixels.


> **Project context (read `../../CLAUDE.md` first).** prosper is a PS5→PC compatibility layer. This is a
> graphics/GPU-translation work log: getting *The Messenger*'s intro cutscene to render, in-process, on the
> developer's own machine. Standard emulator graphics work (same class as RPCS3/Dolphin/DXVK).

## The target (authoritative — from playing it on real hardware)

The intro cutscene is: a **black letterbox** (black bars top + bottom, like widescreen on a 4:3 TV), a
**wide mostly-blue image in the middle (an island)**, and **white text below the island image**. It plays
*after* the loading screen (the dark "monster-horde" art), which is a separate, earlier screen.

## Status — the pipeline reaches the cutscene; its draws render nothing

Everything from boot to the cutscene's draw submission now works:

| Stage | State |
|---|---|
| Boot → IL2CPP → frame loop | ✅ boots through to level0/1/3/4/5 (PSN.prx worker pool, PR #54) |
| Loading-screen art | ✅ renders (1920×1080) |
| Scene shaders recompile | ✅ RDNA2→SPIR-V: alpha-test discard PS, NGG `vccnz` VS, `v_mac_f32` (PR #58) |
| Render survives to the scene | ✅ was SIGSEGV at ~6 frames; `safe_copy` bounds resource reads (PR #59) → 400+ frames |
| Mesh vertex count | ✅ use the bound VB's entry count, not the truncated draw hint (PR #60) |
| **Cutscene composite draws** | ❌ reach us but render nothing → we present our debug-blue clear |

So the game **reaches** the cutscene and submits its draws; they just don't produce pixels.

## What the cutscene draws are (measured)

- `PROSPER_DRAWLOG`: post-title submits are `draws=2, index_counts=[3,3]` — **3-vertex composites** (one
  triangle each), **no vertex buffer**. They run through the *same large NGG vertex shader* as the scene
  mesh (not a trivial fullscreen-triangle VS). `PROSPER_PERDRAW` renders them blue too → it's the geometry,
  not the executor mode.

## Root cause, traced to the SPIR-V

Dump the composite VS and read its position math:
```bash
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 \
  PROSPER_SHADER_DUMP=/tmp/sd PROSPER_RENDER_FIRST=140 PROSPER_DETILE=1 ./boot_trace <dump>
spirv-dis /tmp/sd/frame_vs.spv          # frame_vs.spv = items[0].vs of the LAST render
```
`gl_Position` (the `OpStore` to the `Position` builtin) is a **correct MVP transform**: its x/y/z/w trace
back through real `OpFMul`/`OpFAdd` chains to `OpLoad`s from the constant-buffer StorageBuffers ×
fetched vertex attributes. The **same VS renders the loading art correctly**, so the shader logic is sound.

⇒ The cutscene draw's `gl_Position` lands **off-screen / degenerate because its INPUT DATA differs**: the
MVP constant buffers and/or the fetched vertex attributes for this specific draw are wrong for our
display-space render.

## The next step (open)

**Verify the cutscene draw's constant-buffer (MVP) + vertex data**, two hypotheses:
1. **Zeroed / uncommitted** — if the MVP cbuf is in memory `safe_copy` (PR #59) treats as uncommitted, it
   zero-fills → `w≈0` → everything clips. Check the resolved cbuf bytes are a plausible view-projection
   matrix, not zeros.
2. **Valid but off-display** — the cutscene likely renders the island **to an offscreen atlas** (a 2048×1024
   texture is bound; its dump shows scrambled/partly-committed content), so the MVP is in **atlas space**,
   not display space. We render that draw straight to the display viewport → off-screen. This is the
   render-to-texture / render-target-management gap: execute the scene→atlas pass into a Vulkan image and
   bind it as the composite's texture (see `docs/NEXT_STEP_VERTEX_FETCH.md` "execute the full draw chain").

Also: our executor clears to a **debug blue** (BGR 255,0,0); switch to the game's clear color so the black
letterbox appears even before the island/text draw correctly.

## Repro / tooling

- Repro: `PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 PROSPER_DETILE=1`
  (+ `PROSPER_RENDER_FIRST=140` to skip to the post-loading phase).
- `PROSPER_DRAWLOG` (per-draw counts), `PROSPER_SHADER_DUMP=<dir>` (`frame_vs.spv`/`frame_fs.spv` +
  `exec_*_*.bin` for *failed* recompiles), `PROSPER_DUMP_TEX` (bound textures → BMP),
  `PROSPER_FORCE_COLORWRITE`. (`PROSPER_VCOUNT`/`PROSPER_TOPO` overrides and the quad-fan topology
  heuristic were removed with real indexed-draw support, issue #64.)
- **Disassemble PS5 shaders with `llvm-mc -mcpu=gfx1010`, NOT gfx1030** — the PS5 keeps RDNA1-era encodings
  (e.g. `v_mac_f32` at VOP2 op 0x1f, invalid on gfx1030). See PR #58.
- Frames are BMP; read via PIL→PNG (`ImageFile.LOAD_TRUNCATED_IMAGES=True` for `safe_copy`-truncated tails).
