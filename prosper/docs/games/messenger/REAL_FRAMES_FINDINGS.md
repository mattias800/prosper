# Real game frames — findings & fixes (2026-07-07 nuclear run)

Goal: get *The Messenger* to render real frames in prosper. This documents where we got, the fixes made,
and the two remaining blockers (with concrete next steps). Work done in the `ps5ys-frames` worktree.

## TL;DR

The game **no longer stalls** — after the merged flip/event fixes it runs a healthy frame loop (~23 fps
native; 6435 GPU submits in 280 s, no input polling, no per-frame file I/O). Its per-frame render is a
**fullscreen composite that samples the loaded 1920×1080 title/splash texture**. That composite renders the
**actual title art** (proven: 1992 distinct colors) once three things are handled — two of which are fixed
here, one of which remains:

1. ✅ **`CB_TARGET_MASK` default** was `0` → the composite had color writes disabled → wrote nothing. Fixed
   to `0xF` (`agc_reg_defaults.cpp`). The game reads the register defaults to seed its Dcb, so this
   propagates correctly and the composite now writes color. Validated: normal path + REFVS renders the title.
2. ✅ **Tiled texture auto-detile.** The title T# is `tile_mode=5` (GFX10 `SW_4KB_S`). Threaded `tile_mode`
   through the resource table (`ShaderResource.tile_mode`, set in `agc_shader_layout.cpp`) so the renderer
   auto-detiles instead of needing the `PROSPER_DETILE` gate.
3. ✅ **RESOLVED — real frames now render with the game's OWN shaders** (no REFVS). The "degenerate VS
   output" was two separate recompiler/host bugs, both fixed here:
   - **VS/PS descriptor-set collision.** The recompiler numbered *both* stages' resources from binding 2 in
     descriptor **set 0**, so set-0 binding 3 was simultaneously the VS's storage buffer and the PS's
     texture — an invalid Vulkan layout that corrupted *all* of the VS's buffer reads → every vertex
     collapsed to the origin → a degenerate (zero-area) triangle → the blue clear showed through. Fixed by
     giving each stage its own set (**VS = set 0, PS = set 1**, mirroring the PS5's per-stage resource
     tables): recompiler decorates PS resources with `DescriptorSet 1`, `FrameResource` carries a `set`,
     `render_runner` builds one descriptor set + layout per set and binds both. With this the VS geometry
     is correct (a solid test-PS fills the on-screen triangle).
   - **Vertex color attribute decoded as float32 instead of 8_8_8_8_UNORM.** RDNA2 packs a single 7-bit
     unified FORMAT at V# dword3 bits[18:12] (not the GCN dfmt/nfmt split we were reading). The per-vertex
     modulate color is **format 56 = 8_8_8_8_UNORM** (the `0xff000000→0xffffffff` black→white fade), but the
     legacy decode produced `Unknown` → a raw-float32 read → a garbage PS output **alpha** → the composite
     blended to the blue clear. `decode_buffer_descriptor` now decodes the RDNA2 format field and maps 56 →
     `Unorm8`×4; the recompiler already normalises UNORM8 (÷255), so the color/alpha are now correct.

4. ✅ **Full-quad composite.** The game tiles the fullscreen quad from a 4-corner vertex buffer into two
   triangles via *indexed* triangle-list draws; our single-draw offscreen spine renders `draws[0]` (one
   triangle), so half the frame was the clear. `execute_gpustate` now detects a 4-record quad vertex buffer
   and renders its corners as a triangle FAN (perimeter order BL,TL,TR,BR → tris {0,1,2},{0,2,3} tile the
   quad), so the whole frame fills. Only triggers for a 4-record buffer (real meshes unaffected).

   **Result:** a late frame (post texture-load + fade) renders **the game's full title composite with its
   real VS+PS** — ~137k distinct colors, 0% clear-color. Verify locally with the repro below (frames land
   as BMPs under `PROSPER_FRAME_DIR`); no game imagery is committed to the repo.

### Remaining polish (not blockers)
- **Honor indexed / multi-draw** properly (replace the 4-vertex-fan quad heuristic) so arbitrary scene
  geometry (not just the fullscreen composite) renders — needed once the title/gameplay scene loads.

## The modulate color is a fade-in (not a bug)

The composite's vertex color animates `ff000000 → ff1c1c1c → … → ffffffff` (black→white) over ~24 frames —
the title fade-in. It completes to full white. So the color is correct; only the mask/detile/UV matter.
(This ruled out the earlier "black composite" red herring: the composite is black only at t=0 of the fade.)

## Deeper blocker: the title SCENE never loads (loader-thread SIGSEGV)

The composite is Unity's near-empty **level0** boot scene. After ~80 s at level0 the game organically starts
the async **level1 (title scene)** load — opens `/app0/Media/level1`, streams `resources.assets` (77 MB) in
64 KB preads on a dedicated loader thread. **~60 MB in, the loader thread SIGSEGVs**: it dereferences the
ASCII string `"Rewired_"` (`0x5f64657269776552`, the Rewired input plugin) **as a pointer** during scene
deserialization — a corrupted/shifted object layout. Deterministic across runs (fault after `resources.assets`
block ~966). Crash RIP varies (downstream corruption). So the actual title/gameplay scene never instantiates.

Prime suspects (first-exercised on the load path, all currently returning 0): the libSceAgc trace-only stubs
`H7uZqCoNuWk` (called with a GPU-VA-map shape `a2=0x2000000000 a3=… a4=0x10000`), `-KRzWekV120`,
`tSBxhAPyytQ`; and the one-shot unimplemented `libSceAmpr` trio (`8aI7R7WaOlc`/`a8uLzYY--tM`/`N-FSPA4S3nI` —
PS5 async memory/streaming, directly load-relevant). Repro under gdb:
`PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_FILELOG=1 PROSPER_PREADLOG=1 gdb ./boot_trace`; watch the
`resources.assets` stream reach ~blk 966, then the WORKER-THREAD FAULT.

## What's in this branch

- `agc_reg_defaults.cpp`: `CB_TARGET_MASK` default `0 → 0xF` (color writes enabled; validated to render the title).
- `shader_resources.hpp` + `agc_shader_layout.cpp`: thread the T# `tile_mode` into `ShaderResource`.
- `boot_trace.cpp`: auto-detile tiled sampled surfaces by `r.tile_mode` (was the `PROSPER_DETILE` gate);
  `PROSPER_NODETILE` to disable; widened frame-dump gating to capture the fade-in.
- `videoout_present.cpp`: `PROSPER_DUMP_SCANOUT=<dir>` dumps the raw guest display buffer the game flips
  (proved the game flips *black* buffers — it composites but the composite was a no-op before the mask fix).

## Next steps to a real title frame, in order

1. **Fix the real VS's UV output** (recompiler): the position attribute fetches correctly per-vertex; the UV
   attribute does not. Diff the two fetch paths in `gpu_executor.cpp resolve_dynamic_fetch` /
   `rdna2_to_spirv.cpp`. Then the composite renders the title with the game's own shaders (no REFVS).
2. **Determinism**: the llvmpipe render is ~400× slower than the native loop, so a "good" frame (texture
   loaded + fade complete) is timing-dependent. Consider an on-demand capture at a chosen frame.
3. **Loader SIGSEGV** (for the real title/gameplay scene): watchpoint the slot that receives the `"Rewired_"`
   bytes where a pointer belongs; implement the suspect libSceAgc/libSceAmpr NIDs from live capture,
   guest disassembly, and the firmware symbol map.

## Test recipes (worktree `ps5ys-frames`, build-linux)

```
# Prove the title renders (reference VS isolates the VS-UV bug):
PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 PROSPER_RENDER_REFVS=1 \
  PROSPER_FRAME_DIR=/tmp/t ./boot_trace <dump>    # a late frame → ~1992 distinct colors (the title)
# The game's own render (currently uniform — the VS-UV bug):
… (drop PROSPER_RENDER_REFVS)
```
