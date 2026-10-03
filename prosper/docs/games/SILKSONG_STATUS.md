# Hollow Knight: Silksong (`PPSA12544`) — status

Tracker: [#4121](https://github.com/mattias800/prosper/issues/4121). Engine: Unity 6 (6000.0.50f1).
This document is the technical record the tracker points at; the tracker holds the rung.

**Rung 4.** The owner has played it and describes gameplay as nearly perfect. The missing main-menu
text (#4120) is fixed: every menu string was absent because no dynamic-font glyph ever reached its
atlas.

## Other fixes from the same session

- **Nine unregistered system calls** that prosper answered with a silent `SCE_OK` (#4275). Their
  live argument shapes were captured on the route before they were implemented:
  - `sceVideoOutSubmitChangeBufferAttribute2` re-submits set 0's registered attribute;
  - `sceSaveDataSaveIcon` now writes the save's PNG to `sce_sys/icon0.png`;
  - the six NP Universal Data System property object/array calls now hand out real handles;
  - `sceNpSessionSignalingInitialize` takes an input-only parameter block.

  Before, the UDS `Create*` calls wrote no handle, and the guest set, attached and destroyed
  whatever stale value its out-slot held.
- **Refused shaders now leave their code behind by default** (#4273). #4120's dropped vertex shaders
  could not be examined, because the run that showed them kept nothing.

## Known and explained: `Share.prx`'s start faults at boot

`[prosper] init fn 0x5d0000010 faulted (SIGSEGV at addr=(nil) rip=… (plugin+0x19a)); continuing`
appears on every boot. It is explained and does not affect the game:

- **The fault.** `Media/Plugins/Share.prx`'s `module_start` (`+0x12a0`) passes its `argp` to
  `+0x190`. That function reads `argp` before checking it, expecting Unity's plugin block
  `{u32 0x10, u32 0x200, u64 …}` or a `{0x58, 0x103, …}` variant. prosper auto-links every
  `Media/Plugins` module and starts it at boot with `argp = NULL`, so the first read faults.
- **Why it is harmless.** Unity starts the plugins it uses itself, through
  `sceKernelLoadStartModule(path, 0x34, &{0x10, 0x200, ptr…})`. A probe of the first 60 s of the
  route shows that for `PSN.prx`, `SaveData.prx` and `CommonDialog.prx`, but **never for
  `Share.prx`**. On hardware it is not started at all on this route. prosper's premature start is
  the only one, and nothing the game uses depends on it.

The general fix is prosper's start policy for auto-linked plugins. Deferring every unimported
plugin to `sceKernelLoadStartModule` (PR #4221) regressed three guarded Unity titles, so it needs a
narrower rule. This route is a test case for it.

## Route

A human-authored snaps session (`prosper/tools/snapshot/snaps.py author --name silksong-menu`)
records the title menu at pad flip 1398, profile select at 5519, the brightness setup at 6846 and
gameplay at 11667. It replays headless with fresh save data at 60 paced flips per second.

## How the menu text is drawn, and why it was blank

The menu fonts are legacy Unity dynamic `Font` objects. `fontsmenu_assets_all.bundle` holds 22 of
them, and 21 have a 0x0 `Font Texture`, so glyphs are rasterised at runtime. Unity rasterises each
glyph on the CPU into a CPU-visible Alpha8 upload texture, then copies it into the GPU atlas with a
compute kernel (`0x4011730a00` in the boot that was measured):

```text
s_buffer_load_dwordx2 vcc, s[8:11], 0x8     ; copy extent
v_cmpx_gt_u32 ...                           ; bounds
image_load  v[0:3], v[0:1], s[0:7]   dim:2D        ; upload texture, DST_SEL (0,0,0,X)
image_store v[0:3], v[4:6], s[16:23] dim:2D_ARRAY  ; atlas,          DST_SEL (0,0,0,X)
```

The DST_SEL puts the glyph value in `.w` on the load. prosper applied DST_SEL to sampled views
but ignored it on storage-image stores, so the store wrote `.x`, which is always zero. Every
atlas stayed empty, and every text draw sampled a blank texture. The fix routes `image_store`
through the destination descriptor's DST_SEL, as the inverse of the load mapping (#4120, #3549).

Reproduce offline: capture a submit containing many copies (`PROSPER_GPU_CAPTURE=<path>
PROSPER_GPU_CAPTURE_COMPUTE_ADDR=<copy program> PROSPER_GPU_CAPTURE_MIN_DRAWS=50`), then run
`gpu_replay --recompile-raw --dump-post-compute-resource 0:6 out.bin <capture>`. Before the fix
`out.bin` is all zero. After it, the rectangle the dispatch copies holds an anti-aliased glyph.

## Ruled out

- **The missing menu text is dropped vertex shaders.** #4120's lead was 6,850 vertex-recompile draw
  drops on an older main. On 3650263db, a 250 s session through the menus into gameplay never fires
  `dropped-draws`, and both F9 bundles replay with `dropped=0` (93 and 453 draws). The text draws
  are present with real glyph quads (22 and 9 in the title frame). (#4120)
- **Asset streaming refused the font data.** No APR read is refused in the session. The upload
  textures are never written by a host read, DMA or GPU writeback: `PROSPER_HOSTWRITEWATCH`,
  `PROSPER_DMA_WATCH_DST` and `PROSPER_GUEST_WRITE_WATCH` all stay silent on them, because the guest
  fills them with plain CPU stores. (#4120)
- **Glyphs are rendered into the atlas as a colour target.** `PROSPER_TARGET_WATCH` on eight atlas
  addresses reports `NEVER A COLOUR TARGET` across 16,384 draws, and no draw targets 256x256. (#4120)
- **prosper mis-detiles 8-bpp `SW_4KB_S`.** The upload texture, de-swizzled offline with
  `x0 x1 x2 x3 y0 y1 y2 y3 y4 x4 y5 x5`, is a clean glyph page. That is exactly the order
  `sw4kb_morton` produces at 1 byte per element. The source is read correctly; the store is what
  lost it. (#4120)
- **The copy's arrayed destination or `uint` storage type drops the write.** Replay overrides that
  make the destination non-arrayed, float, or both still wrote nothing. Forcing a constant store
  wrote exactly the dispatch's 22x33 glyph rectangle, which separated "the write lands" from "the
  data is zero". (#4120)
