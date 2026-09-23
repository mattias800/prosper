# Kena: Bridge of Spirits (`PPSA01802`) — status

Unreal Engine 4 (Ember Lab), one 28.5 GB `kena-ps5.pak` (no IoStore), Wwise, SDK `0x03000000`. Tracker
[#3787](https://github.com/mattias800/prosper/issues/3787). Brought up on Windows 11 / RTX 4090;
Linux/AMD title-menu investigations are recorded below.

## Current state — rung 2, gameplay reached with the world absent (2026-09-23)

On `main` (which has both crash fixes, [#3817](https://github.com/mattias800/prosper/pull/3817) and
[#3814](https://github.com/mattias800/prosper/pull/3814)), `screenshot.exe` with the route
`prosper/scripts/kena/reach-first-gameplay-prompt.pad` reaches, in order:

| t (one 540 s run) | what renders |
| --- | --- |
| 0-70 s | black (the Ember Lab logo movie is decoded and delivered to the guest but never visible — open) |
| ~80 s | **main menu** — `New Game / Load Game / Options`, `Version 1.16`, over black |
| ~160 s | **Choose Difficulty** |
| ~170-220 s | **brightness calibration** (the map art and slider render) |
| ~240 s | loading spinner — the first level loads |
| ~270-330 s | **intro narration** (*"Unique wooden masks are carved to honor the dead…"*), legible, with the blue *Spirit Guides* highlight |
| ~410-540 s | the first gameplay prompt, *"Press … to Pulse"* — **over a black world** |

The guest is in the game loop and asking for gameplay input, but the scene behind the prompt does not draw
(6,817 non-black pixels of 8.3 M per sample, all of them the prompt). Under the ladder that is **rung 2**, not rung 3.

**Likely cause of the black world, not yet established per draw:** the title's fragment shaders need wave64.
The run logs `[render] skip draw=... fragment shader requires subgroup size 64 (device range 32..32 ...)` for 402
distinct shaders (400 `why=0x2 wave-any`, two `lane-id`), and an NVIDIA device is 32-wide, so every one of those
draws is skipped — the cross-title gap [#2147](https://github.com/mattias800/prosper/issues/2147). The main menu's
missing animated background is probably the same thing. Confirming it needs either a 64-wide (AMD) run or a
census of which draws in a gameplay frame are skipped; neither has been done.

Every menu frame also alternates with a fully black composited frame in about half the samples; not investigated.

### Linux/AMD title-menu boundary (2026-09-24)

The live menu still shows its wordmark and UI over black. A retained 1600×900 RGBA16F target on the normal
render path contains nonzero scene data (one observed target has every RGB channel at half-float maximum), while
the later 3200×1800 compositor output is black. Other same-extent retained images contain recognizable forest
content. The saturated target is not by itself evidence of the correct scene: raw half-float values and converted
BMP previews have different meanings. The ordered submit capsule is explicitly **unverified** because it was not
matched to its own presented frame; its draw/dispatch lineage is useful for localization, not a visual oracle.

In that capsule, the final compositor samples a zero 32³ LUT at PS binding 39. The immediately preceding LUT
producer is unrealized because its merged-NGG vertex chain hits the existing wave/mesh gate
[#3135](https://github.com/mattias800/prosper/issues/3135). A one-instruction replacement of the compositor's
final LUT sample with its existing lookup coordinates, preserving the rest of the fragment shader, leaves the
**live** menu black by itself. An independent header-derived input-default probe also leaves it black by itself;
together the probes expose a washed-out, blocky scene. This is diagnostic output, not a rendering fix, and the
two controls are not a full same-binary 2×2 experiment. Neither substitution belongs in the default path.

The input-default probe logged changes to earlier fragment programs `0x3007c60000` and `0x3007c80000`
paired with vertex program `0x3007060000`; it did **not** change the final compositor's input wiring.
A separate pair trace of `0x3007060000` with fragment `0x30096e0000` found five guest-programmed
`SPI_PS_INPUT_CNTL_0..4=0` values and header-derived constant defaults. The guest writes these registers
through indirect **physical** offsets (17,420 observed writes, none through the virtual AGC bank).
That vertex program exports PRIM/POS0 without PARAM exports; it is a fused front shader, with no hidden
linked back/chain body. This traced pair is also **not** the final compositor. The captured compositor
vertex SPIR-V stores Locations 0–3. Trace the exact earlier pass changed by the probe and establish its
PS5 input contract before changing generic interpolation semantics.

## Reproduction

```bash
PROSPER_NULL_PAGE=1 PROSPER_GUEST_ARGS= \
PROSPER_SAVE0=<FRESH>/save0 PROSPER_SAVEDATA_DIR=<FRESH>/savedata \
PROSPER_PAD_SCRIPT=@scripts/kena/reach-first-gameplay-prompt.pad PROSPER_PAD_SCRIPT_LOG=1 \
  ./build-mingw-app/screenshot.exe <DUMP_ROOT>/PPSA01802-app0 --seconds 10 --count 54 --timeout 560 --out <OUT>
```

`PROSPER_NULL_PAGE=1` and an empty `PROSPER_GUEST_ARGS` are the UE4 recipe (`DRAGON_QUEST_STATUS.md`). The route's
anchors are pad reads — see `prosper/scripts/kena/README.md`. `tools/dump_hygiene.py` on the dump is clean; its
`_original_files/` (the untouched encrypted originals) is retained and never read.

About 2 in 12 launches hang before their first frame (no raw scanout either). Not yet investigated; relaunch.

## What was fixed to get here

1. **`sceAvPlayerStartEx` was unregistered** ([#3781](https://github.com/mattias800/prosper/issues/3781), merged
   as [#3788](https://github.com/mattias800/prosper/pull/3788)). The title adds its logo movie with
   `auto_start = 0` and starts it only through this call; the dispatcher's `return 0` answered `SCE_OK` without
   starting anything, and the title waited forever on a black screen. 0 of 2 → 2 of 2 runs reach the menu.
2. **A fixed `sceKernelBatchMap` MAP_DIRECT failed over pages prosper had lazily committed**
   ([#3812](https://github.com/mattias800/prosper/issues/3812), PR [#3814](https://github.com/mattias800/prosper/pull/3814),
   Windows only). Host-side reads first-touched unmapped pages of the guest's own reservation; the VEH replaced
   those placeholder pieces with private memory, and `MapViewOfFile3` cannot place a view over them. The guest got
   ENOMEM and UE4 aborted (`sceKernelBatchMap failed with error code: 0x8002000c`) 50-65 s in, 5 of 6 runs. A fixed
   map now releases the range's guest-owned contents first, as `mmap(MAP_FIXED)` does on Linux.
3. **The recompiler emitted invalid SPIR-V for a fragment `s_bfm_b64`**
   ([#3816](https://github.com/mattias800/prosper/issues/3816), PR [#3817](https://github.com/mattias800/prosper/pull/3817)).
   Its lane test used `linear_localid`, which is id 0 in a fragment module; NVIDIA's driver dereferenced it while
   compiling the pipeline and the process died inside `nvoglv64.dll` on the first level load, 4 of 4 runs.

## Ruled out

- **The black pre-menu screen is a renderer failure** — false. The composite was black because the logo movie was
  never started (unregistered `sceAvPlayerStartEx`, #3781); registering it reaches the menu with no renderer change.
- **The BatchMap ENOMEM is #2424's too-small free placeholder** — false. `VirtualQuery` on the failing range shows
  `MEM_COMMIT | MEM_PRIVATE` pages in 16 KiB pieces with non-64 KiB allocation bases, and the registry holds
  **0** free placeholders at that moment; it is prosper's own lazy commit, not a placeholder size mismatch (#3812).
- **The guest mapped or touched the failing range before the fatal map** — false. `PROSPER_MEMLOG=1` shows no guest
  map, flexible map or unmap of that range before the fatal MAP_DIRECT, and a RIP-tagged log of every VEH lazy
  commit (5,120+ in 145 s; the first 60 and every 1024th logged, 65 lines) found **all 65** in a host
  system DLL's copy routine and none in guest code (#3812).
- **The level-load crash is an NVIDIA driver bug** — false as the root cause. Of all 11,710 graphics SPIR-V
  modules created in a crashing run, `spirv-val` rejects exactly one — the last one created — and its only
  defect is an operand naming id 0 emitted by prosper's `s_bfm_b64` lowering (#3816). The driver crashing on
  invalid SPIR-V rather than rejecting it is a driver robustness gap, not the cause.
- **Seconds-anchored pad pulses are enough for this title's menus** — false. An 11-press seconds-based route
  delivered 2 presses, because the guest reads the pad about once a second there and a 300 ms pulse mostly falls
  between reads; pad-read anchors (`pN`) deliver every press.
- **The missing 3D LUT alone explains the black live title scene** — false as a complete explanation. Replacing
  only its compositor sample with the existing lookup coordinates left the live menu black; combining that
  diagnostic replacement with a fragment-input-default probe exposed a washed-out scene. The LUT producer is
  still missing in the unverified capsule, so this does not exonerate the merged-NGG gap (#3135).
- **The traced earlier fragment inputs are zero because virtual AGC register translation dropped their writes**
  — false for the traced `0x30096e0000` pair. A path-labelled register watch observed 17,420 direct physical
  indirect-array writes to the controls and zero virtual-bank writes (#3835).
- **The traced earlier vertex program's missing PARAM stores are in an unlinked back half** — false for its
  `0x30096e0000` pair. Exact-pair logging names one fused front program, no chain, and no linked second body;
  its decoded entry ends without a PARAM export. This says nothing about the separate final compositor pair (#3835).

## Next

- Confirm that the black gameplay world (and the menu background) is the wave64 skip, on a 64-wide device or with
  a per-draw skip census of a gameplay frame; if it is, the title waits on #2147.
  A Linux lane has since traced the menu scene ([#3813](https://github.com/mattias800/prosper/pull/3813)): the
  forest is present in a retained MRT3 image that a one-block fragment shader samples at binding 37 — so on Linux
  the background scene is rendered somewhere. The Linux title-menu boundary above now narrows the later loss;
  do not assume the Windows black background is solely the wave64 skip. Resolve the generic missing-PARAM
  contract and merged-NGG LUT producer separately, then compare a live default-path image with the oracle.
- Host-side renderer reads of never-mapped guest memory (#3820), and `SCE_KERNEL_MAP_NO_OVERWRITE` (#3819).
- The invisible logo movie: 150 frames decoded and delivered, none visible; delivered at ~2x real time.
- The early-boot hang (about 2 in 12 launches).
