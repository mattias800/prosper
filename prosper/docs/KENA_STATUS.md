# Kena: Bridge of Spirits (`PPSA01802`) — status

Unreal Engine 4 (Ember Lab), one 28.5 GB `kena-ps5.pak` (no IoStore), Wwise, SDK `0x03000000`. Tracker
[#3787](https://github.com/mattias800/prosper/issues/3787). Brought up on **Windows 11 / RTX 4090** only; no
Linux/AMD run is on record yet.

## Current state — rung 2, gameplay reached with the world absent (2026-09-23)

On `main` plus [#3814](https://github.com/mattias800/prosper/pull/3814) and
[#3817](https://github.com/mattias800/prosper/pull/3817), `screenshot.exe` with the route
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

## Next

- Confirm that the black gameplay world (and the menu background) is the wave64 skip, on a 64-wide device or with
  a per-draw skip census of a gameplay frame; if it is, the title waits on #2147.
- The invisible logo movie: 150 frames decoded and delivered, none visible; delivered at ~2x real time.
- The early-boot hang (about 2 in 12 launches).
