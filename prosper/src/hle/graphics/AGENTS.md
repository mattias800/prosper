# `src/hle/graphics` — the guest-facing graphics libraries

This folder is prosper's reimplementation of the Sony **libraries a title calls** to get pictures on
screen: `libSceVideoOut` (the display, its buffers, flips and vblanks) and the `libSceAgc` driver
surface (command-buffer submission, GPU completion events, register defaults). It is the HLE side of
the boundary — NIDs in, ABI structs out.

**Its boundary against `src/gpu/` is the important one.** Everything here answers the guest; nothing
here draws. When `SubmitFlip` arrives, this folder does the bookkeeping the guest can observe (flip
count, current buffer, flip arg, buffer labels, the completion event) and then hands the buffer index
to `gpu::present_flip`. Translating command streams, recompiling shaders and talking to Vulkan all
live under `src/gpu/`. If you are reaching for a Vulkan type in this folder, the code probably wants
to be somewhere else.

What lives here:

- **`hle_graphics.cpp`** — `libSceVideoOut`. The display prosper advertises, the registered
  framebuffer set, flip submission and completion (from both the API call and the in-stream GPU
  `SetFlip` packet), and the vblank timebase.
- **`hle_agc.cpp`** — the AGC driver entry points a title submits work through, plus the
  `PROSPER_PROGRESS` heartbeat.
- **`agc_reg_defaults.cpp`** — the register state the guest inherits before it sets anything.
- **`display_mode.hpp`** — pure, dependency-free: host display capabilities in, the mode prosper
  advertises out.
- **`render_cadence.hpp`** — the sampling/skip policy for headless capture runs.

## Two things this folder gets wrong in ways that are hard to see

**An answer here is something the guest ACTS on, so a wrong one does not look like a crash.** The
title behaves correctly given what it was told, which means the symptom shows up as the game's own
UI or pacing being wrong — a resolution it would not otherwise pick, a frame rate it caps itself to.
`display_mode.hpp` exists because two constants were hardcoded here for a year and read as normal
(#3017). When adding a query, ask what the guest does with the answer before deciding it is cosmetic.

**Anything describing the display must be DERIVED, and must have exactly one origin.** The
advertised mode is the worked example: what the title *reads*
(`sceVideoOutGetResolutionStatus`) and what actually *paces* it (the vblank grid, which the kernel's
kevent pump in `hle/kernel/hle_kernel_time.cpp` also schedules on) were once independent literals
that agreed only because somebody maintained them by hand. They now come from one resolved
`AdvertisedDisplayMode`, and they should stay that way: if you find yourself writing a second
constant for a quantity that already exists here, that is the bug reappearing.

The refresh rate is an **enum**, not a number, so advertising a rate means asserting the integer Sony
assigned it. `display_mode.hpp` grades each row by evidence and the selection policy carries a floor,
so a value prosper cannot defend is unreachable by default rather than merely discouraged. Keep that
property when adding rows.
