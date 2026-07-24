# Input replay & checkpoints — reaching a game state to reproduce a bug

> **Status (2026-07-19): frame/pad-read anchoring and recording, route loading, and the opt-in deterministic clock have landed.**
> Current master supports inline `PROSPER_PAD_SCRIPT` entries anchored as `fN:`/`fA-B:` (display
> flips) or `pN:`/`pA-B:` (pad reads), alongside the existing seconds axis. Point entries use
> `PROSPER_PAD_FRAME_HOLD` or `PROSPER_PAD_READ_HOLD` (both default 8) on their count axis. It also
> supports `PROSPER_DET_CLOCK=1` (fixed `1/PROSPER_DET_FPS`, default 60, per flip). Before the first flip the
> monotonic clock follows host time so initialization can progress; afterward it intentionally pauses
> between flips. Realtime/RTC clocks remain tied to host time. `PROSPER_PAD_SCRIPT=@path` loads newline-
> separated routes with comments, explicit time/flip/read ranges, and full-deflection stick directions such
> as `left-stick-left`. `PROSPER_PAD_RECORD=path` records the
> final button stream on the flip axis by default. Set `PROSPER_PAD_RECORD_AXIS=pad-read`, or pass
> `prosper-app --record path --record-axis pad-read`, to emit successful-input-read `pA-B:` ranges.
> `PROSPER_PAD_SCRIPT_LOG=1` logs state transitions with the elapsed time, flip, and pad-read index
> as the game observes them.
> `PROSPER_PAD_SCRIPT_RELOAD=1` live-reloads an `@file` route after a changed file remains
> stable across two metadata polls. Existing time, flip, and read anchors are preserved, so an agent can
> append future input windows during a long exploratory run without restarting the title.
> The first checked-in route, `scripts/messenger/reach-intro-story.pad`, repeatedly reaches the opening
> story but still has small narration-phase drift. Deeper route calibration and semantic checkpoint
> validation remain open in #302. The original design follows.

## The problem

Now that titles are becoming playable, a bug is often reported at a *state* — "the shop menu", "level 1 boss", "after the first save". To reproduce or observe it, an agent must **navigate the game to that point** and then apply the full debug arsenal (`PROSPER_GFXLOG`, `PROSPER_DRAWDIAG`, `PROSPER_HWBP`, `boot_trace`, `screenshot`, the app). Hand-driving a controller isn't an option for a headless agent.

## Principle: replay is a *core* capability, not a tool feature

Every debug tool drives **one core process** configured by env flags. So "reach this state" must also be a property of that core run — then it composes with everything for free:

```bash
PROSPER_PAD_SCRIPT=scripts/messenger/reach-shop.pad \
PROSPER_GFXLOG=1 PROSPER_DRAWDIAG=1 PROSPER_HWBP=0x440012ab0 \
  boot_trace <app0>          # or screenshot, or prosper-app to watch live
```

One env var gets the agent to the bug; their entire existing toolset is unchanged. **Do NOT bake replay into the screenshot tool** — it's a pad backend that every tool inherits.

We already have the seed: **`PROSPER_PAD_SCRIPT` (#202)** — a scripted `PadBackend` in `hle_pad.cpp` / `pad.cpp`. We extend it; we don't reinvent it.

## What exists today (`PROSPER_PAD_SCRIPT`, #202)

- Format: `;`- or newline-separated `<anchor>:<action>[+action…]`, where an anchor is seconds,
  `fN`/`fA-B` display flips, or `pN`/`pA-B` pad reads. For example:
  `3:start;f300-340:cross;p1200-1240:left-stick-left+cross`.
- Actions are button names or full-deflection `left-stick-{left,right,up,down}` and
  `right-stick-{left,right,up,down}` directions. Seconds points use `PROSPER_PAD_HOLD` ms
  (default 300); flip/read points use their count-axis holds above. Explicit ranges always use their
  exclusive end.
- **Anchored at the established origins**: seconds and flips retain their existing first-pad-poll
  origin. Pad reads number successful `scePadRead`/`scePadReadState` calls from zero; controller-
  information queries and rejected reads do not advance the pad-read axis.
- Pad reports CONNECTED whenever a script is set. Parse + time-eval are pure and unit-tested in `pad.cpp`.
- Explicit wall-clock ranges are sampled only when the game polls the pad. A short range can fall
  entirely between polls under slow synchronous rendering. Use longer holds with neutral gaps,
  prefer flip-anchored ranges when the title keeps presenting, or pad-read ranges when presentation
  pauses but the game continues polling. Enable
  `PROSPER_PAD_SCRIPT_LOG=1` to verify delivery rather than inferring it from the route text.
- For exploratory `@file` routes, set `PROSPER_PAD_SCRIPT_RELOAD=1`. A changed route is debounced
  for 250 ms and replaces the active route only after a complete stable read; read/stat failures
  retain the last valid route. Reload does not reset the first-poll wall-clock, flip, or read origin.

The remaining work is deeper route calibration and semantic checkpoint validation that does not
depend on presentation speed.

## IME-keyboard scripts

Titles that consume keyboard input through `sceImeUpdate` instead of `scePadRead` can use
`PROSPER_IME_SCRIPT`. Its entries are `fN:HID` or inclusive `fA-B:HID` windows, where HID is a
decimal or `0x`-prefixed USB usage id (`0x28` Enter, `0x2c` Space). The frame axis counts non-null
`sceImeUpdate` handler pumps from zero; in the observed IME titles this is one pump per game frame.
A point holds for two pumps. Semicolon-separated inline routes and newline-separated `@file` routes
with `#` comments are accepted:

```bash
PROSPER_IME_SCRIPT='f30:0x28;f120-124:0x2c' boot_trace <app0>
PROSPER_IME_SCRIPT=@scripts/title/reach-menu.ime screenshot <app0>
```

The route emits one KEY_DOWN at the start of a window and one KEY_UP after it, without repeating the
key while held. It is separate from `PROSPER_PAD_SCRIPT` because an IME-only title never polls the pad.

## Design

### 1. Frame/pad-read-anchored, file-loadable scripts (core) - implemented
- **Anchor to game frames, not wall-time.** Keep the "first pad poll" origin (robust to load time), but measure progress in **flips since first poll** (game logic frames), not elapsed seconds. The Messenger is a fixed-timestep platformer, so wall-time drifts vs game frames on slow llvmpipe vs a fast GPU — frame anchoring makes `f300:cross` reproduce everywhere. Add an `f<frame>:` entry syntax alongside the existing `<seconds>:` (kept for back-compat).
- **Anchor to pad reads when presentation is not the clock.** `p1200-1240:cross` holds Cross for
  successful guest input-state reads 1200 through 1239. This axis advances when synchronous rendering pauses
  display flips and cannot miss between polls like a short wall-time range. The transition log prints
  the live read index so routes can be calibrated without a code change.
- **Load from a file.** `PROSPER_PAD_SCRIPT=@path` reads a multi-line script file, so long routes live in the repo, not an env string.
- **Richer input.** Explicit ranges and full-deflection analog stick directions are implemented;
  proportional axis values remain future work.

### 2. Record mode in `prosper-app` - implemented
`prosper-app --dump <app0> --record <file>` captures the human's keyboard/gamepad button input stamped
by display-flip count and writes a script file. Add `--record-axis pad-read` to stamp it by successful
guest input-state reads instead; controller-information queries and rejected reads neither initialize
nor advance that recording. The flip axis remains the default for backward compatibility. This is how
checkpoint scripts get *created* — play to the point once, get a reusable, committable route.

### 3. Checkpoint library + agent docs
- `prosper/scripts/<title>/reach-*.pad` — **tiny text files, no game imagery, safe to commit** (unlike golden frames). Named by the state they reach: `reach-title-menu.pad`, `reach-level1.pad`, …
- An `AGENTS.md` note: *"to reproduce a bug at state X, prefix your tool with `PROSPER_PAD_SCRIPT=@scripts/<title>/reach-X.pad`."* A bug report then just cites the checkpoint.

The library is seeded with `scripts/messenger/reach-intro-story.pad`. It is a coarse route: repeated
runs reach the named story state, while exact narration-frame verification remains step 4 work.

## The crux: determinism

For a script to *reliably* land on the same state, the guest must be deterministic given the same input. Levers, in order of impact:
1. **Frame-anchoring** (piece 1) — removes render-speed sensitivity. Biggest win.
2. **Deterministic clock option** — derive guest time from frame count rather than the host wall clock, so time-seeded RNG / delta-time logic is reproducible. (Opt-in; may diverge from real behavior — mark CONFIDENCE.)
3. **Verification guard** — a *checkpoint* = `(script + expected frame hash at its end)`. Re-running must reach the same hashed frame; if it stops, that's a **caught regression**. This is the **same comparison layer as the snapshot-regression harness (#248 / #227)** — checkpoints are deep-state snapshot tests, and should share that hashing/threshold code, not duplicate it.

Determinism is iterative: frame-anchoring first, measure drift across repeated runs, then add the clock option / guard as needed.

## Agent workflow this enables

1. Bug reported "at the shop". Report cites `reach-shop.pad`.
2. Agent runs their tool with `PROSPER_PAD_SCRIPT=@scripts/messenger/reach-shop.pad` + whatever debug flags — lands at the shop, observes the bug, keeps every existing tool.
3. New checkpoint needed? Play there once with `prosper-app --record`; add
   `--record-axis pad-read` when input polling is the more stable clock, then commit the `.pad`.
4. Checkpoints double as regression coverage ("can we still reach level 3?") via the #248 harness.

## Build order
1. Design + this doc + tracking issue (agree frame-anchoring / clock decisions). ← here
2. `prosper-app --record` (low-risk, immediately useful, in the frontend).
3. Frame-anchor + file-load the core script (`pad.cpp` / `hle_pad.cpp`), keeping `<seconds>:` back-compat.
4. Seed `scripts/messenger/` + the `AGENTS.md` note; wire checkpoint verification into #248.
