# Earthion Routes

Reusable `PROSPER_PAD_SCRIPT` routes for Earthion (`PPSA28061`, Ancient's own engine on direct AGC).
Run tools from `prosper/` so relative script paths resolve.

## Why this route exists

Earthion was recorded at **rung 1** with the note *"the 320x224 game picture inside the bezel is
missing"*, and #1590 spent several sessions explaining the black rectangle inside the CRT bezel.

Every capture in that history was of the **intro story-text sequence**, and no route had ever pressed
a button. The intro is white text on black — the rectangle is black on purpose. Pressing through it
reaches the title screen and the menus, in full colour, with no code changes.

The lesson generalises past this title: **a rung is a claim about the furthest state reached, so it
cannot be measured from a boot that never sends input.** A no-input capture bounds the title from
below and nothing more.

## `reach-title-menu.pad`

Alternates Cross and Options every five seconds from the game's first pad poll. Which button leaves
the intro was unknown when the route was written and the page timing moves with the rendering policy,
so it covers both rather than guessing an offset; surplus presses land in the menu as select/back and
the route returns to it, which keeps both arms of an A/B on the same states.

```bash
PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 \
PROSPER_RENDER_SCALE=1 PROSPER_RENDER_EVERY=1 \
PROSPER_PAD_SCRIPT=@scripts/earthion/reach-title-menu.pad PROSPER_PAD_SCRIPT_LOG=1 \
PROSPER_VULKAN_LIB=libvulkan.so.1 \
  ./build/screenshot <DUMP_ROOT>/PPSA28061-app0 \
    --seconds 12 --count 26 --out <OUT_DIR> --timeout 640
```

Reached on Linux/RADV at native 3840x2160, unmodified frontend:

* **Title screen / main menu** — logo over a wireframe globe; `How to Play`, `Game Version: English`,
  `Reset`, `Audio`, `Visuals`, `Language: System Default`, `Extras`; selection bar, hint line, Ancient
  and BIT WAVE marks.
* **HOW TO PLAY** — annotated HUD screenshot with callouts, four sprite icons with descriptions, and a
  two-column `Controls` list with correctly rendered PlayStation glyphs.

Measured inside the bezel (2,500x1,850 crop, downsampled to 160x120), 26 frames over 312 s:

| state | interior mean luma | non-black | distinct colours |
|---|---|---|---|
| title / main menu | 16.5 | 13.4 % | 2,563 |
| How to Play | 18.8-19.4 | 20.9-21.6 % | 2,861-2,923 |
| intro / transition | 0.0 | 0.0 % | 1 |

The 26 frames resolve to 6 distinct images as the probe walks in and out of the menu.

## Verifying the route actually reached the guest

`PROSPER_PAD_SCRIPT_LOG=1` prints the states the guest **read**, not the states the script emitted:

```
[pad-script] elapsed=4.003 frame=448 read=448 buttons=cross
[pad-script] elapsed=9.002 frame=1336 read=1337 buttons=options
```

`PROSPER_PADLOG=1` confirms it from the HLE side:

```
[pad] OPEN userId=1 type=0 index=0 -> handle=1
[pad] pad_read call#1024 connected=1 buttons=0x4000 lx=80 ly=80    <- CROSS
[pad] pad_read call#3584 connected=1 buttons=0x8    lx=80 ly=80    <- OPTIONS
```

Check both before concluding that a title ignores input. A route whose presses never reach a pad read
is indistinguishable, from the screenshots alone, from a title that reads them and does nothing.

## Not yet routed

**Rung 3.** This section previously said that starting a game "needs Up/Down navigation plus Cross"
and was "a menu-navigation problem against a working frontend, not a graphics one". **Both halves are
falsified** (2026-08-21, `docs/EARTHION_STATUS.md`):

* The list above is the **wrapper's** menu and contains no start item — every row is a wrapper
  control (`Reset`, `Game Version`, `Audio`, `Visuals`, `Language`, `Extras`, `How to Play`). No
  amount of directional navigation reaches gameplay from it.
* The title's own **HOW TO PLAY page states the binding**: `Open Menu` is **OPTIONS** and **`Start`
  is the TOUCHPAD**. `pad_button_by_name` accepts `touchpad`.
* Start has been delivered and is not the missing step. A route pressing it every 8 s ran 320 s;
  `PROSPER_PAD_SCRIPT_LOG=1` shows the guest **read** 13 `buttons=touchpad` states, and the picture
  inside the bezel is black in every frame bar one narration line.

The real frontier is that the game picture never leaves the narration sequence, which recurs on a
**~60 s cycle**. Read `docs/EARTHION_STATUS.md` § Ruled out before forming a hypothesis.

## Known-unrelated diagnostic noise

One pixel shader (`ps=0x4101c1f00`) is rejected every frame throughout this route, reaching
`occurrence=32768` in five minutes, because the guest binds a render-target index its own group never
created and hands the descriptor slot 32 bytes of uninitialised stack (#1590 — a guest defect).
**The menu renders in full colour regardless.** Do not read those reject lines as the reason for
anything black on screen; that inference is what cost #1590 its first several sessions.

#1773, the separate recompiler gap that the same draw exercised, is **fixed** — a direct user-data
descriptor staged with `s_mov_b32` now keeps its provenance, and on this title the recompiler
recovers `ud_alias=s9` (the declared `offset_dw`). The draw is still declined, because the resource
at that offset is dropped upstream by the guest defect above, and **the composite does not change**:
the route's three stable states (menu, HOW TO PLAY, black) are pixel-identical and no new content
appears. Two runs of the same binary differ in 7 of 28 frames purely because the narration cycle's
phase is not deterministic, so do not use a whole-run frame-identity count as a render oracle here.
Both facts are in `docs/EARTHION_STATUS.md`.

Those counts are for **this** route. A route that never opens the wrapper menu exercises a different
program set: the touch-pad route produces exactly one `[mimg-unresolved]` line and zero
`[recompile-reject]` lines in a whole run. Name the route before quoting a reject volume.
