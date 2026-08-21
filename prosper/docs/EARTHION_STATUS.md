# Earthion (`PPSA28061`) — status

Ancient's own engine on direct AGC, presented inside a CRT bezel with a retro-collection style
wrapper (the guest creates `Rewind`, `Save Thread`, `Leaderboards Thread` and `TrophyThread`).

**Rung 2** — the committed route reaches the wrapper's title/main menu and its HOW TO PLAY page in
full colour at native 3840x2160. Tracker [#1880]; route `scripts/earthion/reach-title-menu.pad`.

Read this before forming a hypothesis about why the *game picture* is black. The single most
expensive mistake available on this title is to attribute that blackness to the recompiler; that
inference has now been tried and measured twice, and it has been wrong both times (#1590, and
#1773 below).

## What is established

### The wrapper's control mapping — from the guest's own HOW TO PLAY page

The page renders correctly and lists the bindings in the title's own words. It is primary evidence
and it settles a question that three routes had been guessing at:

| Action | Button |
|---|---|
| Open Menu | **OPTIONS** |
| **Start** | **TOUCHPAD** |
| Select | Cross |
| Cancel / Next Sub Weapon | Circle |
| Main Shot | Square |
| Sub Weapon | Cross |
| Main & Sub / Prev Sub Weapon | R1 / R2 |

Consequences, both confirmed by measurement:

* **OPTIONS toggles the wrapper menu.** It is what the committed route's rung-2 evidence depends on.
  A Cross-only route (28 frames / 336 s) never produced the menu in any frame, which is what
  isolates OPTIONS as the opener rather than Cross.
* **The menu contains no "start game" item.** Its rows are `How to Play`, `Game Version: English`,
  `Reset`, `Audio`, `Visuals`, `Language: System Default`, `Extras` — wrapper controls. So the
  rung-3 route is *not* reached by adding Up/Down navigation to the committed route, which is what
  the route README and `GAME_COMPAT_ORCHESTRATION.md` both previously assumed.

`pad_button_by_name` accepts `touchpad` (`src/input/pad.cpp`), and its own comment records why the
name was added: a screen gated on the touch-pad click is unreachable by any scripted run, and the
failure is silent because an unknown name parses as neutral.

### The game picture is black except for the narration crawl

Measured inside the bezel interior (crop, downsample to 160x120, non-black threshold luma > 8):

| route | frames / span | states observed inside the bezel |
|---|---|---|
| committed menu route | 28 / 336 s | menu (36.5 % non-black), HOW TO PLAY (35.4 %), otherwise black |
| Cross only, every 4 s | 28 / 336 s | black; one narration line at 324 s and 336 s |
| Touchpad (Start) every 8 s + Cross | 40 / 320 s | black; one narration line at 288 s |
| Touchpad + Cross every 10 s | 75 / 900 s | black; one narration line recurring on a **~60 s cycle** |

The narration text renders, so the picture path is not dead — the crawl is white text on black by
design (this is the #1590 finding, and it still holds). What has **not** been observed in any run is
the game's own title screen, an attract/demo scene, or gameplay.

Pressing Start is not the missing step: `PROSPER_PAD_SCRIPT_LOG=1` shows the guest **read** 13
touch-pad presses in the 320 s arm (`buttons=touchpad`), and every frame of that run is black inside
the bezel apart from the one narration line.

### #1773 is fixed, and it is not what drops Earthion's draw

`sreg_ud_alias` (this PR) closes the generic gap: a **direct** user-data descriptor staged into a
scratch SGPR range with `s_mov_b32` now keeps its provenance, where previously the copy destroyed
its only key. Verified on the live title — the diagnostic went from

```
[mimg-unresolved] program=0x0 pc=28 srsrc=s20 srt_tag=NONE 0x0 key_res=null pc_res=null (1 res)
```

to

```
[mimg-unresolved] ... ud_alias=s9 alias_res=null written=1 (1 res)
```

`ud_alias=s9` is the recompiler independently recovering `offset_dw=9`, which is exactly what the
front half declares for `ro[1]` and exactly what #1773's disassembly said the copy sources. The
provenance half is done.

**The draw is still declined, for the other, independent reason:** `alias_res=null` with `(1 res)`.
The descriptor the copy points at is not in the resource table at all — the front half declares
`ro[0] offset_dw=1`, `ro[1] offset_dw=9`, `samp[0] offset_dw=17`, and `ro[1]` is removed by the
degenerate-T# guard because the guest's user-data words hold **stack residue**. That is #1590: the
guest asks a render-target group for index 1 when it created the group with one target, ignores the
resulting `0x8A6C0010`, and binds the untouched stack slot. It is a guest defect with no HLE in the
chain.

**The composite is unchanged: 28 of 28 frames pixel-identical** between the committed route run
before and after the fix. Stated explicitly because the tempting reading of "a decline was cleared"
is that something now draws, and here nothing does.

## Ruled out

One line per dead hypothesis, the evidence, and where it lives. Do not re-derive these.

| Hypothesis | Verdict and evidence | Source |
|---|---|---|
| The rung-3 wall is **menu navigation** — the main menu needs Up/Down plus Cross to start a game | **Falsified.** The menu is the wrapper's options menu and has no start item; its seven rows are all wrapper controls. The game's own HOW TO PLAY page binds **Start to the TOUCHPAD**, not to any menu row. Adding directional navigation to the committed route cannot reach gameplay. | this PR |
| Pressing Start (touchpad) leaves the narration and reaches the title screen | **Not observed, and the route is proven to have been delivered.** 13 touch-pad presses read by the guest (`PROSPER_PAD_SCRIPT_LOG=1`, `buttons=touchpad`) across 320 s; every frame black inside the bezel bar one narration line. Distinguish this from a route that never arrived — that check is why the log flag exists. | this PR |
| The committed route's **OPTIONS presses are what hold the title at the menu**, so a Cross-only route would reach gameplay | **Falsified.** Cross-only, 28 frames / 336 s: no menu ever appears (which is the evidence that OPTIONS is the opener), and the picture is black throughout except one narration line at the end. Removing OPTIONS removes the menu, not the blackness. | this PR |
| **#1773** — the `s_mov_b32` descriptor-copy provenance gap — is what drops Earthion's `pc=28` draw | **Falsified by fixing it.** With `sreg_ud_alias` live the alias resolves correctly to `s9`, and the draw is still declined because `by_sgpr_base(9)` finds nothing: the resource is dropped upstream by the degenerate-T# guard (#1590, a guest defect). Composite 28/28 pixel-identical before and after. Two independent causes; this removed one. Same shape as instrument trap 54. | this PR, #1773, #1590 |
| A recompiler reject census on the gameplay route will name the blocker | **Void as run — there is nothing to census.** On the touch-pad route the entire run produces exactly **one** diagnostic line (`[mimg-unresolved]`) and **zero** `[recompile-reject]` / `[exec-recompile-reject]` lines. The route README's "rejected every frame, `occurrence=32768` in five minutes" describes the **menu** route, which exercises a different program set. Name the route before quoting a reject volume. | this PR |

## Instrument notes specific to this title

* **Do not sample screenshots on an interval that divides the title's own cycle.** The picture cycles
  with a ~60 s period; a 12 s interval yields five phases and reported the identical narration frame
  seven consecutive times, which reads as a frozen title. See instrument trap 211.
* **Measure inside the bezel.** The renderer-drawn CRT chrome is identical in every state and
  dominates any whole-frame metric. Two of the three "states" a whole-frame CRC distinguishes on this
  title are the same black picture with a small animated indicator near the bottom edge
  (`~(1839, 2112)-(1991, 2121)` at 3840x2160) — a CRC change there is not a change in the picture.
* `PROSPER_SHARPLOG=1` prints what the stage **declared** before the drop guards run; reach for it
  before concluding anything about descriptor recovery here (`docs/RESOURCE_BINDING.md` § Ruled out).

## What the next lane should do

The frontier is **not** the recompiler and **not** input routing. It is that the game picture never
leaves the narration sequence. Two concrete openings, in order of cheapness:

1. **Find out whether the wrapper ever starts the inner game.** The narration recurs on a ~60 s
   cycle and the picture is otherwise black; establish whether that is an attract loop whose other
   phases fail to draw, or a sequence that restarts. A dense, non-commensurate sampling sweep plus a
   draw census per phase answers it. Do not reuse a 12 s grid.
2. **`libScePad::n3kSX62fgNo` is unimplemented** (returns 0) and is not in `../PS5-3.20_Libs/`, so it
   is newer than 3.20. Given that Start is the touch-pad click, a pad function this title calls and
   prosper answers with 0 is worth naming before anything else is attempted. `libSceVideoOut`'s
   `MCJ8SkzsQxY` / `eb-gvTYQcoY` and `libSceAgcDriver`'s `U9ueyEhSkF4` / `JQc0956gCf0` /
   `F0ZXt5q0ZTA` are unresolved in the same way.

`#1590` remains the record of the descriptor defect (closed, routed past in #1775). It is a guest
defect, so the fix is on prosper's side only in the sense that the render-target group must answer
the guest's index-1 request the way hardware does.

[#1880]: https://github.com/mattias800/prosper/issues/1880
