# Tomb Raider I-III Remastered (`PPSA16901`) — status

Tracker: [#2990](https://github.com/mattias800/prosper/issues/2990).
Route files and the menu map: `prosper/scripts/tomb-raider-PPSA16901/AGENTS.md`.

**Rung 2** — the Tomb Raider I title screen is reached and rendered on a default launch.

The headline fact about this title is that **no prosper code change was needed to get here.** It
booted and rendered real 1920x1080 frames on the first attempt, and everything below rung 3 was
reached with pad input alone. Prior to 2026-08-26 the title had no record anywhere in this
repository — no `COMPATIBILITY.md` row, no tracker, no mention in any source file or script.

## The dump

6.1 GB. A launcher `eboot.bin` plus three self-contained games under `1/`, `2/` and `3/`, each with
its own `tombN.prx`, and `pros.sdk.Prospero-9.000.prx` alongside. **SDK 9**, so this title is on the
pre-13 side of the post-submit completion-visibility contract (#2219) — worth remembering if a
submit-race signature ever shows up here, though nothing so far points that way.

## What renders

| State | Result |
| --- | --- |
| EULA gate (40 pages) | renders and scrolls; body text garbled (below) |
| Publisher/developer logos | correct — the Saber Interactive logo is pixel-clean |
| Tomb Raider I title screen | logo, Lara portrait, animated ring menu and labels all correct |
| Tomb Raider III title screen | logo, Lara model and background art correct; ring area is a solid violet block |
| Croft Manor (`Lara's Home`) | live 3D scene, animating; **severely degraded** (below) |

## Open defects

1. **Gameplay world geometry is shattered and untextured.** In Croft Manor the character models are
   correct and recognisable — Lara's silhouette and Winston carrying the tea tray both read
   immediately — while the static level geometry is torn into long stretched triangles and every
   surface draws untextured. This is what holds the title at rung 2 rather than rung 3.
2. **Text is intermittently garbled.** The EULA body and the Tomb Raider III title labels draw the
   wrong glyphs; the *same font* renders correctly for the `EULA` heading, the `1/40` page counter
   and every Tomb Raider I ring label. Right font, right layout, wrong glyph selection — so this is
   a selection or source-string problem, not a rasterization one.
3. **Solid untextured quads on the menus** — small magenta quads on the Tomb Raider I ring and a
   large violet block over the whole Tomb Raider III ring. Plausibly the same root cause as (1),
   since both are "a surface that should be textured is not", but that link is a hypothesis and has
   not been tested.
4. **No savedata is written, and the ring menu has no passport item.** No `savedata0` directory
   appears across a full boot, and the five-item ring carries no New Game or Load Game. Whether
   those are one defect or two is unestablished.

## Ruled out

- **The renderer is not rejecting anything.** A full boot-to-gameplay run produces **zero**
  `[recompile-reject]` lines and **zero** `[compute] skip` lines, so neither the missing textures
  nor the torn geometry is an unsupported-op gap. Whatever is wrong is semantic — prosper is
  executing the title's draws and getting a wrong answer, not declining to execute them.
  (run08, 2026-08-26, `606fd6ae`.)
- **The EULA hold is not a hang, and not a renderer stall.** A no-input launch sits on a single
  distinct frame for its whole run, which reads exactly like a stalled title; it is the game waiting
  for input on page 1 of 40. The tell is that the *content* is a correctly rendered EULA page.
  Cross does not clear it — only reaching page 40 does.
- **Acceptance is not persisted, so route desync is not a savedata-staleness effect.** No
  `savedata0` directory is created at all, and the EULA re-shows identically on every cold start.

## Not yet investigated

- Whether the missing New Game item is caused by the absent savedata, or is independent.
- Whether the garbled text and the untextured quads share a root cause.
- Tomb Raider II, beyond confirming that Up/Down reaches its title screen.
