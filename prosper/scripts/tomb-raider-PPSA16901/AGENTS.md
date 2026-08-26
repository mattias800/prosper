# AGENTS.md — scripts/tomb-raider-PPSA16901

Input routes for **Tomb Raider I-III Remastered** (`PPSA16901`). One `.pad` file per reached
state, in the syntax `docs/INPUT_REPLAY.md` defines.

The dump is a launcher plus three self-contained games under `1/`, `2/` and `3/`, so a route here
has to say *which* game it drives as well as which screen. Everything below is Tomb Raider I unless
the filename says otherwise.

## What a route has to get past

**The EULA gate.** A default launch opens on a 40-page EULA and holds there forever — it is the
first thing any route must clear, and it is the reason a no-input run of this title looks like a
hang. Cross is **inert** until the last page: pulses on page 1 and again on page 32 both left the
frame byte-identical. Down auto-repeats, so one long hold walks 1/40 to 40/40 and Cross then
accepts. Circle and Options are inert throughout.

**Nothing persists.** No `savedata0` directory is created across a full boot, so the EULA re-shows
on every run and these routes stay reproducible from a cold start. Do not "optimise" the EULA hold
out of a route on the assumption that acceptance was saved — it is not.

## The title-screen menu map

Two independent axes, which is what makes this menu easy to misread:

- **Up/Down switches game** — Tomb Raider I, II, III. The art and logo change with it.
- **Left/Right rotates the ring**, a five-item cycle:
  `Lara's Home` → `Controls` → `Extras` → `Game` → `Sound` → back to `Lara's Home`.
  Left and right traverse it in opposite directions; `Lara's Home` is the item selected on arrival.
- **Cross selects the ring item.**

**The item labelled `Game` IS the New Game entry** — not a settings category. An earlier version of
this file said the ring carried no New Game item; that was wrong, and it was wrong in the expensive
direction, because it sent a lane looking for a missing passport item that is not missing. Three
LEFTs from the default `Lara's Home` reach it, and Cross starts a new game with its opening cutscene.
`reach-fmv.pad` does exactly that.

`reach-gameplay.pad` still enters through `Lara's Home` because that reaches a playable level in a
few seconds with no cutscene to sit through — a route choice, not a limitation.

**The cutscenes are the title's own Theora files and the guest decodes them itself**: a run of
`reach-fmv.pad` logs zero AvPlayer and zero videodec activity. Their pacing is therefore the game's
clock, not prosper's video backend, so a timing defect seen in a cutscene is not an AvPlayer bug.
Ground truth for timing work is in the dump — `ffprobe` the FMV and count its packets.

## Timing

The title screen arrives at roughly 38 s on the development host, after the EULA accept and the
logo sequence. That is wall-clock on one machine and not a contract: if a route desyncs, re-derive
the arrival time from a `--seconds 1` capture rather than nudging the numbers blind, and prefer
poll-safe holds with neutral gaps over point presses.
