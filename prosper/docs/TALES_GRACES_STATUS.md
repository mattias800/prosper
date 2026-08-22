# Tales of Graces f Remastered (`PPSA19991`) — status

Unity / IL2CPP, Bandai Namco. Tracker: [#1889](https://github.com/mattias800/prosper/issues/1889).

## Current rung

**Rung 3 — gameplay with real GPU draws.** A default boot renders the publisher and CRIWARE
logos, plays the opening movie to completion and reaches the title screen with no input at all;
`scripts/talesgraces/reach-gameplay.pad` then drives the privacy policy, the main menu, the
new-game settings screen and the Grade Shop into the **Lhant Hill prologue**, which renders as a
full-colour 3D field with character models and in-world dialogue and holds for the rest of the run.

## What was blocking rung 3, and it was not the renderer

`scripts/talesgraces/reach-options.pad` could never leave the new-game settings screen, and the
project record read that screen as the title's frontier. It is not: **two screens in the new-game
flow bind their "go" action to the OPTIONS button, and each then raises a Yes/No dialog whose
default selection is No.**

| screen | footer, as the game draws it | dialog it raises |
| --- | --- | --- |
| new-game settings | `[options] Confirm Settings   [left stick] Change` | `Confirm settings?`  Yes / **No** |
| Grade Shop | `(cross) Select   [options] Start Game   L2 R2 Scroll` | `Start the game?`   Yes / **No** |

Cross is not bound to "confirm" on either screen, so a Cross-only route sits on the settings
screen indefinitely — which is exactly what every previous run observed. Pressing OPTIONS raises
the dialog, and Cross there answers **No** and returns to the same screen, so even an
OPTIONS+Cross route loops.

**The second half of the gate is a route-authoring hazard worth reusing.** A `up` window written
in this repo's usual whole-second form (`110-111:up`) **auto-repeats** inside a two-entry list and
lands back on No; three separate one-second `up` windows left `Confirm settings?` reading No. The
same window shortened to **0.15 s** (`485-485.15:up`) moved the cursor to Yes on the first try.
Any route that has to pick a non-default entry in a short list should tap, not hold.

## Reproduction

```bash
PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 \
PROSPER_SAVE0=~/<per-title-save-dir> \
PROSPER_PAD_SCRIPT=@scripts/talesgraces/reach-gameplay.pad \
  ./build-linux/screenshot <DUMP_ROOT>/PPSA19991-app0 --seconds 10 --count 90
```

`PROSPER_SAVE0` matters here: prosper's save-data root is not namespaced by title (#2734), so a
shared default directory lets another title's save present as this one's state.

## The content-level progression oracle: an Addressables map-bundle pair

**The cross-title Unity discriminator in `GAME_COMPAT_ORCHESTRATION.md` — `Media/levelNN` as the
build-settings scene index — does not apply to this title, and the way it fails is quiet.**
`Media/` holds exactly **one** level file (`level0`) and `globalgamemanagers` names exactly one
build-settings scene, `Assets/Scenes/NobleGameMainScene.unity`. The whole game is one Unity scene
plus Addressables, so a scene-index oracle here reads a constant and would report "no progress" on
a run that reached the prologue.

The equivalent signal is the Addressables bundle the guest opens.
`Media/StreamingAssets/aa/PS5/` holds 16,400 bundles, of which **459 are
`togf_mapmodel_assets__map_map_<code>.bundle`** and **375 are
`togf_maptexture_assets__map_map_<code>.bundle`**. The pairing is what makes it self-checking:
**every one of the 375 maptexture codes has a mapmodel partner** (`comm -13` over the two sorted
code lists is empty), so a mis-derived code surfaces as an unpaired name rather than as a
plausible wrong answer. `PROSPER_FILELOG=1` names each one as it is opened
(`hle_file.cpp:651`/`:682` log every guest open unconditionally, so a zero is exhaustive over the
closed set of opens rather than an absence of evidence).

**The exclusion set is measured, not assumed.** Two map codes are opened *before* gameplay and
must not be counted:

| code | where it appears | side |
| --- | --- | --- |
| `map_map_zone_d01` | boot / title screen | maptexture only |
| `map_map_sysm_d01` | main menu and new-game settings screen (`sysm` = system menu) | both |

A 60 s default boot opens **zero** `togf_mapmodel_*` bundles, and a 750 s routed run that stopped
on the settings screen opened only those two codes across its whole length. Reaching the prologue
opens **`map_map_mont_d01` on both sides within the same second** — a matched pair, outside the
exclusion set, at the exact sample where the frames become a 3D field.

So the success condition, stated before the run that satisfied it: *a matched
`togf_mapmodel` + `togf_maptexture` pair for one `map_map_<code>` other than `zone_d01` and
`sysm_d01`.*

## Ruled out

- **"The new-game Options screen is the last screen before gameplay."** Two more gates follow it:
  the `Confirm settings?` dialog and the entire Grade Shop, itself gated by `Start the game?`.
  Both are OPTIONS-button gates with a No default. Evidence: routed run reaching
  `map_map_mont_d01` and the Lhant Hill prologue, 2026-08-20.
- **"The prompt before gameplay is *Begin game with selected settings?*."** That string belongs to
  *Crisis Core –Final Fantasy VII– Reunion* (`PPSA07809`) — `docs/CRISIS_CORE_STATUS.md` and
  `scripts/crisis-core/reach-title.pad`. This title's prompts are `Confirm settings?` and
  `Start the game?`. No run of `PPSA19991` has ever produced the Crisis Core string; it reached
  this title's briefing by transplant.
- **"A directional press in a menu needs the same window length as a Cross press."** Falsified on
  the `Confirm settings?` dialog: 1 s `up` windows left the cursor on No three times; 0.15 s moved
  it to Yes. Auto-repeat inside a two-entry list.
- **"`Media/levelNN` is a progression oracle for this title."** There is one level file. See the
  section above.
- **"Any `togf_mapmodel_*`/`togf_maptexture_*` open means a field map loaded."** `zone_d01`
  (title) and `sysm_d01` (system menu) are opened before gameplay; both are menu backdrops.
- **The recompiler is not implicated.** `PROSPER_DBG`-class `[recompile-reject]` and
  `[compute] skip` lines: **0** across the 750 s run that reached the prologue.

## Known defects on this title

- ~~[#2731](https://github.com/mattias800/prosper/issues/2731) — decoded movie frames composite with
  their chroma components collapsed (`Cr == Cb`).~~ **Fixed.** This title stages the decoded NV12 in
  two allocations of its own, `0x111000` bytes apart, and the renderer's chroma-plane test required
  the luma plane to END at the chroma plane's address — so every movie frame took the legacy narrow
  coverage broadcast, which repeats the first byte into every channel. Measured on the default route
  with `tools/screenshot`: `corr(Cb, Cr)` over the ten opening-movie samples of a 120 s run went from
  `+0.9992..+1.0000` to `-0.95..+0.53`, and the sky is blue again. See `docs/RESOURCE_BINDING.md`
  § Ruled out.
- [#1688](https://github.com/mattias800/prosper/issues/1688) — `sceVideodec2` has no real AVC
  decoder for the later movie path.
- [#1673](https://github.com/mattias800/prosper/issues/1673) — APR completion delivery; measured
  latent on this title's route (#2638).
- Route reliability: boot depth on the development host varies by more than 200 s with machine
  load — the settings screen has been observed at t=90 s and at t=441 s on the same binary and the
  same route. `reach-gameplay.pad` covers this by repeating both gates once, ~170 s later; a
  route-timing anchor that is genuinely load-independent does not exist here, because the guest
  polls the pad and flips frames throughout the boot. This is the same class as
  [#2743](https://github.com/mattias800/prosper/issues/2743). The repeated pass is benign when
  it is not needed: OPTIONS during the prologue raises a `[triangle] Skip` overlay, and the
  route's following `up`/Cross do not accept it, so the scene carries on.
