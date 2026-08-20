# Bendy and the Dark Revival (`PPSA27624`) — status

Tracker: [#1897](https://github.com/mattias800/prosper/issues/1897). Engine: Unity 2022.3.51f1 /
IL2CPP. This document is the technical record the tracker points at; the tracker holds the rung.

**Rung 3 — gameplay with real GPU draws**, reached 2026-08-20 on a scripted route with **no prosper
code change**: the title was one input route away. `prosper/scripts/bendy-dark-revival-PPSA27624/`
holds the routes.

## The progression oracle: `Media/levelNN` is the Unity build-settings scene index

Derived offline from the dump, with no prosper involved, and self-checking:

* `Media/globalgamemanagers` holds exactly **37** build-settings scene paths, at contiguous offsets
  183432..185960.
* `Media/` holds exactly **37** `levelNN` files, `level0`..`level36`. 37 == 37.
* The section scenes are `Section_S101`..`Section_S133` — a **gapless, monotonic** run of 33, plus 4
  core scenes = 37. A mis-derived index therefore breaks the `S1xx` sequence rather than yielding a
  plausible wrong answer, which is what makes the mapping self-checking rather than merely plausible.
* `sharedassetsN` gives an independent second reading: a scene load opens `levelN` **and**
  `sharedassetsN`, so the pairing cross-checks the index without re-using the same derivation.

| file | scene | file | scene |
| --- | --- | --- | --- |
| `level0` | `Core/1_Initialize/InitializeGame` | `level4` | `Core/3_Sections/Section_S101_JoeyDrewApartment` |
| `level1` | `Core/2_General/Empty` | `level5` | `Section_S102_ArchGateOffices` |
| `level2` | `Core/2_General/Game` | ... | ... |
| `level3` | `Core/2_General/Reset` | `level36` | `Section_S133_Archives` |

**`level2` is NOT a gameplay signal.** `Core/2_General/Game` is the persistent scene that carries the
menu UI, and the title screen already reaches it. The gameplay discriminator is **any `levelN` with
N >= 4**, i.e. a `Core/3_Sections/Section_S1xx_*` scene; `level4`
(`Section_S101_JoeyDrewApartment`) is the one a new game loads first.

This matters because aggregate frame metrics cannot separate this title's animated menu from a level
— see `GAME_COMPAT_ORCHESTRATION.md`, "Cross-title: aggregate frame metrics cannot tell an animating
menu from gameplay". Read frame counts, pixel-distinctness and `status=ok` as checks that a run was
**valid**, never as evidence of what it **reached**.

## The sibling positive control: *Bendy and the Ink Machine* (`PPSA27616`)

Same studio, same engine family, tracker [#1881](https://github.com/mattias800/prosper/issues/1881),
already at rung 3. Its build settings hold **11** scenes against **11** `levelNN` files:

| file | scene | file | scene |
| --- | --- | --- | --- |
| `level0` | `Core/InitializeGame` | `level6` | `_GameEnding/Apartment` |
| `level1` | `Core/Chapters/CH1` | `level7` | `Core/Chapters/Archives` |
| `level2`..`level5` | `Core/Chapters/CH2`..`CH5` | `level8` | `Core/TheEnd` |
| | | `level9` / `level10` | `Core/Empty` / `Core/Reset` |

So the two titles are **not** structured alike, and the difference is the reason Dark Revival needs a
different discriminator: Ink Machine puts a whole chapter in one scene (`level1` == gameplay), while
Dark Revival keeps a persistent `Game.unity` and streams 33 section scenes on top of it.

Ink Machine's committed route lives in `prosper/scripts/bendy/` — note that directory is named for
the **franchise** and drives `PPSA27616`, not this title (#2756). Dark Revival's routes are in
`prosper/scripts/bendy-dark-revival-PPSA27624/`.

## Route

`prosper/scripts/bendy-dark-revival-PPSA27624/reach-gameplay.pad`, native Linux/RADV, unmodified
`tools/screenshot` frontend, no snapshot acceleration:

```bash
PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 PROSPER_FILELOG=1 \
PROSPER_SAVE0=~/<per-title-save-dir> \
PROSPER_PAD_SCRIPT=@prosper/scripts/bendy-dark-revival-PPSA27624/reach-gameplay.pad \
  ./build/screenshot <DUMP_ROOT>/PPSA27624-app0 \
    --seconds 15 --count 20 --out ~/<outdir> --timeout 420 --no-manifest
```

**Set `PROSPER_SAVE0` to a per-title directory.** Save data is not namespaced by title (#2734), and
the default `/tmp/prosper-savedata0` is shared with every other game, so another title's save can
present as this one's corruption. The variable is read at `prosper/src/hle/fs/hle_file.cpp:486`;
`PROSPER_SAVEDATA` does not exist and is silently ignored.

Observed timeline, consistent to within ~1 s across three runs on one machine (times are from the
first pad poll, which this title issues at boot):

| t | what |
| --- | --- |
| ~0-20 s | opening movie; #1981's media clock ends it |
| ~50-65 s | title screen with `BEGIN`; `draws_last` 2 -> 3 -> 7 -> 12 |
| ~70 s | `Media/level2` + `Media/level4` open together; `draws_last` 23 |
| ~77-87 s | `draws_last` **1014-1204** — `Section_S101_JoeyDrewApartment` renders |
| ~108-118 s | `Media/level5` + `Media/level6` (`Section_S102`, `Section_S103`) |
| ~150-165 s | with Square: the `STAND` interaction completes and the objective HUD appears |

### The route is save-state dependent, and this is the thing to get right when re-running it

Measured 2026-08-20 across four bounded runs. With a **fresh** `PROSPER_SAVE0`, the button sequence
takes `BEGIN` -> empty slot -> NEW GAME and opens `level4` first, then `level5`+`level6`, running at
`draws_last` ~1,000-1,200 while the opening section streams in. Re-run **against the previous run's
save directory**, the identical sequence takes the CONTINUE path instead: it opens `level5`+`level6`
with **no `level4` at all**, runs at `draws_last` ~200, and the frames carry the title's own
**`Game Loaded.`** notice.

Both arms reach gameplay, so this is a determinism requirement rather than a failure — but a route
"validated" without controlling for it is measuring two different things on alternate runs, and the
scene set is the only signal that shows the difference. The aggregate metrics do not:
`guest=running status=ok`, `stop=request-satisfied` and full sample counts in every arm.

A useful side effect: the CONTINUE arm is direct evidence that prosper's save-data round-trip works
for this title end to end — run B wrote the save and run C read it back and resumed in the right
section.

### Button bindings this route depends on

* **Cross** carries the entire menu path, and also skips the opening cutscenes — their `SKIP` prompt
  is bound to Cross.
* **Square** is the in-world interaction, and the first prompt (`STAND`) is a **hold**. A Cross-only
  route reaches the sections and then holds at that prompt indefinitely; that is a route limit, not a
  title or emulator blocker. `menu-probe.pad` is kept as the Cross-only arm that demonstrates it.
* The route keeps Square out of the first 120 s so it can never land on the save-slot screen, whose
  Square binding has not been established.

## Ruled out

*(One line each: the dead hypothesis, the evidence that killed it, and the issue/PR. The pre-#1981
stalled-boot falsifications are recorded on tracker #1897 and are not repeated here.)*

- **"Reaching gameplay needs a prosper fix."** Falsified 2026-08-20 on `80c0756e` with no code
  change at all: a pad route reaches `Section_S101`/`S102`/`S103` with `draws_last` up to 1,204 and
  the in-game objective HUD. Three bounded `tools/screenshot` runs, `guest=running status=ok`, exit
  status 0 in each. Tracker #1897.
- **"#1979 (the title-screen background video is never composited) gates progression."** Falsified
  by the same three runs: the defect is unchanged and every run passes the menu and reaches
  gameplay. It is cosmetic. #1979 stays open.
- **"Repeating the route repeats the run."** Falsified 2026-08-20: with a save left over from the
  previous run the same sequence takes CONTINUE instead of NEW GAME, opens a different scene set
  (`level5`+`level6`, no `level4`) and runs at a fifth of the draw rate — while every aggregate the
  capture tool produces stays identical. Re-run with a fresh `PROSPER_SAVE0`. See the section above.
- **"The sibling's scene layout transfers."** Falsified by the two dumps' own bytes: *Bendy and the
  Ink Machine* (`PPSA27616`) has 11 scenes and puts a chapter in one scene, so `level1` is its
  gameplay signal; Dark Revival has 37 and streams 33 sections onto a persistent `Game.unity`.
  Reusing the sibling's discriminator here marks `level1` (`Empty`) as gameplay and `level2`
  (`Game`, reached at the title screen) as a level. See the sibling section above.
