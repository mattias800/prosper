# Bendy and the Dark Revival (`PPSA27624`) — status

Tracker: [#1897](https://github.com/mattias800/prosper/issues/1897). Engine: Unity 2022.3.51f1 /
IL2CPP. This document is the technical record the tracker points at; the tracker holds the rung.

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

## Ruled out

*(Falsified hypotheses go here: one line each, with the evidence that killed it and the issue/PR.
The pre-#1981 stalled-boot falsifications are recorded on tracker #1897 and are not repeated here.)*
