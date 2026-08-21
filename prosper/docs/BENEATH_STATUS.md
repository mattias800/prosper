# Beneath (`PPSA27640`) — status

Tracker: [#1898](https://github.com/mattias800/prosper/issues/1898). Engine: Unity / IL2CPP.
This document is the technical record the tracker points at; the tracker holds the rung.

**Rung 3 — gameplay with real GPU draws**, reached 2026-08-21 on a scripted route with **no prosper
code change**: the title was one input route away. The route lives in
`prosper/scripts/beneath-PPSA27640/`.

## The one thing to know before running this title

**`PROSPER_NULL_PAGE=1` is required, and it is not papering over a heap bug.** Without it the
guest's primary thread dies **1.1 s into boot**, before any frame:

```text
[shot] guest thread ended: kind=2 detail=SIGSEGV at addr=0x8 rip=0x410e44346 (image+0xe44346)
[shot] done: 0/45 screenshot(s); stop=guest-fault ... guest=faulted status=GUEST-FAULT
```

The fault is fully explained, and the explanation is what makes the flag defensible rather than
superstitious. `eboot+0xe44346` sits inside an **unrolled frame-pointer-chain unwinder**:

```asm
e44277:  xor    eax,eax                       ; depth = 0
e44280:  cmp    eax,0x13                      ; 20-entry computed jump, one entry per depth
e4428b:  movsxd rdx,DWORD PTR [rcx+rdx*4]
e44292:  jmp    rdx
...
e44310:  mov    rdx,QWORD PTR [rdx]           ; \  saved-rbp hops, one per depth,
e44313:  mov    rdx,QWORD PTR [rdx]           ;  >  UNCHECKED
   ...                                        ; /
e44346:  mov    rdx,QWORD PTR [rdx+0x8]       ; <-- faults: the return-address slot
e4434a:  lea    rbx,[rax+0x1]                 ; depth + 1
e4434e:  mov    QWORD PTR [rbp+rax*8-0x540],rdx
e44356:  test   rdx,rdx
e44359:  je     0xe44368                      ; the loop's ONLY terminator: a NULL return address
```

So the walk terminates on a **null return address**, never on a null frame pointer — it takes one
hop past the end of the chain by design and expects the read to yield zero. prosper enters the guest
with `xor %rbp,%rbp` (`src/host/image/exec_image_linux.cpp:3809`), the correct SysV convention, so
the outermost frame's saved rbp *is* 0 and the extra hop lands on `[0x0 + 8]`.

Three independent checks say this is that unwinder and nothing else:

* The instrument dump names the instruction and the register: `insn @rip: 48 8b 52 08`
  (`mov rdx,[rdx+8]`) with `rdx=0x0` — and `rax=0x11`, i.e. **depth 17**, inside the 20-entry table.
* The dump's `rcx` is the **jump table itself**: `rcx=0x411d06740`, and the function loads
  `lea rcx,[rip+0xec24de]` = image-relative `0x1d06740`. Image base + `0x1d06740` is exactly that
  value, so the faulting instruction is provably inside this dispatch and not merely near it.
  `rbx=0x11` alongside `rax=0x11` is the `lea rbx,[rax+1]` of the previous iteration read back at
  the top of this one, which is what makes 17 a *depth* rather than an unrelated value.
* **Exactly two** emulated low reads occur in a 450 s run, both at that one rip and that one
  address. Heap corruption or a missing HLE return value would scatter across rips and addresses;
  this does not. The guest collects this backtrace twice, deterministically, while loading its
  `Media/Plugins/*.prx` native plugins.

`PROSPER_NULL_PAGE=1` therefore restores exactly the platform property the guest is relying on — a
low page that reads as zero — and the run is otherwise unmodified. Whether that should become the
Linux default is a separate, much wider decision and is **not** proposed here. The full diagnosis,
including why the wider default question is left open, is #2813.

## The progression oracle: `Media/levelNN` is the Unity build-settings scene index

Derived offline from the dump, with no prosper involved, and self-checking:

* `Media/globalgamemanagers` holds exactly **46** build-settings scene paths at contiguous offsets
  218784..221088.
* `Media/` holds exactly **46** `levelNN` files, `level0`..`level45`. 46 == 46, index-for-index.
* **The self-check is the `.resS` partition.** Twelve `levelNN` files above `level5` carry a
  `.resS` streamed-resource sidecar: 14, 15, 16, 20, 24, 25, 26, 27, 30, 31, 39, 44. Under this
  mapping **every one of the twelve is a world scene and not one is a loading screen** — and the
  dump is more than half loading screens, so a mis-derived index would break that partition rather
  than produce a plausible wrong answer. The four `ivanov_level1..4` scenes landing on the four
  consecutive `.resS` files 24-27 is the sharpest instance.

| file | scene | file | scene |
| --- | --- | --- | --- |
| `level0` | `Assets/Preload` | `level7` | `LOADING_SCREENS/LoadShip` |
| `level1` | `Menu/Health_W` | `level14` | `SCIENCE_SHIP/Science_Remake2` |
| `level2` | `Menu/ShaderPreload` | `level15`/`level16` | `BAUER/Bauer Lucas v2` / `externalElevator` |
| `level3` | `Menu/Launcher` | `level20` | `WIKNER/Wikner_45_4` |
| `level4` | `Menu/MenuD` (main menu) | `level24`..`level27` | `IVANOV/ivanov_level1..4_IDs` |
| `level5` | `INTRO/Intro` | `level42` | `LOST CITY/Lost City v4` |
| `level6` | `LOADING_SCREENS/backMainMenu` | `level44` | `BAUER/Bauer END` |

**The gameplay discriminator is any world scene**, i.e. `levelN` for N in
{14, 15, 16, 18, 20, 24, 25, 26, 27, 28, 30, 31, 34, 39, 42, 44}. A new game loads **`level14`**
(`Science_Remake2`) first, and `level7` (`LoadShip`) is the loading screen that immediately precedes
it — the pair is itself a small consistency check on the mapping.

### The runtime cross-check: FMOD banks name the scene the guest just opened

The `.resS` partition is derived from the same file listing as the index itself. A **runtime** check
that shares nothing with it falls out of the same `PROSPER_FILELOG=1` trace, because this title
loads a per-area FMOD bank immediately after the scene that needs it:

| scene file opens | the very next bank opened |
| --- | --- |
| `Media/level5` | `Media/StreamingAssets/`**`Intro`**`.bank` |
| `Media/level14` | `Media/StreamingAssets/`**`Science_Ship`**`.bank` |

`level5` is `INTRO/Intro.unity` and `level14` is `SCIENCE_SHIP/Science_Remake2.unity` under the
offline mapping, and the guest independently names both at runtime. Two indices derived one way,
confirmed by a completely different mechanism.

**The oracle discriminates here, and that had to be established rather than assumed.** It has
failed twice on other titles: *Tales of Graces f* ships exactly one level file, so the index is a
constant and the oracle is void; *Bendy and the Ink Machine*'s layout does not transfer to its own
sequel. 46 files across menu, loading and world scenes is what makes it usable for this title.

This matters because aggregate frame metrics cannot separate this title's animated menu from a
level — see `GAME_COMPAT_ORCHESTRATION.md`, "Cross-title: aggregate frame metrics cannot tell an
animating menu from gameplay". Read frame counts, pixel-distinctness and `status=ok` as checks that
a run was **valid**, never as evidence of what it **reached**.

### A second content check that does not reuse the derivation

Every gameplay sample carries the title's **waypoint HUD marker** — a pin glyph and a distance
readout (`39m`), the `_WaypointO` / `WAYPOINT DISTANCE` element from the localisation table. It is
drawn only in-world. It is independent of the scene-index derivation, so the two can disagree; they
do not.

## Route

`prosper/scripts/beneath-PPSA27640/reach-gameplay.pad`, native Linux/RADV, unmodified
`tools/screenshot` frontend, no snapshot acceleration:

```bash
PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 PROSPER_NULL_PAGE=1 PROSPER_FILELOG=1 \
PROSPER_PAD_SCRIPT=@prosper/scripts/beneath-PPSA27640/reach-gameplay.pad \
PROSPER_PAD_SCRIPT_LOG=1 \
PROSPER_SAVE0=~/<per-title-save-dir> \
  ./build/screenshot <DUMP_ROOT>/PPSA27640-app0 \
    --seconds 10 --count 58 --out ~/<outdir> --timeout 700
```

`PROSPER_SAVE0` is a **root** under which saves are namespaced by title id (#2734); set it to an
isolated directory per run.

### What the route has to get past, in the title's own words

The whole opening ladder is **Cross**, and every gate preselects the entry the route wants, so the
ladder is self-synchronising — surplus presses simply wait on whatever screen the boot is on. The
strings are from `Media/sharedassets4.assets`, the title's own localisation table:

| screen | text | answer |
| --- | --- | --- |
| title | ` Press Any Button To Start ` (`_Start`) | Cross |
| main menu | `NEW GAME` / `LOAD GAME` / `OPTIONS` / `CREDITS` / `EXIT` | Cross (NEW GAME preselected) |
| GAME MODE | `SELECT DIFFICULTY` — `TOURIST PASS` / `COLD REALITY` / `HARD BASTARD` | Cross (TOURIST PASS preselected, and drawn in green in the capture) |
| INTRO | `INTRO/Intro.unity` narration | Cross |

This is unusual for this project and worth stating plainly: **five titles reached gameplay in the
week before this one and the wall was a mis-mapped button every time.** Here it was not — Cross
answers all four gates. The route's real work is covering a variable boot depth, not finding a
button.

### Why the route stops pressing Cross at t=240

The stick windows are movement in-world, but they are **list navigation** on the main menu, and
`up` on a preselected top entry can wrap onto `EXIT`. Ending the Cross ladder before the sticks
start means a slow boot can still *land* a selection on `EXIT` but can never **confirm** it: the
route has no button left that would commit it.

Cross itself is **measured harmless in-game** for this title — the exploratory arm mashed it from
t=87 s (level14 open) to t=360 s without leaving or changing the scene and without faulting. That
measurement is what lets the ladder run long enough to cover a slow boot.

### Observed timeline

`PROSPER_FILELOG=1` open lines, timestamped against `PROSPER_PAD_SCRIPT_LOG=1` `elapsed=`:

| t | scene | what |
| --- | --- | --- |
| ~0 s | `level0` + `level1` | preload |
| ~6.5 s | `level2` + `level3` | shader preload, launcher |
| ~18 s | `level4` | main menu |
| ~48 s | `level5` | INTRO |
| ~84.5 s | `level7` | `LoadShip` |
| ~87 s | `level14` | **`Science_Remake2` — gameplay** |

## Ruled out

*(One line each: the dead hypothesis, the evidence that killed it, and the issue/PR.)*

- **"Reaching gameplay needs a prosper fix."** Falsified 2026-08-21 on `e653f271` with no code
  change: a pad route reaches `Media/level14` (`SCIENCE_SHIP/Science_Remake2`) with the in-world
  waypoint HUD drawn, in bounded `tools/screenshot` runs ending `guest=running status=ok`.
  Tracker #1898.
- **"The `addr=0x8` boot fault is heap corruption or a missing HLE return value."** Falsified by the
  instrument's own dump: **two** emulated low reads in a 450 s run, both at the single rip
  `eboot+0xe44346`, both at `addr=0x8`, with `rdx=0x0` and `rax=0x11` — a frame-pointer unwinder at
  depth 17 taking its designed one hop past a NULL frame pointer. A corruption or a wrong return
  value scatters across rips and addresses. See the section above.
- **"`PROSPER_NULL_PAGE=1` is optional for this title now."** Falsified 2026-08-21 on `e653f271`:
  the identical route without it dies at 1.1 s, `0/45` samples, `status=GUEST-FAULT`. The
  2026-08-03 intake on #1898 saw the same fault at the same rip.
- **"Aggregate frame metrics can confirm the route reached gameplay."** Not tested here and not
  usable here — the title's menus animate. The scene index and the in-world HUD are the
  discriminators; the aggregates are only a validity check. Cross-title record in
  `GAME_COMPAT_ORCHESTRATION.md`.

## Open, not blocking

- **1 of 4** compute programs is dropped as `reason=recompile-reject mode=unresolved-operand` at
  `pc=17 words=bf880035` — SOPP `op=0x8`, `S_CBRANCH_EXECZ`. The denominator is the census header
  (`[compute-census] 32768 dispatch decisions over 4 program(s)`), because per-program detail
  prints only for programs that skipped (#2745) and only at power-of-two totals (#2746). It does
  not gate progression — both validated runs reach and render `Science_Remake2` — but **what the
  program computes is not established**, so "no visible defect" is weak evidence here rather than
  strong. #2815, plausibly the same family as #590.
