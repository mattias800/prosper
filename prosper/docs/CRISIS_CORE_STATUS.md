# Crisis Core –Final Fantasy VII– Reunion (`PPSA07809`) — status

Unreal Engine 4.27 + IoStore. Tracker: [#1894](https://github.com/mattias800/prosper/issues/1894).

**Rung 2 — the title screen is reached and rendered.** Verified 2026-08-06 on a native
Linux/Vulkan run (RADV, AMD Radeon 8060S), unmodified `tools/screenshot`, images inspected by eye.

<p align="center"><img src="../../assets/screenshots/crisis-core-title.png" alt="Crisis Core Reunion — title screen"></p>
<p align="center"><img src="../../assets/screenshots/crisis-core-main-menu.png" alt="Crisis Core Reunion — main menu"></p>

The boot sequence that now runs end to end: white splash → the autosave-notice dialog → (one
Cross) → **"Press Any Button" title screen** with `Ver. 1.0.4` and the Square Enix copyright, held
stably to t=75 s. Pressing Cross again opens the **main menu** — `NEW GAME` / `LOAD GAME` /
`CONTINUE` (correctly greyed out, no save present) / `OPTIONS` / `CONVERT SAVE DATA` — and pressing
on through it opens the new-game settings flow: voice language, then "Begin game with selected
settings? Yes/No". All of these render correctly.

Route arms, stated exactly, because the second one is why the checked-in route has two input
pulses: with a **single** Cross at flip 120-140, 1 of 2 arms reached the title screen and the other
sat on the autosave dialog to t=90 s — the flip at which that dialog appears varies more than
expected between arms. With **two** pulses (the checked-in form), the one arm run reached the title
screen and then its main menu. So: title screen observed in 2 arms across 3, and never a fault in
any arm that had the throttle.

## The route, and the one thing that is not honest about it

```bash
PROSPER_GUEST_ARGS= PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_SUBMIT_STALL_US=3000 \
  PROSPER_PAD_SCRIPT=@prosper/scripts/crisis-core/reach-title.pad \
  ./build-linux/screenshot <DUMP_ROOT>/PPSA07809-app0 --seconds 2 --count 90 --timeout 220
```

`PROSPER_SUBMIT_STALL_US=3000` is a plain sleep in the guest's own submit call, and the title should
not need it. It is in the route because it is what gets past #1945 — and understanding *why* is the
substantive result below. **A default run still dies 5–12 s into the boot**, so this is rung 2 on a
throttled route, not rung 2 on the default one. Say so wherever the rung is quoted.

The first captures were taken with `PROSPER_MB3_POISON=1` instead, which reaches the same screens.
The route names the sleep because the measurement below shows the scan works only through its
duration: a route built on a diagnostic would silently change behaviour the day that diagnostic gets
cheaper — and it did get roughly an order of magnitude cheaper inside this very PR, when the scan
was narrowed from 32 size classes to one and then made double-observing.

## #1945 on this title is decided by submit duration — and is not a late write

Every arm of a default run ends in the guest's own MallocBinned3 bundle-list pop at
`eboot+0x174c206` (`mov rax,[rcx]`), either as a prosper `WORKER-THREAD FAULT` or as the guest's
own `FMallocBinned3 Attempt to free/realloc an unrecognized block …`. The bin array is the
per-thread `FFreeBlockList` fetched through `scePthreadGetspecific` (NID `eoht7mQOCmo`, GOT slot
`0x822b300`), 32 bytes per size class as `{partial.head, partial.count, full.head, full.count}`,
and **only the 32-byte class (bin 1 — the class the consumed-marker labels are allocated from) is
ever corrupt**; in two independent fault dumps every other bin in the same array held a valid
pointer.

What separates the arms is **how long the guest's own submit call takes**. The lever is a plain
`nanosleep` at the end of the submit fold (`PROSPER_SUBMIT_STALL_US`), so the arms differ in nothing
but duration. Twelve arms, three per dose, 60 s bound, one binary, otherwise the default route:

| stall per submit | survived the bound | died (either costume) |
| --- | --- | --- |
| none (default) | **0 / 3** | 3 / 3 |
| 500 us | **0 / 3** | 3 / 3 |
| 1500 us | **3 / 3** | 0 / 3 |
| 3000 us | **3 / 3** | 0 / 3 |

**0 of 6 below the threshold, 6 of 6 above it** (Fisher exact, two-tailed, p ≈ 0.002), with the
threshold between 500 us and 1500 us. Surviving arms reach 300-480 flips and 14,000-23,000 draws;
dying arms never get past the engine's startup.

The diagnostic arms agree and are why the lever was looked for at all. **All three were measured
against the pre-narrowing, single-pass scan and are not expected to reproduce on merged master** —
the shipped scan walks one size class instead of 32 and walks it twice, so it is roughly an order of
magnitude cheaper, and by this section's own thesis a cheaper probe may simply stop rescuing the
title. They are kept because they are what pointed at duration; do not quote them as current
behaviour, and use `PROSPER_SUBMIT_STALL_US` for anything that needs to reproduce.

| arm (pre-narrowing scan) | outcome |
| --- | --- |
| `PROSPER_MB3_POISON=1` (deep interior chain walk, thousands of reads/submit) | **survives**, reaches the title screen — 2/2 |
| `PROSPER_MB3_POISON=1 PROSPER_MB3_POISON_HOPS=1` (same probe, shallow) | faults 5 s in |
| `PROSPER_POOLSHIFT_WINDOW=1` (the pre-existing per-submit bin scan) | faults |

So **nothing any probe reads matters — only how long it takes.** The corruption is a timing race
that disappears when the guest is held inside its submit call for one to three milliseconds.
`CONFIDENCE: HIGH` — not because the mechanism is understood (it is not), but because the claim
being made is only that duration decides the outcome, and that is what 12 arms across four doses
measure, with a monotone threshold and a one-line lever.

Independent corroboration from another title: `OREGON_TRAIL_STATUS.md` records nine arms of
`PPSA19244` on the same binary as an earlier 10/10 exit-90 census producing **zero** faults and
reaching that title's screen, and concludes "the difference is **timing, not code**". This
dose-response is the controlled form of that observation.

That also makes `PROSPER_MB3_POISON` a **confound**: any measurement taken with it armed is taken
on a title that would otherwise have died, so it must never be used as a passive observer here.
Use `PROSPER_SUBMIT_STALL_US` when the intent is a throttle and the probe when the intent is a
scan, and never read a result from an arm that had both.

**What this does not say.** It does not identify the race, and in particular it is not evidence for
"the guest outruns prosper" over any other ordering story — a stall changes thread interleaving
globally. It also does not transfer to another title without its own dose-response.

## 2026-08-07: the race is named, and it is a version-gated contract this title is outside

The ArcRunner lane's per-fold census (`ARCRUNNER_STATUS.md` § *2026-08-07 (arc7)*) localised the same
family's corruption on `PPSA21406` to the guest's builder thread being released **mid-fold**: prosper
applies a submit's completion writes while it is still executing the rest of the same command buffer,
so the guest's chunk recycler sees its consumed markers and rebuilds labels whose init packets prosper
has not reached yet. prosper's post-submit visibility model exists to make exactly that impossible —
and it is armed by `if (version >= 13)` in `agc_reg_defaults.cpp`.

**This title requests SDK version 10**, so the contract is unarmed for it. That made "forcing it on
rescues Crisis Core too" a *prediction* rather than a second search, and it holds. Same binary,
alternated control/armed, `PROSPER_GUEST_ARGS= PROSPER_RENDER=1`, 40 s bound, exact zero peer-process
censuses before and after all six arms:

| arm | runs | faulted | notes |
| --- | --- | --- | --- |
| default | 3 | **3 of 3**, exit 90 | one arm also carries 4 `FMallocBinned3`/fatal lines |
| `PROSPER_POST_SUBMIT_VISIBILITY=1` | 3 | **0 of 3**, `rc=124` at the bound | lever witnessed on each (`POST-SUBMIT-VISIBILITY FORCED ON`); zero faults, zero fatals |

```bash
PROSPER_GUEST_ARGS= PROSPER_RENDER=1 PROSPER_POST_SUBMIT_VISIBILITY=1 \
  timeout 40 ./build-linux/boot_trace <DUMP_ROOT>/PPSA07809-app0
```

Read the scope narrowly. This is a 40 s survival endpoint, not a progression claim for this title,
and the gate is **not** removed — three rung-6 snapshot-guarded titles are also pre-13 (*Dead Cells*
10, *Blasphemous 2* 10, *Alex Kidd* 8), so removing it needs the matrix
([#2220](https://github.com/mattias800/prosper/issues/2220)). What it does settle is that
the `(500, 1500]` dose-response this document records is not a Crisis Core peculiarity: two
independently brought-up titles, both outside one unarmed contract, both rescued by arming it.

## Ruled out

- **"prosper's completion writes land late, so they hit a recycled label."** Measured directly:
  `PROSPER_PEND_AGE=1` instruments the pend queue's real residency (enqueue → landing). Over 5,632
  completion writes in a dying arm the **peak was 3 ms** and **not one** exceeded the 20 ms warn
  threshold. The write path is prompt; whatever the race is, it is not the pend queue holding
  writes. (2026-08-06, this lane.)
- **"Suppressing both prosper-authored halves of the forged pointer stops the corruption."** The
  combined arm named as unrun in `src/gpu/command_processor.cpp` (§forge_trip) and in
  `ARCRUNNER_STATUS.md` was run here: `PROSPER_REL1_FORGE_SUPPRESS_ALL=1 PROSPER_INIT_SUPPRESS=ptr`,
  a **valid** arm by its own criterion (`FORGE-DECISION-TOTALS candidates=127 suppressed=127
  landed=0`, `INIT-SUPPRESS` reaching #4096). The title still faults at the same pop, with
  `0x0002400100024001` in the bin head instead of `0x30016000` — the same two-pops-farther value
  ArcRunner's fence-only arm produced. So the two known label writes are not the sole author on
  this title. (Note the arm deliberately drops live fences, so it bounds authorship, not
  correctness.)
- **"The `0x30016000` fault face and the `0xff000000ff000000` face are different bugs."** Same
  slot (`bin1.head`), same site, same thread role, different contents; see #1945.
- **"The corruption is the `%fs` leak of #1155."** Retired by #2018 — the thread is on its guest
  TCB, measured.
- **`PROSPER_MB3WATCH` reporting nothing meant the head was not being stomped.** The hook armed on
  a stale `[0x20_0000_0000, 0x21_0000_0000)` window while this title's per-thread cache base is
  `0x3001af0000`, so it armed **nothing** on any current title — the same silent false negative
  #1998 recorded for `PROSPER_WATCH_LABEL`/`PROSPER_WATCH_HOT`. Fixed in this lane; any null result
  quoted from it before 2026-08-06 is **void, not negative**. (It is also not usable on this title
  even when armed: the head is written by every 32-byte malloc/free, and the SIGTRAP storm stops
  the run from making any progress at all — 0 frames in 180 s.)

## Open defects, in the order they matter

1. **#1945** — the bundle-list corruption. The whole reason the default route dies. Narrowed above.
2. **The title screen's key art is missing** (#2057). The "Press Any Button" prompt, version string
   and copyright render; the CRISIS CORE logo/character art that occupies the left of the screen on
   hardware is absent (flat black). Rung 5 will need this.
3. **Most published frames select a source that is not the screen** (#2058). Before this lane's
   throttled route, the frames delivered ahead of the fault were a **single flat colour, alternating
   black and white**, while intermediate render targets carried real image content — that
   measurement (previously in `COMPATIBILITY.md`) is preserved here because it is distinct from what
   follows and may be the same defect at an earlier boot stage. On a run that survives to the menus
   the same path presents a flat blue with magenta blocks while the real content exists —
   3 to 6 of every 90 samples carry the actual screen (measured across four 90-sample runs). The
   dialog text is visible *through* the blue in some frames, so this is a publish/composite source
   selection problem, not a lost draw. Filed as #2058; it makes any small-sample content measurement
   on this title a lottery.

## Reproduction notes

- No `-force-gfx-direct`: `PROSPER_GUEST_ARGS= PROSPER_NULL_PAGE=1`. (Guest `%fs` TLS is on by
  default and is *not* disabled by omitting a switch; `PROSPER_NO_GUEST_FS=1` is the opt-out.)
- `#1982` (the declined ordered-DMA submit) does **not** reproduce on this title on current
  master — 0 occurrences across every arm since #1987.
- `#2027` (the frame-bundle writer never completing) still blocks the F9/`gpu_replay` loop here.
- The dump's per-thread pool base and the fault site have been stable across runs
  (`0x301ac50000` / `0x3001af0000`, `eboot+0x174c206`), which makes this the cheapest reproducer in
  the library for the #1945 family: no input, no gating, dead inside ~10 s.
