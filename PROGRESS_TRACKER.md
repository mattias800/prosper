# Progress tracker

<!--
    GENERATED FILE -- DO NOT EDIT BY HAND.
    Produced by prosper/tools/docs/gen_progress_tracker.py from the `tracker:game` GitHub
    issues. Regenerate with:  python3 prosper/tools/docs/gen_progress_tracker.py
    CI regenerates this file and fails if it differs from the committed copy.
-->

**This file is generated. The GitHub game trackers are authoritative.**
It is produced by [`prosper/tools/docs/gen_progress_tracker.py`](prosper/tools/docs/gen_progress_tracker.py)
from the `tracker:game` issues, and exists so per-title progress is answerable from a
checkout -- with `grep`, offline, in one command -- rather than only through the GitHub API.
Editing a row here changes nothing: **edit the tracker issue**, then regenerate.

```bash
python3 prosper/tools/docs/gen_progress_tracker.py          # rewrite this file
python3 prosper/tools/docs/gen_progress_tracker.py --check   # fail if it is stale
```

It is a projection, not a second source of truth, and it is deliberately narrow: it carries
only fields the trackers hold in a **fixed, parseable form**. `COMPATIBILITY.md` remains the
user-facing overview, and its markers are a chart, not a rung scale.

> **What this file can be wrong about, and it is not a defect in the generator.**
> It reads each tracker's **body**. The charter records that a tracker body is routinely days
> behind its own comment thread, and that a fresh `updated_at` does not mean the body was
> re-read -- the timestamp moves on every comment. So a row here is exactly as current as the
> body it came from, and no more. This file makes that staleness **visible and greppable**
> instead of leaving it to be rediscovered per reader, but it cannot repair it: the repair is
> editing the tracker body. **Before quoting a rung from this table, read the tracker's last
> comments**, and prefer the title's status doc when the two disagree.
>
> A dated instance, found the day this caveat was written. #2736 merged at `22:58:42Z` on
> 2026-08-19, putting *Asterix & Obelix: Babylon Mission* gameplay on master. Tracker #1884's
> `updatedAt` then moved to `23:16:55Z` -- **after** the merge -- while its ladder still read
> rung 2, because what changed was a comment and nobody had re-read the body. It was ticked to
> rung 3 at `23:29:19Z`, once a human opened the screenshot and confirmed the scene. For those
> 31 minutes the tracker was demonstrably stale **and** demonstrably fresh-looking, and every
> timestamp-based freshness audit would have passed it. That is the whole reason the rule above
> is "read the comments", not "check the timestamp".

## How to read the columns

| Column | Derivation |
| --- | --- |
| **Rung** | The highest **ticked** rung, 0 if none. |
| **Ladder** | Every ticked rung, `-` for unticked. **The ladder is legitimately non-contiguous** on some titles -- PR #1696 and #1676 deliberately took titles from rung 3/4 to rung 6 without rung 5, because a reviewed gameplay guard is evidenced by its own route and thresholds and never depended on a hardware oracle. `1234-6` is a real state, not an editing slip. |
| **Guard** | From `prosper/tools/snapshot/snapshots.json`, matched on title ID -- not from the tracker's prose, so it cannot disagree with the registry. |
| **FPS** | The tracker's `FPS record:` line: the rate while the title was producing frames, and the share of the run that was. `-` means **no tracker line exists**; `none` means somebody looked and there is no measurement; `--` means the title produced nothing. See below. |
| **Oracle** | The tracker's `Oracle record:` line, verbatim. `none` means **no PS5 hardware comparison is on record** (see #2730). Not to be confused with `snapshots.json`'s `structural_references`, which are luminance signatures generated from prosper's own runs -- a *regression* reference, not a hardware oracle. |
| **Open blockers** | Issues/PRs cited in the tracker's `## Current blocker(s)` section that are still open. Cited-and-closed entries are omitted; a tracker citing nothing shows `-`. |
| **Status doc** | First `prosper/docs/*_STATUS.md` referenced by the tracker, else its first `prosper/docs/*.md`. `-` means the tracker references neither. |

A `-` is an **explicit absence**, never a parse failure: the generator aborts on anything it
cannot parse and writes no file at all, so a row that is present is a row that was read.

### The FPS column: the rate while producing, and how much of the run that was

`**18.5** fps · 62% active` reads **18.5 frames per second while the title was producing frames**,
which it did for **62%** of the measured run. Both halves are required, and the second is a
percentage rather than a rate precisely so it cannot be quoted as a rival framerate.

Three states, and the column has to keep them apart:

| Cell | Reading |
| --- | --- |
| `**19.8** fps · 97% active` | A homogeneous window. This is what a record should be made from. |
| `**1.0** fps · 98% active` | Homogeneous and genuinely slow. The "we have work to do" bucket. |
| `**18.5** fps · 62% active` | A **mixed** window -- real, but it should not have been filed. Narrow the window and re-measure. |
| `**--** fps · 0% active` | The title produced nothing. The R-Type Delta shape (#2783) -- see below. |

**Measure over a window where the title was doing ONE thing.** The line names a scene, so it has
already committed to that: measure `gameplay` over gameplay. Mixing regimes is what makes a
framerate meaningless, and no choice of statistic repairs it -- *The Messenger* measured 3.0 fps
averaged over 380 s while alternating between ~15-23 fps and **exactly zero**, including 120
consecutive seconds across which 25,015 publications carried exactly one change. The
July performance pass measured that title's first level at 12-24 fps, so filing 3.0 would have
manufactured a regression that never happened.

**That is what the active share is for: it is a verdict on your window, not on the title.** Near 100%
means the window was homogeneous and the number is worth filing. Well below it means the window mixed
a menu with gameplay -- narrow the window and measure again rather than filing the mixture. **If a
route never reaches the scene you want to record, file `none` and no number at all.** An explicit
absence is worth more than a figure that describes a title screen.

**Why the `--` matters.** prosper re-publishes the frame it retained whenever a submit produces no
usable present source, so a title whose picture is completely frozen keeps publishing at the
display's rate. A framerate counted from publications reads **full speed for a frozen title** --
instrument trap 90, and the R-Type Delta regression #2783, which hid for nine days behind a
healthy-looking present rate. A title that produced fewer than two distinct frames therefore has no
rate at all, written `--`, and the `0% active` beside it says why. It is never rendered as `0.0`,
which would be a measurement. (`prosper/src/gpu/present/present_frame_rate.hpp` carries the argument
in full, including why the headline is a median over frame intervals and needs no threshold.)

The rest of the cell is not decoration: a framerate means nothing without its conditions. Resolution,
what was on screen, and which frontend measured it all move the number by more than the differences
anybody is trying to see, and a date is what stops a figure from a fixed-since regression being read
as current.

To record one, add exactly one line anywhere in the tracker body:

```
FPS record: 18.5 fps while producing, 62% active; 3840x2160; gameplay; screenshot; 2026-08-21
FPS record: -- fps while producing, 0% active; 1920x1080; title screen; screenshot; 2026-08-21
FPS record: none
```

The line is **optional** -- a tracker without one renders `-` and parses fine -- but it is **strictly
validated when present**, and a malformed one fails the whole run and names the tracker. That is on
purpose: a required field would take the entire projection down the day it landed, while a loosely
parsed one would decay into bare numbers, and a bare framerate is not a measurement. Take both values
straight from `tools/screenshot`'s summary line, or from the `typical_fps` / `active_fraction` fields
of its manifest.

| Title | Title ID | Rung | Ladder | FPS | Guard | Oracle | Open blockers | Tracker | Status doc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Alex Kidd in Miracle World DX | `PPSA02664` | 6 | `123456` | - | `alexkidd-gameplay` | [comment](https://github.com/mattias800/prosper/issues/710#issuecomment-4972898604) | [#710](https://github.com/mattias800/prosper/issues/710) | [#1875](https://github.com/mattias800/prosper/issues/1875) | [`PPSA02664_BLACK_WORLD.md`](prosper/docs/PPSA02664_BLACK_WORLD.md) |
| Asterix & Obelix: Slap Them All! | `PPSA08576` | 6 | `1234-6` | - | `asterix-gameplay` | none | - | [#1877](https://github.com/mattias800/prosper/issues/1877) | - |
| Blasphemous 2 | `PPSA13579` | 6 | `1234-6` | - | `blasphemous2-gameplay` | none | - | [#1867](https://github.com/mattias800/prosper/issues/1867) | - |
| Blue Prince | `PPSA25009` | 6 | `123456` | **4.7** fps · 95% active · 1080p · opening cinematic · screenshot · 2026-08-21 | `blue-prince-hall`, `blue-prince-title` | [`issue-1287-hall-live-vs-oracle.png`](prosper/docs/screenshots/issue-1287-hall-live-vs-oracle.png) | [#1287](https://github.com/mattias800/prosper/issues/1287), [#1284](https://github.com/mattias800/prosper/issues/1284), [#1733](https://github.com/mattias800/prosper/issues/1733), [#1178](https://github.com/mattias800/prosper/issues/1178) | [#1808](https://github.com/mattias800/prosper/issues/1808) | [`BLUE_PRINCE_STATUS.md`](prosper/docs/BLUE_PRINCE_STATUS.md) |
| Dead Cells | `PPSA15552` | 6 | `123456` | - | `dead-cells-gameplay` | [comment](https://github.com/mattias800/prosper/issues/566#issuecomment-4952168468) | - | [#1866](https://github.com/mattias800/prosper/issues/1866) | [`DEAD_CELLS_STATUS.md`](prosper/docs/DEAD_CELLS_STATUS.md) |
| Evergate | `PPSA01885` | 6 | `1234-6` | - | `evergate-gameplay`, `evergate-title` | none | - | [#1868](https://github.com/mattias800/prosper/issues/1868) | - |
| Greak: Memories of Azur | `PPSA02849` | 6 | `1234-6` | - | `greak-gameplay` | none | - | [#1887](https://github.com/mattias800/prosper/issues/1887) | - |
| GRIS | `PPSA09804` | 6 | `1234-6` | - | `gris-gameplay` | none | - | [#1869](https://github.com/mattias800/prosper/issues/1869) | [`GRIS_SONIC_COBRA_BRINGUP.md`](prosper/docs/GRIS_SONIC_COBRA_BRINGUP.md) |
| New Joe & Mac: Caveman Ninja | `PPSA02801` | 6 | `1234-6` | - | `joe-mac-gameplay` | none | - | [#1876](https://github.com/mattias800/prosper/issues/1876) | - |
| Rugrats: Adventure in Gameland | `PPSA23396` | 6 | `123--6` | - | `rugrats-gameplay` | none | - | [#1888](https://github.com/mattias800/prosper/issues/1888) | - |
| Space Adventure Cobra — The Awakening | `PPSA17337` | 6 | `1234-6` | - | `cobra-gameplay` | none | - | [#1870](https://github.com/mattias800/prosper/issues/1870) | [`GRIS_SONIC_COBRA_BRINGUP.md`](prosper/docs/GRIS_SONIC_COBRA_BRINGUP.md) |
| Summer Sports Games | `PPSA03416` | 6 | `1234-6` | - | `summer-sports-gameplay` | none | - | [#1878](https://github.com/mattias800/prosper/issues/1878) | - |
| The Messenger | `PPSA24651` | 6 | `123456` | - | `messenger-scene` | [comment](https://github.com/mattias800/prosper/issues/530#issuecomment-4947961624) | [#298](https://github.com/mattias800/prosper/issues/298) | [#1865](https://github.com/mattias800/prosper/issues/1865) | [`MESSENGER_BLACK_RENDER.md`](prosper/docs/MESSENGER_BLACK_RENDER.md) |
| Worms Armageddon: Anniversary Edition | `PPSA20052` | 6 | `1234-6` | - | `worms-armageddon-gameplay` | none | - | [#1879](https://github.com/mattias800/prosper/issues/1879) | - |
| Tales of Graces f Remastered | `PPSA19991` | 4 | `1234--` | - | - | none | [#1688](https://github.com/mattias800/prosper/issues/1688), [#1673](https://github.com/mattias800/prosper/issues/1673) | [#1889](https://github.com/mattias800/prosper/issues/1889) | - |
| Terminator 2D: NO FATE | `PPSA25872` | 4 | `1234--` | - | `terminator-boot` | none | - | [#1872](https://github.com/mattias800/prosper/issues/1872) | - |
| Asterix & Obelix - Babylon Mission | `PPSA30490` | 3 | `123---` | - | - | none | [#1599](https://github.com/mattias800/prosper/issues/1599), [#2743](https://github.com/mattias800/prosper/issues/2743), [#2738](https://github.com/mattias800/prosper/issues/2738) | [#1884](https://github.com/mattias800/prosper/issues/1884) | [`ASTERIX_BABYLON_STATUS.md`](prosper/docs/ASTERIX_BABYLON_STATUS.md) |
| Bendy and the Dark Revival | `PPSA27624` | 3 | `123---` | - | - | none | [#1979](https://github.com/mattias800/prosper/issues/1979), [#1955](https://github.com/mattias800/prosper/issues/1955) | [#1897](https://github.com/mattias800/prosper/issues/1897) | - |
| Bendy and the Ink Machine | `PPSA27616` | 3 | `123---` | - | - | none | [#1178](https://github.com/mattias800/prosper/issues/1178), [#1177](https://github.com/mattias800/prosper/issues/1177) | [#1881](https://github.com/mattias800/prosper/issues/1881) | - |
| Beneath | `PPSA27640` | 3 | `123---` | - | - | none | [#2813](https://github.com/mattias800/prosper/issues/2813), [#2815](https://github.com/mattias800/prosper/issues/2815) | [#1898](https://github.com/mattias800/prosper/issues/1898) | [`BENEATH_STATUS.md`](prosper/docs/BENEATH_STATUS.md) |
| R-Type Delta: HD Boosted | `PPSA26414` | 3 | `123---` | - | - | none | [#1746](https://github.com/mattias800/prosper/issues/1746), [#1591](https://github.com/mattias800/prosper/issues/1591) | [#1810](https://github.com/mattias800/prosper/issues/1810) | [`R_TYPE_DELTA_STATUS.md`](prosper/docs/R_TYPE_DELTA_STATUS.md) |
| Syberia: Remastered | `PPSA30140` | 3 | `123---` | - | - | none | [#1790](https://github.com/mattias800/prosper/issues/1790), [#1627](https://github.com/mattias800/prosper/issues/1627), [#1737](https://github.com/mattias800/prosper/issues/1737), [#1628](https://github.com/mattias800/prosper/issues/1628) | [#1811](https://github.com/mattias800/prosper/issues/1811) | [`SYBERIA_STATUS.md`](prosper/docs/SYBERIA_STATUS.md) |
| Tactics Ogre: Reborn | `PPSA03839` | 3 | `123---` | - | - | none | [#1913](https://github.com/mattias800/prosper/issues/1913), [#1784](https://github.com/mattias800/prosper/issues/1784) | [#1892](https://github.com/mattias800/prosper/issues/1892) | - |
| The House of the Dead 2: Remake | `PPSA24203` | 3 | `123---` | - | - | none | [#1907](https://github.com/mattias800/prosper/issues/1907) | [#1896](https://github.com/mattias800/prosper/issues/1896) | - |
| Astro Bot | `PPSA21564` | 2 | `12----` | - | - | none | [#1732](https://github.com/mattias800/prosper/issues/1732), [#1459](https://github.com/mattias800/prosper/issues/1459), [#1730](https://github.com/mattias800/prosper/issues/1730), [#1731](https://github.com/mattias800/prosper/issues/1731) | [#1809](https://github.com/mattias800/prosper/issues/1809) | [`GAME_COMPAT_ORCHESTRATION.md`](prosper/docs/GAME_COMPAT_ORCHESTRATION.md) |
| Crisis Core –Final Fantasy VII– Reunion | `PPSA07809` | 2 | `12----` | - | - | none | [#1945](https://github.com/mattias800/prosper/issues/1945), [#2057](https://github.com/mattias800/prosper/issues/2057), [#2058](https://github.com/mattias800/prosper/issues/2058), [#2027](https://github.com/mattias800/prosper/issues/2027) | [#1894](https://github.com/mattias800/prosper/issues/1894) | [`CRISIS_CORE_STATUS.md`](prosper/docs/CRISIS_CORE_STATUS.md) |
| Dragon Quest VII Reimagined | `PPSA17942` | 2 | `12----` | - | - | none | [#1486](https://github.com/mattias800/prosper/issues/1486), [#1588](https://github.com/mattias800/prosper/issues/1588), [#1706](https://github.com/mattias800/prosper/issues/1706), [#2778](https://github.com/mattias800/prosper/issues/2778) | [#1874](https://github.com/mattias800/prosper/issues/1874) | [`DRAGON_QUEST_STATUS.md`](prosper/docs/DRAGON_QUEST_STATUS.md) |
| Earthion | `PPSA28061` | 2 | `12----` | - | - | none | - | [#1880](https://github.com/mattias800/prosper/issues/1880) | - |
| Grand Theft Auto V | `PPSA04263` | 2 | `12----` | - | - | none | [#2429](https://github.com/mattias800/prosper/issues/2429) | [#1873](https://github.com/mattias800/prosper/issues/1873) | [`GAME_COMPAT_ORCHESTRATION.md`](prosper/docs/GAME_COMPAT_ORCHESTRATION.md) |
| Hi-Fi RUSH | `PPSA17168` | 2 | `12----` | **0.5** fps · 9% active · 4K · title screen, default launch, no pad · tools/screenshot · 2026-08-21 | - | none | - | [#2891](https://github.com/mattias800/prosper/issues/2891) | - |
| Little Nightmares III | `PPSA05143` | 2 | `12----` | - | - | none | [#2014](https://github.com/mattias800/prosper/issues/2014), [#2022](https://github.com/mattias800/prosper/issues/2022), [#2028](https://github.com/mattias800/prosper/issues/2028), [#1977](https://github.com/mattias800/prosper/issues/1977) | [#1893](https://github.com/mattias800/prosper/issues/1893) | [`LITTLE_NIGHTMARES_3_STATUS.md`](prosper/docs/LITTLE_NIGHTMARES_3_STATUS.md) |
| Nikoderiko: The Magical World | `PPSA23760` | 2 | `12----` | - | - | none | [#1607](https://github.com/mattias800/prosper/issues/1607) | [#1885](https://github.com/mattias800/prosper/issues/1885) | [`NIKODERIKO_STATUS.md`](prosper/docs/NIKODERIKO_STATUS.md) |
| Sonic Frontiers | `PPSA03831` | 2 | `12----` | - | - | none | [#2206](https://github.com/mattias800/prosper/issues/2206), [#657](https://github.com/mattias800/prosper/issues/657) | [#1891](https://github.com/mattias800/prosper/issues/1891) | [`SONIC_FRONTIERS_STATUS.md`](prosper/docs/SONIC_FRONTIERS_STATUS.md) |
| Sonic Racing: CrossWorlds | `PPSA08804` | 2 | `12----` | - | - | none | [#2361](https://github.com/mattias800/prosper/issues/2361), [#2362](https://github.com/mattias800/prosper/issues/2362), [#2363](https://github.com/mattias800/prosper/issues/2363), [#2309](https://github.com/mattias800/prosper/issues/2309), [#2303](https://github.com/mattias800/prosper/issues/2303) | [#1895](https://github.com/mattias800/prosper/issues/1895) | [`SONIC_CROSSWORLDS_STATUS.md`](prosper/docs/SONIC_CROSSWORLDS_STATUS.md) |
| The Forgotten City | `PPSA03026` | 2 | `12----` | - | - | none | [#1961](https://github.com/mattias800/prosper/issues/1961), [#1945](https://github.com/mattias800/prosper/issues/1945), [#1226](https://github.com/mattias800/prosper/issues/1226) | [#1890](https://github.com/mattias800/prosper/issues/1890) | - |
| The Oregon Trail | `PPSA19244` | 2 | `12----` | - | - | none | [#1945](https://github.com/mattias800/prosper/issues/1945), [#1606](https://github.com/mattias800/prosper/issues/1606), [#1641](https://github.com/mattias800/prosper/issues/1641), [#1634](https://github.com/mattias800/prosper/issues/1634) | [#1886](https://github.com/mattias800/prosper/issues/1886) | [`OREGON_TRAIL_STATUS.md`](prosper/docs/OREGON_TRAIL_STATUS.md) |
| The Pathless | `PPSA01826` | 2 | `12----` | - | - | none | [#1570](https://github.com/mattias800/prosper/issues/1570), [#1213](https://github.com/mattias800/prosper/issues/1213) | [#1883](https://github.com/mattias800/prosper/issues/1883) | [`GAME_COMPAT_ORCHESTRATION.md`](prosper/docs/GAME_COMPAT_ORCHESTRATION.md) |
| The Plucky Squire | `PPSA15319` | 2 | `12----` | - | - | none | [#1390](https://github.com/mattias800/prosper/issues/1390) | [#1882](https://github.com/mattias800/prosper/issues/1882) | [`GAME_COMPAT_ORCHESTRATION.md`](prosper/docs/GAME_COMPAT_ORCHESTRATION.md) |
| Sonic Origins | `PPSA05325` | 1 | `1-----` | - | - | none | [#2267](https://github.com/mattias800/prosper/issues/2267), [#1905](https://github.com/mattias800/prosper/issues/1905), [#1720](https://github.com/mattias800/prosper/issues/1720) | [#1871](https://github.com/mattias800/prosper/issues/1871) | [`GRIS_SONIC_COBRA_BRINGUP.md`](prosper/docs/GRIS_SONIC_COBRA_BRINGUP.md) |
| ArcRunner | `PPSA21406` | 0 | `------` | - | - | none | [#1226](https://github.com/mattias800/prosper/issues/1226), [#2084](https://github.com/mattias800/prosper/issues/2084) | [#1817](https://github.com/mattias800/prosper/issues/1817) | [`ARCRUNNER_STATUS.md`](prosper/docs/ARCRUNNER_STATUS.md) |
| BALAN WONDERWORLD | `PPSA02058` | 0 | `------` | none | - | none | - | [#2882](https://github.com/mattias800/prosper/issues/2882) | - |
| Judgment | `PPSA02739` | 0 | `------` | none | - | none | - | [#2880](https://github.com/mattias800/prosper/issues/2880) | - |
| Little Nightmares II | `PPSA02154` | 0 | `------` | none | - | none | - | [#2884](https://github.com/mattias800/prosper/issues/2884) | - |
| Metaphor: ReFantazio | `PPSA20800` | 0 | `------` | none | - | none | - | [#2876](https://github.com/mattias800/prosper/issues/2876) | - |
| PGA TOUR 2K25 | `PPSA17952` | 0 | `------` | none | - | none | [#2894](https://github.com/mattias800/prosper/issues/2894) | [#2895](https://github.com/mattias800/prosper/issues/2895) | [`PGA_TOUR_2K25_STATUS.md`](prosper/docs/PGA_TOUR_2K25_STATUS.md) |
| Sifu | `PPSA03001` | 0 | `------` | none | - | none | - | [#2885](https://github.com/mattias800/prosper/issues/2885) | - |
| Sniper Ghost Warrior Contracts 2 | `PPSA03130` | 0 | `------` | none | - | none | - | [#2867](https://github.com/mattias800/prosper/issues/2867) | - |
| Spacebase Startopia | `PPSA02846` | 0 | `------` | none | - | none | - | [#2887](https://github.com/mattias800/prosper/issues/2887) | - |
| Stray | `PPSA02101` | 0 | `------` | none | - | none | - | [#2883](https://github.com/mattias800/prosper/issues/2883) | - |
| The Lord of the Rings: Gollum | `PPSA06367` | 0 | `------` | **--** fps · 0% active · 1440p · one flat white composited clear before the boot dies at ~4 s · screenshot · 2026-08-22 | - | none | [#2898](https://github.com/mattias800/prosper/issues/2898) | [#2900](https://github.com/mattias800/prosper/issues/2900) | [`GOLLUM_STATUS.md`](prosper/docs/GOLLUM_STATUS.md) |
| Unbound: Worlds Apart | `PPSA03274` | 0 | `------` | none | - | none | - | [#2886](https://github.com/mattias800/prosper/issues/2886) | - |
| Yakuza Kiwami | `PPSA31334` | 0 | `------` | none | - | none | - | [#2864](https://github.com/mattias800/prosper/issues/2864) | - |

## Counts

| Highest rung reached | Titles |
| --- | --- |
| 6 -- reviewed automatic gameplay snapshot guard | 14 |
| 4 -- manual visual verification | 2 |
| 3 -- gameplay with the scene rendering | 8 |
| 2 -- title screen | 14 |
| 1 -- any real graphics | 1 |
| 0 -- not started | 13 |

**4 of 52** trackers record a PS5 hardware-oracle comparison; the rest carry
`Oracle record: none`. That ratio is the reason this column exists -- before #2730 it took a
scan of 6,224 issue comments to establish, and it was wrong by nine titles.
