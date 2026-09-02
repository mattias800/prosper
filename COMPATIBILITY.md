# Game compatibility

prosper is an experimental compatibility layer, not a general-purpose game runner. The results below
describe specific, user-supplied PS5 dumps tested primarily on Linux. A milestone means that the
documented route is reproducible; it does **not** mean that the entire game is playable or free of
bugs. Different title revisions may behave differently.

Detailed investigation notes, measurements, known defects, and next steps live in the linked
[game-tracker issues](https://github.com/mattias800/prosper/issues?q=is%3Aissue+%22%5BGame+tracker%5D%22).

**Looking for what is new?** [`BLOG.md`](BLOG.md) is the progress blog: every checked-in screenshot
newest-first, with the story around the ones worth a story. Read down from the top and stop at one
you have seen. This file is organised by title instead, so it answers "how far does X get" rather
than "what changed lately".

This page is the **user-facing overview** and is written by hand. For the same titles as a
machine-readable index — rung, ladder, snapshot guard, hardware-oracle record, open blockers —
see [`PROGRESS_TRACKER.md`](PROGRESS_TRACKER.md), which is **generated from the tracker issues**
and kept in step with them by CI. Neither file is authoritative over a tracker; when this page and
a tracker disagree, the tracker wins.

Last updated: 2026-08-28

## Summary

| Title | Title ID | Engine | User-visible milestone | Tracker |
| --- | --- | --- | --- | --- |
| *The Messenger* | `PPSA24651` | Unity / IL2CPP | ✅ First level gameplay | [#1865](https://github.com/mattias800/prosper/issues/1865) |
| *Dead Cells* | `PPSA15552` | Custom | ✅ Prisoners' Quarters gameplay | [#1866](https://github.com/mattias800/prosper/issues/1866) |
| *Blasphemous 2* | `PPSA13579` | Unity | ✅ First playable room | [#1867](https://github.com/mattias800/prosper/issues/1867) |
| *Evergate* | `PPSA01885` | Unity | ✅ First tutorial-room gameplay | [#1868](https://github.com/mattias800/prosper/issues/1868) |
| *GRIS* | `PPSA09804` | Unity / IL2CPP | ✅ Opening gameplay | [#1869](https://github.com/mattias800/prosper/issues/1869) |
| *Space Adventure Cobra — The Awakening* | `PPSA17337` | Unity / IL2CPP | 🚧 Tutorial combat **on the reviewed revision; black on current master** ([#2899](https://github.com/mattias800/prosper/issues/2899)) | [#1870](https://github.com/mattias800/prosper/issues/1870) |
| *Sonic Origins* | `PPSA05325` | Hedgehog Engine | 🚧 4K title screen on a one-button route; a default launch stops at the game's own auto-save notice, which waits for Cross | [#1871](https://github.com/mattias800/prosper/issues/1871) |
| *Sonic Frontiers* | `PPSA03831` | Hedgehog Engine 2 (Needle) | 🚧 Full 4K opening sequence, title screen and main menu; a route reaches Cyber Space gameplay in the guest, but the world does not render behind the HUD | [#1891](https://github.com/mattias800/prosper/issues/1891) |
| *Sonic Racing: CrossWorlds* | `PPSA08804` | Unreal Engine 5 | 🔬 4K title screen and menus with a pad route; needs input to advance past the logos | [#1895](https://github.com/mattias800/prosper/issues/1895) |
| *Terminator 2D: NO FATE* | `PPSA25872` | Unity / IL2CPP | ✅ Main menu and attract-mode gameplay | [#1872](https://github.com/mattias800/prosper/issues/1872) |
| *Blue Prince* | `PPSA25009` | Unity | 🚧 Manor entrance-hall gameplay | [#1808](https://github.com/mattias800/prosper/issues/1808) |
| *Grand Theft Auto V* | `PPSA04263` | RAGE | 🚧 Story Mode reached in the guest; HUD and radar render over an absent 3D world | [#1873](https://github.com/mattias800/prosper/issues/1873) |
| *Dragon Quest VII Reimagined* | `PPSA17942` | Unreal Engine 4 | 🚧 Rung 3 — **the field state in Pilchard Bay**: minimap, party block, area banner and the player character, 144 frames over 588 s. Locomotion is measured, not inferred: the minimap changes in **8 of 8** stick windows against **0 of 8** neutral. The blocker was never a control but the route's patience — the opening chapter needs ~450 confirms, and routes giving it ~40 reached the field **zero** times. **Geometry and the 2D/UI path are correct; the lit-material shading is not** — buildings, cliffs and the boat blow to white while the water crushes dark, and 25% (run 3) to 56% (run 4) of field frames render a recognisable scene (#1486 / #1588). Route: `scripts/dragon-quest-vii/reach-field-control.pad` | [#1874](https://github.com/mattias800/prosper/issues/1874) |
| *Alex Kidd in Miracle World DX* | `PPSA02664` | Unity / IL2CPP | ✅ First-level gameplay | [#1875](https://github.com/mattias800/prosper/issues/1875) |
| *New Joe &amp; Mac: Caveman Ninja* | `PPSA02801` | Unity / IL2CPP | ✅ Level 1 gameplay | [#1876](https://github.com/mattias800/prosper/issues/1876) |
| *Asterix &amp; Obelix: Slap Them All!* | `PPSA08576` | Unity / IL2CPP | ✅ First forest level | [#1877](https://github.com/mattias800/prosper/issues/1877) |
| *Summer Sports Games* | `PPSA03416` | Unity / IL2CPP | ✅ Live 3D athletics | [#1878](https://github.com/mattias800/prosper/issues/1878) |
| *Worms Armageddon: Anniversary Edition* | `PPSA20052` | Custom | ✅ Live match gameplay | [#1879](https://github.com/mattias800/prosper/issues/1879) |
| *Earthion* | `PPSA28061` | Custom (Ancient) | 🚧 Title screen and menus | [#1880](https://github.com/mattias800/prosper/issues/1880) |
| *Bendy and the Ink Machine* | `PPSA27616` | Unity / IL2CPP | 🚧 Chapter 1 gameplay | [#1881](https://github.com/mattias800/prosper/issues/1881) |
| *The Plucky Squire* | `PPSA15319` | Unreal Engine 4 | 🚧 Title and save/play-style menus | [#1882](https://github.com/mattias800/prosper/issues/1882) |
| *The Pathless* | `PPSA01826` | Unreal Engine 4 | 🚧 Title screen | [#1883](https://github.com/mattias800/prosper/issues/1883) |
| *ArcRunner* | `PPSA21406` | Unreal Engine 4 | 🚧 Intro cinematic and title screen; needs one default-off switch, not the throttle | [#1817](https://github.com/mattias800/prosper/issues/1817) |
| *Asterix &amp; Obelix: Babylon Mission* | `PPSA30490` | Unity 6 / IL2CPP | 🚧 `World_3_10` harbour-level gameplay, reached 1 run in 5 | [#1884](https://github.com/mattias800/prosper/issues/1884) |
| *R-Type Delta: HD Boosted* | `PPSA26414` | Custom | 🚧 Stage 1 gameplay on a scripted route | [#1810](https://github.com/mattias800/prosper/issues/1810) |
| *Nikoderiko: The Magical World* | `PPSA23760` | Unreal Engine 4 | 🚧 Title screen and EULA | [#1885](https://github.com/mattias800/prosper/issues/1885) |
| *The Oregon Trail* | `PPSA19244` | Unreal Engine 4 | 🚧 Title screen reached and rendered | [#1886](https://github.com/mattias800/prosper/issues/1886) |
| *Greak: Memories of Azur* | `PPSA02849` | Unity / IL2CPP | ✅ First-level gameplay | [#1887](https://github.com/mattias800/prosper/issues/1887) |
| *Rugrats: Adventure in Gameland* | `PPSA23396` | Unity / IL2CPP | ✅ First nursery level | [#1888](https://github.com/mattias800/prosper/issues/1888) |
| *Syberia: Remastered* | `PPSA30140` | Unity / IL2CPP | 🚧 First playable scene | [#1811](https://github.com/mattias800/prosper/issues/1811) |
| *Tales of Graces f Remastered* | `PPSA19991` | Unity / IL2CPP | 🚧 Lhant Hill prologue gameplay; movie frames render with collapsed chroma ([#2731](https://github.com/mattias800/prosper/issues/2731)) | [#1889](https://github.com/mattias800/prosper/issues/1889) |
| *Astro Bot* | `PPSA21564` | ASOBI (in-house) | 🚧 Opening sequence and title screen | [#1809](https://github.com/mattias800/prosper/issues/1809) |
| *The Forgotten City* | `PPSA03026` | Unreal Engine | 🚧 Title screen | [#1890](https://github.com/mattias800/prosper/issues/1890) |
| *Tactics Ogre: Reborn* | `PPSA03839` | — | 🚧 First tutorial battle | [#1892](https://github.com/mattias800/prosper/issues/1892) |
| *Little Nightmares III* | `PPSA05143` | Unreal Engine 4 | 🚧 Boot splash sequence and title screen; most title frames carry a yellow tint | [#1893](https://github.com/mattias800/prosper/issues/1893) |
| *Crisis Core –Final Fantasy VII– Reunion* | `PPSA07809` | Unreal Engine 4 | 🚧 Title screen, on a throttled route — a default run still dies in the guest allocator within seconds | [#1894](https://github.com/mattias800/prosper/issues/1894) |
| *The House of the Dead 2: Remake* | `PPSA24203` | — | 🚧 Training 1 gameplay | [#1896](https://github.com/mattias800/prosper/issues/1896) |
| *Bendy and the Dark Revival* | `PPSA27624` | Unity / IL2CPP | 🚧 Chapter 1 gameplay; the menu's background video is not composited | [#1897](https://github.com/mattias800/prosper/issues/1897) |
| *Beneath* | `PPSA27640` | Unity / IL2CPP | 🚧 Opening dive gameplay aboard the science ship | [#1898](https://github.com/mattias800/prosper/issues/1898) |
| *Yakuza Kiwami* | `PPSA31334` | Ryu Ga Gotoku (PAR) | 🔬 Rung 0 — boots without faulting and loads every shader archive, scenario file and database table; the frame loop runs but every draw is skipped by the recompiler's NGG vertex prologue ([#2922](https://github.com/mattias800/prosper/issues/2922)), so all frames are black | [#2864](https://github.com/mattias800/prosper/issues/2864) |
| *Sniper Ghost Warrior Contracts 2* | `PPSA03130` | CryEngine | 🔬 Rung 0 — boots in 91 ms and drives a 4K present loop at ~21 flips/s, but every frame is black: no pass produces a present source ([#2871](https://github.com/mattias800/prosper/issues/2871)). The boot deadlock in a preloaded PRX the title never imports is fixed | [#2867](https://github.com/mattias800/prosper/issues/2867) |
| *The Lord of the Rings: Gollum* | `PPSA06367` | Unreal Engine 4 | 🔬 Rung 0 — boots, links every module and composites a 2560x1440 frame, but the only frame is a flat white clear. The startup-movie crash is fixed: `sceVideodec2GetPictureInfo` now fills its struct from the movie's own SPS/VUI ([#2898](https://github.com/mattias800/prosper/issues/2898)); the boot advances through Electra's picture consumer and stops at the record-ownership contract ([#2967](https://github.com/mattias800/prosper/issues/2967)). Its AAC movie audio decodes | [#2900](https://github.com/mattias800/prosper/issues/2900) |
| *The First Berserker: Khazan* | `PPSA20447` | Unreal Engine 4 | 🔬 Rung 0 — boots, mounts and enumerates all thirty save slots, but composites only a flat white 4K clear; the guest then exhausts prosper's direct-memory pool and calls its own OOM handler ([#2908](https://github.com/mattias800/prosper/issues/2908)). The save-data event-drain code that parked its game thread forever is fixed | [#2909](https://github.com/mattias800/prosper/issues/2909) |
| *Metaphor: ReFantazio* | `PPSA20800` | Atlus GFD | 🔬 Loading mascot and the language-selection screen, twelve languages drawn legibly in Latin, Cyrillic, Japanese, Chinese and Korean; the background art behind the menu does not draw ([#2952](https://github.com/mattias800/prosper/issues/2952)). The SIGFPE that killed every boot at five seconds is fixed ([#2951](https://github.com/mattias800/prosper/issues/2951)), as is the CRI Mana crash before it ([#2934](https://github.com/mattias800/prosper/issues/2934)) | [#2876](https://github.com/mattias800/prosper/issues/2876) |
| *Judgment* | `PPSA02739` | Ryu Ga Gotoku (PAR) | 🔬 Rung 0 — boots and runs indefinitely without faulting, executes real GPU draws and presents 4K frames with zero recompiler rejections, but every frame is pure black ([#2923](https://github.com/mattias800/prosper/issues/2923)) | [#2880](https://github.com/mattias800/prosper/issues/2880) |
| *BALAN WONDERWORLD* | `PPSA02058` | Unreal Engine 4 | 🔬 Rung 2 — a routed run answers the language menu's own *"Are you sure you want to change the game language to English?"* modal, which needs **Down** and not Cross, and reaches the title screen at t≈15 s, the main menu, and the opening story cutscene, which is a **decoded 4K H.264 movie** (two VA-API access-unit decoders open at t≈126 s, 3070 pictures) composited by prosper. No stage loads in 717 s, and the wrong composite still takes most frames ([#2932](https://github.com/mattias800/prosper/issues/2932)). Route: `prosper/scripts/balan-PPSA02058/` | [#2882](https://github.com/mattias800/prosper/issues/2882) |
| *Stray* | `PPSA02101` | Unreal Engine 4 | 🔬 Rung 2 — a Cross-only route accepts the brightness-calibration screen its own `✕ Accept` prompt names, and reaches the first map load (`hk_project_mainstart`, t≈37 s, absent from every default run). The world then composites as a flat **letterboxed** clear, so no scene renders ([#2932](https://github.com/mattias800/prosper/issues/2932)). Route: `prosper/scripts/stray-PPSA02101/` | [#2883](https://github.com/mattias800/prosper/issues/2883) |
| *Little Nightmares II* | `PPSA02154` | Unreal Engine 4 | 🔬 Rung 1 — a 4K logo sequence (Bandai Namco → Tarsier → Unreal) renders for ~130 s, then the composite is a flat white 4K clear for the remaining 260 s of a 390 s run and no title screen is reached ([#2932](https://github.com/mattias800/prosper/issues/2932)); it also calls the unregistered `sceAgcDcbDrawIndirect` ([#2929](https://github.com/mattias800/prosper/issues/2929)) | [#2884](https://github.com/mattias800/prosper/issues/2884) |
| *Spacebase Startopia* | `PPSA02846` | Unity 2020.3.1 / IL2CPP | 🔬 Rung 0 — boots in 447 ms, publishes 3 flips and one black 1080p frame, then stops submitting while its own threads and FMOD audio keep running ([#2933](https://github.com/mattias800/prosper/issues/2933)) | [#2887](https://github.com/mattias800/prosper/issues/2887) |
| *Sifu* | `PPSA03001` | Unreal Engine 4 | 🔬 Rung 0 — every frame is a flat 4K clear, white then magenta. Two further defects on the same boot: a GPU hard recovery from a compute submit, in 2 of 3 runs and both naming the same program ([#2935](https://github.com/mattias800/prosper/issues/2935)), and the guest's own out-of-memory assert in the third ([#2908](https://github.com/mattias800/prosper/issues/2908), shared with *Khazan*) | [#2885](https://github.com/mattias800/prosper/issues/2885) |
| *Unbound: Worlds Apart* | `PPSA03274` | Unreal Engine 4 | 🔬 Rung 2 — the title screen renders continuously on a default launch (40/40 samples over 200 s), and a routed run passes it on Cross and plays the **complete intro cinematic** in full colour to t≈65 s. The first level does not render. **The route needs an isolated `PROSPER_SAVE0`/`PROSPER_SAVEDATA_DIR`**: resuming a stale save at the shared default location turns the whole run black and looks exactly like a composite defect ([#2932](https://github.com/mattias800/prosper/issues/2932)). Route: `prosper/scripts/unbound-PPSA03274/` | [#2886](https://github.com/mattias800/prosper/issues/2886) |
| *PGA TOUR 2K25* | `PPSA17952` | Unity 6 / IL2CPP | 🔬 Rung 0 — boots in 437 ms, streams its Unity assets and submits real draws, but every frame is black and a worker thread dies parsing a NULL HTTP response header ([#2894](https://github.com/mattias800/prosper/issues/2894)). The PSN `module_start` handshake that killed it at 1.2 s is fixed | [#2895](https://github.com/mattias800/prosper/issues/2895) |
| *Beast of Reincarnation* | `PPSA29343` | Unreal Engine 5 | 🔬 Rung 1 — the GAME FREAK logo and the game's own Digital Deluxe bonus dialog render at 4K, but only with `PROSPER_CB_EFC_NO_COLOR=1`: on a default launch prosper's unmodelled ELIMINATE_FAST_CLEAR passes paint over the composite and every frame is a flat clear ([#1588](https://github.com/mattias800/prosper/issues/1588)). The pixel shader that writes both scanout buffers now recompiles | [#2916](https://github.com/mattias800/prosper/issues/2916) |
| *Tomb Raider I-III Remastered* | `PPSA16901` | Custom (Saber) | 🚧 Rung 3 — a pad route clears the title's own 40-page EULA gate (Cross is inert until page 40), reaches the rendered title screen, and enters **Croft Manor**, which now renders with correct geometry — steps, walls, hedges, trees, Lara and Winston all correctly shaped and animating. **The world now renders correctly textured** — Croft Manor's assault course draws its brickwork, sandstone, mossy platforms, gravel and foliage, with Lara and Winston (screenshot: `assets/screenshots/tomb-raider-croft-manor-assault-course.webp`, a genuine render confirmed against the dump's own picture assets). The wrong-texture defect is fixed ([#2998](https://github.com/mattias800/prosper/issues/2998)): the decode cache validated one surface of a 256-layer array — 0.29% of the atlas — so a decode taken while it was nearly empty was reused all run; some text draws the wrong glyphs ([#2999](https://github.com/mattias800/prosper/issues/2999)). The shattered world was one defect: the title's 32-bit index buffers are never announced and were read as 16-bit. Route: `prosper/scripts/tomb-raider-PPSA16901/` | [#2990](https://github.com/mattias800/prosper/issues/2990) |

## At a glance

Derived from the table above by reading each row's **milestone text** against the six-rung bring-up
ladder in `CLAUDE.md`. It is *not* derived from the ✅/🚧/🔬 markers, which are not a rung scale:
twelve of the twenty-five titles that reach gameplay are marked 🚧 rather than ✅, and the fifteen 🔬
rows sit at three different rungs — four at rung 2, three at rung 1 and eight at rung 0 — with none
unrun. Counting markers gives a different — and wrong — answer.

**"Not yet booted" is a real category, not a rung.** A title can be tracked and never measured, and
that is different from having been measured and found wanting. It is counted separately so an
unmeasured title is never mistaken for a failing one; newly tracked titles start here.

| Where the title stops | Titles |
| --- | --- |
| **Gameplay reached**, with the scene rendering (rung 3 or better) | 26 |
| **Title screen or menu** reached, or gameplay reached without a rendered world (rung 2) | 17 |
| **Below a title screen** — logo or splash only (rung 1) | 3 |
| **Boots, but no frame with content** (rung 0) | 8 |
| **Not yet booted** — tracked, no run attempted yet | 0 |
| Total tracked | 54 |

Every figure above is re-derived from the rows each time this table is touched, and the buckets now
sum to the total. They did not before: **rung 0 had no row at all**, so the titles that boot and
render nothing were counted under whichever neighbouring bucket a previous editor reached for, and
the sub-counts came to 48 against 49 rows. A missing category does not show up as a wrong number in
one cell — it shows up as an arithmetic error nobody can localise, which is why the fix is a new row
rather than an adjusted one. Rung 0 is now the third-largest bucket, so the gap it was hiding was not
a rounding error.

**The "not yet booted" bucket emptied on 2026-08-22.** Seven of its eight titles were run by the
never-booted survey
([`prosper/docs/NEVER_BOOTED_SURVEY_2026_08.md`](prosper/docs/NEVER_BOOTED_SURVEY_2026_08.md)) —
three landed at rung 2, one at rung 1 and three at rung 0 — and *Judgment* (`PPSA02739`) was booted
the same day by a separate lane ([#2923](https://github.com/mattias800/prosper/issues/2923)).

The row stays at 0 rather than being deleted, for the reason above: it is a **category, not a rung**,
and the next title onboarded starts in it. The distribution is also the argument for having kept it
separate — the unmeasured group turned out neither uniformly promising nor uniformly stuck, so
folding it into rung 0 would have been wrong in both directions.

"Gameplay reached" is the ladder's rung 3 and says nothing about how complete the rendered scene is.
**Rung 3 requires the gameplay scene to actually render, not merely to be reached.** The bar is
deliberately loose — a scene a person would recognise as the game — so *degraded* still counts:
*Syberia: Remastered*'s composite is degraded and *The House of the Dead 2*'s world carries severe
defects, and both are rung 3. **Absent does not count.** *Grand Theft Auto V* and *Sonic Frontiers*
each reach the game loop in the guest, with a correct HUD over a world that never draws; both are
**rung 2**, and their milestone text says what they reached. They have genuinely got further than a
title stuck at a menu — the text carries that, not the number.

### Where the titles accumulate

The 17 titles at rung 2 — a title screen or menu, or gameplay reached without a rendered world — by
the engine recorded in the table:

| Engine | Titles |
| --- | --- |
| Unreal Engine — 10 × UE4, 1 × UE5, 1 unversioned | 12 |
| Hedgehog Engine, Hedgehog Engine 2, RAGE, Custom (Ancient), ASOBI — one each | 5 |

**Unreal dominates this group, and it no longer accounts for all of it.** Twelve of the 17 rung-2
rows are Unreal, against 18 Unreal rows in the table overall — the other six are one at gameplay
(*Dragon Quest VII Reimagined*, whose world renders), two at rung 1 (*Little Nightmares II* and
*Beast of Reincarnation*) and three at rung 0 (*The Lord of the Rings: Gollum*, *The First Berserker:
Khazan* and *Sifu*). So "every Unreal title stops at a title screen", which this section used to say,
is not true in either direction: one has passed it and five have not reached it.

The distribution on the other side is the mirror image: the titles at gameplay are overwhelmingly
Unity-family, and **no Unity title remains at rung 2 for want of a rendered world.** The rung-2 group
is twelve Unreal titles plus *Sonic Origins*, *Earthion*, *Astro Bot*, and the two that reach the
game loop without a world, *Grand Theft Auto V* and *Sonic Frontiers*.

**This is an observation about where titles accumulate, not a claim that the twelve Unreal titles
share one root cause** — and this is no longer merely an untested hypothesis in either direction.
[#2747](https://github.com/mattias800/prosper/issues/2747) is a census over eight of them showing
that four lose the *same* UE volumetric-fog compute pass to one recompiler gap and **none of them is
stuck because of it**, and the never-booted survey found the same shape again: its eight titles share
symptoms and share latent defects, and every rung-0 wall in it is title-specific. What bounds
*Nikoderiko*, *ArcRunner*, *Crisis Core*, *Little Nightmares III* and *The Oregon Trail* is still
recorded separately for each, in the sections below and in their tracker issues. What the grouping does say
is that the shared UE bring-up surface —
[`prosper/docs/UE4_APR_IOSTORE_BRINGUP.md`](prosper/docs/UE4_APR_IOSTORE_BRINGUP.md) and
[`prosper/docs/CROSS_ENGINE_UE4.md`](prosper/docs/CROSS_ENGINE_UE4.md) — carries the largest single
block of titles waiting to reach gameplay.

## Screenshots and short descriptions

The images below are representative user-visible checkpoints. The linked tracker issue for each game
is the authoritative place for current compatibility status and development history.

## The Messenger — `PPSA24651`

<p align="center"><img src="assets/screenshots/messenger-title.webp" alt="The Messenger — title screen"></p>
<p align="center"><img src="assets/screenshots/messenger.webp" alt="The Messenger — first level gameplay"></p>

The opening route reaches the first level with player, terrain, effects, dialogue, and foreground
composition at native 1920×1080. See the [tracker](https://github.com/mattias800/prosper/issues/1865).

## Dead Cells — `PPSA15552`

<p align="center"><img src="assets/screenshots/dead-cells-title.webp" alt="Dead Cells — main menu"></p>
<p align="center"><img src="assets/screenshots/dead-cells.webp" alt="Dead Cells — Prisoners' Quarters gameplay"></p>

The route reaches a controllable, full-colour Prisoners' Quarters scene. See the [tracker](https://github.com/mattias800/prosper/issues/1866).

## Blasphemous 2 — `PPSA13579`

<p align="center"><img src="assets/screenshots/blasphemous2-title.webp" alt="Blasphemous 2 — title screen"></p>
<p align="center"><img src="assets/screenshots/blasphemous2.webp" alt="Blasphemous 2 — first playable room"></p>

The opening route passes the cinematic and reaches a complete first room at native 1920×1080. See the [tracker](https://github.com/mattias800/prosper/issues/1867).

## Evergate — `PPSA01885`

<p align="center"><img src="assets/screenshots/evergate-title.webp" alt="Evergate — title screen"></p>
<p align="center"><img src="assets/screenshots/evergate.webp" alt="Evergate — first tutorial room gameplay"></p>

The scripted route reaches the first tutorial room. See the [tracker](https://github.com/mattias800/prosper/issues/1868).

## GRIS — `PPSA09804`

<p align="center"><img src="prosper/docs/screenshots/issue-1356-gris-title.png" alt="GRIS — title screen"></p>
<p align="center"><img src="assets/screenshots/gris.webp" alt="GRIS — opening gameplay"></p>

The opening fall leads into native 1920×1080 gameplay with scripted input and audio. See the [tracker](https://github.com/mattias800/prosper/issues/1869).

## Space Adventure Cobra — The Awakening — `PPSA17337`

<p align="center"><img src="prosper/docs/screenshots/issue-1356-space-adventure-cobra-title.png" alt="Space Adventure Cobra — title screen"></p>
<p align="center"><img src="assets/screenshots/space-adventure-cobra.webp" alt="Space Adventure Cobra — tutorial combat"></p>

The route reaches the native 1920×1080 desert tutorial combat scene with audio. See the [tracker](https://github.com/mattias800/prosper/issues/1870).

**Regressed on current master ([#2899](https://github.com/mattias800/prosper/issues/2899)).** The
`cobra-gameplay` guard now renders one uniformly black frame for the whole 199.6 s route. Bisected to
`ff72e77c` (#1974, `sceAvPlayerJumpToTime`): with the seek implemented, Unity's `PS5VideoMedia`
prepare *succeeds* instead of timing out, and the title then stops driving the player. The two
screenshots above are the reviewed pre-regression state, not a capture of master.

## Sonic Origins — `PPSA05325`

<p align="center"><img src="assets/screenshots/sonic-origins-title-screen.webp" alt="Sonic Origins — title screen"></p>
<p align="center"><img src="assets/screenshots/sonic-origins-autosave-notice.webp" alt="Sonic Origins — the boot auto-save notice, which waits for Cross"></p>
<p align="center"><img src="assets/screenshots/sonic-origins-sonic-team-logo-blue.webp" alt="Sonic Origins — the SONIC TEAM logo from the decoded intro"></p>
<p align="center"><img src="assets/screenshots/sonic-origins-sega-logo.webp" alt="Sonic Origins — SEGA logo"></p>

*All four are direct, unmodified `tools/screenshot` captures — headless Linux/RADV, native 3840×2160,
no render-scale or frame-skip acceleration, master `f856e7a8`. The SEGA logo, SONIC TEAM logo and
auto-save notice are from a **default launch with no input** (42 samples at 10 s); the title screen is
from the same launch driven by `prosper/scripts/sonic/dismiss-boot-notice.pad`, sample 77 of 84 at
t=385 s.*

**The title screen is reached, and the wall in front of it was a dialog box.** A default launch runs
black → a cyan loading element → the 4K SEGA logo → the decoded 4K intro → the SONIC TEAM logo → the
third-party legal plate (Retro Engine / Headcannon / CRIWARE) → and then, at t≈200 s, the game's own
**"This title supports auto save."** notice, whose only control is a `[X] Close` button. It holds
there for the rest of a 420 s run: ten consecutive 10 s samples with a per-pixel mean delta of
0.03–0.12 out of 255, the guest running and frames still being produced at 4.7/s. One `cross` closes
it, and the title screen above comes up and stays up for the remaining 140 s of the run, cycling
Sonic, Tails and Knuckles through the emblem. This is the fourth title here whose progression wall
turned out to be input mapping rather than rendering.

Two open defects on that title screen, neither of which stops it being reached:

- Its **"SONIC ORIGINS" wordmark never draws** on the banner, and no *press any button* prompt
  appears. The banner, emblem, characters and background are all correct.
- About **a third of its frames composite with their two chroma components equal**, so they render
  magenta and green — 10 of 28 samples in the title-screen window, `corr(Cb, Cr) = +0.997` against
  `−0.78` on the clean frames in the same run. This is the [#2731](https://github.com/mattias800/prosper/issues/2731)
  signature surviving that issue's fix on one path: the same run's 42 pre-title frames are clean,
  0 of 42.

The title had previously produced nothing but black. Its boot state machine waits for a save-data job
that could never finish, because `sceSaveDataCreateTransactionResource` returned 0 when it must return
the id of the transaction resource it creates
([#1905](https://github.com/mattias800/prosper/issues/1905)); with a real id the boot advances, the
frontend loads its menu resource set and opens its logo movie, and the SEGA logo renders. What moved
it from there to the title screen was two further fixes:
[#2901](https://github.com/mattias800/prosper/pull/2901) (a decoded video's chroma plane is recognised
by its geometry) and [#2910](https://github.com/mattias800/prosper/pull/2910)
(`sceSaveDataGetEventResult` no longer answers a drained queue with the "still in flight" code — this
title has a `sceKernelSleep(1)` re-poll loop at `eboot+0x940380` that nothing but a different return
value can leave).

See [`prosper/docs/GRIS_SONIC_COBRA_BRINGUP.md`](prosper/docs/GRIS_SONIC_COBRA_BRINGUP.md) and the
[tracker](https://github.com/mattias800/prosper/issues/1871).

## Sonic Frontiers — `PPSA03831`

<p align="center"><img src="assets/screenshots/sonic-frontiers-title-screen.webp" alt="Sonic Frontiers — title screen"></p>
<p align="center"><img src="assets/screenshots/sonic-frontiers-main-menu.webp" alt="Sonic Frontiers — main menu"></p>
<p align="center"><img src="assets/screenshots/sonic-frontiers-sega-logo.webp" alt="Sonic Frontiers — SEGA logo"></p>
<p align="center"><img src="assets/screenshots/sonic-frontiers-middleware-credits.webp" alt="Sonic Frontiers — middleware credits"></p>
<p align="center"><img src="assets/screenshots/sonic-frontiers-opening-sequence.webp" alt="Sonic Frontiers — opening logo sequence"></p>
<p align="center"><img src="assets/screenshots/sonic-frontiers-sonic-team-logo.webp" alt="Sonic Frontiers — Sonic Team logo"></p>
<p align="center"><img src="assets/screenshots/sonic-frontiers-autosave-notice.webp" alt="Sonic Frontiers — auto-save notice"></p>

A default launch with no input reaches the game's whole opening sequence at 3840×2160 — the SEGA
logo, the Cyber Space intro, the Sonic Team logo and the middleware credits — then the auto-save
notice, the title screen and the main menu. The title composites into the display buffer it flips
rather than through render passes, so publishing it needs the flipped buffer itself to be a present
source.

The boot previously stopped dead after the opening sequence, at a black screen. It was waiting on
`sceSaveDataTransferringMountPs4` — the call behind the menu's own "Carry over from PlayStation®4"
entry. prosper did not implement it, and the unimplemented default answers `SCE_OK`, so the title
believed a PS4 save had been mounted and spent every frame trying to open a mount point that was
never created. **Known defects:** the menu heading draws the string "Try Again" where the game's
logo belongs, and the six main-menu entries do not always render their text
([#2206](https://github.com/mattias800/prosper/issues/2206)).

An input route now takes the title past the menu into gameplay:
[`scripts/sonic-frontiers-PPSA03831/reach-gameplay.pad`](prosper/scripts/sonic-frontiers-PPSA03831/reach-gameplay.pad)
clears the twelve-page boot notice queue that a no-input arm sits behind forever, moves the
main-menu cursor from "Extras" to "New Game", and reaches `GameModeStage` on the Cyber Space stage
`w6d01` — one hundred streamed terrain sectors, the stage HUD, Cyber Space BGM, and a stage clock
that runs. **The world behind that HUD is black**: sixteen of the stage's thirty-two compute
programs never execute, three of them full-screen passes over the scene target
([#2790](https://github.com/mattias800/prosper/issues/2790)). That is the frontier for this title,
and the Needle stack is shared with *Sonic Origins* and *Sonic Racing: CrossWorlds*. See
[`docs/SONIC_FRONTIERS_STATUS.md`](prosper/docs/SONIC_FRONTIERS_STATUS.md) and the
[tracker](https://github.com/mattias800/prosper/issues/1891).

## Sonic Racing: CrossWorlds — `PPSA08804`

<p align="center"><img src="assets/screenshots/sonic-crossworlds-title.webp" alt="Sonic Racing: CrossWorlds — title screen"></p>

The title screen, composited by the live renderer at 3840×2160 — the full 3D scene, every character,
the track and the UI.

**It needs controller input to get there.** A default launch with no pad stops at the SEGA logo, which
is what this title was recorded as doing for months: the engine keeps producing frames and the composite
stays black, so every renderer diagnostic reads healthy while nothing advances. The title is simply
waiting for a button. With `prosper/scripts/sonic-crossworlds/advance-boot-logos.pad` it walks the whole
sequence — SEGA logo, Unreal Engine, CRIWARE, the licensor screen, the auto-save notice, the title
screen, and into the player-profile menu.

One detail that matters if you write your own route: **a held button is not a press.** Holding Cross
advances nothing; the guest needs a neutral→pressed transition, so the route uses short pulses.

Beyond the title screen the profile menu renders its UI correctly but leaves the central content panel
black, and the sequence eventually holds on white. See
[`docs/SONIC_CROSSWORLDS_STATUS.md`](prosper/docs/SONIC_CROSSWORLDS_STATUS.md) and the
[tracker](https://github.com/mattias800/prosper/issues/1895).

## Terminator 2D: NO FATE — `PPSA25872`

<p align="center"><img src="assets/screenshots/terminator-title.webp" alt="Terminator 2D: NO FATE — main menu"></p>
<p align="center"><img src="assets/screenshots/terminator.webp" alt="Terminator 2D: NO FATE — attract-mode gameplay"></p>

The main menu and attract-mode gameplay are verified. See the [tracker](https://github.com/mattias800/prosper/issues/1872).

## Blue Prince — `PPSA25009`

<p align="center"><img src="assets/screenshots/blue-prince-title.webp" alt="Blue Prince — title screen"></p>
<p align="center"><img src="assets/screenshots/blue-prince-hall.webp" alt="Blue Prince — Mount Holly entrance-hall gameplay"></p>

The manor entrance hall renders with real 3D gameplay content. See the [tracker](https://github.com/mattias800/prosper/issues/1808).

## Grand Theft Auto V — `PPSA04263`

<p align="center"><img src="assets/screenshots/gta5-title.webp" alt="Grand Theft Auto V — title screen"></p>
<p align="center"><img src="assets/screenshots/gta5-main-menu.webp" alt="Grand Theft Auto V — main menu"></p>

The title and STORY/ONLINE main menu render. A checked-in pad route also reaches Story Mode gameplay:
the HUD, radar and tutorial text are visible, but the 3D world is still black. See the
[tracker](https://github.com/mattias800/prosper/issues/1873) and the exact compute-failure census in
[#2481](https://github.com/mattias800/prosper/issues/2481).

## Dragon Quest VII Reimagined — `PPSA17942`

<p align="center"><img src="assets/screenshots/dragon-quest-vii-title.webp" alt="Dragon Quest VII Reimagined — title screen"></p>
<p align="center"><img src="assets/screenshots/dragon-quest-vii-name-entry.webp" alt="Dragon Quest VII Reimagined — name entry"></p>
<p align="center"><img src="assets/screenshots/dragon-quest-vii-name-confirmation.webp" alt="Dragon Quest VII Reimagined — name confirmation"></p>
<p align="center"><img src="assets/screenshots/dragon-quest-vii-onboarding.webp" alt="Dragon Quest VII Reimagined — first-run setup"></p>

<p align="center"><img src="assets/screenshots/dragon-quest-vii-opening-chapter.webp" alt="Dragon Quest VII Reimagined — the Estard coast in the opening chapter"></p>
<p align="center"><img src="assets/screenshots/dragon-quest-vii-pilchard-bay-gameplay.webp" alt="Dragon Quest VII Reimagined — free field control in Pilchard Bay: the player character stands outside a harbour house with a quest marker over its door and a rowing boat beached to the right, the circular minimap at bottom-left and the party status block (Lv.1, HP 22, MP 7) at bottom-right. Colour is severely degraded — the composite blows the buildings to white and crushes the ground to navy — but the scene is structurally complete."></p>
<p align="center"><img src="assets/screenshots/dragon-quest-vii-walked-to-cliff.webp" alt="Dragon Quest VII Reimagined — the same Pilchard Bay harbour after a left-stick window: the quest-marker house that stood centre-left is now upper-right and a cliff face has entered from the left, with the minimap scrolled to match. The player has walked."></p>
<p align="center"><img src="assets/screenshots/dragon-quest-vii-field-hud.webp" alt="Dragon Quest VII Reimagined — the Pilchard Bay location banner on entering the area, with the field HUD live: minimap, quest marker and party status. The world behind it is largely lost to the composite collapse."></p>

The title, new-save flow, and first-run setup render at native 3840×2160 with audio, and a scripted route continues past them: the game creates a real adventure log (`GameSaveData000.dat`), loads Estard’s persistent level and the chapter-1 sequence, and runs the opening story over a rendered 3D world. **`reach-field-control.pad` now reaches the field state in Pilchard Bay** — minimap, party block, area-entry banner and the player character — across 144 frames spanning 588 s, and `probe-locomotion.pad` measures free movement: the minimap changes in 8 of 8 stick windows against 0 of 8 neutral. What was in the way was the route’s patience, not a control: the opening chapter needs ~450 confirms and every earlier route gave it ~40, which yielded **zero** field frames. **The geometry and the 2D/UI path are correct; the lit-material shading is not** — buildings, cliffs and the boat blow to white while the water crushes far too dark. 25% of field frames render a recognisable scene in run 3 and 56% in run 4; the rest collapse to uniform white, black, or a flat blue speckle. See the [tracker](https://github.com/mattias800/prosper/issues/1874).

## Alex Kidd in Miracle World DX — `PPSA02664`

<p align="center"><img src="assets/screenshots/alex-kidd.webp" alt="Alex Kidd in Miracle World DX — first level"></p>

The first level renders at native 1920×1080. See the [tracker](https://github.com/mattias800/prosper/issues/1875).

## New Joe &amp; Mac: Caveman Ninja — `PPSA02801`

<p align="center"><img src="assets/screenshots/joe-mac-menu.webp" alt="New Joe &amp; Mac — arcade menu"></p>
<p align="center"><img src="assets/screenshots/joe-mac.webp" alt="New Joe &amp; Mac — level 1 gameplay"></p>

The title, menus, and level 1 render at native 1920×1080. See the [tracker](https://github.com/mattias800/prosper/issues/1876).

## Asterix &amp; Obelix: Slap Them All! — `PPSA08576`

<p align="center"><img src="assets/screenshots/asterix-slap-them-all.webp" alt="Asterix &amp; Obelix: Slap Them All! — first forest level"></p>

The first forest level renders on the initial route with no title-specific changes. See the [tracker](https://github.com/mattias800/prosper/issues/1877).

## Summer Sports Games — `PPSA03416`

<p align="center"><img src="assets/screenshots/summer-sports-games.webp" alt="Summer Sports Games — athletics event"></p>

The mode-select flow reaches live 3D athletics at native 3840×2160. See the [tracker](https://github.com/mattias800/prosper/issues/1878).

## Worms Armageddon: Anniversary Edition — `PPSA20052`

<p align="center"><img src="assets/screenshots/worms-armageddon-title.webp" alt="Worms Armageddon — title screen"></p>
<p align="center"><img src="assets/screenshots/worms-armageddon-gameplay.webp" alt="Worms Armageddon — match gameplay"></p>

The scripted route reaches a live match at native 1920×1080. See the [tracker](https://github.com/mattias800/prosper/issues/1879).

## Earthion — `PPSA28061`

<p align="center"><img src="assets/screenshots/earthion-title-menu.webp" alt="Earthion — title screen and main menu"></p>

The route reaches the title screen, main menu, and HOW TO PLAY page in full colour at native 3840×2160. See the [tracker](https://github.com/mattias800/prosper/issues/1880).

## Bendy and the Ink Machine — `PPSA27616`

<p align="center"><img src="assets/screenshots/bendy-title.webp" alt="Bendy and the Ink Machine — main menu"></p>
<p align="center"><img src="assets/screenshots/bendy-gameplay.webp" alt="Bendy and the Ink Machine — Chapter 1 gameplay"></p>

The route reaches first-person Chapter 1 gameplay at native 3840×2160. See the [tracker](https://github.com/mattias800/prosper/issues/1881).

## The Plucky Squire — `PPSA15319`

<p align="center"><img src="assets/screenshots/plucky-squire-title.webp" alt="The Plucky Squire — title screen"></p>

The title and save-file/play-style menus render at native 3840×2160; the current route does not reach chapter one. See the [tracker](https://github.com/mattias800/prosper/issues/1882).

## The Pathless — `PPSA01826`

<p align="center"><img src="assets/screenshots/pathless-title.webp" alt="The Pathless — title screen"></p>

The title screen renders at native 2560×1440. Gameplay has not yet been reached. See the [tracker](https://github.com/mattias800/prosper/issues/1883).

## ArcRunner — `PPSA21406`

<p align="center"><img src="assets/screenshots/arcrunner-title-screen.webp" alt="ArcRunner — the title screen at 3840×2160, on the throttled route"></p>
<p align="center"><img src="assets/screenshots/arcrunner-intro-space-station.webp" alt="ArcRunner — the intro cinematic's Titan-class space station (colours are wrong: see #2094)"></p>
<p align="center"><img src="assets/screenshots/arcrunner-intro-city.webp" alt="ArcRunner — the intro cinematic's neon street (colours are wrong: see #2094)"></p>

The whole intro cinematic renders at 4K — the Titan-class station against its nebula, the rainy neon street, and the population card — with 1,901 of 1,908 video frames succeeding. **Past the cinematic the title reaches its title screen and holds it**, unchanged for 176 s of a 288 s run.

**The colours in these frames are wrong.** Decoding the same movie outside prosper gives an *orange* nebula with *cyan* thrusters; prosper produces green and magenta. Geometry, composition, timing and frame pacing are correct, so the images are valid evidence for those — but not for colour. Tracked in [#2094](https://github.com/mattias800/prosper/issues/2094).

**This needs `PROSPER_SUBMIT_STALL_US=1500`, and that is a diagnostic lever rather than a fix.** On the default route the title faults before the cinematic: 17 runs out of 17, against 0 out of 4 with the stall applied. Every graphics subsystem the cinematic exercises works; what remains is a submit-timing race, tracked as [#1226](https://github.com/mattias800/prosper/issues/1226) with a dose-response experiment in [#2084](https://github.com/mattias800/prosper/issues/2084). See the [tracker](https://github.com/mattias800/prosper/issues/1817).

<p align="center"><img src="assets/screenshots/arcrunner-title-screen-default-route.webp" alt="ArcRunner — the title screen at 3840×2160 on the default route, with the post-submit visibility contract armed"></p>

**The race is now named, and the throttle is no longer needed to get past it.** prosper's post-submit completion-visibility contract — which holds a submit's completion writes private until the submit call returns, so a guest can never observe a half-retired frame — is armed only for titles that request SDK version 13 or later. ArcRunner requests version 10, so on this title prosper's own label writes become visible *in the middle of the fold that produced them*, and the guest's command-chunk recycler is released early: it rebuilds labels whose initialisation packets prosper has not executed yet. With the contract forced on and **no** throttle, the default route runs 260 s, delivers 1,977 of the movie's 1,908 video frames, faults zero times, and renders the title screen above. The same change rescues *Crisis Core*, the other title with this failure, which is also SDK 10.

This is not yet counted as a title screen on a default launch, because it still needs `PROSPER_POST_SUBMIT_VISIBILITY=1`. Removing the version gate is a one-line change, and three titles already at full snapshot coverage are also pre-13 — so it is scheduled behind a cross-title regression pass rather than taken on this title's evidence alone.

## Asterix &amp; Obelix: Babylon Mission — `PPSA30490`

<p align="center"><img src="assets/screenshots/asterix-babylon-intro-cutscene.webp" alt="Asterix &amp; Obelix: Babylon Mission — narrated intro cutscene"></p>
<p align="center"><img src="assets/screenshots/asterix-babylon-title.webp" alt="Asterix &amp; Obelix: Babylon Mission — title screen and main menu"></p>
<p align="center"><img src="assets/screenshots/asterix-babylon-gameplay.webp" alt="Asterix &amp; Obelix: Babylon Mission — the World_3_10 harbour level, both character portraits with heart meters, Asterix and Obelix among NPCs with parallax and water"></p>

Both publisher logo movies, the narrated intro cutscene and the title screen with its `ADVENTURE` / `OPTIONS` menu render at native 1920×1080, and the route crosses them into the `World_3_10` harbour level — both character portraits with heart meters, Asterix and Obelix in a composed scene with NPCs, parallax, water and rotating Roman-helmet collectibles, inspected over 43 samples in 430 s with no black or flat frame.

**Gameplay here is demonstrated rather than reproducible**, which is why the row is 🚧: the committed route reaches it **1 run in 5**, and the other four stop on the `SoloCoopMenu` character-select screen while the guest is *observed* receiving 66 Cross presses ([#2743](https://github.com/mattias800/prosper/issues/2743)). The nondeterminism sits below the route, so a rerun is not a fix. See the [tracker](https://github.com/mattias800/prosper/issues/1884).

## R-Type Delta: HD Boosted — `PPSA26414`

<p align="center"><img src="assets/screenshots/rtype-delta-opening-movie-colour.webp" alt="R-Type Delta — the opening movie's R-9 hangar shot in full colour"></p>
<p align="center"><img src="assets/screenshots/rtype-delta-title.webp" alt="R-Type Delta: HD Boosted — title screen"></p>
<p align="center"><img src="assets/screenshots/rtype-delta-force-select.webp" alt="R-Type Delta — attract-mode demonstration, the R-9 and its Force device"></p>
<p align="center"><img src="assets/screenshots/rtype-delta-stage1-restored.webp" alt="R-Type Delta — stage 1 gameplay: the R-9 and its Force device over the sunset cityscape, enemy formations, and the BEAM and score HUD"></p>

The Clear River Games publisher logo and the full opening movie — the R-9 fighter in its hangar — render live at 1920×1080 from the real GPU command stream, in full colour, and the run continues into the **title screen**, the attract-mode demonstration and **stage 1 gameplay**.

Gameplay needs the scripted route [`prosper/scripts/rtype-delta-PPSA26414/reach-gameplay.pad`](prosper/scripts/rtype-delta-PPSA26414/reach-gameplay.pad): the title screen's prompt is the PS5 **OPTIONS** glyph rather than Cross, so a Cross-only arm never clears it. Player control is distinguished from the attract loop — which plays stage 1 by itself with the pad untouched — by three things a neutral-input control arm never produces: `loadsel_Release.prx`, `loads1_Release.prx` and a `SaveData.dat` write.

Reaching this needs the game's files **evicted from the host page cache** first, which takes one command and no change to how the title is launched:

```bash
python3 prosper/tools/dropcache.py <DUMP_ROOT>/PPSA26414-app0    # then launch normally
```

The title's input worker sleeps 400 ms before its first `sceUserServiceGetEvent` drain, and on a fast host prosper finishes the asset load well inside that window — so the shell runs first, dereferences an empty user vector and dies. Reading the assets from storage rather than from cache stretches the load past the 400 ms mark and the title survives its own race. (Repeat the eviction before each launch: the run itself re-warms the cache.) Every guest sleep is honoured exactly and every service answer is faithful — a PS5's slower core is what makes this title ship working, and prosper simply executes the game's own loader faster than the console it is emulating. See the [tracker](https://github.com/mattias800/prosper/issues/1810) and [#1746](https://github.com/mattias800/prosper/issues/1746).

## Nikoderiko: The Magical World — `PPSA23760`

<p align="center"><img src="assets/screenshots/nikoderiko-title.webp" alt="Nikoderiko — title screen"></p>

The warning screen, publisher logo, title screen, and EULA render at native 3840×2160. See the [tracker](https://github.com/mattias800/prosper/issues/1885).

## The Oregon Trail — `PPSA19244`

<p align="center"><img src="assets/screenshots/oregon-trail-gameloft-splash.webp" alt="The Oregon Trail — Gameloft splash"></p>
<p align="center"><img src="assets/screenshots/oregon-trail-health-warning.webp" alt="The Oregon Trail — health/epilepsy warning screen"></p>
<p align="center"><img src="assets/screenshots/oregon-trail-title-screen.webp" alt="The Oregon Trail — title screen"></p>

A default launch renders the legal popup, the Gameloft splash and the health warning at native 3840×2160, and reaches the **title screen** with its "Press any button" prompt, correctly composited. See the [tracker](https://github.com/mattias800/prosper/issues/1886).

## Greak: Memories of Azur — `PPSA02849`

<p align="center"><img src="assets/screenshots/greak-title.webp" alt="Greak: Memories of Azur — title screen"></p>
<p align="center"><img src="assets/screenshots/greak.webp" alt="Greak: Memories of Azur — first-level gameplay"></p>

The scripted route reaches sustained first-level gameplay at native 1920×1080. See the [tracker](https://github.com/mattias800/prosper/issues/1887).

## Rugrats: Adventure in Gameland — `PPSA23396`

<p align="center"><img src="assets/screenshots/rugrats-title.webp" alt="Rugrats — title and game-mode selector"></p>
<p align="center"><img src="assets/screenshots/rugrats.webp" alt="Rugrats — first nursery level"></p>

The route reaches the first nursery level at native 1920×1080. See the [tracker](https://github.com/mattias800/prosper/issues/1888).

## Syberia: Remastered — `PPSA30140`

<p align="center"><img src="assets/screenshots/syberia-profile.webp" alt="Syberia: Remastered — profile-select menu"></p>
<p align="center"><img src="assets/screenshots/syberia-title.webp" alt="Syberia: Remastered — title screen"></p>
<p align="center"><img src="assets/screenshots/syberia-gameplay.webp" alt="Syberia: Remastered — first playable scene"></p>

The validated route reaches the title screen and first playable scene with real GPU draws. See the [tracker](https://github.com/mattias800/prosper/issues/1811).

## Tales of Graces f Remastered — `PPSA19991`

<p align="center"><img src="assets/screenshots/tales-graces-f-gameplay.webp" alt="Tales of Graces f Remastered — Lhant Hill prologue gameplay"></p>
<p align="center"><img src="assets/screenshots/tales-graces-f-publisher.webp" alt="Tales of Graces f Remastered — publisher logo"></p>
<p align="center"><img src="assets/screenshots/tales-graces-f-criware.webp" alt="Tales of Graces f Remastered — CRIWARE logo"></p>
<p align="center"><img src="assets/screenshots/tales-graces-f-title-no-input.webp" alt="Tales of Graces f Remastered — title screen reached with no input"></p>
<p align="center"><img src="assets/screenshots/tales-graces-f-options.webp" alt="Tales of Graces f Remastered — options screen"></p>

*The title-screen frame is a direct, unmodified `tools/screenshot` capture — headless Linux/RADV,
**default launch with no input route**, no render-scale or frame-skip acceleration, master
`4c8b77c8`, native 1920×1080, t=260 s of a 300 s run. It supersedes the earlier
`tales-graces-f-title.png`, which is the same screen captured on the pad route and is still on
master; this one carries the `Press ✕` prompt and, more importantly, was reached with the pad
unplugged.*

The route [`prosper/scripts/talesgraces/reach-gameplay.pad`](prosper/scripts/talesgraces/reach-gameplay.pad) reaches the **Lhant Hill prologue** — Asbel in a live 3D field at native 1920×1080 — reproduced 2 of 2, with no rejected shader or skipped dispatch in any run. A default launch with no input route still reaches the **title screen** on its own at roughly 220 s, then the EULA, main menu and new-game Options screen.

**What held this title at a menu was input, not the renderer.** Two new-game screens bind their confirm action to the **OPTIONS** button and each raises a Yes/No dialog defaulting to **No**, so a Cross-only route loops forever and an OPTIONS+Cross route answers *No*. Movie frames composite with their chroma collapsed ([#2731](https://github.com/mattias800/prosper/issues/2731)). See [`prosper/docs/TALES_GRACES_STATUS.md`](prosper/docs/TALES_GRACES_STATUS.md) and the [tracker](https://github.com/mattias800/prosper/issues/1889).

## Astro Bot — `PPSA21564`

<p align="center"><img src="assets/screenshots/astro-bot-title.webp" alt="Astro Bot — title screen"></p>
<p align="center"><img src="assets/screenshots/astro-bot-opening-cinematic.webp" alt="Astro Bot — opening sequence"></p>
<p align="center"><img src="assets/screenshots/astro-bot-worldmap-background.webp" alt="Astro Bot — world-map backdrop"></p>

The opening sequence and ASTRO BOT title card render at native 3840×2160. See the [tracker](https://github.com/mattias800/prosper/issues/1809) for current visual and performance work.

## Metaphor: ReFantazio — `PPSA20800`

<p align="center"><img src="assets/screenshots/metaphor-language-select.webp" alt="Metaphor: ReFantazio — the language-selection screen, twelve languages in white serif type with English highlighted by a blue brush-stroke"></p>
<p align="center"><img src="assets/screenshots/metaphor-loading-mascot.webp" alt="Metaphor: ReFantazio — the loading screen's winged fairy perched on an open book"></p>

A default 3840×2160 launch reaches the loading mascot and then the language-selection screen, and
holds it: 12 of 12 samples distinct over 60 s with the guest still running. Every glyph above is
rasterized from the title's own TrueType file, which it hands prosper through
`sceFontOpenFontMemory` — Latin, Cyrillic, Japanese, Traditional and Simplified Chinese and Korean.
The **background art behind the menu does not draw** ([#2952](https://github.com/mattias800/prosper/issues/2952)).
Until 2026-08-23 the title died of a divide-by-zero five seconds into every boot
([#2951](https://github.com/mattias800/prosper/issues/2951)). See the
[tracker](https://github.com/mattias800/prosper/issues/2876).

## The Forgotten City — `PPSA03026`

<p align="center"><img src="assets/screenshots/forgotten-city-title.webp" alt="The Forgotten City — title screen"></p>

The direct native 3840×2160 frontend capture reaches the title screen and its `Press Any Button` prompt, through the full white splash, Unreal Engine and Film Victoria logo cards, legal text and autosave notice. Gameplay has not yet been reached, and the run is bounded by two intermittent guest-heap faults. See the [tracker](https://github.com/mattias800/prosper/issues/1890).

## Tactics Ogre: Reborn — `PPSA03839`

<p align="center"><img src="assets/screenshots/tactics-ogre-title.webp" alt="Tactics Ogre: Reborn — title screen"></p>
<p align="center"><img src="assets/screenshots/tactics-ogre-reborn-gameplay.webp" alt="Tactics Ogre: Reborn — first tutorial battle"></p>

The route reaches the first tutorial battle with real GPU draws at native 1920×1080. See the [tracker](https://github.com/mattias800/prosper/issues/1892).

## Little Nightmares III — `PPSA05143`

<p align="center"><img src="assets/screenshots/little-nightmares-3-boot-splash.webp" alt="Little Nightmares III — developer splash from the boot sequence"></p>
<p align="center"><img src="assets/screenshots/little-nightmares-3-title-screen.webp" alt="Little Nightmares III — title screen"></p>

The boot splash sequence and the title screen render at native 3840×2160 on a default launch, with the
logo, the player slot and the Start prompt all legible. The render-thread stall that used to end the run
before the title is gone.

The title screen is still degraded: most composited frames arrive with the red and green channels forced
to maximum, which reads as a flat yellow background under otherwise-correct content.

**A default launch used to be blocked by a defect of ours, now fixed.** prosper's `/savedata0` mount
was a single host directory shared by every title, so this game read an `OptionSettings` slot written
by another Unreal title, declared it corrupt and deleted it — and then held on a modal
`Your options save has corrupted and has been deleted` dialog that a run with no input cannot
dismiss. Save data is now namespaced by title id
([#2734](https://github.com/mattias800/prosper/issues/2734),
`prosper/docs/SAVE_DATA_LAYOUT.md`), so a default launch no longer meets another title's save. Saves
written before that change stay in the old flat directory, untouched and unread; prosper reports
where they are on first use. See the [tracker](https://github.com/mattias800/prosper/issues/1893).

## Crisis Core –Final Fantasy VII– Reunion — `PPSA07809`

<p align="center"><img src="assets/screenshots/crisis-core-title.webp" alt="Crisis Core Reunion — title screen"></p>
<p align="center"><img src="assets/screenshots/crisis-core-main-menu.webp" alt="Crisis Core Reunion — main menu"></p>
<p align="center"><img src="assets/screenshots/crisis-core-voice-language.webp" alt="Crisis Core Reunion — voice-language selection"></p>

The title is Unreal Engine 4.27 with IoStore packaging. It boots into a native Linux/Vulkan run, completes
engine bootstrap, and streams real content from the 8.49 GB IoStore container. The declined-GPU-submit freeze
that used to stop it here is gone, fixed by [#1987](https://github.com/mattias800/prosper/pull/1987). It now
reaches the white splash, the autosave-notice dialog and then the **title screen** — "Press Any Button" with the
version string and the Square Enix copyright — which holds stably. One button press opens the main menu
(`NEW GAME` / `LOAD GAME` / `CONTINUE`, correctly greyed with no save present / `OPTIONS` / `CONVERT SAVE DATA`),
and pressing on through it opens the new-game settings flow: voice-language selection and the "Begin game with
selected settings?" prompt both render correctly.

**This is rung 2 on a throttled route, not on a default launch.** What ends a default run is a guest memory
fault a few seconds into the boot — on the render thread, the RHI thread or a worker pool thread depending on
the run, and in some runs the game's own allocator catches it first. Holding the guest's own submit call for a
few extra milliseconds avoids it, which is how the captures above were taken; that is a stopgap while the race
is fixed. Two rendering defects remain on the reached screens: the title screen's key art never draws, and most
published frames select a non-scanout composite source while the real screen exists. See
[`docs/CRISIS_CORE_STATUS.md`](prosper/docs/CRISIS_CORE_STATUS.md) and the
[tracker](https://github.com/mattias800/prosper/issues/1894).

## The House of the Dead 2: Remake — `PPSA24203`

<p align="center"><img src="assets/screenshots/house-of-the-dead-2-remake-title.webp" alt="The House of the Dead 2: Remake — title screen"></p>
<p align="center"><img src="assets/screenshots/house-of-the-dead-2-remake-gameplay.webp" alt="The House of the Dead 2: Remake — Training 1 gameplay"></p>

The route reaches Training 1 with a live rail-shooter camera, HUD, crosshair, and interaction effects at native 1920×1080. See the [tracker](https://github.com/mattias800/prosper/issues/1896).

## Bendy and the Dark Revival — `PPSA27624`

<p align="center"><img src="assets/screenshots/bendy-dark-revival-gameplay.webp" alt="Bendy and the Dark Revival — Chapter 1 gameplay with the NEW OBJECTIVE HUD"></p>
<p align="center"><img src="assets/screenshots/bendy-dark-revival-title.webp" alt="Bendy and the Dark Revival — title screen"></p>

The route [`prosper/scripts/bendy-dark-revival-PPSA27624/reach-gameplay.pad`](prosper/scripts/bendy-dark-revival-PPSA27624/reach-gameplay.pad)
crosses the multi-language photosensitivity warning, the title screen and the BEGIN menu into **Chapter 1** at native
3840×2160 — Audrey's opening dialogue, the hold-to-interact prompt, a first-person reticle and the `NEW OBJECTIVE`
banner. **No prosper code change was required**; the title was one input route away. Reproduced across five bounded
runs, two of which share 5 of 8 byte-identical frame CRCs.

The menu's background video is still not composited, so the BEGIN menu sits on a flat grey background. See
[`prosper/docs/BENDY_DARK_REVIVAL_STATUS.md`](prosper/docs/BENDY_DARK_REVIVAL_STATUS.md) and the
[tracker](https://github.com/mattias800/prosper/issues/1897).

## Beneath — `PPSA27640`

<p align="center"><img src="assets/screenshots/beneath-gameplay.webp" alt="Beneath — the opening dive aboard the science ship, with the waypoint HUD reading 21m and a character subtitle on screen"></p>

A direct, unmodified native 1920×1080 frontend capture reaches gameplay: the scripted route
(`prosper/scripts/beneath-PPSA27640/reach-gameplay.pad`) drives the title screen, `NEW GAME`, the
`SELECT DIFFICULTY` screen and the intro, then walks the opening dive in
`Assets/LEVELS/SCIENCE_SHIP/Science_Remake2.unity`. The capture above is from that route; the
waypoint HUD counts down as the route moves, and the title's own dialogue subtitles play.

This title needs `PROSPER_NULL_PAGE=1` to boot — the Unity player walks its frame-pointer chain one
hop past the terminal NULL frame pointer and reads the return-address slot at address `0x8`.
See [`prosper/docs/BENEATH_STATUS.md`](prosper/docs/BENEATH_STATUS.md) and the
[tracker](https://github.com/mattias800/prosper/issues/1898).

## Unbound: Worlds Apart — `PPSA03274`

<p align="center"><img src="assets/screenshots/unbound-worlds-apart-title-screen.webp" alt="Unbound: Worlds Apart — the title screen at 3840x2160: the UNBOUND / Worlds Apart wordmark in a pale carved typeface over a dark blue forest, a cloaked figure standing left of a glowing blue portal, with a Cross-button prompt below"></p>

A direct, unmodified `tools/screenshot` capture at 3840×2160 on a default launch with no pad input.
The title screen renders complete — wordmark, character, portal and prompt glyph — and on current
master it renders **continuously**: 40 of 40 samples over 200 s, with 354 distinct frames produced.
The ~9% duty cycle and the 5 s cadence recorded in the 2026-08-22 survey
([#2932](https://github.com/mattias800/prosper/issues/2932)) no longer reproduce here. The prompt is
waiting for Cross.
See [`prosper/docs/NEVER_BOOTED_SURVEY_2026_08.md`](prosper/docs/NEVER_BOOTED_SURVEY_2026_08.md) and
the [tracker](https://github.com/mattias800/prosper/issues/2886).

<p align="center"><img src="assets/screenshots/unbound-worlds-apart-intro-cinematic-village.webp" alt="Unbound: Worlds Apart — a later moment of the intro cinematic at 3840x2160: a sunlit village clearing of thatched huts strung with orange bunting, tall trees and fireflies, pink mushrooms in the foreground grass and the small red-cloaked character at the right, with a prompt reading Press Square to skip"></p>

A second, later frame of the same cinematic, captured with an isolated save on master `4ce6e11e` —
the evidence that this title's black frames were the save and not the renderer.

<p align="center"><img src="assets/screenshots/unbound-worlds-apart-intro-cinematic.webp" alt="Unbound: Worlds Apart — the intro cinematic at 3840x2160: the cloaked blue-hooded character Soli standing beside a golden deer in a teal moonlit forest clearing, thatched huts and bunting behind them, purple mushrooms and grass in the foreground, and a prompt reading Press Square to skip"></p>

A direct, unmodified `tools/screenshot` capture at 3840×2160, driven by
[`prosper/scripts/unbound-PPSA03274/reach-first-level.pad`](prosper/scripts/unbound-PPSA03274/reach-first-level.pad).
Cross clears the title screen, and the screen after it names a **different** button: its own prompt
reads `Press ▢ to skip`. A cross-only ladder sat on this cinematic for 180 s; Square skips it and
the first level's map load (`normalvillage`) follows at t ≈ 78 s. The level itself does not render.

**This route requires an isolated save.** With a stale save directory at the shared default location
the title advances into a state that renders nothing, and the run goes black about 12 s after the
first press — indistinguishable from a composite defect, and it is what the black-frame readings on
this title had been measuring. A matched A/B on the same commit: 4 of 20 samples carried content with
the shared save, 17 of 20 with a fresh `PROSPER_SAVE0` + `PROSPER_SAVEDATA_DIR`. Both arms are on
master `4ce6e11e`; building the 2026-08-22 commit the route was validated against reproduced the black
too, so there is no regression here — only a dirty save
([#2932](https://github.com/mattias800/prosper/issues/2932)).

## BALAN WONDERWORLD — `PPSA02058`

<p align="center"><img src="assets/screenshots/balan-wonderworld-language-select.webp" alt="BALAN WONDERWORLD — the language-select screen at 3840x2160: a Language Settings heading over a red and gold theatre interior, with 24 language buttons in three columns, English highlighted in cyan, and a prompt bar reading Change Language and Select Language"></p>

A direct, unmodified `tools/screenshot` capture at 3840×2160 on a default launch with no pad input.
This is the game's own first-boot language menu, not its title screen: all 24 labels render,
including CJK, Thai, Greek, Cyrillic and Arabic, and the prompt bar names the buttons it wants. About
82% of frames are a flat white 4K clear instead
([#2932](https://github.com/mattias800/prosper/issues/2932)). See
[`prosper/docs/NEVER_BOOTED_SURVEY_2026_08.md`](prosper/docs/NEVER_BOOTED_SURVEY_2026_08.md) and the
[tracker](https://github.com/mattias800/prosper/issues/2882).

<p align="center"><img src="assets/screenshots/balan-wonderworld-main-menu.webp" alt="BALAN WONDERWORLD — the main menu at 3840x2160: a red and gold theatre interior with a lit proscenium and drawn curtains, a menu column on the right reading Language Settings, Settings, Credits and Copyright Information, and a 2021 Square Enix copyright line"></p>

<p align="center"><img src="assets/screenshots/balan-wonderworld-prologue.webp" alt="BALAN WONDERWORLD — the opening story cutscene at 3840x2160: Leo and Emma standing in a city park at golden hour, a basketball court with graffiti-covered fencing behind them, children playing, trees and a brick building in the background, and speaker cabinets flanking the frame"></p>

Direct, unmodified `tools/screenshot` captures at 3840×2160, driven by
[`prosper/scripts/balan-PPSA02058/reach-title-and-prologue.pad`](prosper/scripts/balan-PPSA02058/reach-title-and-prologue.pad).
**The language screen is not waiting for Cross.** Pressing Cross raises its own modal — *"Are you
sure you want to change the game language to English?"* — and 109 Cross presses over 330 s never
leave the screen, loading no asset a default run does not. **Down** answers the modal; `/app0/title`
then opens at t ≈ 15.5 s. Behind it the title screen and main menu render, and then the opening
story cutscene, which is real GPU draws at 4K rather than a decoded movie — the boot logs no video
decode of any kind. No stage asset loads in a 717 s run, so this is not yet gameplay.

## Stray — `PPSA02101`

<p align="center"><img src="assets/screenshots/stray-brightness-calibration.webp" alt="Stray — the brightness-calibration screen at 3840x2160: three grey cat-head silhouettes at increasing brightness on black, instruction text above them, a 16-step slider below, and Defaults and Accept prompts in the lower right"></p>

A direct, unmodified `tools/screenshot` capture at 3840×2160 on a default launch with no pad input.
The BlueTwelve Studio logo plays and the game's own brightness-calibration screen renders in full,
then holds. Its prompt reads `✕ Accept`, and on this default route nothing presses it.
[`prosper/scripts/stray-PPSA02101/reach-first-map.pad`](prosper/scripts/stray-PPSA02101/reach-first-map.pad)
does, and it is the one of these three titles whose on-screen glyph really is sufficient: Cross
accepts the screen and carries the main menu behind it, and the first map load
(`hk_project_mainstart`) follows at t ≈ 36.5 s — absent from every default run measured. There is
no capture of what is behind it because there is nothing to show: the frames after the load are
letterboxed with a flat single-colour fill between the bars, and a flat clear is not progression
evidence. See
[`prosper/docs/NEVER_BOOTED_SURVEY_2026_08.md`](prosper/docs/NEVER_BOOTED_SURVEY_2026_08.md) and the
[tracker](https://github.com/mattias800/prosper/issues/2883).

## Little Nightmares II — `PPSA02154`

<p align="center"><img src="assets/screenshots/little-nightmares-2-tarsier-logo.webp" alt="Little Nightmares II — the Tarsier Studios logo at 3840x2160: a white condensed serif wordmark centred on a dark grey vignetted background"></p>

A direct, unmodified `tools/screenshot` capture at 3840×2160 on a default launch with no pad input.
The boot logo sequence advances — Bandai Namco, then Tarsier Studios, then Unreal Engine — through
the first ~130 s. After that the composite is a flat white 4K clear for the remaining 260 s of a
390 s run, and no title screen is reached
([#2932](https://github.com/mattias800/prosper/issues/2932)). See
[`prosper/docs/NEVER_BOOTED_SURVEY_2026_08.md`](prosper/docs/NEVER_BOOTED_SURVEY_2026_08.md) and the
[tracker](https://github.com/mattias800/prosper/issues/2884).

## Reproducible routes

Where available, reusable input routes live under [`prosper/scripts`](prosper/scripts). The tracker
issue for each title identifies the current route and the exact tested revision.
