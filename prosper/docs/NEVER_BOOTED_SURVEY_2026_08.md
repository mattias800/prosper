# The never-booted eight — a breadth survey (2026-08-22)

Eight tracked titles had never been run. This records what one build and one-to-four bounded runs
each found, and then the part that is the point: **which of them stop for the same reason.**

The deliverable is the [blocker ranking](#the-blocker-ranking), not the per-title sections. A fix
that unblocks four titles is worth more than a perfect fix for one, and that comparison cannot be
made until every title has been measured — which is the whole reason this was run as breadth rather
than as eight investigations.

**Nothing here was fixed.** Every finding is filed; depth is the next task, chosen from the ranking.

## How everything below was measured

One build of master `bffa40d4`, `-DPROSPER_APP=ON`, built inside the project's distrobox. Every run
used the frontend **`tools/screenshot`** on the **default route** — no pad input, no snapshot
acceleration, `PROSPER_RENDER=1 PROSPER_GUEST_ARGS=-force-gfx-direct`, `PROSPER_BOOTPHASE=1`,
Linux / AMD Radeon 8060S (RADV STRIX_HALO). Name the frontend when quoting any of it (instrument
trap 127).

Every run was wrapped in an outer `timeout` at a **different** value from the tool's own
`--timeout`, because `--timeout` cannot bound a boot (instrument trap 214) — the deadline lives in
the sampling loop and is unreachable until `boot_program()` returns. Where a limit fired, the
sections below say which one.

Two corpus-wide greps appear here. Both use `grep -a` and both were validated against a known
positive before their result was believed (instrument trap 218).

## Results

| Title ID | Title | Engine | AGC SDK | Rung | Where it stops |
| --- | --- | --- | --- | --- | --- |
| `PPSA02058` | BALAN WONDERWORLD | Unreal Engine 4 (`.pak`) | 7 | **2** | The game's own **4K language-select menu** renders, with full theatre art and button glyphs — but on only ~18% of frames; the rest are a flat white 4K clear ([#2932](https://github.com/mattias800/prosper/issues/2932)). The menu waits for CROSS. |
| `PPSA02101` | Stray | Unreal Engine 4 (`.pak`) | 9 | **2** | BlueTwelve logo, then the game's **4K brightness-calibration screen**, held for the rest of the run. Its own prompt reads `✕ Accept` — the flow waits for CROSS. |
| `PPSA02154` | Little Nightmares II | Unreal Engine 4 (`.pak`) | 8 | **1** | A **4K logo sequence** — Bandai Namco → Tarsier Studios → Unreal Engine — for the first ~130 s. Then the composite is a **flat white 4K clear for the remaining 260 s** of a 390 s run, and the last content frame is the last sample before the unregistered `sceAgcDcbDrawIndirect` call ([#2929](https://github.com/mattias800/prosper/issues/2929)). Also shows both wrong-frame signatures of [#2932](https://github.com/mattias800/prosper/issues/2932). |
| `PPSA02846` | Spacebase Startopia | Unity 2020.3.12f1 / IL2CPP | 8 | **0** | Boots in 447 ms, publishes **3 flips and one black 1080p frame**, then never submits again. The guest stays alive and audible ([#2933](https://github.com/mattias800/prosper/issues/2933)). |
| `PPSA03001` | Sifu | Unreal Engine 4 (`.pak`) | 8 | **0** | Flat white then flat magenta 4K clear, never any content. Two further defects on the same boot: the guest's own **out-of-memory assert** ([#2908](https://github.com/mattias800/prosper/issues/2908), 2 of 3 runs) and a **GPU hard recovery** ([#2935](https://github.com/mattias800/prosper/issues/2935), 1 of 3). |
| `PPSA03130` | Sniper Ghost Warrior Contracts 2 | CryEngine | 9 | **0** | A 4K present loop at ~9 flips/s in which **no pass produces a present source** — 2231 flips, 0 published ([#2871](https://github.com/mattias800/prosper/issues/2871), independently reproduced). |
| `PPSA03274` | Unbound: Worlds Apart | Unreal Engine 4 (`.pak`) | 8 | **2** | The **4K title screen**, wordmark, character and Cross prompt — on ~9% of frames, at a strict 5 s cadence; a near-black diagonal wedge otherwise ([#2932](https://github.com/mattias800/prosper/issues/2932)). Waits for CROSS. |
| `PPSA20800` | Metaphor: ReFantazio | Atlus GFD + CRIWARE | 12 | **0** | **Zero frames.** Dies ~7 s in: SIGSEGV in the `CRI Server Mana` movie thread on a non-canonical pointer ([#2934](https://github.com/mattias800/prosper/issues/2934)). |

**Three of the eight reach rung 2, one reaches rung 1, four are rung 0.** Every one of them boots,
links and runs guest code; the slowest boot in the set is 447 ms.

### Two notes on how these rungs are assigned

**`PPSA03130` was not actually un-booted.** It carries a milestone and a named blocker
([#2871](https://github.com/mattias800/prosper/issues/2871)) in `COMPATIBILITY.md` from another
lane, while its tracker *body* still says "no boot attempted" — the body-versus-comments staleness
the charter warns about, in the wild. It is kept in the survey because an independent reproduction
of a rung-0 claim is worth having, and it did produce one qualification (see its section).

**BALAN and Stray reach a pre-title first-boot menu, not the title screen.** The ladder's rung 2 is
"title screen"; `COMPATIBILITY.md`'s own bucket is "title screen **or menu** reached", and that is
the reading applied here. Both render an interactive screen of the game's own UI at 4K, past the
logos. Only *Unbound* reaches a literal title screen. The milestone text in each row says which.

## The blocker ranking

Ranked by **how many titles a blocker holds**, not by how many call into it. Those are different
questions, and #2747 is the standing reminder of why: the stuck Unreal titles share a *defect* and
not a *wall*.

| # | Blocker | Titles held | Which |
| --- | --- | --- | --- |
| 1 | **The finished frame loses most presents to a wrong composite** — flat white 4K clear, or a near-black diagonal wedge ([#2932](https://github.com/mattias800/prosper/issues/2932)) | **3 of 8**, and 3 more already recorded | BALAN, Little Nightmares II, Unbound — plus *Beast of Reincarnation* ([#1588](https://github.com/mattias800/prosper/issues/1588)), *Gollum* ([#2898](https://github.com/mattias800/prosper/issues/2898)), *Khazan* ([#2908](https://github.com/mattias800/prosper/issues/2908)), all rung 0 on "a flat white clear" |
| 2 | **The game's own screen waits for CROSS and nothing presses it** | **3 of 8** | BALAN (language select), Stray (`✕ Accept`), Unbound (title-screen prompt) |
| 3 | **UE4's allocator asserts after the halving probe** ([#2908](https://github.com/mattias800/prosper/issues/2908)) | **2** | Sifu (new carrier), *Khazan* `PPSA20447` |
| 4 | **`sceAgcDcbDrawIndirect` unregistered** ([#2929](https://github.com/mattias800/prosper/issues/2929)) | 0 confirmed; **2 confirmed callers**, NID in **49 of 54** dumps | Little Nightmares II, Sifu |
| 5 | **HTTP entry points answer a false success** ([#2930](https://github.com/mattias800/prosper/issues/2930) v1, [#2894](https://github.com/mattias800/prosper/issues/2894) v2) | 0 here; **7 of 8** call one | all but Spacebase — v1 on six, v2 on Metaphor |
| 6 | **A GPU hard recovery from a compute submit** ([#2935](https://github.com/mattias800/prosper/issues/2935)) | 1 | Sifu — the *GTA V* pattern ([#2481](https://github.com/mattias800/prosper/issues/2481)) on a second title |
| 7 | **No pass produces a present source** ([#2871](https://github.com/mattias800/prosper/issues/2871)) | 1 | Sniper Ghost Warrior Contracts 2 |
| 8 | **The guest stops submitting after three flips** ([#2933](https://github.com/mattias800/prosper/issues/2933)) | 1 | Spacebase Startopia |
| 9 | **SIGSEGV in the CRI Mana movie thread** ([#2934](https://github.com/mattias800/prosper/issues/2934)) | 1 | Metaphor: ReFantazio |
| 10 | **A guest PRX dlopened after boot does not satisfy already-bound eboot imports** ([#2931](https://github.com/mattias800/prosper/issues/2931)) | 0 | Unbound — 16 live `libfmod`/`libfmodstudio` calls return 0 |

### Reading the ranking

**Rows 1 and 2 are the two that pay.** Between them they hold every title in this survey that is
not rung 0, and row 1 additionally names the exact symptom three *already-tracked* rung-0 titles
are stuck on. Six titles is the largest single cluster this survey found, and the only one where a
fix would move more than two rows of `COMPATIBILITY.md`.

**Row 2 is cheap to settle and has faked a wall five times already** — most recently *Sonic
Origins*, whose four-month wall was a modal waiting for CROSS while every frame-count instrument
reported a healthy title. Three of these eight are sitting on a screen whose own on-screen prompt
names the button. A pulsed-pad route on each is one run apiece and would separate "the title is
waiting" from "the title is stuck" without any code.

**Rows 4 and 5 are large and hold nothing, and that is the finding.** `sceAgcDcbDrawIndirect` is a
one-line registration gap whose NID sits in 49 of 54 dumps; the HTTP false success is called by
seven of these eight. Neither is why any surveyed title stops. They are worth fixing on their own
terms — a missing indirect draw deletes real content silently, and an id-returning contract must
never answer 0 — but a ranking that ordered by call count would put them first and be wrong.

**Row 3 is the one that changed a shared blocker's shape.** See its section.

### What the survey did NOT find

- **No title in this set is blocked on a recompiler gap.** Sifu is the only one with rejected
  shader programs at all (five), and the program that lost its device is not among them.
- **No title is blocked on an unimplemented library that another surveyed title also blocks on.**
  Every rung-0 wall here is title-specific. The sharing is in the *symptoms* (rows 1-3), not in the
  missing surface.
- **No cross-reference to [#2872](https://github.com/mattias800/prosper/issues/2872)** (the APR
  gather/scatter read holding *Yakuza Kiwami* and *Judgment*). Metaphor calls one `libSceAmpr` NID —
  `sceAmprCommandBufferGetNumCommands`, `gzndltBEzWc` — and BALAN calls
  `sceAmprMeasureCommandSizeWriteAddressOnCompletion` (`C+IEj+BsAFM`); neither is #2872's
  `sceAmprAprCommandBufferReadFileGatherScatter`, and neither title stops there.
- **All eight are pre-SDK-13** (AGC register-defaults requests: 7, 9, 8, 8, 8, 9, 8, 12), so all
  eight are inside the [#2219](https://github.com/mattias800/prosper/issues/2219) post-submit
  completion-visibility cohort. **That is a shared property, not a shared diagnosis** — no title
  here was tested against `PROSPER_POST_SUBMIT_VISIBILITY=1`, and none showed the mid-boot death
  that lever addresses on *ArcRunner* and *Crisis Core*.

## Per-title findings

### `PPSA02058` — BALAN WONDERWORLD (Unreal Engine 4, `.pak`, no IoStore)

Boots in **232 ms**. Renders its own **language-select menu at 3840x2160** — the theatre
background, the game's font, all 24 language buttons with English highlighted, and the
`○ ✕ △ □ Change Language` / `Select Language` prompt bar. Runs at 17.9 fps while producing frames,
97% of the run active, guest alive at 140 s.

It is rung 2 on that menu and not on a title screen: the menu is the game's first-boot flow and
waits for CROSS.

**~82% of frames are a flat white 4K clear instead** ([#2932](https://github.com/mattias800/prosper/issues/2932)).
A 1 s grid over 110 samples put content at t = 16, 18, 32, 33, 36, 45, 49, 50, 54, 56, 60, 69, 72,
73, 77, 82, 86, 98, 103, 105 s — 20 of 110, with no cadence. Earlier 7 s and 5 s grids reported
"3 of 20" and "3 of 24"; those numbers are the grid, not the title (instrument trap 211), and are
recorded here only so they are not re-derived as a discrepancy.

Calls the unregistered `sceAmprMeasureCommandSizeWriteAddressOnCompletion`, `sceHttpInit`,
`sceHttpCreateTemplate`, `sceSaveDataSetParam`, `sceSaveDataSaveIcon`,
`sceNpWebApi2PushEventCreateHandle`, `sceNetResolverCreate`, `sceVoiceQoSInit`,
`sceAjmBatchJobDecodeSingle`, `pthread_setschedparam`. None is established as a blocker.

### `PPSA02101` — Stray (Unreal Engine 4, `.pak`)

Boots in **239 ms**. Renders the **BlueTwelve Studio logo** and then the game's own **4K
brightness-calibration screen** — three cat silhouettes at increasing brightness, the instruction
text, a 16-step slider, and `△ Defaults` / `✕ Accept`. Holds there for the rest of a 140 s run,
10.1 fps active.

The screen names its own button. This is the input-mapping shape, not a renderer wall.

Only unregistered call worth noting beyond the shared set: `sceCoredumpRegisterCoredumpHandler`.

### `PPSA02154` — Little Nightmares II (Unreal Engine 4, `.pak`)

Boots in **245 ms**. Renders a **4K logo sequence** that genuinely advances — **Bandai Namco
Entertainment**, **Tarsier Studios**, **Unreal Engine** — through the first ~130 s of a 390 s run.
6.9 fps while producing frames, 19% of the run active, guest alive at the end.

**Then it goes white and stays white.** On a 13 s grid over 30 samples, the last frame carrying
content is at **t=130 s**; every sample from **t=143 s to t=390 s** — 20 consecutive samples,
260 seconds — is a flat white 4K clear (`distinct_rgb_colors == 1`, `nonblack_rgb_pixels ==
8294400`). No title screen is reached.

**And the transition brackets the `sceAgcDcbDrawIndirect` call.** In this run the single
`[prosper] unimplemented: libSceAgc::1q1titRBL6o` line falls between the t=130 s sample (the last
with content) and the t=143 s sample (the first of the 260 s of white). An earlier run on a 15 s
grid put the first call at t≈100 s with the two following samples, at 105 s and 120 s, both white.

**Two runs, the same ordering — and that is a correlation, not a cause.** `CONFIDENCE: LOW`. Two
samples of an ordering is weak, the log line is emitted once per NID so it marks the *first* call
rather than the interesting one, and #2932's flat white appears on this title well before t=130 s
too. It is recorded because it is cheap to test properly and nobody would think to look: registering
the NID ([#2929](https://github.com/mattias800/prosper/issues/2929)) and re-running this exact route
answers it in one run.

Shows **both** wrong-frame signatures of [#2932](https://github.com/mattias800/prosper/issues/2932):
flat white 4K clears, and near-black frames carrying a diagonal smeared band in the upper-left. The
wedge frames are the same kind of picture as *Unbound*'s, which is what put the two titles in one
issue.

`PROSPER_PROGRESS_UNIMPL=1` gives the rest of its unimplemented traffic as
`47 x pthread_setschedparam`, `3 x sceSaveDataSetParam`, and one each of `sceHttpInit`,
`sceHttpCreateTemplate`, `sceNpWebApi2PushEventCreateHandle`, `sceVoiceQoSInit`,
`sceNetResolverCreate`, `sceSaveDataSaveIcon`.

> **A measurement of this title was wrong in an earlier draft of this document and is corrected
> here.** It claimed "no title screen in a 380 s run" on the strength of a run configured
> `--seconds 5 --count 24 --timeout 380`: the tool stops at `request-satisfied`, so 24 samples 5 s
> apart end the run at **120 s** and the 380 s deadline was never approached. `--timeout` is an upper
> bound, and `--seconds x --count` is what actually sets a run's length. The genuine 390 s run above
> is what produced the flat-white finding, which the short run could not have seen.

### `PPSA02846` — Spacebase Startopia (Unity 2020.3.12f1 / IL2CPP)

Boots in **447 ms**, publishes **3 flips and one black 1920x1080 composited frame**, and then never
submits again — reproduced on four runs of 110-140 s, all `status=ok guest=running`, all
`0% of the run active`. Rung 0. Full detail and the `hle_calls` census in
[#2933](https://github.com/mattias800/prosper/issues/2933).

The short version: the guest is **running, not deadlocked**. Over a 200-`k_usleep` window it makes
6928 HLE calls across 10 distinct handlers, all of them mutex / clock / semaphore / audio, and
**not one service or graphics handler**. Audio is still streaming through live FMOD threads.
`GfxFlipThread` is parked in `futex_do_wait`. The title prints Unity's own
`Forcing call to sce::Agc::suspendPoint to avoid TRC R4089 breach` 46 times in 140 s — its watchdog
reacting to a frame that never completes, not a call prosper is failing.

The single black frame means the display buffer was flipped before anything drew into it, so the
failure is upstream of the composite.

### `PPSA03001` — Sifu (Unreal Engine 4, `.pak`)

Boots in **266 ms**. **Never renders content.** All 20 samples of a 140 s run are a flat 4K clear,
one colour: RGB `(255,255,255)` to t≈35 s, then RGB `(255,0,255)` from t≈42 s onward. Rung 0.

Three runs produced two different endings, and both are filed:

- **2 of 3 runs: the guest's own out-of-memory assert**, at t≈45 s, in `FAsyncLoadingTh`. This is
  [#2908](https://github.com/mattias800/prosper/issues/2908), previously a single-title issue on
  *Khazan* — see the next section.
- **1 of 3 runs: a RADV hard recovery** at t≈105 s, losing the Vulkan device from a compute submit
  and disabling live compute process-wide
  ([#2935](https://github.com/mattias800/prosper/issues/2935)). The *GTA V* shape
  ([#2481](https://github.com/mattias800/prosper/issues/2481)) on a second title.

Five compute programs are rejected before either ending — three `recompile-reject`, two
`compute-cfg-reject … reason=wave64-ambiguous-mask-read`. **The program that lost the device is not
among them**: it recompiled and ran. And the composite is already a flat clear long before either
event, so neither is what holds the title at rung 0.

### `PPSA03130` — Sniper Ghost Warrior Contracts 2 (CryEngine)

Independently reproduced at rung 0, matching [#2871](https://github.com/mattias800/prosper/issues/2871).
Boots in **272 ms**, then drives a 4K present loop that publishes nothing:
`[rtt] PRESENT SOURCE EXTENT MISMATCH: no pass produced a 3840x2160 present source`, 64+
occurrences, `present_count` 2231 over 244 s, all 24 samples `raw_scanout` and fully black.

**One qualification for the next lane:** the title needs a long bound. A first attempt with
`--timeout 170` inside a 300 s outer limit saved **0 of 20** — the present loop is not running yet
at that point. The run above used `--timeout 520` inside 560 s. This is consistent with the
warm-versus-cold-cache asymmetry already recorded as instrument trap 216 on this title, and it is
the difference between "rung 0, no frames" and "the run was killed before it started".

Flip rate here is ~9/s against the ~21/s in the issue body; the outcome is identical.

### `PPSA03274` — Unbound: Worlds Apart (Unreal Engine 4, `.pak`)

Boots in **413 ms**. Renders its **title screen at 3840x2160** — the `UNBOUND / Worlds Apart`
wordmark, the cloaked character, the portal, the forest, and a Cross prompt. Rung 2, and the only
literal title screen in this survey.

It appears on **10 of 110** 1 s samples, at t = 49, 54, 59, 64, 69, 74, 79, 84, 92, 98 s — a
**strict 5.0 s cadence**, visible for well under a second each time. The other ~91% of frames are
near-black with a diagonal smeared band ([#2932](https://github.com/mattias800/prosper/issues/2932)).
Unbound's cadence is periodic and BALAN's is not, which is why the issue is explicit that one
mechanism should not be assumed for both.

Separately: the title loads its own `libfmod.prx` (1093 exports) and `libfmodstudio.prx` (359
exports) at runtime, **after** `STUBS_INSTALLED`, and **16 of the eboot's calls into them still
reach prosper's return-0 dispatcher** ([#2931](https://github.com/mattias800/prosper/issues/2931)).
The NIDs are present in the modules that were just loaded, checked with `grep -a` against a positive
control, so this is not a version mismatch.

### `PPSA20800` — Metaphor: ReFantazio (Atlus GFD + CRIWARE)

The largest dump tracked (71 GB) and the **fastest boot in the survey — 161 ms**. It then dies
about 7 s in, with **zero frames written**: `WORKER-THREAD FAULT: sig=11` in the thread named
`CRI Server Mana` — CRIWARE's `.usm` movie player — at `eboot+0x1221b60`, on
`mov eax,[rdi] ; mfence ; ret` with `rdi = 0x20000000e0c5df73`, an address that is not a guest
pointer. Rung 0. Detail in [#2934](https://github.com/mattias800/prosper/issues/2934).

Its tracker predicted CRIWARE would be the surface that mattered here, and it was, on the first run.

Four unregistered NIDs are called before the fault, none established as the cause:
`sceHttp2CreateTemplate` (this is [#2894](https://github.com/mattias800/prosper/issues/2894)),
`sceAmprCommandBufferGetNumCommands`, `sce::Json::InitParameter2`'s constructor plus two of its
setters, and `sceRtcGetDayOfWeek`.

## What this survey changed about an existing blocker

[#2908](https://github.com/mattias800/prosper/issues/2908) — "the guest exhausts prosper's
direct-memory pool and calls its own OOM handler" — was a single-title issue on *Khazan*. Sifu is a
second carrier, with the same guest-side report and the same UE4 halving probe under
`PROSPER_MEMLOG=1`.

But a control the survey ran incidentally sharpens it. *Unbound* runs the **identical** probe, takes
**19** `ENOMEM (pool exhausted)` answers in a 95 s run, never prints `PS5 Out of Memory`, never
asserts, and goes on to render its title screen.

**So the probe consuming the pool is ordinary UE4 behaviour on this backend, not the defect.** What
separates Khazan and Sifu from Unbound is a **later, small** allocation failing after the probe has
settled — Sifu dies on 16385 bytes while its own report says 13.6 GB is available. A pool resize
that still failed that request would move the symptom without fixing it. Recorded on the issue.

## One finding that is about the machine, not the titles

Running `ctest` on the survey build turned up something the survey was not looking for and which is
recorded here because it is true of the ground everything above stands on.

**Five Vulkan-execution tests fail on RADV at `bffa40d4`, and every failing assertion is an indexed
draw** — while CI is **green on the same SHA**, because `.github/workflows/ci.yml` runs the suite
under Mesa **lavapipe** on a runner with no GPU. 296 of 301 pass. The discriminator is internal to
one process:

```text
  [FAIL] indexed vertexOffset selects records 1..3 from a shared vertex pool
  [ok]   non-indexed firstVertex selects records 1..3 from a shared vertex pool
  [ok]   dropping the indices changes the picture (indices are really applied)
```

Same vertex pool, same shader, same target, same frame — the non-indexed arm is right and the
indexed arm is not, and the index buffer demonstrably reaches the GPU.

Filed as [#2937](https://github.com/mattias800/prosper/issues/2937), with the wedged-device and
`-j4`-contention explanations ruled out there.

**Whether this touches [#2932](https://github.com/mattias800/prosper/issues/2932) is not
established.** `CONFIDENCE: LOW`, recorded as a lead and nothing more: three of the surveyed titles
render a correct frame and then lose most of their presents to a wrong composite, and real engines
draw almost everything indexed — but "indexed draws are wrong" and "the composite alternates" are
different observables and no experiment has linked them. The reason to write it down is narrower and
solid: anyone chasing #2932 on this box should know this is true of the machine underneath it.

## Ruled out

One line per hypothesis this survey killed, so nobody re-derives it at full cost.

- **`PROSPER_CB_EFC_NO_COLOR` / [#1588](https://github.com/mattias800/prosper/issues/1588) is not
  the mechanism behind the flat clears on these titles.** The `[gpu] resolve_pipeline_state`
  diagnostic reports **MODE=0 (DISABLE)** on all six titles that emit it and never MODE=2
  (ELIMINATE_FAST_CLEAR) — BALAN >=131072 draws, Sifu >=16384, Stray >=4096, Little Nightmares II
  >=2048, Unbound >=32, Spacebase 1. There is no EFC population for that lever to act on, whatever
  it does for *Beast of Reincarnation*. #2932.
- **`PROSPER_LEGACY_CB_DISABLE_MASK=1` does not rescue BALAN and makes it strictly worse.** Matched
  A/B, identical `--seconds 5 --count 24` in both arms with only the lever differing: default **3 of
  24** samples carry the menu, lever arm **0 of 24**. Restoring #919's zeroed write mask removes the
  content rather than the white, so "CB DISABLE draws are painting over the composite" is falsified
  as the mechanism on this title. #2932.
- **The UE4 halving probe exhausting the direct-memory pool is not by itself the OOM defect.**
  Unbound takes 19 `ENOMEM (pool exhausted)` answers on the same probe and renders its title screen
  anyway. #2908.
- **Sifu's rejected compute programs are not what lost its device.** Five programs are skipped;
  the program named in `[compute] fatal Vulkan device loss` is not one of them — it recompiled and
  ran. #2935.
- **A green CI is not evidence that prosper's Vulkan execution tests pass.** CI runs them on
  lavapipe; five fail on RADV at the same SHA. #2937.
- **`grep -rl` without `-a` is not usable for a dump census**, restated because this survey used it
  twice: `sceAgcDcbDrawIndirect`'s NID reads as absent from every dump without `-a` and is present
  in 49 of 54 with it. Both greps here were validated on a known positive first. Instrument trap 218.

## Where a depth lane should start

In order, and the first two are the ones that pay:

1. **[#2932](https://github.com/mattias800/prosper/issues/2932)** — six titles show its symptom,
   three of them here. Start with Unbound: its 5.0 s cadence is the most tractable handle in the
   set, and an F9 grab (`PROSPER_GRAB_BUNDLE_AT_FRAME`, aimed with a cheap screenshot sweep) on a
   content frame and on a wedge frame gives two replayable bundles that differ by exactly the thing
   in question.
2. **A pulsed-pad route on BALAN, Stray and Unbound** — three runs, no code, and it separates
   "waiting for CROSS" from "stuck" on three rung-2 titles at once.
3. **[#2929](https://github.com/mattias800/prosper/issues/2929)** — a one-NID registration gap with
   a 49-of-54 upper bound on exposure and a sibling handler to copy the shape from.
4. **[#2908](https://github.com/mattias800/prosper/issues/2908)** — now two carriers and a control
   that says which half of it to chase.
