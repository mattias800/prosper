# The Oregon Trail (`PPSA19244`) — status and evidence

Unreal Engine 4. **Rung 2 — title screen**, and since #1946 the whole UI layer composites correctly
(`assets/screenshots/oregon-trail-title-screen.png`). Since #1933 landed the ErrorDialog foreground
lifecycle, the title renders its startup legal/EULA popup (`WBP_System_Disclaimer_Popup`) at native
3840x2160 from the live renderer. Tracked on
[#1606](https://github.com/mattias800/prosper/issues/1606) and
[#1641](https://github.com/mattias800/prosper/issues/1641).

## Where it stands (2026-08-05, later)

**Rung 2 reached: the title screen renders.** At master `c77c66b4`, nine default-launch arms on an
idle host produced **zero** #1945 faults and zero exit-90 (bounds 12–60 s; see `## Ruled out`); the 60 s
arm ran 3,689 guest frames, every sample pixel-distinct, ending on the rendered *The Oregon Trail*
title screen with its "Press any button" prompt
(`assets/screenshots/oregon-trail-title-screen.png`). Neither the ordered-DMA stall below (closed by
#1987) nor #1945 bounds a default launch any more. The #1945 non-reproduction is a **timing**
result, not a code one — the same binary previously produced 10/10 exit-90 — see `## Ruled out`.
The next frontier here is a route past the title screen.

### The three title-screen defects were one defect, and it is fixed (#1946)

The three things that screenshot showed — **#1946's solid-block Slate glyphs, the unblended black
panel behind the title logo, and the flat pink lower field** — were a single cause: **every draw in
the title rendered with alpha blending disabled.** Not a font, atlas, coverage or UV defect at all.

The chain, end to end:

* `sceAgcGetRegisterDefaults2` hands the guest a hash-keyed table of `{register offset, default
  value}` records; the guest's AGC pipeline builder searches it linearly and caches each record.
* Render target **0's** blend record has its own key, `0xA6D12629` — *not* the `0xEF550356` key that
  the same builder uses for RT1..7 (it derives those offsets as `record.offset + rtIndex`). The
  Oregon Trail eboot shows both: the search at `eboot+0x4ab2e00`, the miss path at `eboot+0x4ab36be`
  that caches `0xffffffffffffffff`, and the consumer at `eboot+0x138caf0` that loads the cached
  qword as RT0's record while `eboot+0x138cac7` loads the `0xEF550356` one for RT1..7.
* prosper offered `0xA6D12629` **only for SDK version >= 13** (it was added for *Dragon Quest VII*,
  which asks for 13). **The Oregon Trail asks for version 12** — `[agc] register defaults requested
  for SDK version 12`, 8 of 8 arms — so its lookup missed, its RT0 blend write went out with offset
  `0xffffffff`, and the command processor correctly dropped it as a nonexistent register.
  `CB_BLEND0_CONTROL` therefore never left its default. (The run does print
  `[agc] out-of-range indirect reg write dropped … off=0xffffffff val=0xffffffff`, which is the same
  signature — but that print is capped at four lines and the #1264 unfilled-placeholder family shares
  it, so treat it as consistent with this, not as the count. The register watch is the measurement.)
* Measured, before: `PROSPER_REGWATCH=Cx:0x1E0` recorded **133 writes, every one `0x20010001`**
  (`ENABLE=0`), and `PROSPER_RTTLOG=1` recorded **0 of 1,231 draws with blending enabled.**
  After: `0x65010504` appears — `ENABLE=1`, `SRC_ALPHA`/`ONE_MINUS_SRC_ALPHA` — and blend-enabled
  draws appear. `0x65010504` is byte-identical to what *Dragon Quest VII* programs.

Making the key available on every SDK version restores the whole layer at once: the logo composites
on the night-sky gradient with no black box, the sky/ground gradient and clouds are back, the three
"solid white rectangles" resolve into the HarperCollins Productions / Gameloft / Unreal Engine
logos, and Slate text draws its real glyph coverage. A default-launch title-screen sample goes from
a handful of flat colours to **66,460 distinct colours**.

`assets/screenshots/oregon-trail-title-screen.png`, `…-gameloft-splash.png` and
`…-health-warning.png` are all re-captured from a default-environment run on the fixed build.
`…-legal-popup.png` is **not** — it is a pre-fix capture and still shows the blocky text; four dense
arms (0.5 s sampling) all reached the Gameloft splash by 2.0 s without catching the popup, so it was
dropped from `COMPATIBILITY.md`'s gallery rather than shipped alongside three corrected frames. The
file is kept because the earlier sections above cite it.

## Where it stood earlier on 2026-08-05

The startup sequence now runs **past** the EULA stage. On a default launch the title presents, in
order: the legal popup, the **Gameloft splash** (`assets/screenshots/oregon-trail-gameloft-splash.png`),
one corrupted blue/magenta frame, and the **health/epilepsy warning screen**
(`assets/screenshots/oregon-trail-health-warning.png`) — then **stalls with the frame counter frozen**
(frame 252 and frame 396 in two arms), immediately after:

```text
[agc] ordered DMA submit rejected: Jump reads target/predication memory after retained DMA (order=…)
[agc] ordered DMA submit not executed: unsupported eager/deferred guest-memory dependency
```

That rejected submit is the current frontier — a *hang*, not a fault. Three of four arms on
`ff72e77c` + the mutex-encoding fix reached it and survived their whole 12 s bound; one hit #1945
instead, so **#1945 is no longer what bounds every run** (1 of 4 here, against 3 of 4 in the
2026-08-04 census).

What unblocked the EULA stage was **not** a graphics or a heap change. `scePthreadMutex{Lock,
Trylock,Timedlock}` returned the bare FreeBSD errno while the Sony contract is the libkernel-encoded
`SCE_KERNEL_ERROR_*` form, and the shipped guest `libc.prx` compares against the encoded constants
exactly (`_Mtx_trylock` → `0x80020010`, `_Mtx_lock` → `0x8002000b`, `_Mtx_timedlock` →
`0x8002003c`). An unrecognised value becomes Dinkumware `_Thrd_error`, and `std::_Throw_C_error(4)`
raises an **uncaught** `std::system_error("invalid argument")`. On master the title died on that
`std::terminate` ~2 s in, deterministically (4/4 arms), with a black frame — the popup was no longer
being reached at all. See `## Ruled out`.

Two further defects remain visible in the new frames: the corrupted blue/magenta frame between the
splash and the warning, and #1946 (Slate glyphs draw as solid blocks — clearly legible in the
warning screen capture).

Read `## Ruled out` before forming a hypothesis. Several separate leads on this title have been
measured and killed, including the GPU-side questions and the two service candidates that the issue
body originally called highest-value.

## Historical — where it stood after #1933 (2026-08-04)

The black-frame era is over. On master `9dcb6c4b` the default environment
(`PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1`, no diagnostics, no
route) produces real content within two seconds: a green legal panel titled *"Welcome to The Oregon
Trail"*, three legal links, and a focused confirm button, at 3840x2160 with **4,009,938 non-black
pixels** (48 % of the frame). `assets/screenshots/oregon-trail-legal-popup.png` is that frame.

Two defects now bound the title, in this order:

1. **A guest heap corruption ends the run after ~4-6 s (#1945).** `RenderThread 1` faults in the
   guest's own small-block allocator at `eboot+0x14600df` (`mov rax,QWORD PTR [rcx]` — popping the
   free-list head of the 32-byte size class). Three of four default-environment runs died this way
   within six seconds; the pad route below never even reaches its first press. Exit status is 90.
2. **Slate text renders as solid blocks (#1946).** Layout, kerning and word shapes are correct, so
   the font is loaded and measured; every glyph quad draws fully opaque instead of sampling its
   coverage.

Nothing here needs more base-pass or colour-state work. **The empty base pass was never a renderer
defect** — the guest was gated on account login, exactly as the shipped Blueprint contract below
says, and the 4K `R16G16B16A16_SFLOAT` surface with "no producer" is simply the scene target of a
frame whose only content is a UMG popup.

### The A/B that establishes this

One binary, one instrument, one variable — `sceSystemServiceGetStatus`'s
`isInBackgroundExecution` byte, which is what #1933 changed:

| Arm | `st[5]` | Frames | Fonts / ICU loaded | Fault |
|---|---|---|---|---|
| A — master as shipped | `sample_error_dialog_background()` | Legal popup, 32 distinct colours, 4,009,938 non-black px | `FF_Exo2_Bold.ufont`, `FF_PressStart2P_Condensed.ufont`, `icudt64l/brkitr/line.brk` | **yes**, ~4-6 s, `eboot+0x14600df` |
| B — control, pre-#1933 behaviour | forced `0` | 12/12 samples, 1 distinct colour, **0** non-black px, 2,911 frames over 60 s | none, over a **12x longer** run | none, exit 0 |

Arm B reproduces the historical rung-0 signature exactly, so the difference is attributable to the
one byte. The **font and ICU line-break loads are the progression witness**: Slate pulls a `.ufont`
and the ICU break iterator when it first *lays out* text, and the far longer control run never
touches them. (The widget *classes* are not a witness — see `## Ruled out`.)

## Where the GPU investigation stood before that (historical)

* The complete UE4 post-process chain is present and healthy: a 1920x1080 `R11G11B10F` HDR scene
  colour, a bloom pyramid from 960x540 down to 60x34, tonemap, a 32x32x32 grading LUT, and the
  3840x2160 front-buffer composite. A retained capture replays byte-exactly.
* That chain post-processes a **3840x2160 `R16G16B16A16_SFLOAT` surface that has no producer.** Over
  ~515 frames (10,796 realized target passes) it does not appear once. Every draw that targets it is
  suppressed, and nothing else writes it.
* The guest **clears** that scene target every frame and then draws nothing into it: the retained
  capture shows a compute dispatch writing `0x40404040` across the exact 294,912-byte DCC metadata
  range of `0x3012370000`, at 73,728 threads for 73,728 dwords, per frame — and then no colour draw
  ever targets it.
* Prosper decodes **~23 draws per frame in total** — ~21 realized (exactly the post chain plus the
  front-buffer composite) plus ~2 suppressed colour-disabled draws. For a UE4 title rendering a 4K
  scene that is menu-scale, not gameplay-scale.
* Only `/Game/Maps/L_GameloftSplash` is loaded during the measured black-frame run. Its shipped level
  Blueprint waits on account-login, free-space and EULA state before adding the Gameloft splash
  widget; that widget's own completion path loads `/Game/Maps/L_Main`. The empty base pass is
  therefore correct for the content state that actually loaded, rather than a missing GPU draw list.

**The GPU-side questions are answered. The account gate is now fixed:** the signed-out account path
opens and finishes an ErrorDialog, and Prosper exposes the system-dialog background→foreground
lifecycle that dispatches the game's account-completion poll. Two same-instrument CPU-only arms
recorded both lifecycle samples before the writer fired; the default-behaviour arm carries the exact
foreground-broadcast stack. Do not return to base-pass or further GPU instrumentation without
contradictory new evidence.

**That frontier is now closed by observation, not by inference.** The 2026-08-04 rendering arms show
the EULA stage reached and its popup *presented* — so both halves of the old open question ("prove the
EULA stage and splash `AddToViewport`", "content gate versus UMG presentation failure") are answered:
the gate was the content gate, and UMG presents. The frontier moved to #1945 (the guest heap
corruption that ends the run) and #1946 (glyph coverage). `WBP_Splash_Gameloft` and
`/Game/Maps/L_Main` have **not** been observed yet — the crash lands before a route can accept the
popup — so the splash and title screen remain unproven, not disproven.

## Shipped front-end progression contract

Offline package and guest-code analysis identifies the exact startup contract; this is stronger than
inferring progression from service-call volume:

| Blueprint predicate/action | Native implementation | Completion state |
|---|---|---|
| `IsAccountLoginRequired` | `0x88af00` | Always required. |
| `IsAccountLoginChecked` | `0x88aef0` | Reads byte `+0x109`. The ErrorDialog poll at `0x88ad6b` writes the two-byte value `0x0100` at `+0x108` on every non-`RUNNING` result, making `+0x108 = 0` and **`+0x109 = 1`**. Before the fix that poll was undispatched; the ErrorDialog lifecycle fix now reaches the writer after the observed foreground edge. |
| `IsNoFreeSpaceCheckRequired` | `0x8896c0` | Always required. |
| `CheckNoFreeSpace` | `0x8896d0` | Its synchronous completion path unconditionally writes byte `+0xb0 = 1` at `0x889722`, then invokes and clears the callback. `IsNoFreeSpaceChecked` (`0x87d070`) reads that byte. |
| `IsEULACheckRequired` | `0x88ab80` | Always required. |
| `IsEULAChecked` / `SetEULAChecked` | `0x87d640` / `0x886a60` | The getter reads byte `+0xc8`; the setter writes it to one. The shipped `BP_EULA_Controller` creates `WBP_System_Disclaimer_Popup`, adds it to the viewport, and calls the setter from its close path. |

`L_GameloftSplash` contains the state flags `bLoginChecked`, `bNoFreeSpaceCheckStarted`,
`bEULACheckStarted` and `bSplashStarted` and calls the predicates above. Exact retained-package
bytecode order is: start/check free space; wait for `IsNoFreeSpaceChecked`; wait for
`IsAccountLoginChecked`; call `Begin`; run the EULA branch; create `WBP_Splash_Gameloft`; then
`AddToViewport`. That widget polls `IsSplashMoviePlaying`, and its `LoadNextLevel` path opens
`/Game/Maps/L_Main`.

The automatic native account stage is now mapped too. `0x87d100` checks
`IsAccountLoginRequired`, wraps its incoming continuation, and calls the account-state method
`0x88abe0` through vtable slot `+0x78`. Completion invokes wrapper slot `+0x58`
(`0x8a54e0`) and enters the EULA stage at `0x87d350`; EULA completion similarly invokes
`0x8a5600` and continues at `0x879dd0`. In the signed-out arm, `0x88abe0` calls
`ShowAccountLogin`, whose observed successful ErrorDialog Open returns true and leaves account state
active. The constructor registers `0x88ad20` through wrapper `0x887110` on multicast
`0x867af10`; the platform status loop broadcasts that multicast when status byte 5
(`isInBackgroundExecution`) transitions back to zero. Its other route is the NP-state callback, but
the shipped handler returns immediately for Prosper's delivered `SIGNED_OUT` value. Prosper's
`sceSystemServiceGetStatus` previously wrote byte 5 as zero on every call, so an auto-finished system
dialog could not produce the required nonzero-to-zero edge. A same-binary, same-instrument pair made
that gap live: `PROSPER_BP=0x88ae00` hit `ShowAccountLogin` with immediate return `0x88ac99` in the
account-state method, while `PROSPER_BP=0x88ad75` remained at zero after ErrorDialog Open and more
than 12 seconds of advancing frames. The lifecycle discriminator then logged background 1 and 0
before the writer hit; the production-default implementation repeated the exact ordering and stack.
All arms printed the progression-only no-compute witness and no Vulkan-device line.

The general fix models ErrorDialog as system-owned UI. A backend-owned dialog reports background
while its frontend status is `RUNNING` and a foreground return when it finishes. The instantaneous
headless path preserves the same two observable samples instead of collapsing them. It contains no
title identity or guest address. The two writer arms prove account completion **2/2**, but not stable
downstream progression: the discriminator arm continued for its 16-second bound, while the final
default-behaviour arm hit an unrelated worker null fault after the writer and exited early. Treat
post-account stability as mixed `1/2`; do not claim EULA, splash, title screen, or gameplay yet.

## Ruled out

One line per dead hypothesis, the evidence that killed it, and where that evidence lives. Do not
re-derive these without contradictory new evidence.

| Hypothesis | Verdict and evidence | Source |
|---|---|---|
| #1945 bounds **every** run on this title, so it is the reproducible member of that issue's four-title family | **Falsified at master `c77c66b4` on an idle host — and the title now reaches the TITLE SCREEN.** **Nine** arms on 2026-08-05 — three clean 12 s, one 15 s (`PROSPER_MB3_FREELIST_GUARD`+`PROSPER_GENERATION_GUARD`), one 12 s (`PROSPER_WRITE_TRAP=0x1`+`PROSPER_FORGE_TRIP`), two clean 18 s under `taskset -c 0,1`, one clean 32 s in the exact env the census quoted (`PROSPER_NULL_PAGE=1`, no `-force-gfx-direct`), one clean 60 s — produced **zero** `WORKER-THREAD FAULT` lines, zero exit-90, every sample pixel-distinct. The longest arm ran **60 s / 3,689 guest frames** and its last sample is the rendered *The Oregon Trail* title screen with the "Press any button" prompt (`assets/screenshots/oregon-trail-title-screen.png`) — a **rung 2** this doc previously recorded as never reached. The earlier 10/10 exit-90 census was measured on the same binary (`278c9b1f`; `c77c66b4` is its docs-only child, sole diff `GAME_COMPAT_ORCHESTRATION.md`), so the difference is **timing, not code** — that much is established. *Why* the timing differs is a **hypothesis, not a result**: contention would explain both this and the 2/7, 3/4 and 1/4 rates on the other three titles, but the two `taskset -c 0,1` arms here were a deliberate attempt to widen the window by CPU starvation and did **not** reproduce it, which cuts mildly against the simplest form of that reading. Do not plan work here on the assumption that a default launch reproduces #1945. | #1945 |
| The `R11G11B10F` compute write-back notifier gap (Syberia #1619) explains the black scene here | **Falsified.** The precondition holds — the scene-colour target really is `R11G11B10_FLOAT` — but the notifier is only reached from `live_compute.cpp` under `bi.mirror_result_to_imported`, i.e. when a dispatch writes a *renderer-owned* storage image. A validated live trace with `PROSPER_COMPUTELOG=1` (which does **not** clear `live_gpu_targets`) produced **11,440 `[compute]` lines and zero** `mirrored storage result into renderer RTT` lines; every storage image in the run logs `renderer_owned=0`. **The notifier is never invoked by this title.** CONFIDENCE: HIGH. | #1606 |
| Prosper's `CB_COLOR_CONTROL` tracking is stale, unwritten, or folds wrongly, and the `MODE` decode is wrong | **Falsified — the tracking is exact.** `PROSPER_REGWATCH` over ~14 s records **12,390** `CB_COLOR_CONTROL` writes and 11,358 `CB_TARGET_MASK` writes, all on the indirect (`Set*RegsIndirect`) path. `MODE=1` (`CB_NORMAL`) is the **dominant** value at 9,346 writes versus 1,335 `MODE=0`, a 7:1 ratio. Joined correctly against each draw's own `command_order`, all **1,050 of 1,050** suppressed draws resolve exactly the `CB_COLOR_CONTROL` and `CB_TARGET_MASK` the guest wrote at or before that draw — **zero mismatches**. The guest deliberately programs `CB_DISABLE` on these draws, immediately before each one. The decode is sound in the other direction too: the same title's post-process draws decode `MODE=1` and render. **Do not force `MODE=1`** — it is not a fix, and `PROSPER_FORCE_COLORWRITE=1` leaves the frame byte-identically black. **Amended by #1724:** this row's finding stands (the decode is exact and the guest deliberately programs `CB_DISABLE`), but prosper no longer *acts* on `MODE` when deriving the colour write mask — the mask comes from `CB_TARGET_MASK & CB_SHADER_MASK` alone. If this title's suppressed draws carry a non-zero mask they now write; re-measure before reasoning from the old suppression counts. | #1606 |
| This is Astro Bot's colour-state signature (#1585), or #305's stale/unwritten-register family | **Falsified.** #1585 records `MODE=1` occurring *zero* times in an entire Astro frame; here it is dominant 7:1. Do not correlate these investigations on that basis. | #1606 |
| The `[agc] WaitRegMem … dependency violated` spam is the loudest lead | **Falsified — it is a print-volume artifact.** The log site is rate-limited (first 40 lines, then every 1,024th) and the `#N` in the message is that counter, so the true total is readable off the log: **18 and 21 unsatisfied waits for entire runs**, spread evenly (~1.5/s at ~50 fps, roughly one per 35 frames), never reaching `#1023`. The code comment at the site already says an unsatisfied wait is normal, handled state. Recorded as instrument trap #13 in `GAME_COMPAT_ORCHESTRATION.md`. | #1606, PR #1638 |
| The base pass is decoded but attributed to a different target, MRT slot, or writer class | **Falsified, positive-controlled.** Arm A: `PROSPER_GPU_CAPTURE_MIN_DRAWS=60` over 30 s (~1,500 frames) produced **no capture**. Arm B (control): `MIN_DRAWS=20` over 20 s produced a 185,766,978-byte capture, proving the mechanism fires. The filter counts *decoded* draws (`semantic_draw_count`), before realization, so an all-suppressed submit still passes it. **No submit in the run decodes 60 or more draws.** The silent-drop paths were audited too: `retained_draw_selected` cannot silently drop, zero `indexed indirect draw skipped` lines, and the only capped drop message is the unfilled-placeholder `#1264` family. | #1606 |
| Prosper is failing to walk some submission path — a second queue/DCB, an IB2 chain, an unimplemented submit entry point | **Falsified by four independent measurements.** (1) `[agc] SubmitDcb` accounting over 1,700 submits / ~884 frames: 3,599,703 dwords produced 582,171 applied packets = **6.18 dwords per applied packet**, an ordinary PM4 density with **no unconsumed remainder**; an entire 4K frame arrives as ~4,310 dwords (~17 KB of PM4). (2) **Zero** `[pm4] unknown raw type-3 opcode` lines in the retained full boot log. (3) `PROSPER_PROGRESS_UNIMPL=1` reports 17 unimplemented functions, **not one of them AGC, GPU or submit related** (Coredump/Posix/Http/NpWebApi2/Net/SaveData telemetry only); both known submit entry points are implemented and exercised. (4) Compute flows normally through the same machinery — 7,629 dispatches over 884 presents. **Prosper decodes 100% of what the guest submits; the guest is not issuing the base pass.** CONFIDENCE: HIGH. | #1641 |
| `~23 draws/frame` is implausibly few (an assertion) | **Now a measurement, against a calibrated control.** Blue Prince — the only title in the comparison that demonstrably renders a 3D world — shows both regimes in one run, same build, same instruments: **7–13 draws/frame on its own menus, 1,500–3,200 in its 3D scene.** Oregon Trail sits flat at 22.4 (peak 23) for an entire run while running a gameplay-scale post chain. Read the trap first: a whole-run average describes neither regime (Blue Prince's is 622.7). | #1641, PR #1645 |
| The same signature confirms the defect on Nikoderiko or Asterix | **Neither is currently in a phase that can test it.** Nikoderiko's 35.9 draws/frame (peak 53) was measured **parked on the title screen and EULA** — a 2D UI screen — and a working title's own menus measure 7–13, so 53 is *normal*, not a signature; its rise lands exactly when the rich screen appears (samples 10–11 carry 160,300 and 161,248 distinct colours, samples 3–9 are uniformly black). Asterix `PPSA30490` sits at exactly 4.00 draws/frame (3 realized + 1 suppressed), dead flat across 14,472 frames, never reaching any content phase — a title that never starts rendering, a different failure. **Do not widen #1641 to those titles on this evidence.** | #1641 |
| Headless ErrorDialog's immediate `FINISHED` value is itself the wrong completion value | **Falsified by exact control flow.** At `0x903041` the guest compares `UpdateStatus` against `FINISHED` (`3`); equality terminates the dialog, while every other state returns to wait. The account-specific poll at `0x88ad6b` returns only for `RUNNING` (`2`); any other result reaches `0x88ad75`, writes `0x0100` at object offset `+0x108` (clearing active and setting account-checked at `+0x109`), then invokes and clears its callback. `FINISHED` is therefore the exact advancing value **if the account poll is dispatched**. The remaining defect is dispatch/lifecycle, not the value. Earlier text named DOLL as a cross-title ErrorDialog-Open positive control; retract that claim: DOLL validates the honest offline NP branch, but its retained run explicitly says no ErrorDialog Open-class NID fired. | #1606; `DOLL_LOADING_PROGRESSION.md` |
| The retained ErrorDialog Open line proves the game-specific account poll at `0x88ad20` ran | **Falsified; it proves a different owner.** The Open import has one direct guest caller, `0x901ed8`, in a Sony online-identity virtual method inherited unchanged by the PS5 identity class. A Vulkan-free live positive at `0x901edd` fired once and its frame recovered the exact return site `caller_rbp=eboot+0x88ae5a`, immediately after `ShowAccountLogin`'s virtual call. Dialog completion is polled separately by the Sony online-subsystem tick (`0x97f9e0` → `0x9036a0` → `0x903020`). The earlier zero at `0x88ad70` remains void, and the static `0x88ad20` semantics remain conditional; neither says that function executed. | #1606; #1932 |
| The Sony online-subsystem ErrorDialog tick directly completes the game account gate | **Falsified by the exact delegate flow.** `ShowAccountLogin` passes an empty delegate to the identity method at `0x901de0`; that method stores it at identity offset `+0x60`. The generic tick at `0x903020` correctly terminates a `FINISHED` dialog and invokes slot `+0x58` only when that stored delegate is non-empty. It is empty on this path, so the Sony tick cannot write account byte `+0x109`. The separate account poll at `0x88ad20` owns that write and is reached from foreground return (or a signed-in NP-state notification), not from the generic dialog callback. | #1606 |
| `L_GameloftSplash` directly calls `ShowAccountLogin`, so its Blueprint bytecode should expose the missing call | **Falsified by the retained package import table.** The level imports `CheckNoFreeSpace`, `IsAccountLoginChecked`, `IsAccountLoginRequired`, the EULA predicates/setter, `Begin`, `Create` and `AddToViewport`, but no `ShowAccountLogin` or account-init function. Account initialization is an automatic native stage (`0x87d100` → `0x88abe0`); the Blueprint only polls its checked predicate. | #1606 |
| The account completion writer runs but rejects Prosper's dialog status | **Falsified, positive-controlled.** On the same executable, a clean software-breakpoint control at `ShowAccountLogin` (`0x88ae00`) fired once and its first recovered return was `0x88ac99`, directly inside the account-state method. A second clean arm at the writer (`0x88ad75`) printed the same no-compute witness, armed the same instrument, logged ErrorDialog Open/auto-`FINISHED`, and advanced for more than 12 seconds with zero hits. The status comparison is never reached; changing `FINISHED` cannot repair an undispatched poll. | #1606 |
| A foreground-return pulse changes the lever but still does not dispatch the account writer | **Falsified twice, with exact event order.** The opt-in discriminator and the production-default build each logged ErrorDialog Open, background `1`, foreground `0`, then exactly one breakpoint hit at writer `0x88ad75`. The latter hit's stack includes platform transition site `0x191c324` and multicast `0x867af10`, proving the expected broadcaster owned the hit. The first arm ran for its 16-second bound; the second later faulted on a separate worker, after all mechanism evidence was already recorded. | #1606 |
| The asset load trace proves the guest reached `Create` / `AddToViewport` on the splash and EULA widgets | **Falsified by a control arm — those loads are package dependencies, not execution.** `tools/re/pak_index.py --distinct` over the shipped-master run resolves `WBP_Splash_Gameloft`, `BP_EULA_Controller`, `WBP_System_Disclaimer_Popup`, `WBP_EULA_NotificationScroll_Popup` and the `T_Logo_*` atlas — and the **pre-#1933 control arm loads the identical set at the identical ordinals** (lines 2190-2239 of both traces) over a run **12x longer**, while never reaching the popup. `L_GameloftSplash.umap`'s import table pulls those classes when the map loads, before any Blueprint gate runs. What *does* discriminate is `FF_Exo2_Bold.ufont`, `FF_PressStart2P_Condensed.ufont` and `icudt64l/brkitr/line.brk`: Slate loads a font and the ICU break iterator only when it lays text out, and only the shipped-master arm loads them (3 assets present in A and absent from B; B's 486 extra assets are streamed `.vaff` audio from its longer run). **A class load is not an instantiation.** Any load-trace progression claim needs a duration-matched or longer control. | this doc, #1606 |
| The 4K `R16G16B16A16_SFLOAT` "surface with no producer" is a renderer or base-pass defect | **Resolved, not by GPU work.** It was the scene target of a frame whose only content is a UMG popup, in a process still gated on account login. With the gate open the same code path renders 4,009,938 non-black pixels. Do not reopen the producer question. | this doc, #1933 |
| `PROSPER_FAULT_SKIP` over the corrupt free-list pop is a usable workaround for #1945 | **Falsified.** `PROSPER_FAULT_SKIP=0x14600df:0x14600f3` (the allocator's own general-allocator fallback) does survive the pop, but the run then wedges — `frame_seq` frozen at 223 from 6 s to 33 s, frames black — and dies in `std::terminate`'s `ud2` at `0x5c00359af` on a worker. The corruption is upstream and must be fixed at its source. | #1945 |
| #1944 (the reserved-range lazy-commit handler zero-filling a guest page) is the same bug as #1945, or is what corrupts the free list here | **Falsified on this title.** `[lazy-commit]` is printed unconditionally by `exec_image_linux.cpp`, so every run reports it. Across **ten** `PPSA19244` arms on 2026-08-05 — three default render arms on `ff72e77c`, three with the mutex-encoding fix, four with `PROSPER_FAULTMEM`/`PROSPER_PEEK` — the count is **0 in every single one**, including the arms that faulted at `eboot+0x14600df` / `eboot+0x146027d` on `0x30016000`. The handler never fires on this title, so it cannot be the writer. This agrees with the third ArcRunner row already recorded there (its `0x30016000` arm also had no lazy-commit line, while both `addr=(nil)` arms did). Do not merge the two issues. | #1945, #1944 |
| The `std::system_error("invalid argument")` / `std::terminate` face is guest heap corruption, or the same defect as #1945 | **Falsified — it is prosper's own error encoding, and it is fixed.** `scePthreadMutexTrylock` returned the bare FreeBSD `EBUSY` (16); the shipped `libc.prx` `_Mtx_trylock` (export `k6pGNMwJB08` at `libc.prx+0x5ef0`) compares the result against `0x80020010` and maps anything else non-zero to Dinkumware `_Thrd_error` (4), which `std::_Throw_C_error` turns into an uncaught `std::system_error`. Encoding the Sony spellings flipped the named check `Terminating due to uncaught exception 'invalid argument: invalid argument'` from **3/3 present to 0/3**, in the same binary and environment, and the title went from a black frame at frame 5 to the Gameloft splash and the health warning. `_Mtx_lock` (`0x8002000b`) and `_Mtx_timedlock` (`0x8002003c`) carry the same contract; `_Mtx_unlock` and `_Cnd_wait` discard the result entirely. | #1945, #1612 |
| PlayGo `GetLocus` is a sustained chunk-0 poll or progression gate | **Falsified; the earlier rate interpretation was an instrumentation mistake.** The retained trace's 1,000 calls carry chunk IDs **0, 1, 2, …, 999 exactly once**, with no reset. Guest code at `0x13f28a4` is a bounded startup cache-population loop: it calls `GetLocus` for each ID, stores only successful `LOCAL_FAST` (`3`) results, cleanly skips absent IDs, stops at `0x3e8`, and sets its initialized byte at `0x13f29e7`. Chunk 0 succeeds and the remaining nonexistent IDs fail as expected. This is neither periodic polling nor a wait for those IDs to become local; changing PlayGo behaviour is not justified. | #1606, #1641 |
| #1946's solid-block glyphs are a Slate **font-atlas** defect — a blank/unpopulated atlas, wrong glyph UVs, a single-channel format/swizzle, or a mip-tail tiling error | **Falsified — the atlas is correct and was read by eye.** `PROSPER_RESOURCE_HASH_DIM=2048x2048 PROSPER_DUMP_RESOURCE_VERSION=1` dumped the decoded 2048x2048 UI atlas at `0x30252f0000` (`[tdump]`: `fmt=130` = `8_8_8_8_SRGB`, `tile=9`, DST_SEL `BGRA`); the image is crisp and readable — the *"The Oregon Trail"* logo, the Gameloft and HarperCollins/HMH marks, the ornamental divider. Half its texels are content (`rgb_nonblack = alpha_nonzero = 2,138,372` of 4,194,304), so its alpha is real and prosper decodes it. `PROSPER_TEXCOMMIT=1` also shows the title samples **no single-channel texture at all** — every sampled surface is `f9` (`Unorm8`) at 4 B/texel. The defect was lost **alpha blending**, not lost coverage. | #1946 |
| Every UE4 title with UMG text is exposed to the same defect (the #1946 issue body's premise) | **Falsified by a same-build positive control.** *Dragon Quest VII Reimagined* (`PPSA17942`, UE4) renders anti-aliased Slate text and alpha-blended logos correctly on the identical binary, and `PROSPER_RTTLOG=1` records **2,069 blend-enabled draws** for it against **0 of 1,231** for Oregon Trail. The exposure is not "UE4 + UMG" but "**asks `sceAgcGetRegisterDefaults2` for an SDK version below 13 and uses render target 0 blending**" — DQ asks for 13 and was never affected. | #1946 |
| Offering the RT0 blend key on every SDK version changes the other titles that are on the pre-13 path | **Falsified by a same-binary pre/post pair.** Ten titles were surveyed with the `[agc] register defaults requested for SDK version N` line: only *Dead Cells* (10), *Blasphemous 2* (10) and Oregon Trail (12) ask below 13; every other title asks for 13 and is bit-for-bit unaffected. The two version-10 titles were then run against a build carrying master's pre-fix table and against the fixed one, and their `CB_BLEND0_CONTROL` value **distributions are unchanged** — Dead Cells 43.96/39.01/17.03 % → 43.63/39.09/17.28 %, Blasphemous 2 46.00/27.31/15.44/10.77/0.47 % → 45.35/27.28/15.86/10.80/0.71 % over the same five values. Both already program `ENABLE=1` blend on RT0 through another path in their SDK-10 builder, so the key is inert for them. Their title screens were opened on the fixed build and match their committed captures. **A static scan for the key constant sharpens this rather than replacing it, and rules out the tempting shortcut explanation:** `0xA6D12629` is *present* in both — one `cmp`-immediate site in Dead Cells' `eboot.bin` (file offset `0x174de69`) and two in Blasphemous 2 (`eboot.bin` `0x3a9963`, `sce_module/libSceJobManager.prx` `0x6b73`) — so "those titles never search for the key" is **false** and was never why the change is inert for them. (The scan reads a known answer correctly: it lands Oregon Trail's own site at `0x4ab2e03`, inside the `cmp` at the documented `eboot+0x4ab2e00`.) The measured distributions above remain the evidence: their RT0 blend does not come from this record. | #1946 |
| `PROSPER_DUMP_ATLAS=1` producing no font atlas means the title never sampled one | **Void as evidence — the diagnostic is doubly filtered.** It writes only textures with `tw <= 2048 && th <= 1024` **and** a decoded `VK_FORMAT_R8G8B8A8_UNORM`, so a 2048x2048 atlas and every natively-uploaded `R8_UNORM` surface are silently skipped, and its 24-bit BMP output drops alpha entirely — the channel this investigation was actually about. Use `PROSPER_TEXCOMMIT=1` to enumerate what is sampled and `PROSPER_RESOURCE_HASH_DIM` + `PROSPER_DUMP_RESOURCE_VERSION` to dump it. Note `PROSPER_DUMP_ATLAS` also switches the renderer onto the CPU RTT-copy diagnostic path, so it is not a neutral observer. | #1946 |

**Note on the Asterix figure.** 4.00 is the #1641 census number — *decoded* draws per frame, realized
plus suppressed, over the whole run. A separate Asterix plugin-link A/B, run while investigating the
Tales of Graces `DllNotFoundException` family (#1609), reported an unchanged draws-per-frame count on
both arms using a different instrument at a different phase, and is the basis for "Asterix's black
frame is **not** a silent `DllNotFoundException`". **That A/B is not in the project record** — it
exists only in a report to an orchestrator and an unmerged branch — so it is recorded here as an
outstanding claim, not as a falsification. Whoever lands the plugin-autolink work should re-run it and
file the numbers on #1599, stating which quantity was counted; do not assume the two figures measure
the same thing, and do not treat either as superseding the other until that is checked.

### Corrections to the record

* The two unrealized operations in the retained capture are **draw indices 0 and 1**, not `draw[2]`
  (the 1920x1080 HDR scene colour) and `draw[3]` (the 1x1 exposure) — both of those **realized**. For
  a `Draw` operation, `source_index` is `DrawItem::draw_index`, not an array slot, and `--inspect`
  prints that same `draw_index` while `item=` is the array slot. The captured list runs
  `draw[2] item=0` … `draw[22] item=20` with no gaps, and the header's `draws=21 computes=2
  operations=25` confirms the arithmetic. Nothing about the dropped operations can be inferred from
  `draw[2]`'s or `draw[3]`'s state. (#1606)
* The reason each draw failed was **discarded** before the capture was written: `realize_retained_draw()`
  had no `OperationRealizationFailure*` parameter while `realize_retained_compute()` did. Fixed by
  #1636 / PR #1644 — pre-fix captures carry empty `Unknown` records and cannot be re-read for this.

## Instrument notes

* `PROSPER_COMPUTELOG`, `PROSPER_EXECLOG`, `PROSPER_GFXLOG`, `PROSPER_REGWATCH`,
  `PROSPER_RENDER_TIMING`, `PROSPER_RTT_TIMING` and `PROSPER_PROVENANCE_DIM` are **not** in the
  `live_gpu_targets` disable list (`live_renderer.cpp:1008-1015`), so none of them forces the CPU readback
  RTT path. `PROSPER_RTTLOG`, `PROSPER_RESOURCE_HASH_DIM` and `PROSPER_GPU_CAPTURE` **do** — and
  `PROSPER_RTTLOG` therefore cannot observe the compute→renderer-RTT mirror path at all.
* `PROSPER_REGWATCH` reports at command-processor **decode** time while `PROSPER_EXECLOG` reports at
  **realization** time, and they interleave per submit. Joining them naively produces an apparent
  future-read for 1,049 of 1,050 draws, which is an artifact. Join on the draw's own `command_order`
  (added by #1633) and take the last watched write with order ≤ the draw's.
* `PROSPER_GFXLOG` re-decodes and dumps every packet after each submit, so it is far too slow to run
  against a title submitting thousands of draws per frame.
* The retained capture replays byte-exactly (`hash=ccc433ff6d980383`) but contains **only realized
  draws**, so the dropped base pass cannot be studied offline from it. That work had to be done live.
* A finite startup enumeration reported over a long, instrumented wall-clock interval is not a call
  rate. The 1,000 PlayGo `GetLocus` records are IDs 0 through 999 from one bounded loop, not a
  sustained ~17 calls/s chunk-0 ticker. Always inspect arguments and sequence boundaries before
  converting a trace count into frequency.
* The first live CPU discriminator on 2026-08-04 is **VOID, not a falsification**. In a single
  bounded 15-second arm, the all-thread HWBP at `0x995ed0`
  (`GameloftOnlineFeaturesSubsystem::Begin`) and the process-wide BP at `0x34cba70`
  (`UUserWidget::AddToViewport`) both armed, but neither fired. Because the promised `Begin` positive
  control was also zero, the run cannot distinguish a missed temporal window or wrong native owner
  from an unexecuted UI path. It must not be cited as evidence that either function is unreachable.
  That arm also omitted `PROSPER_RENDER` but not `PROSPER_NO_COMPUTE`, so boot_trace registered the
  Vulkan compute backend; future CPU-only arms require `PROSPER_NO_COMPUTE=1` **and** the
  `[compute] progression-only no-op backend registered` witness before interpreting any hit count.
* The corrected 2026-08-04 account-dialog arm is also **VOID, and corrects an owner conflation**.
  `PROSPER_NO_COMPUTE=1` printed the required progression-only banner, the title emitted
  `sceErrorDialogOpen(errorCode=0x80550006)`, and the HWBP/BP at `0x88ad70`/`0x88ad75` both armed;
  however, neither breakpoint fired. The `0x88ad70` positive was immediately after one
  game-specific `UpdateStatus` call, so the adjacent account-checked writer's zero is not
  interpretable. Offline owner recovery explains why: the ErrorDialog Open import has one direct
  guest caller, `0x901ed8`, in the Sony online-identity base method at `0x901de0`. The base vtable
  (`0x7d4a438`) and its PS5-derived vtable (`0x7d51100`) both put that method at slot `+0x10`;
  the PS5 constructor at `0x972960` installs the latter. The game's `ShowAccountLogin` method
  (`0x88ae00`) obtains the identity interface and calls that exact slot at `0x88ae57`. Dialog
  completion is polled independently by the Sony online-subsystem tick: `0x97f9e0` calls
  `0x9036a0`, which obtains the same interface and calls the `0x903020` UpdateStatus/Terminate path.
  Thus the retained Open line proves `0x901de0`, not `0x88ad20`; the static statement remains valid
  only conditionally — **if** `0x88ad20` runs, any non-`RUNNING` result writes checked=true.
* The bounded owner discriminator resolved that owner, but also exposed a generic instrument bug.
  Exact process census was zero before and after; `PROSPER_NO_COMPUTE=1` printed the progression-only
  banner and no Vulkan-device line. The genuine all-thread HWBP at `0x901edd` fired once and its own
  frame recorded `caller_rbp=eboot+0x88ae5a`, directly proving the observed Open returned to
  `ShowAccountLogin`. The simultaneous int3 at `0x88ae5a` is **not** a second confirmation: while an
  HWBP fd was live, the Linux handler misrouted that int3 as a second `[hwbp]` event at `0x88ae5b`,
  skipped the software-breakpoint restore/step state machine, and the subject then aborted with heap
  corruption. Therefore the first pre-int3 HWBP owner record is valid, but the int3 record and every
  progression outcome are void. Tracked as #1932 and instrument trap 84; do not combine
  `PROSPER_HWBP` with `PROSPER_BP` until that routing defect is fixed.
* The replacement account pair used **one instrument at a time** on the exact same executable.
  Both arms had an exact zero process census before launch, printed
  `[compute] progression-only no-op backend registered`, printed no Vulkan-device line, and had a
  final zero census. The positive `PROSPER_BP=0x88ae00` arm hit `ShowAccountLogin` once; its first
  recovered return was account-state call site `0x88ac99`. That arm later faulted, so only the
  already-recorded pre-fault BP hit is usable. The negative `PROSPER_BP=0x88ad75` arm logged
  ErrorDialog Open/auto-`FINISHED` and continued through guest timestamps past 12 seconds without a
  hit. Its outer distrobox timeout left the exact child alive; the mandatory post-run census caught
  and terminated it before the title-run gate was released. That wrapper behaviour is an instance
  of instrument trap 66, not title evidence.
* The lifecycle fix used two additional clean, single-instrument CPU-only arms. Both had exact zero
  pre/post process censuses, printed the progression-only no-compute banner, printed no Vulkan-device
  line, and logged ErrorDialog Open → background `1` → foreground `0` → writer `0x88ad75`. The
  opt-in discriminator reached its 16-second timeout. The production-default arm recorded the same
  ordering and broadcaster stack, then a different worker faulted through null `rip` and the subject
  exited `90`. Thus the lifecycle mechanism is reproduced `2/2`, while subsequent stability is
  explicitly only `1/2` and no later content milestone is claimed.
* Unit coverage exercises both ownership models: headless Open emits one bounded enter/return pair;
  a backend-owned ErrorDialog remains backgrounded while its frontend reports `RUNNING` and returns
  foreground after `FINISHED`. The accepting backend is pinned with the registry lease; unregister
  or replacement abandons safely to `FINISHED`, never calls the stale/replacement backend, and cannot
  remain backgrounded. Four defect-shaped mutations are named: collapsing only the headless edge
  fails `headless ErrorDialog exposes a background sample`; making the first backed sample advance
  directly to return fails `backend ErrorDialog persists background across repeated RUNNING
  samples`; leasing the replacement backend fails the abandonment/reroute check; and leasing the
  replacement on direct Term fails the close-reroute check. Restoration passed both `platform_ui`
  and `service_getters`.

* The `[fault] … on-guest-TCB=NO(host-%fs leak?)` line printed at the #1945 fault is **not** evidence of
  a `%fs` leak. That heuristic assumes `rax` still holds the `%fs` self-pointer from a nearby
  `mov %fs:0x0`; at `eboot+0x14600df` `rax` is the guest allocator's per-thread bin-array base
  (`0x30209c0000` / `0x3020140000` in the two observed instances). Do not open a #1155-class
  investigation from it without an independent `%fs` witness.
  **The independent witness now exists** (#2018): the line reports `guest_fs_to_host_scoped()`'s
  return, which is this thread's real `%fs` base and is returned only after `TCB_MAGIC` is verified
  there, so `on-guest-TCB` is a fact rather than an inference and `rax` is printed beside it as raw
  evidence carrying no claim. Measured on Crisis Core (`PPSA07809`), the same `RenderThread 1` fault
  that used to print `NO(host-%fs leak?)` now prints `on-guest-TCB=yes`. Reports from **before**
  #2018 still carry the old heuristic and must be read the way this entry says.
* **A guest `std::terminate` is worth one disassembly pass before it is treated as heap damage.**
  For an *uncaught* exception the C++ runtime calls `std::terminate` from `__cxa_throw` **without
  unwinding**, so prosper's own frame-pointer walk at the `ud2` still carries the throwing stack.
  Here that walk named `eboot+0x4ca716c`, three instructions of disassembly identified the shipped
  "throw unless the result is 0 or `ignore`" helper, and the two PLT slots resolved through
  `self_dump`'s `DT_JMPREL`/`DT_PLTGOT` to `_Mtx_trylock` and `std::_Throw_C_error`. That is the
  whole distance from "the guest terminated" to "prosper returned the wrong error encoding" — no
  live instrument, no re-boot. The recipe is reusable: read the PLT thunk's
  `jmp QWORD PTR [rip+…]` target, subtract `DT_PLTGOT`, `/8`, `-3` for the relocation index, and
  index `DT_JMPREL` to get the NID, then resolve it in `../PS5-3.20_Libs/`.
* `PROSPER_PEEK` produced **no output** on the `WORKER-THREAD FAULT` path in four armed arms. It is
  consumed by the primary-thread fault report only, so a zero from it on a worker fault is void, not
  a negative. `PROSPER_FAULTMEM` does print on the worker path.
* In a 5-second arm that provably decoded and realized draws (32 `CB_COLOR_CONTROL` diagnostics, a
  folded `WaitRegMem`, and rendered popup frames in the sibling arms), `PROSPER_RTT_TIMING=1` emitted
  **zero** `[rtt-timing]` lines. If a draw census depends on that stream, confirm it emits inside your
  window before reading a zero as "no draws".

## Reproduction

**The blend-state measurement (#1946), and how to re-take it on any title.** Three cheap instruments,
each of which detects its own invalidity, in the order they were useful:

```bash
# 1. Which SDK version does the title ask for?  One line, printed unconditionally at AGC init.
#    Below 13 was the exposed path before #1946.
… ./build-linux/screenshot <DUMP_ROOT>/<TITLE>-app0 …    # grep '[agc] register defaults requested'

# 2. Is any draw blending at all?  Per-draw enable + factors + every sampled texture.
#    Count with a form that PRINTS a zero instead of exiting non-zero on one — see below.
PROSPER_RTTLOG=1 … > rttlog.txt 2>&1            # bound with PROSPER_RTTLOG_{MIN,MAX}_SUBMIT
grep -o 'blend=1(' rttlog.txt | wc -l           # blend-enabled draws  (0 prints as 0)
grep -o 'blend=[01](' rttlog.txt | wc -l        # total draws — the denominator

# 3. What did the guest actually write to the blend register, and by which path?
PROSPER_REGWATCH=Cx:0x1E0,Cx:0x1E1 …   # 0x1E0 = CB_BLEND0_CONTROL, 0x1E1 = CB_BLEND1_CONTROL
```

**Do not count this with `grep -c`.** Zero is the *expected* answer for the defect being measured, and
`grep -c` exits 1 on zero matches — so in an `&&` chain it aborts the run that was supposed to report
the finding, and its "failure" is indistinguishable from a real one (`CLAUDE.md`). `grep -o … | wc -l`
prints `0` and exits 0. Take the denominator from the same file in the same way: a bare enabled-count
is not interpretable without the total. Match the **trailing `(`** as shown: the per-draw RTTLOG line
is the only one that prints `blend=N(src=…` (`live_renderer.cpp:4739`), while a bare `blend=N` also
appears on the `PROSPER_GFXLOG` item line and in the target-step summary — so an unanchored
`blend=1` over a log with either of those enabled counts lines, not draws.

**`PROSPER_RTTLOG` is itself in the `live_gpu_targets` disable list**
(`live_renderer.cpp:1008-1015`), so arming it forces the CPU-readback RTT path for the whole run. Two
consequences: it cannot observe the compute→renderer-RTT mirror path at all, and its run is not
frame-for-frame or timing-comparable with a default run — so use it to compare **two RTTLOG arms**
(before/after, or subject/control), never an RTTLOG arm against a default one.

A `blend=1` count of zero is only evidence when a **positive control** on the same binary produces a
non-zero one: *Dragon Quest VII* (`PPSA17942`) is the calibrated control used here — 2,069 against
Oregon Trail's 0. And to attribute a register value to prosper rather than to the guest, change the
value in `src/hle/agc_reg_defaults.cpp` to a marker and re-run `PROSPER_REGWATCH`: if the marker
appears in the register, prosper's table is the source. That is what proved this one.

**Whether a title searches for the key at all is a static question**, answerable without booting it:
the search is a plain `cmp DWORD PTR [reg], imm32` against the key, so the 4-byte little-endian
constant appears verbatim in the module that contains the builder. `29 26 d1 a6` is `0xA6D12629` and
`56 03 55 ef` is `0xEF550356`. Validate the scan on a known answer before trusting it — on
`PPSA19244-app0/eboot.bin` the first lands at file offset `0x4ab2e03`, i.e. inside the `81 3c 3e …`
at the documented search site `eboot+0x4ab2e00`.

**Current — the popup, and the fault that follows it (2026-08-04, master `9dcb6c4b`):**

```bash
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 \
  ./build-linux/screenshot <DUMP_ROOT>/PPSA19244-app0 \
      --seconds 2 --count 8 --out <OUT> --timeout 60
```

Samples at 2.0 s and 4.0 s carry the legal popup (3840x2160, 32 distinct colours, 4,009,938 non-black
pixels); the run then usually dies with `WORKER-THREAD FAULT … rip=…+0x14600df` and exit 90 (#1945).
Sample every 1-3 s: a 5-second interval is longer than the whole run in most arms. The route
`scripts/oregon-trail/reach-past-eula.pad` is committed for when #1945 is fixed — it has **not** yet
been observed to act, because the crash lands before its first press.

**Historical — the black-frame draw census (pre-#1933):**

```bash
PROSPER_NO_FRAME_DUMPS=1 PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 \
PROSPER_RENDER_TIMING=1 PROSPER_RTT_TIMING=1 PROSPER_EXECLOG=1 \
  ./build-linux/screenshot <DUMP_ROOT>/PPSA19244-app0 --seconds 10 --count 2 --out <OUT> --timeout 250
```

Count decoded draws per **sample interval** — sum `draws=N` from `[rtt-timing]`, count
`[exec] skip draw`, split at each `[shot]` line, divide by that interval's frame delta. Averaging over
a whole run hides the phase transition that matters.
