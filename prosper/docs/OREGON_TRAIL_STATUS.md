# The Oregon Trail (`PPSA19244`) — status and evidence

Unreal Engine 4. **Rung 0 — no real graphics.** The title boots cleanly and holds a steady ~50 fps
frame loop with frames provably advancing, but every one is uniformly black. Tracked on
[#1606](https://github.com/mattias800/prosper/issues/1606) and
[#1641](https://github.com/mattias800/prosper/issues/1641).

Read `## Ruled out` before forming a hypothesis. Several separate leads on this title have been
measured and killed, including the GPU-side questions and the two service candidates that the issue
body originally called highest-value.

## Where it actually stands

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

**The GPU-side questions are answered. The remaining frontier is CPU/UI-side:** the signed-out
account path opens and auto-finishes an ErrorDialog, but its account-completion poll is tied to a
system foreground-return callback. Prosper currently reports no background transition at all. Prove
that lifecycle edge (or its absence) with a clean, single-instrument arm before changing service
behaviour, then distinguish EULA/splash `AddToViewport` from a UMG presentation failure. Do not
return to base-pass or further GPU instrumentation without contradictory new evidence.

## Shipped front-end progression contract

Offline package and guest-code analysis identifies the exact startup contract; this is stronger than
inferring progression from service-call volume:

| Blueprint predicate/action | Native implementation | Completion state |
|---|---|---|
| `IsAccountLoginRequired` | `0x88af00` | Always required. |
| `IsAccountLoginChecked` | `0x88aef0` | Reads byte `+0x109`. The ErrorDialog poll at `0x88ad6b` writes the two-byte value `0x0100` at `+0x108` on every non-`RUNNING` result, making `+0x108 = 0` and **`+0x109 = 1`**. The missing discriminator is whether that poll is dispatched after Prosper's headless dialog. |
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
`sceSystemServiceGetStatus` currently writes byte 5 as zero on every call, so an auto-finished system
dialog cannot itself produce the required nonzero-to-zero edge. This is the current, bounded
candidate—not yet a justified behaviour change.

## Ruled out

One line per dead hypothesis, the evidence that killed it, and where that evidence lives. Do not
re-derive these without contradictory new evidence.

| Hypothesis | Verdict and evidence | Source |
|---|---|---|
| The `R11G11B10F` compute write-back notifier gap (Syberia #1619) explains the black scene here | **Falsified.** The precondition holds — the scene-colour target really is `R11G11B10_FLOAT` — but the notifier is only reached from `live_compute.cpp` under `bi.mirror_result_to_imported`, i.e. when a dispatch writes a *renderer-owned* storage image. A validated live trace with `PROSPER_COMPUTELOG=1` (which does **not** clear `live_gpu_targets`) produced **11,440 `[compute]` lines and zero** `mirrored storage result into renderer RTT` lines; every storage image in the run logs `renderer_owned=0`. **The notifier is never invoked by this title.** CONFIDENCE: HIGH. | #1606 |
| Prosper's `CB_COLOR_CONTROL` tracking is stale, unwritten, or folds wrongly, and the `MODE` decode is wrong | **Falsified — the tracking is exact.** `PROSPER_REGWATCH` over ~14 s records **12,390** `CB_COLOR_CONTROL` writes and 11,358 `CB_TARGET_MASK` writes, all on the indirect (`Set*RegsIndirect`) path. `MODE=1` (`CB_NORMAL`) is the **dominant** value at 9,346 writes versus 1,335 `MODE=0`, a 7:1 ratio. Joined correctly against each draw's own `command_order`, all **1,050 of 1,050** suppressed draws resolve exactly the `CB_COLOR_CONTROL` and `CB_TARGET_MASK` the guest wrote at or before that draw — **zero mismatches**. The guest deliberately programs `CB_DISABLE` on these draws, immediately before each one. The decode is sound in the other direction too: the same title's post-process draws decode `MODE=1` and render. **Do not force `MODE=1`** — it is not a fix, and `PROSPER_FORCE_COLORWRITE=1` leaves the frame byte-identically black. **Amended by #1724:** this row's finding stands (the decode is exact and the guest deliberately programs `CB_DISABLE`), but prosper no longer *acts* on `MODE` when deriving the colour write mask — the mask comes from `CB_TARGET_MASK & CB_SHADER_MASK` alone. If this title's suppressed draws carry a non-zero mask they now write; re-measure before reasoning from the old suppression counts. | #1606 |
| This is Astro Bot's colour-state signature (#1585), or #305's stale/unwritten-register family | **Falsified.** #1585 records `MODE=1` occurring *zero* times in an entire Astro frame; here it is dominant 7:1. Do not correlate these investigations on that basis. | #1606 |
| The `[agc] WaitRegMem … dependency violated` spam is the loudest lead | **Falsified — it is a print-volume artifact.** The log site is rate-limited (first 40 lines, then every 1,024th) and the `#N` in the message is that counter, so the true total is readable off the log: **18 and 21 unsatisfied waits for entire runs**, spread evenly (~1.5/s at ~50 fps, roughly one per 35 frames), never reaching `#1023`. The code comment at the site already says an unsatisfied wait is normal, handled state. Recorded as instrument trap #13 in `GAME_COMPAT_ORCHESTRATION.md`. | #1606, PR #1638 |
| The base pass is decoded but attributed to a different target, MRT slot, or writer class | **Falsified, positive-controlled.** Arm A: `PROSPER_GPU_CAPTURE_MIN_DRAWS=60` over 30 s (~1,500 frames) produced **no capture**. Arm B (control): `MIN_DRAWS=20` over 20 s produced a 185,766,978-byte capture, proving the mechanism fires. The filter counts *decoded* draws (`semantic_draw_count`), before realization, so an all-suppressed submit still passes it. **No submit in the run decodes 60 or more draws.** The silent-drop paths were audited too: `retained_draw_selected` cannot silently drop, zero `indexed indirect draw skipped` lines, and the only capped drop message is the unfilled-placeholder `#1264` family. | #1606 |
| Prosper is failing to walk some submission path — a second queue/DCB, an IB2 chain, an unimplemented submit entry point | **Falsified by four independent measurements.** (1) `[agc] SubmitDcb` accounting over 1,700 submits / ~884 frames: 3,599,703 dwords produced 582,171 applied packets = **6.18 dwords per applied packet**, an ordinary PM4 density with **no unconsumed remainder**; an entire 4K frame arrives as ~4,310 dwords (~17 KB of PM4). (2) **Zero** `[pm4] unknown raw type-3 opcode` lines in the retained full boot log. (3) `PROSPER_PROGRESS_UNIMPL=1` reports 17 unimplemented functions, **not one of them AGC, GPU or submit related** (Coredump/Posix/Http/NpWebApi2/Net/SaveData telemetry only); both known submit entry points are implemented and exercised. (4) Compute flows normally through the same machinery — 7,629 dispatches over 884 presents. **Prosper decodes 100% of what the guest submits; the guest is not issuing the base pass.** CONFIDENCE: HIGH. | #1641 |
| `~23 draws/frame` is implausibly few (an assertion) | **Now a measurement, against a calibrated control.** Blue Prince — the only title in the comparison that demonstrably renders a 3D world — shows both regimes in one run, same build, same instruments: **7–13 draws/frame on its own menus, 1,500–3,200 in its 3D scene.** Oregon Trail sits flat at 22.4 (peak 23) for an entire run while running a gameplay-scale post chain. Read the trap first: a whole-run average describes neither regime (Blue Prince's is 622.7). | #1641, PR #1645 |
| The same signature confirms the defect on Nikoderiko or Asterix | **Neither is currently in a phase that can test it.** Nikoderiko's 35.9 draws/frame (peak 53) was measured **parked on the title screen and EULA** — a 2D UI screen — and a working title's own menus measure 7–13, so 53 is *normal*, not a signature; its rise lands exactly when the rich screen appears (samples 10–11 carry 160,300 and 161,248 distinct colours, samples 3–9 are uniformly black). Asterix `PPSA30490` sits at exactly 4.00 draws/frame (3 realized + 1 suppressed), dead flat across 14,472 frames, never reaching any content phase — a title that never starts rendering, a different failure. **Do not widen #1641 to those titles on this evidence.** | #1641 |
| Headless ErrorDialog's immediate `FINISHED` result prevents the account-login flow from advancing | **Falsified, with exact control flow and a cross-title positive control.** At `0x903041` the guest compares `UpdateStatus` against `FINISHED` (`3`); equality terminates the dialog and invokes its completion callback, while every other state returns to wait. The account-specific poll at `0x88ad6b` returns only for `RUNNING` (`2`); any other result reaches `0x88ad75`, writes `0x0100` at object offset `+0x108` (clearing active and setting account-checked at `+0x109`), then invokes and clears its callback. Prosper's immediate `FINISHED` is therefore the exact advancing state, not a stall. DOLL is the cross-title positive control: the same honest offline `SIGNED_OUT` and ErrorDialog lifecycle immediately streams title assets and reaches its interactive title screen. Changing ErrorDialog semantics is not justified. | #1606; `DOLL_LOADING_PROGRESSION.md` |
| The retained ErrorDialog Open line proves the game-specific account poll at `0x88ad20` ran | **Falsified; it proves a different owner.** The Open import has one direct guest caller, `0x901ed8`, in a Sony online-identity virtual method inherited unchanged by the PS5 identity class. A Vulkan-free live positive at `0x901edd` fired once and its frame recovered the exact return site `caller_rbp=eboot+0x88ae5a`, immediately after `ShowAccountLogin`'s virtual call. Dialog completion is polled separately by the Sony online-subsystem tick (`0x97f9e0` → `0x9036a0` → `0x903020`). The earlier zero at `0x88ad70` remains void, and the static `0x88ad20` semantics remain conditional; neither says that function executed. | #1606; #1932 |
| The Sony online-subsystem ErrorDialog tick directly completes the game account gate | **Falsified by the exact delegate flow.** `ShowAccountLogin` passes an empty delegate to the identity method at `0x901de0`; that method stores it at identity offset `+0x60`. The generic tick at `0x903020` correctly terminates a `FINISHED` dialog and invokes slot `+0x58` only when that stored delegate is non-empty. It is empty on this path, so the Sony tick cannot write account byte `+0x109`. The separate account poll at `0x88ad20` owns that write and is reached from foreground return (or a signed-in NP-state notification), not from the generic dialog callback. | #1606 |
| `L_GameloftSplash` directly calls `ShowAccountLogin`, so its Blueprint bytecode should expose the missing call | **Falsified by the retained package import table.** The level imports `CheckNoFreeSpace`, `IsAccountLoginChecked`, `IsAccountLoginRequired`, the EULA predicates/setter, `Begin`, `Create` and `AddToViewport`, but no `ShowAccountLogin` or account-init function. Account initialization is an automatic native stage (`0x87d100` → `0x88abe0`); the Blueprint only polls its checked predicate. | #1606 |
| PlayGo `GetLocus` is a sustained chunk-0 poll or progression gate | **Falsified; the earlier rate interpretation was an instrumentation mistake.** The retained trace's 1,000 calls carry chunk IDs **0, 1, 2, …, 999 exactly once**, with no reset. Guest code at `0x13f28a4` is a bounded startup cache-population loop: it calls `GetLocus` for each ID, stores only successful `LOCAL_FAST` (`3`) results, cleanly skips absent IDs, stops at `0x3e8`, and sets its initialized byte at `0x13f29e7`. Chunk 0 succeeds and the remaining nonexistent IDs fail as expected. This is neither periodic polling nor a wait for those IDs to become local; changing PlayGo behaviour is not justified. | #1606, #1641 |

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
  `live_gpu_targets` disable list (`live_renderer.cpp:933`), so none of them forces the CPU readback
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

## Reproduction

```bash
PROSPER_NO_FRAME_DUMPS=1 PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 \
PROSPER_RENDER_TIMING=1 PROSPER_RTT_TIMING=1 PROSPER_EXECLOG=1 \
  ./build-linux/screenshot <DUMP_ROOT>/PPSA19244-app0 --seconds 10 --count 2 --out <OUT> --timeout 250
```

Count decoded draws per **sample interval** — sum `draws=N` from `[rtt-timing]`, count
`[exec] skip draw`, split at each `[shot]` line, divide by that interval's frame delta. Averaging over
a whole run hides the phase transition that matters.
