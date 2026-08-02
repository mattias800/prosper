# The Oregon Trail (`PPSA19244`) — status and evidence

Unreal Engine 4. **Rung 0 — no real graphics.** The title boots cleanly and holds a steady ~50 fps
frame loop with frames provably advancing, but every one is uniformly black. Tracked on
[#1606](https://github.com/mattias800/prosper/issues/1606) and
[#1641](https://github.com/mattias800/prosper/issues/1641).

Read `## Ruled out` before forming a hypothesis. Four separate leads on this title have been measured
and killed, including both of the ones the issue body originally called highest-value.

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

**The GPU-side questions are answered. The next step is CPU-side:** find where UE4 decides its
base-pass draw list is empty (`PROSPER_HWBP` / `boot_trace` on the render thread, or symbol recovery
around the scene-renderer entry), not further GPU instrumentation.

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

## Reproduction

```bash
PROSPER_NO_FRAME_DUMPS=1 PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 \
PROSPER_RENDER_TIMING=1 PROSPER_RTT_TIMING=1 PROSPER_EXECLOG=1 \
  ./build-linux/screenshot <DUMP_ROOT>/PPSA19244-app0 --seconds 10 --count 2 --out <OUT> --timeout 250
```

Count decoded draws per **sample interval** — sum `draws=N` from `[rtt-timing]`, count
`[exec] skip draw`, split at each `[shot]` line, divide by that interval's frame delta. Averaging over
a whole run hides the phase transition that matters.
