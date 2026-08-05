# Little Nightmares III (`PPSA05143`) — status and evidence

Unreal Engine 4. **Rung 2 — title screen** (2026-08-05, master `4d7a2ded`;
`assets/screenshots/little-nightmares-3-title-screen.png`). Tracked on
[#1893](https://github.com/mattias800/prosper/issues/1893).

## Where it stands (2026-08-05)

**Rung 2 reached: the title screen renders.** On a default launch the title now plays its whole
splash sequence — Bandai Namco → Supermassive Games → Unreal Engine → Wwise — and then presents its
own title screen at native 3840x2160: the *Little Nightmares III* logo, the `Player` slot bottom
left, the copyright line, and the `⊗ Start` prompt bottom right, all legible.

The render-thread stall that used to end every run before the title is **gone**. It was #1962, and
its cause was #1982's ordered-DMA `Jump` decline, fixed by
[#1987](https://github.com/mattias800/prosper/pull/1987). A 360 s bounded arm on `4d7a2ded` ran to
its own end with 36/36 samples source-distinct, 26 pixel-distinct, no `LowLevelFatalError`, no
watchdog abort, and **zero** `[agc] ordered DMA submit rejected` lines — against a hard wall at
t≈60 s before.

```text
[shot] done: 36 screenshot(s); source-distinct=36 pixel-distinct=26
       max-source-stale=0.0s max-pixel-stale=30.0s status=ok
[progress] t=356.9s submits=9817 draws_cum=376986 dispatches=80095 flips=1067 presents=6606
```

### The open defect: the UI's background clears to yellow instead of black

Menu frames are intermittently degraded: the background behind the UI is `RGB(255,255,0)` where it
should be black. **The content itself is correct** — every layer composites over that background
with its own alpha, exactly as it does over black.

> **Superseded framing — do not reuse.** This was first written up as "red and green forced to
> maximum", `out = (255, 255, in.b)`. That fitted the title screen, whose content is all
> white-on-black at one alpha, and it is **wrong**. The EULA screen falsifies it: its dialog panel
> is `(0,0,0)` clean and `(102,102,0)` degraded, where `out = (255,255,in.b)` demands `(255,255,0)`.
> #2014.

The background-colour model fits every measured pixel on **two different screens with three
different alphas**. With content `C` at alpha `a` over background `B`, `out = a·C + (1-a)·B`, and
only `B` differs — `(0,0,0)` clean against `(255,255,0)` degraded:

| element | clean | degraded | implied `(1-a)` |
| --- | --- | --- | --- |
| title-screen background | `(0,0,0)` ×8,090,540 | `(255,255,0)` ×8,090,521 | 1.0 |
| title logo, fully opaque | `(255,255,255)` ×151,941 | `(255,255,255)` ×151,941 | 0.0 |
| title-screen footer text | `(102,102,102)` ×14,539 | `(255,255,102)` ×14,537 | 0.6 |
| EULA dialog panel | `(0,0,0)` | `(102,102,0)` | 0.4 |
| EULA body text | `(153,153,153)` ×31,976 | `(194,194,153)` ×31,949 | 0.16 |

Two of those rows are load-bearing. **Fully opaque white is byte-identical in both frames** — 126,501
pixels on the EULA screen — so nothing is being clamped or channel-swizzled; opaque content is
already correct. And the blue channel is *never* touched in any row, in either frame, which is what
identifies the wrong value as a background colour rather than a per-channel transform.

This is a much narrower place to look than a channel map: the question is why the surface the UI
composites onto is cleared to `(1,1,0)` instead of `(0,0,0)` on roughly two thirds of frames.

What is and is not bounded:

- **It tints frames that contain no content at all, so it is NOT switched on by the post-process
  chain.** It is tempting to read "the logos are clean and the title screen is not" as an onset, and
  an earlier draft of this doc did. Two independent observations kill it, and the second needs no
  cross-build argument:
  - The earliest tinted frame on record is the *empty* composite at `frame_seq=4`, t≈4.0 s (#1962,
    build `ff72e77c`); a frame with nothing drawn over a yellow background is exactly the pure
    `RGB(255,255,0)` seen there. Clean logo frames at t=20/40/60/80 s are interleaved with it.
  - On **`4d7a2ded` itself**, ten consecutive samples (t=120–210 s) are pure `(255,255,0)` across
    all 8,294,400 pixels — one colour, no content — while that arm did not composite the title
    screen until t≈230 s. Ten of its twenty-four tinted samples therefore carry nothing that any
    post-process chain could have acted on.

  So the tint is present and intermittent from the first seconds and applies to empty frames, and
  **the post-process/tonemap stage is not localised by this evidence** — do not narrow to it on the
  strength of the splash frames.
- **It is not frontend-specific.** The native SDL3 `prosper-app` window shows it too, and there it
  is *persistent* — three grabs 40 s apart at the title screen are byte-identical yellow. Headless
  `screenshot` sees it on roughly two thirds of samples, with correct black frames interleaved.
- **The rate is 24/36 in both 360 s arms, but that is not yet a calibrated discriminator.** The two
  arms are the default and the `PROSPER_NO_PACKED_R11_STORAGE=1` arm — i.e. the same pair an A/B
  would compare, so their agreement cannot also serve as the baseline's run-to-run variance. With
  n=1 per configuration the null spread is unknown. Worse, the two 24s are not even counting the
  same thing: the treated arm reached the title screen ~120 s later, so most of its tinted samples
  are empty composites where the default arm's are title-screen frames. Establish a null with two
  **same-configuration** arms, and compare within a fixed content window, before reading any future
  "unchanged at 24/36" as a negative.

Open as [#2014](https://github.com/mattias800/prosper/issues/2014).

### Past the title screen: the EULA, via a checked-in input route

`scripts/little-nightmares-3/reach-gameplay.pad` presses `cross` at the title screen and the title
advances to its **End-User License Agreement** screen — Bandai Namco's EULA text, a `Decline` /
`Accept` selector with `Accept` highlighted, and the `Navigate / Skip / Select / Back` prompt row.
It renders completely and, in the clean frames, correctly —
`assets/screenshots/little-nightmares-3-eula.png`.

This is still **rung 2**, not rung 3: a EULA is a menu, not gameplay with real GPU draws. It is
recorded because it establishes two things the next lane would otherwise have to re-derive — that
guest input is reaching this title at all, and where the route stands.

The route is written to be self-invalidating. Run it with `PROSPER_PAD_SCRIPT_LOG=1` and check the
`[pad-script]` transitions before believing any null result:

```text
[pad-script] elapsed=120.249 frame=730 read=733 buttons=cross
[pad-script] elapsed=124.219 frame=737 read=740 buttons=neutral
…8 press/release pairs, each observed at a real pad poll…
```

Without that log, "the guest ignored the input" and "the input was never applied" are the same
observation. Note also that the route's wall-clock windows are calibrated against a **default-route,
lightly-loaded** boot: under GPU contention the title screen arrives later and the presses can land
early. Re-check the log rather than the frames when the route appears to do nothing.

**What remains for rung 3:** accepting the EULA and reaching gameplay. Whether the remaining presses
in the route already accept it is not established — the run that discovered the screen was still on
it at its bound.

### #2003 is not inert here — but it is not the reason the wall went

This title calls both APIs #2003 changed. Under `PROSPER_SVCLOG=1` (9,270 `[svc]` lines in the run,
which is the positive control that the logger was live):

```text
[svc] sceAppContentAppParamGetInt(0x1, …)        once, before the t=20 s sample
[svc] sceNpEntitlementAccessGetSkuFlag(…)        x2, between the t=100 s and t=120 s samples
```

Both `GetSkuFlag` calls land in the window where the title screen composites. The dump answers them
from its own bytes: `sce_sys/param.json` declares `applicationDrmType: "upgradable"`, which
`derive_sku_flag` (`hle_addcontent.cpp:372`) maps to `Full`, and `userDefinedParam1: 0`. Before
#2003 the NID was unregistered and fell to the dispatcher's stub, which reports success while
leaving the out pointer untouched — `hle_service.cpp:3941` names `PPSA05143` as one of the titles
that then acts on uninitialized stack residue.

**Not established:** whether #2003 is *necessary* for rung 2. The wall was at t≈60 s, before either
call, and its removal is attributable to the ordered-DMA rejection going 1 → 0 (#1987, measured).
Settling it needs a revert arm, which has not been run. Do not restate "#2003 is inert for this
title" — that claim was made on this tracker from an unarmed `svc_log` and withdrawn.

### Other things visible in the same run

- Two of 36 samples are a blue/magenta noise band rather than content (t=150 s, t=190 s). *The
  Oregon Trail* shows a "corrupted blue/magenta frame" in its own startup sequence
  (`OREGON_TRAIL_STATUS.md`), so this may be one shape across UE4 titles rather than two defects —
  filed together on that hypothesis as
  [#2028](https://github.com/mattias800/prosper/issues/2028). It is **not** #2014: #2014 changes only
  the background the UI composites onto and leaves the content correct, this replaces the content
  entirely, and both occur in different samples of the same run.
- 17 distinct compute programs are dropped with `[compute] skip unsupported program 0x…`. Each is
  logged once, so the line count is **not** a dispatch count. Per the charter this is a fatal gap
  regardless of whether it turns out to relate to the tint —
  [#2022](https://github.com/mattias800/prosper/issues/2022).
- `sceAgcDcbDrawIndirect` is still unimplemented ([#1977](https://github.com/mattias800/prosper/issues/1977)).

## Reproduction

```bash
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 PROSPER_NULL_PAGE=1 \
  ./build-linux/screenshot <DUMP_ROOT>/PPSA05143-app0 \
      --seconds 10 --count 36 --timeout 380 --out <OUT>
```

The title screen first appears at roughly t=110 s on the **default** route, so any arm shorter than
about two minutes will miss it entirely and see only the splash sequence. `PROSPER_NULL_PAGE=1`
matches the earlier arms on this title and does not change the outcome either way. There is no input
route: `scripts/little-nightmares-3/reach-gameplay.pad` reaches the EULA screen (see above), and
nothing past that has been driven.

**Length is not sufficient on its own, only on the default route.** Three of three default-route
arms reached the title screen. Both `PROSPER_NO_PACKED_R11_STORAGE=1` arms ran markedly further
behind: the 360 s one did not composite the title screen until roughly t=230 s, and a 200 s one of
the same configuration **never reached it at all** and is void for any question about the title
screen. That arm is recorded here rather than dropped, because a bound chosen from the default
route's t≈110 s would have looked generous and still produced nothing. The GPU was shared with peer
lanes throughout, so this is not offered as a timing measurement of the switch — only as the reason
that arm answers nothing.

## Ruled out

Read this before forming a hypothesis.

| Hypothesis | Verdict |
| --- | --- |
| The stall before the title is the undelivered-GPU-completion family (#232 / #208 / #210 / #984) as its own defect | **Superseded.** The completion was never generated, because the submit owing it was declined: `PROSPER_EOPLOG=1` censused 2,706 `FIRE`, 1 `SKIP(rejected)`, 0 `OWE`. Same root cause as #1982, fixed by #1987. #1962. |
| The flat yellow frame is a real guest screen whose text layer is missing (e.g. a health warning) | **Falsified.** It is pure `RGB(255,255,0)` over the whole 4K frame, first composited at `frame_seq=4` with the identical crc — before any title content exists — and the frame the freeze landed on varied between runs. #1962. |
| The yellow tint is a per-scanout-buffer defect (one flip buffer composited wrong) | **Falsified.** The manifest's `front_index` does not correlate with the tint: 14 tinted / 6 untinted on `front_index=0` and 10 / 6 on `front_index=1` over 36 samples. (Two of the twelve untinted are the blue/magenta noise frames below, not clean content; moving them to either bucket leaves the two columns uncorrelated.) #2014. |
| The yellow tint comes from the packed-R11G11B10 compute storage path | **NOT falsified — the arm is inconclusive, and is recorded here so nobody counts it as a negative.** `PROSPER_NO_PACKED_R11_STORAGE=1` over a 360 s arm gives **24 / 36** tinted samples against the default arm's 24 / 36. But the switch only reaches the *compute* path (`gpu_executor.cpp:4719`) and the packed emission is further gated on a 3-component `Float10_11_11` storage image (`rdna2_to_spirv.cpp:9516`); nothing logs whether that path was ever taken, so "the switch moved nothing" cannot be told apart from "the switch was never in circuit". Needs a counter of dispatches compiled with `packed_r11=true` before it means anything. #2014. |
| `[agc] WaitRegMem … dependency violated` is the lead for the stall | **Falsified — instrument noise.** 31 events on *The Pathless* (`PPSA01826`, UE4), which renders its title screen for a full 140 s arm without stalling, against 40 on this title, and on this title they all stop *before* the stall began. #1962. |
| `crc=666f7b3f` fingerprints this title's wall | **No — it is just "black 3840x2160".** The same crc was the frozen frame of Crisis Core (#1982) and Sonic Frontiers (#1968), which have different causes. Do not group titles by it. |
| The `0x30016000` UE pooled-allocator fault (#1945 / #1226) bounds this title | **Not seen** in any arm of six on `ff72e77c`, nor in any arm on `4d7a2ded`. |
| #2003 is inert for this title (no `sceNpEntitlementAccessGetSkuFlag` line in the run logs) | **Void, not negative — the instrument was never armed.** `svc_log()` is gated on `PROSPER_SVCLOG`, which those arms did not set. With it armed the title calls `GetSkuFlag` twice, in the same window the title screen composites. Claimed and withdrawn on #1893. |
| The boot dies in the guest at `addr=0x80` | **Falsified — the faulting instruction was prosper's own.** `k_ef_create` (`hle_kernel.cpp:1792`) storing through a guest out-pointer of `0x80`; the `rip=eboot+0x…` label was instrument-trap 22. The `0x80` itself came from `sceAjmInitialize` rejecting this title's config revision, fixed by #1966. Filed as #1963. |

## Ladder

- [x] Rung 1 — real graphics from the live renderer (splash sequence, native 3840x2160)
- [x] Rung 2 — title screen reached and rendered (degraded by #2014)
- [ ] Rung 3 — gameplay with real GPU draws — route reaches the EULA screen, not gameplay
- [ ] Rung 4 — manual visual verification
- [ ] Rung 5 — PS5 hardware-oracle comparison
- [ ] Rung 6 — reviewed automatic gameplay snapshot guard
