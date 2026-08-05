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

### The open defect: most title frames arrive yellow

The title screen is reached but degraded. Most composited frames are the **correct frame with the
red and green channels forced to maximum**, which reads as a flat yellow background under otherwise
perfect white content. The mapping is exact, measured over two 4K frames 10 s apart in one run:

| correct frame | degraded frame |
| --- | --- |
| `(0,0,0)` ×8,090,540 | `(255,255,0)` ×8,090,521 |
| `(255,255,255)` ×151,941 | `(255,255,255)` ×151,941 |
| `(102,102,102)` ×14,539 | `(255,255,102)` ×14,537 |
| `(1,1,1)` ×1,263 | `(255,255,1)` ×1,249 |

So `out = (255, 255, in.b)`, with the blue channel carrying the true value. Equivalently it is an
additive/max composite against opaque yellow — the two are indistinguishable when one operand is
0 or 255, so do not claim one over the other without a frame containing a non-grey colour.

Two facts bound where it can live:

- **The splash logos are correct and the title screen is not.** Frames up to t≈40 s (Bandai Namco,
  Supermassive, Unreal, Wwise) are clean black-and-white in both frontends; the tint starts once
  the engine's real post-process chain engages.
- **It is not frontend-specific.** The native SDL3 `prosper-app` window shows it too, and there it
  is *persistent* — three grabs 40 s apart at the title screen are byte-identical yellow. Headless
  `screenshot` sees it on roughly two thirds of samples, with correct black frames interleaved.
- **The rate is stable and is a usable discriminator.** Two independent 360 s arms both give
  **24 / 36** tinted samples. That reproducibility is what makes an A/B on this defect worth
  running: a switch that does not move 24/36 has not touched the cause.

Open as [#2014](https://github.com/mattias800/prosper/issues/2014).

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
  (`OREGON_TRAIL_STATUS.md`), so this may be one shape across UE4 titles rather than two defects.
- 17 distinct compute programs are dropped with `[compute] skip unsupported program 0x…`. Each is
  logged once, so the line count is **not** a dispatch count. Per the charter this is a fatal gap
  regardless of whether it turns out to relate to the tint.
- `sceAgcDcbDrawIndirect` is still unimplemented ([#1977](https://github.com/mattias800/prosper/issues/1977)).

## Reproduction

```bash
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 PROSPER_NULL_PAGE=1 \
  ./build-linux/screenshot <DUMP_ROOT>/PPSA05143-app0 \
      --seconds 10 --count 36 --timeout 380 --out <OUT>
```

The title screen first appears at roughly t=110 s, so any arm shorter than about two minutes will
miss it entirely and see only the splash sequence. `PROSPER_NULL_PAGE=1` matches the earlier arms on
this title and does not change the outcome either way. There is no input route yet — nothing past
the title screen has been driven.

## Ruled out

Read this before forming a hypothesis.

| Hypothesis | Verdict |
| --- | --- |
| The stall before the title is the undelivered-GPU-completion family (#232 / #208 / #210 / #984) as its own defect | **Superseded.** The completion was never generated, because the submit owing it was declined: `PROSPER_EOPLOG=1` censused 2,706 `FIRE`, 1 `SKIP(rejected)`, 0 `OWE`. Same root cause as #1982, fixed by #1987. #1962. |
| The flat yellow frame is a real guest screen whose text layer is missing (e.g. a health warning) | **Falsified.** It is pure `RGB(255,255,0)` over the whole 4K frame, first composited at `frame_seq=4` with the identical crc — before any title content exists — and the frame the freeze landed on varied between runs. #1962. |
| The yellow tint is a per-scanout-buffer defect (one flip buffer composited wrong) | **Falsified.** The manifest's `front_index` does not correlate with the tint: 14 yellow / 6 clean on `front_index=0` and 10 / 6 on `front_index=1` over 36 samples. #2014. |
| The yellow tint comes from the packed-R11G11B10 compute storage path | **Falsified by a one-variable A/B.** `PROSPER_NO_PACKED_R11_STORAGE=1` over a 360 s arm gives **24 / 36** tinted samples — *identical* to the default arm's 24 / 36, on the same binary, route and host. Both arms also reach the same title screen and both show the same occasional clean frame, so the switch moved nothing. #2014. |
| `[agc] WaitRegMem … dependency violated` is the lead for the stall | **Falsified — instrument noise.** 31 events on *The Pathless* (`PPSA01826`, UE4), which renders its title screen for a full 140 s arm without stalling, against 40 on this title, and on this title they all stop *before* the stall began. #1962. |
| `crc=666f7b3f` fingerprints this title's wall | **No — it is just "black 3840x2160".** The same crc was the frozen frame of Crisis Core (#1982) and Sonic Frontiers (#1968), which have different causes. Do not group titles by it. |
| The `0x30016000` UE pooled-allocator fault (#1945 / #1226) bounds this title | **Not seen** in any arm of six on `ff72e77c`, nor in any arm on `4d7a2ded`. |
| #2003 is inert for this title (no `sceNpEntitlementAccessGetSkuFlag` line in the run logs) | **Void, not negative — the instrument was never armed.** `svc_log()` is gated on `PROSPER_SVCLOG`, which those arms did not set. With it armed the title calls `GetSkuFlag` twice, in the same window the title screen composites. Claimed and withdrawn on #1893. |
| The boot dies in the guest at `addr=0x80` | **Falsified — the faulting instruction was prosper's own.** `k_ef_create` (`hle_kernel.cpp:1767`) storing through a guest out-pointer of `0x80`; the `rip=eboot+0x…` label was instrument-trap 22. The `0x80` itself came from `sceAjmInitialize` rejecting this title's config revision, fixed by #1966. Filed as #1963. |

## Ladder

- [x] Rung 1 — real graphics from the live renderer (splash sequence, native 3840x2160)
- [x] Rung 2 — title screen reached and rendered (degraded by #2014)
- [ ] Rung 3 — gameplay with real GPU draws — no input route yet
- [ ] Rung 4 — manual visual verification
- [ ] Rung 5 — PS5 hardware-oracle comparison
- [ ] Rung 6 — reviewed automatic gameplay snapshot guard
