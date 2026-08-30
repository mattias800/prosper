# Stray (`PPSA02101`) — status

Tracker: [#2883](https://github.com/mattias800/prosper/issues/2883). Engine: Unreal Engine 4.
Route: [`prosper/scripts/stray-PPSA02101/`](../scripts/stray-PPSA02101/README.md).

**Rung 2.** A Cross-only route accepts the brightness-calibration screen and reaches the first map
load (`hk_project_mainstart`, t ≈ 37 s, absent from every default run). Two separate things are wrong
after that, and they are worth keeping apart because they have different evidence:

| what | issue |
| --- | --- |
| the world composites as a flat **letterboxed** clear — no scene renders | [#2932](https://github.com/mattias800/prosper/issues/2932) |
| the **title screen** renders its menu but not its background | [#3126](https://github.com/mattias800/prosper/issues/3126) |

## Two routes, and which to use

`reach-first-map.pad` drives past calibration to the map load. `reach-title-flip.pad` stops at the
title screen and holds there; it is **anchored on flip ordinals, not wall-clock**, and it is the one
to use for anything reproducible. The route README records why: a timed route reached the title only
sometimes, and the flip anchor works because the target is a steady state rather than a moment.

## Performance: this title is not GPU-bound

Measured 2026-08-29 with an F8 `.prperf` capture over a 5.02 s window at the title screen, read with
`tools/perf/performance_capture_report.py`:

| bucket | ms |
| --- | --- |
| `gpu-device` | 221.7 (≈ 4% of the window) |
| compute | 2023.8 |
| texture materialisation | 1137.4 |
| buffer copy | 602.0 |

One compute program, `0x3011300000`, accounts for ~605 ms of that at roughly 32 ms per dispatch at
3840×2160. So the cost is on the host side of the boundary, and pointing a GPU profiler at this title
answers a question it does not have. Detail on [#3126](https://github.com/mattias800/prosper/issues/3126).

## The title-screen defect, as far as it is established

`[mimg-unresolved]` reports five image ops on one routed boot whose descriptor resolves to nothing;
every draw using those shaders is discarded. What is **established**: for the shader stages that were
dumped, the resource table is complete and correct — each declared read-only T# is present, the
SGPR-resident one under DIRECT (`sgpr_base`) provenance and the EUD-resident ones under INDIRECT
(`srt_offset = (offset_dw - num_user_sgprs) * 4`) provenance, which is exactly what the table shows.

What is **not** established is which stage the five failures belong to. Every line reports
`program=0x0`, because `recompile_fragment_impl` hardcodes a zero program address
([#3130](https://github.com/mattias800/prosper/issues/3130)) — so no fragment-shader recompile failure
on any title can currently be attributed to a shader. Fixing that turns this investigation from an
inference into a lookup, and is the cheapest next step.

Separately, four stages declare a *writable* 8-dword T# that reaches no resource table at all
([#3128](https://github.com/mattias800/prosper/issues/3128)); one of them has four image ops against a
completely empty table. Whether that is what its image ops want is untested — see Ruled out.

## Ruled out

One line per falsified hypothesis, the evidence, and the link. Do not restart these without
contradictory new evidence.

- **"The descriptor is at the right SGPR base and mis-classified as a buffer."** Falsified. A
  whole-table dump (SRSRC set + finished table + the guest's four raw sharp arrays, per stage) shows
  the tables for the dumped stages are complete: five declared T#s, five present. #3126.
- **"The table being consulted and the code being scanned belong to different stages."** Falsified by
  the same dump — same stage, and the join is exact once the sharp arrays are printed alongside. #3126.
- **"The lost `sreg_srt` tag explains the `srsrc=s8/s16` failures."** Void, not merely wrong: the
  diagnostic's own `written=0` says the shader never wrote those SGPRs, so nothing was `s_load`ed and
  no tag could be lost. The resolver correctly falls through to `by_sgpr_base`, which is what returns
  null. #3126.
- **"Admitting the dropped writable T#s (#3128) fixes the title screen."** One arm says no: admitting
  them as `StorageImage` left both `mimg-unresolved` (5) and `max_nonblack` (0.1097) exactly
  unchanged. That arm may have used a class/provenance the MIMG resolver does not match on, so the
  question is open — but do not assume the fix moves this title. #3128.
- **"A wall-clock pad route reaches the title screen."** Falsified: 0.1095 once, then 0.0063 / 0.0000
  / 0.0063 on three unmodified reruns. The single lucky sample was adopted as an A/B baseline and four
  hypotheses were run against it before the reruns exposed it. Use `reach-title-flip.pad`. #3127.
- **MIMG `SRSRC` is not a user-SGPR index.** It names any SGPR, so a scan that treats every SRSRC as
  a user-data slot reports mostly false positives — scratch registers the shader loaded a descriptor
  into. An earlier list of "bases missing from the table" derived that way is discarded. #3126.
