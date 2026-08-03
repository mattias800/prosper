# Nikoderiko: The Magical World (`PPSA23760`) — status and evidence

Unreal Engine 4. **Rung 2 — title screen.** The epilepsy warning, the Knights Peak publisher logo and
the full title screen render at native 3840x2160 on unmodified master with no code changes, and a
scripted route continues into the legible MY.GAMES EULA dialog. **The 3D world behind the title is
black.** Tracked on [#1607](https://github.com/mattias800/prosper/issues/1607); the blocker itself is
[#305](https://github.com/mattias800/prosper/issues/305).

Read `## Ruled out` before forming a hypothesis. This issue's original premise was falsified and the
issue was retitled; so was #305's.

## Where it actually stands

**The blocker is #305, in graphics register state — not the recompiler and not descriptor
provenance.** For 11 of 13 traced vertex stages, the dword pair that *both* the AGC header
(`direct_resource_offset`) *and* the shader's own `SBASE` name as the descriptor-table pointer is not
a mapped guest address, under every canonicalization the scalar fold tries. The shader dereferences a
V# `num_records`/`dword3` tail as a pointer — the `0x0004dfac…` / `0x0001d22c…` constant family — the
const-fold correctly refuses to invent a descriptor, and the draw is skipped fail-visibly.

The condition that discriminates the #305 signature is stage-agnostic: **the required user-data
block is larger than the bound pipeline's user-SGPR window** (`SPI_SHADER_PGM_RSRC2_{GS,PS}.USER_SGPR`,
which equals the shader's own `user_data_range_end`). The early sample looked vertex/GS-specific
because every sampled pixel stage with declared pointers had a 30-dword window, equal to the largest
observed write. The later failure-replay inventory contains vertex stages with 30/32-dword windows
whose pointers remain readable and smaller 8/12/28-dword vertex windows that show the signature.
Window fit, not stage identity, predicts it. Full measurement, instruments and the complete
falsification list are in [`RESOURCE_BINDING.md`](RESOURCE_BINDING.md) § `Ruled out`.

Submit provenance is no longer the missing instrument. `PROSPER_UDPROV` records order, direct versus
indirect path, submit origin, fold and Jump depth for every SH write and carries them into draw
snapshots; `PROSPER_SUBMITORDER` separately records pre-mutex call order and submit-thread identity.
The trace localizes the mismatch (the bound program came from a q1/Dcb fold and the larger write from
the following q3/DcbFinal fold), but does not by itself establish a cross-queue root cause. Giving
DcbFinal a separate register file was tested as a default-off A/B and did not restore the world or
collapse rejects. The underlying ordering/ring contract remains unproven.

**Nikoderiko is the loud reproduction #305 has been missing.** On DOLL / Dragon Quest VII
(`PPSA17942`) the same defect costs 0–33 UI draws per 7-minute run; here it costs the world — 25
distinct dropped `(es, ps)` pipelines. Fix #305 against this title.

Baseline reject inventory on unmodified `961a6cdd`, 440 s boot with `PROSPER_DBG=1
PROSPER_DYNTRACE_FAIL=1`, title screen held from t=112 s: 172 `[recompile-reject]`, 207
`[exec-recompile-reject]`, 25 distinct dropped `(es, ps)` pipelines, 20 / 15 distinct failing VS / PS
programs, 95 `[vertex-recompile-reject]`, 90 + 7 `[mubuf-unresolved]`, 75 `[mimg-unresolved]`, 70
`[compute-struct-reject]`.

## Ruled out

One line per dead hypothesis, the evidence that killed it, and where that evidence lives. Do not
re-derive these without contradictory new evidence.

| Hypothesis | Verdict and evidence | Source |
|---|---|---|
| The 8-SGPR `SRSRC` range is **computed inside the shader**, so `sreg_range_written` is set, the user-data fallback is skipped, and descriptor provenance is defeated | **Falsified — this was #1607's founding premise and its original title.** Of the 27 failing stages dumped, replayed with `PROSPER_DYNTRACE_FAIL=1` and disassembled with `shader_inspect`, **every failing *vertex* stage** uses the **canonical bindless-dynamic vertex fetch** (the pixel-stage failures in this title are a separate matter — see the pixel-stage row below): an ordinary descriptor-table read whose index is wave-uniform (`(s18 << 4) & 0x1f0`) and whose base is a user-data pointer — the exact shape `resolve_dynamic_fetch` was written for, plus the AGC format-patch tail. `sreg_range_written` is the *symptom* of that legitimate reload. **The recompiler's three-step ladder is not the defect.** | #1607 |
| A wrong **seeding origin** for the user-data window | **Falsified.** `user_data_range_start` is `0` for every shader and `user_data_range_end` equals (last declared direct offset + 2), so the metadata is self-consistent with seeding at `SPI_SHADER_USER_DATA_GS_0 + 0`, which is what the executor does. The hypothesis that Nikoderiko was the first live `range_start != 0` shader was falsified by printing the raw field — that was the point of printing it. | #1607 |
| **Header decode** is wrong | **Falsified.** The shader's own `SBASE` register equals the header's declared direct offset in all ten stages disassembled (e.g. `0x300e710000` reads `s[14:15]` = dword 6; the header says `[10]=6`). | #1607 |
| The defect is specific to **vertex/GS**, so GS-versus-PS asymmetry is an acceptance test | **Falsified as a stage property.** The original #1607 sample did find every traced PS direct pointer readable, but all such PS windows were 30 dwords, equal to the largest observed write. A later failure-replay inventory found the #305 signature in 8/12/28-dword vertex windows and not in 30/32-dword vertex windows. Pixel stages were unaffected in that run because their windows fit, not because they were pixel stages. Their separate all-zero T# / unresolved-MIMG failures remain separate. | #305 |
| This is a general **UE4** defect | **Does not hold as stated.** Same build, same recipe: The Pathless `PPSA01826` has **13 of 13** declared direct pointers readable with `implied=0` everywhere, 0 `mubuf-unresolved` and 2 `mimg-unresolved`; its own rejects are missing opcodes (`fmt=14 op=0x68` ×6, `op=0x47`, `fmt=0 op=0x25`, `fmt=7 op=0x1`) — a different problem. The Plucky Squire `PPSA15319` shows 0 failed stage builds. **ArcRunner `PPSA21406` is inconclusive, not negative** — 175 s produced one presented frame and a 153-line log, so it never reached content that could exercise the path; do not read that row either way. The confirmed set is **Nikoderiko + DOLL**. | #1607 |
| Its 35.9 decoded draws/frame (peak 53) is #1641's "implausibly few" signature | **Falsified — that run was parked on a 2D UI screen.** A working title's *own menus* decode 7–13 draws/frame, so 53 for a title/EULA screen is normal. The 31→53 rise lands exactly when the rich screen appears: samples 10–11 carry 160,300 and 161,248 distinct colours while samples 3–9 are uniformly black. Testing this title against that signature needs a route into a 3D scene. | #1641, PR #1645 |
| The EULA route is a prosper defect | **No — it is route work.** `left-stick-down` reaches the guest and the document visibly advances (distinct-colour counts move 24,765 → 23,258 → 23,328 → 23,361 → 23,344; t=186 s shows a different paragraph than t=120 s), but ~20 presses at the default `PROSPER_PAD_HOLD` (300 ms) advance only about two paragraphs, `Apply` stays greyed out, and `cross` does nothing while it is disabled (byte-identical samples across the press window). A working route needs a much longer scroll phase or a raised `PROSPER_PAD_HOLD`. | #1607 |

## Instrument warnings

* **Diagnostics serialize draw realization** (`parallel_draw_diagnostic_active`), so a `PROSPER_DBG`
  boot reaches the title screen at ~112 s rather than ~90 s. **A 200 s window is not enough** — it
  times out in the pre-title load with *zero* rejects logged and looks deceptively as though the issue
  were fixed. Budget ~440 s.
* The first `[udmap]` probe tested only the raw 64-bit pointer value, so tagged-but-valid pointers read
  as `UNMAPPED`. It was corrected to use the fold's own 48-/40-bit canonicalization **before any number
  from it was trusted**. `UNMAPPED` now means the fold could not reach that address either.
* `[udmap]` is emitted only during failure replay. Every row therefore describes a stage that already
  failed; a readable pointer means that stage failed for another reason, not that it is a passing
  population sample. See instrument trap 24 in `GAME_COMPAT_ORCHESTRATION.md`.
* `PROSPER_PAD_SCRIPT` timings anchor to the **first pad poll**, not process start. Nikoderiko polls
  the pad during its logo sequence, so script time ≈ wall time here — do not assume that on another
  title.

## Reproduction

```bash
PROSPER_NO_FRAME_DUMPS=1 PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 \
PROSPER_DBG=1 PROSPER_DYNTRACE_FAIL=1 PROSPER_UDPROV=1 \
  ./build-linux/screenshot <DUMP_ROOT>/PPSA23760-app0 --seconds 8 --count 55 --out <OUT> --timeout 460
```

Then grep `[udcand]`, `[udmap]`, `[udprov]` and `[dynfail] replaying VS`. The instruments
(`PROSPER_UDPROV`, `PROSPER_BINDTRACE`, `PROSPER_SHADER_HEADER_NEWEST`, `PROSPER_UD_TAIL_ALIGN`) are
on PR #1639 — reuse them rather than rebuilding the measurement.
