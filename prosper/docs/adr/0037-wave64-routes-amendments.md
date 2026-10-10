---
kind: adr
status: proposed
date: 2026-10-08
---

# ADR 0037: Amend ADR 0028's Wave64 routes with what migration steps 1-3 measured

## Context

ADR 0028 is accepted (#4761) and frozen, so its text cannot be corrected in place. Steps 1-3 of its
migration order have since merged: step 1 is the `s_cbranch_execz` any-vote certificate (#4755), step
2 the `route=` field and per-route counters (#4754), and step 3 the compute cross-lane classification
and `PROSPER_WAVE64_EXCHANGE` (#4760, #4763). (#4749, the #4714 scalar-pair fabricated-zero fix, is
not a migration step.) This ADR records the deltas those steps exposed and the gaps the ADR leaves,
and changes no accepted rule.

The Black Flag census quoted below is **pre-step evidence**: the 2026-10-08 comment on #4131 ran
`prosper-app` built on 2026-10-07, before #4754, #4755, #4760 and #4763 merged, on Windows with an
RTX 4070 SUPER, with no input route, a cold pipeline cache and a thin early-boot sample of one flip
window, so it gives classes and not counts. It is what prompted this ADR. It does not meet the bar
Decision 3 sets, and the current-`main` census Decision 3 requires replaces it.

What is now known:

- **Route 3 largely exists.** Compute workgroup exchange for uniform control flow was found already
  present on the default path (recorded in `docs/gpu/RECOMPILER_REMAINING.md` § Ruled out). ADR 0028
  describes it as work to build. The pre-step census agrees: of four refused compute programs, none was
  refused for `subgroup-contract`; all four are `compute/recompile` defects (a MIMG NSA address, a
  scalar-pair `s_cselect_b64`, an `s_cbranch_scc1` the structurizer cannot place, and an
  `s_cbranch_execz` inside a counted loop).
- **Migration step 4 (N lanes per invocation) is not motivated** by any evidence so far on Black
  Flag, and step 5 (fragment promotion) is unmeasured: whether the refused fragment draw
  `0x407edfaf00` is a full-screen pass is not known.
- **The fragment frontier is a proof-domain question.** The Black Flag `unproved-vote` draw is clean
  at the guest-side classifier and is refused only at the SPIR-V neutral proof, because its body
  holds a storage-buffer load and the proof admits no loads. The question of widening it (option B) or keeping loads out (option A) was posted as a comment on #4755,
  which merged without an answer.
- **Route 2's rewrites have only an estimator.** The `s_cbranch_execz` certificate exists; the
  uniform `readlane`, compaction-ballot and uniform-vote rewrites do not, and the yield estimate has
  never run on a real title.

## Decision

1. **ADR 0028 stands.** This ADR amends it; it does not supersede it. Where the two differ, the
   differences below govern.
2. **Route 3 is "exists, extend by evidence".** New work on it starts from a refusal that a census
   attributes to `compute/subgroup-contract`, not from the ADR's description of it as unbuilt.
3. **Steps 4 and 5 are gated on a census.** Neither starts until a run with an input route, a build
   from current `main` and a warm pipeline cache shows refused programs they would admit, with the
   count per class. Until then the next work is the route-2 rewrites and the four `compute/recompile`
   defects, each as its own issue with a red-without-fix test.
4. **Loads stay out of the neutral proof (option A) as the working default.** The merged step-1 certificate
   admits no loads, and this ADR does not change that. A draw refused only because its skipped region
   holds a load stays visibly refused and is served later by the exact routes (the owned wave of #4384,
   or step 5 promotion). Rationale: no surveyed emulator proves loads neutral in an EXEC-skipped region,
   a wrong verdict corrupts pixels silently, and the exact routes need no such proof. **Option B may be
   adopted later** by a follow-up amendment, if a census of the refused fragment set shows the
   load-bearing shapes are few and structurally alike. It would be narrow: robust2 word-buffer contract,
   bounded index, result reaching only an identity-masked export or a dead value, each condition with a
   mutation arm, and the cross-title A/B before any default.
5. **Programs outside the ADR's coverage are named, not implied.** The ADR gives no route to vertex,
   geometry or NGG programs (the Kena refusals, #4427), and says nothing about the reverse direction
   (a guest Wave32 program on a wider host). Both stay visibly refused under the ADR's route 6 until
   a separate decision names a route.
6. **Admitted programs get a self-check mode.** A default-OFF, diagnostic-class switch compares an
   admitted program's output against the exact path (workgroup exchange or the owned wave) on the
   same inputs and logs a mismatch. It changes nothing the guest sees. Today it can compare only
   pairs whose exact path exists on `main`: a compute program admitted by a route other than the
   exchange, against the exchange. The owned fragment wave (#4384) is not merged, so proven fragment
   programs have no exact path to compare against yet. It is the control that lets a
   route be trusted for a default flip, and it is what ADR 0028's `GPU-5` lacks as an enforcement
   other than review.
7. **Default flips need the cross-title A/B first.** `PROSPER_WAVE64_EXCHANGE` and any route default
   wait for a same-binary A/B on Kena, GTA V, Sonic Frontiers and the guarded titles, run on a
   machine that holds those dumps, with the manifest slice of ADR 0021/0023 so
   each arm records head, build and switches.

## Consequences

- No accepted text changes, so CI's frozen-ADR rule is untouched.
- The next work is ordered by evidence: route-2 rewrites and the four recompile defects first, then
  the longer census, then steps 4-5 only if the census asks for them.
- `GPU-5` remains violated until each of these is closed or a route names it: the Wave32 mask forms
  (`allow_b32_masks`) and the merged-NGG shell's VCC reconstruction, both listed as uncovered by
  #4749 in the spec's `GPU-5` "Violated today" line, and a fabricated scalar word routed through a
  VGPR (`v_mov` + `v_readfirstlane`, #4787). This ADR does not hide that.
- Enforcement: review, the `[wave64-route]` and `[wave64-unsupported]` lines, and the self-check
  mode once it exists. No tool can verify a census claim; the census comment on the tracker is the
  record.

## Alternatives considered

- **Edit ADR 0028.** Rejected by CI and by the lifecycle: a changed decision is a new ADR.
- **Supersede ADR 0028.** Rejected: nothing in its decision is wrong; its ordering and one premise
  (route 3 as unbuilt) are stale. A supersession would re-open an accepted decision for a delta.
- **Keep the deltas in a status doc only.** Done for the evidence (`RECOMPILER_REMAINING.md`), but
  an accepted ADR with no pointer to its corrections misleads the next reader; this ADR is that
  pointer.

## Approval

Requires the project owner's acceptance. It unblocks the census-gated ordering above and the
self-check mode; it makes option A the recorded default and leaves option B open for a later amendment.
