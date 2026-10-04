# General Wave64 / Kena checkpoint — 2026-10-04

This is a durable **partial-work handoff**, not acceptance evidence. The helper/launch code is
not accepted. Both confirmed review findings have correction attempts. After two recorded build
failures and reviewed compile corrections, the focused baseline built and ran41PASS/14FAIL.
The canonical-EXEC correction improved this to50PASS/5FAIL and exposed two warm-cache negatives.
The subsequent cache/fixture correction reached53PASS/2FAIL. Authentic shared-registration and
physical-profile fixture corrections now reach **55/55 actual focused CPU cases PASS,
CTest0/helper0**: the first passing four-PE focused baseline, NOT the full helper campaign.
The first integrated connected run reached **96/99PASS:89/92 CPU,7/7 Vulkan; CTest8/helper1**.
The diagnostic-contract correction now reaches **99/99PASS:92/92 CPU,7/7 Vulkan;
CTest0/helper0** at exact997243e7. It has scoped independent source clearance, not whole-PR approval.
The original saved-live/WQM/DS helper-pixel oracle actually passes on the Windows RTX4090 at
16->15->16 widths. Three connected production review findings remain; two separate SOURCE-ONLY
correction branches are retained below. No whole-source approval or passing full acceptance baseline.
General Wave64 and Kena title-screen 3D are
NOT complete. No current Kena drop census, FPS improvement or title-screen rendering result was
measured in this checkpoint. The original work was frozen at the maintainer's checkpoint request;
subsequent bounded goal follow-ups are explicitly recorded below.
The earlier source-only sections describe their own historical freezes, not the later execution
status. Do not retroactively assign the new native result to those older source identities.

## Latest: companions integrated, all three blockers corrected (2026-10-04, later session)

**Read this section first; the sections below it are history.** A new owner (Claude Code)
adopted the retained worktree and its sole native build after verifying it clean, at
`0b2ff27b`, with no live build/test process. Merges on the PR branch, all conflict-free:
#4399 `ef87c026`, #4398 `8e0f2064`, then main `a524d233`; merge result `884c8780`.

| Commit | What |
| --- | --- |
| `30e478bb` | #4398 did not compile (`SharedShaderWords` is not visible in `fragment_draw_plan.hpp`). #4399's regression never reached its own path: center+POS ENA does not make the realizer derive GS. Only P10/P20, mixed P0+smooth or an enabled **pull-model plane (ENA bit3)** do. Both cases failed their "realizer must generate GS" precondition. The fixture now uses the pull plane; isolated-key expectations are corrected to 3 compiles/2 hits, and the changed-layout query must return `draws[2]`'s own GS. |
| `a50e022b` | **#4397.** `fragment_raster_workitem_gap` refuses `DISABLE_COLOR_WRITES_ON_DEPTH_PASS` (bit31) by name, and refuses bits outside the defined gfx10.3 field map (`0x3f8ff800`). Bit30 is admitted as inert, because Z/stencil/bounds are already required off (CONFIDENCE: MED). The general renderer does **not** apply either bit yet; changing every title on a field name repeats the #1724 MODE=DISABLE lesson. `resolve_pipeline_state` now counts resolutions carrying them (exit summary `DB_DEPTH_CONTROL colour-on-depth totals`) so exposure can be measured first. |
| `32d41578` | Review finding on #4399: the derived-GS key now includes the float-controls verdict for nonexplicit transports. |
| following | Opt-in SOURCE retention of coefficient GS, collector FS and derived pull GS; clang-format. |
| `14ea1abe` | Review REJECTED `e2e378f6` because my per-thread rationale was false. Draws realize on a worker pool, but the only `cached_fragment_draw_program` caller is the render-thread backend, so a `thread_local` GS gave up to N cold plans and N collector pipelines per profile. Residence is now process-wide: the lock covers lookup/insert, the compile runs outside it, and nothing is filed if the verdict changed mid-compile. New `OtherRealizingThreadSharesTheGeneration` (`std::thread`) fails under `thread_local`. The #4397 test's vacuous pending assertions were dropped. The input-free packet recipe and the shipping colour state also ignore bits 30/31: filed as #4419 (on `main`, outside this draft). |

Native evidence, Windows/Clang Release, RTX 4090. The full cohort ran at `32d41578` plus test-only retention, with cases run directly, not via CTest, because this build dir registers nothing for `ctest -N`. Each later commit rebuilt and reran its affected executables: `e2e378f6` four executables; `14ea1abe` derived-geometry 3/3 and raster launch 24/24, plus a `thread_local` omission control.

- **103/103 cases pass** in 12 executables: the 99 baseline, plus 2 `DerivedInterpolationGeometry`, 1 `FragmentCollectorGeometry` (Vulkan) and 1 new `FragmentRasterLaunch.ProgrammedDepthColorControlRefusesBeforePendingAuthority`.
- **Omission controls, each restored green:** (A) per-draw fresh GS compile fails derived lines 74/76/77, with a distinct GS and a cold plan per observation. (B) a 2-stage collector fails the 3-module count and the **numerical centre-IJ oracle on the 4090** (I read 0 where the independent vertex oracle expects 0.2656 and similar). (C) without the bit31 check, exactly the two bit31 arms fail at gap, plan reason and capacity; without the reserved check, exactly the two reserved arms fail.
- **SDK validation:** `spirv-val --target-env vulkan1.3` accepts all **151/151** retained modules. A corrupted copy is rejected (control).
- Not run: app/game/Kena, Vulkan validation layers, and the broader old capture/bank/default-emitter neighbour population.

Raw receipts are under `.codex/claude-wave64-20261004/` (local, not committed). The same reviewer reviews every correction; see the PR's registered reviews for the verdict on the current head.

**Kena is `PPSA01802`.** The build dir's `GAME_DUMP` points at `PPSA12544`, which is a different title. The 2026-09-30 Windows Kena run (`.codex/kena-title-20260930/`) gives the real drop mix to aim at. Over the run: `fragment/subgroup-contract` 37,118 draw uses, `shader-recompile/vertex` 9,639, `shader-recompile/fragment` 2,819, compute recompile 305. The helper recipe admits 0/32 retained originals, with first unmet classes SMEM 18 and M0 12. So the next general step is composing the accepted scalar-bank (SMEM) authority from #4374 with the raster helper launch, rather than another Kena run. A Kena run is only worth doing once some original can actually be admitted.

## Source-only follow-up after the frozen checkpoint

The original3fcb/2775280b checkpoint and its registered REJECTED review remain historical.
A protected successor attempts the PREFETCH correction: a shared canonical encoding predicate
now connects cold proof, complete effects, mask/scalar-origin inventories and Architectural final
packet admission. Reserved modes and unproved CLAUSE have named refusals. LegacyRaw is not
expanded; Architectural resource callers share the hint admission, without new resource/input
authority. The underlying synchronous emitter already treats the
I-cache hint as a no-op; it is no longer accepted by only the early launch proof.

Three new source-declared regressions (UNEXECUTED):

- `FragmentRasterLaunch.CanonicalPrefetchRetainsTheRegisteredOriginalThroughFinalCompilation`
  covers valid modes1..3, real registration/private association, masks, final kernel and warm reuse.
- `FragmentRasterLaunch.ReservedPrefetchAndUnprovedClauseDoNotMintPendingOriginalPermission`
  covers reserved0/4/high-bit modes and CLAUSE named cold-proof refusals.
- `FragmentScalarReads.CanonicalPrefetchPreservesOriginsButReservedHintsAndClausesRefuse`
  covers original descriptor origins/shifted PCs and atomic manifest refusal.

This is an authored source attempt, not a verified repair or review approval. The future campaign
must include these cases in addition to the frozen inventory below, and independently check the
final dispatcher negative and LegacyRaw/resource neighbors. No native build/test/SPIR-V/GPU or
game was run for this follow-up. The position-independent entry-schema attempt is described below;
the incomplete connected review and full-original frontier remain. Source/gate identities
and outcomes are recorded in the PR follow-up comment, not borrowed from the frozen3fcb receipt.

## Latest integration checkpoint

The PREFETCH source attempt is commit `f657f651816b9727c71debc353f8407245ee236a`, parent
`2775280b4d26cb5ed44bb1f44ee7a499fc9f9bae`, tree
`ea3d5834c2b1facf282810af05ac81ccbaabed44`. Its scoped formatting, architecture, GTest-policy,
contribution-shape and diff gates passed. These are source gates only; the three new cases remain
UNEXECUTED and the successor has no independent approval.

Latest fetched main `ee9dcb1e1e2c695426acb25575e7c6c320971d1d` was then integrated by normal
merge `4b4a8ba696a9f055b461dce9442f2eb6bdc9af72`, parents f657f651 and ee9dcb1e, tree
`ea309c0b6e2b23d5e5a50b54a08f1cd827ee58cc`. The complete incoming #4382 diff was inspected:
only `fragment_scalar_bank_fixture.hpp` and `test_ordered_graphics_read_point.cpp` changed,
replacing allocator-location-dependent fixture layouts with owned contiguous programs and
SDK-relative offsets. The merge had no conflicts and session startup reported matching current
instructions. No native build/test was run for the integration. The reviewer still needs to
check this main delta against the helper changes; inspection is not independent approval.

The exact published head and post-merge source-gate outcomes are recorded in the PR integration
comment. Keep the original freeze and its review as historical evidence, not current acceptance.

A subsequent source-only coverage pass adds three UNEXECUTED `FragmentPacketExports` cases:

- `CanonicalPrefetchPreservesArchitecturalPacketAndResourceSinks`: actual compiled module/VM
  and full logical64 raw/typed export oracles for modes1..3, nonzero and genuinely present zero,
  both Architectural routes, shifted original export PC and unchanged commit eligibility.
- `ReservedPrefetchAndClauseRefuseAtTheActualArchitecturalDispatcher`: same-original canonical
  positive control plus named transactional refusals with no SPIR-V/input/output on both routes.
- `ArchitecturalPrefetchDoesNotExpandLegacyRawPacketOrResourcePolicy`: same-original NOP
  positive control and retained LegacyRaw hint/CLAUSE refusals on both routes.

Source inspection corrected the original claim that the separate resource API was wholly
unchanged: its Architectural mode shares the final hint override; its LegacyRaw mode does not.
No functional production change was made in this coverage pass. The source comment and this
handoff now state the actual boundary. No build, configured test discovery, VM execution,
validator, GPU or game run occurred. These three cases are additional to the earlier three;
registered/private-owner negative coverage and independent successor review remain outstanding.

The next source-only attempt addresses the position-free entry-schema finding. The existing
`FragmentDrawEntryRecipe` is now shared with and retained immutably by capacity, instead of
reconstructing live/helper-mask authority from a nonempty position-row vector. Original kernel
entry-mask availability, capacity schema checks, GPU assembly and whole-draw validation select
the explicit recipe, including when ENA=ADDR0. Transaction instantiation checks plan/capacity
recipe agreement. Unknown recipes and rows supplied to the default old recipe refuse. The
private original/launch association is still required; selecting a public enum grants no draw
permission. Existing code/profile/module cache inputs and WAT2 wire layout are unchanged.
The validator factory's helper capacity now also passes the explicit schema; its five existing
helper SOURCE forms remain required, in addition to the position-free retained test forms below.

Two new source-declared `FragmentRasterLaunch` cases are UNEXECUTED:

- `PositionFreeOriginalRetainsItsRegisteredRasterEntrySchema`: real registration, exact raw
  owner/private launch, empty position inputs, actual initial-EXEC schema, complete emitted
  plan, warm reuse, instantiated transaction, unavailable/extra masks and unknown-recipe negatives.
- `PositionFreeLiveMaskSurvivesAssemblyOriginalExecutionAndWholeDrawValidation`: authored
  count/assembly/original-VM/whole-draw validation chain with genuine inline guest writer, nine
  hand-owned offline quads, live/helper separation, upper32 and padding, eighteen expected replay
  indexes, and an upper32 original-site corruption preventing all replay. Six SOURCE forms retain
  distinct `position_free_*` names through the existing opt-in retention path.

The existing raster-row negatives now pass the explicit schema; an old input-free plan neighbor
also asserts its old schema and absent initial mask authority. No native build/discovery/test/VM,
SPIR-V validation, GPU or app/game execution was performed for this attempt. Both repair deltas
still require independent review, calibrated controls and exact-head execution before acceptance.

The SAME independent reviewer subsequently inspected the correction deltas at exact
`6c6235a4b986d688931e3f425f9106bcece5ef1e` and reported **SCOPED NO-FINDINGS**. This covers
the common PREFETCH predicate and checked consumers, the three packet/resource/LegacyRaw test
bodies, explicit position-independent schema/private-launch checks and both new test bodies,
all capacity call sites including the factory, and narrow applicability of main's two fixture
changes. It is not a whole-source CLEAR, test result, registered approval or accepted fix.
All new tests remain UNEXECUTED; the earlier unfinished connected review and native obligations
below still apply. No source change or native activity occurred during the review. The reviewer
stopped after the bounded report; the PR comment retains its exact source identity and limits.

## First focused native attempt: FAILED before test discovery

ROOT subsequently attempted a native Windows/Clang CPU baseline at exact source
`154531e3ab6dcff5092b3b2bbcc8d06d0d102b3e`, parent
`6c6235a4b986d688931e3f425f9106bcece5ef1e`, tree
`10281ce5f1a004dbd9fa1a8ccc00c65992998a76`. Actual configure0 and inspected CPU-only Ninja
graph0, then **build1/helper1**. No binary certification, configured test discovery, CTest,
VM, SPIR-V validation, GPU or game execution followed. The intended population was55 CPU
cases across four explicit targets, NOT55 registered or executed cases:
`test_fragment_raster_launch`, `test_fragment_draw_plan`, `test_fragment_packet_exports`,
`test_fragment_scalar_reads`.

Clang rejects three private accesses in `gpu_execute.hpp`: `selected_empty_words` at3269/3272
and `bind` at3299, observed while compiling both `agc_shader_layout.cpp` and
`capture_collect.cpp`. The current source confirms the concrete signature mismatch:
`fragment_raster_launch.hpp:29` befriends the ten-parameter `realize_draw_item`, while the
actual definition at `gpu_execute.hpp:2060` additionally takes
`shared_ptr<const OrderedScalarBankReadPoint>` and `shared_ptr<const FragmentScalarBank>`.
Those defaulted parameters still belong to its function signature. Fix the friend declaration
to match the actual private producing function; **do not make the authority methods public**.
That was the first next action at the failed freeze. The limited correction result below does
not change or replace the failed cohort.

The bounded SAME reviewer source-cleared only the repaired native apparatus, SHA256
`0ab615e5efae194ab357b5e65cd54a824ff8104e9b554b84d5127ca34acc34db`.
Before execution the reviewer found that a certified PE with a wrong filter could run zero
GoogleTests yet satisfy outer CTest counts. ROOT repaired the exact suite-to-PE/three-argument
filter join and matching per-case RUN/result body checks. Actual hand-built discriminator
calibrations accepted the valid command/body and refused wrong filter/PE, zero tests, foreign
body and skip markers. This calibrates the apparatus only, NOT production shader controls.
The reviewer performed no execution and gave no whole-source or PR approval.

Raw configure/graph/build logs, apparatus calibration and `build-precertificate.json` are in
`.codex/issue-4235-native-20261004/154531e3-focused-baseline/`; the reviewed runner is
`.codex/issue-4235-helper-cpu-native-20261004.ps1` (`-ExecuteNativeProof` was the actual invocation).
APP SHA256 `313eb7fdfe7729c71bb6515a4bb10cf73e538518f2345da419240716c8d4e662` and
process0 remained unchanged before/after; no app/DLL target was in the inspected build graph.
The native handle finished with exit1 and the final native process census was empty. Source
remained clean/exact154 throughout; the later documentation commit supplies no native credit.

## Friend-signature correction: two failed consumers compile

Commit `f3773be0cfcd7daeb79c001374773ae6716c9b27`, parent
`1c26cbbae1ed962df472586fbdc102b208b838c6`, tree
`9acca561183751d10d17f070300ec506a45285d9`, changes only
`fragment_raster_launch.hpp` (+5/-1). It adds matching class forward declarations and replaces
the obsolete ten-parameter private friend with the actual twelve-parameter realizer signature.
The authority methods and constructor remain private; no runtime body or public permission changed.
The SAME reviewer independently inspected the complete header delta, realizer signature and
class definitions: **SCOPED NO-FINDINGS**, not whole-source CLEAR or PR approval.

ROOT then actually compiled both previously failing native translation units,
`agc_shader_layout.cpp` and `capture_collect.cpp`, under exact clean f377. Configure0, inspected
two-object graph0, build0/helper0, two fresh requested compiler commands and object outputs.
The build also regenerated/compiled its revision object and static revision archive; no PE was
linked or certified. **Zero tests executed**, no SPIR-V validation, GPU or app/game execution.
APP313/process0 were unchanged before/after; final handle exit0 and native process census empty.
This demonstrates correction of the observed private-access compilation errors in these two
consumers, NOT a successful full core/test build or runtime acceptance of either helper repair.

Raw receipts: `.codex/issue-4235-native-20261004/f3773be0-private-friend-objects/`.
Actual invocation: `.codex/issue-4235-private-friend-objects-20261004.ps1 -ExecuteNativeCompile`;
SAME source-cleared compile-only apparatus SHA256
`546603ad309c2a83fb2a8c5f99cf52605563490245aff77571e57c214eb128e6`.
Changed-line formatting initially failed and was corrected by scoped formatting; final exact
source formatting, architecture, GTest-policy, contribution-shape and diff gates all exited0.
The next native action is a newly bound/reviewed successor of the four-PE/55-intended-case
baseline, with fresh strict certificates and actual discovery; preserve the original154 runner
and failed receipts rather than changing their identities or borrowing the two-object result.

## First completed focused CPU result: 41PASS/14FAIL, not acceptance

The successor at exact392f3c3a/Pf3773be0/Tf551f5fa configured0/graph0, but build1/helper1:
`fragment_raster_launch.cpp` called nonexistent `RegisterFile::contains`. No certificate,
discovery or tests followed in that cohort. Its raw receipts remain in
`.codex/issue-4235-native-20261004/392f3c3a-focused-baseline/`.

Reviewed correction `de2037b2800dc17731bbe8411a4ef785fa353d21`, parent
`392f3c3aca46d94dd3a36b83ebc675091b6fe320`, tree
`a3c29943d55255775d4c2341cc2e9f0582b36f88`, changes only that query (+2/-1) to const
`find()!=end()`. The actual `RegisterFile` implementation checks keys, preserves present-zero
versus absent, and inserts nothing. SAME scoped source review found no defect; exact source
formatting/architecture/GTest-policy/contribution/diff gates exited0. No whole-PR approval.

ROOT then completed exact clean de203 configure0/graph0/build0 and strict-dirty revision
certification0 for all four requested PEs. Actual unique discovery and execution both55:

| Actual case family | PASS | Ordinary FAIL |
| --- | ---: | ---: |
| FragmentRasterLaunch | 9 | 14 |
| FragmentDrawPlan / FragmentDrawResidency | 6 | 0 |
| FragmentPacketExports | 17 | 0 |
| FragmentScalarReads | 9 | 0 |

CTest8/helper1,1.58s real test time; no timeout, XML error, skip, disabled/inverted case or
missing full body. Every JUnit system-out occurs in retained LastTest. Four PE names, strict
certificate rows and hashes are retained in `cert.log` and `observed.json`; actual registration
joins each name to its correct certified PE and exact filter arguments. Five of the eight new
follow-up cases PASS: the reserved cold-proof negative, scalar-origin case and all three
packet/resource/LegacyRaw PREFETCH cases. The registered helper PREFETCH positive and both
position-free cases FAIL early; their later assertions/VM chain did NOT execute.

Confirmed next actions, independently checked by SAME against actual bodies and source:

1. **Canonical incoming EXEC is rejected by the helper proof.** Eleven failure bodies print
   `fragment-raster-mask-source-unproved:pc=0` or `:pc=1`. The `mask` lambda in
   `fragment_raster_program` requires SGPR before its value126 branch, but
   `decode_src_field(126)` returns `OperandKind::Special` with value126, and SOP1 sources use
   that decoder. Thus that incoming-EXEC branch is unreachable for canonical decoded EXEC.
   Fix the exact canonical mask representation, preserving ordinary saved-mask aliases and
   VCC/M0/numeric/unknown-special refusals; do NOT accept all special registers or invent masks.
   Retain the real red cases and require their positive/negative restored results on a new head.
2. **Two other fixture/cache failures remain unisolated.**
   `SameOriginalProgramCannotBorrowAnotherDrawsEntryObservation` sees different raw-code owners
   where it expects the same one. `WiderParameterProgramStillNeedsItsCompleteEntryAndExecutionRecipe`
   gets `fragment-draw-original-float-mode-unavailable` before its expected vector refusal and
   sees distinct warm plans. Check the actual registered source/profile fixtures and lifecycle;
   do not collapse foreign-owner identity or fabricate float availability to satisfy assertions.
   The helper PREFETCH positive also fails its early cold-proof assertion without printing its
   rejection tag; do not claim a measured tag for that case from inference alone.
3. **The body checker confuses zero passes with zero executions.** Preserved `observed.json`
   reports `WrongGtestBodies=14`: original runner0ab rejects `[  PASSED  ] 0 tests`, which is
   present in each genuine failing one-test run. All55 real bodies have exactly one matching
   RUN and timed OK/FAILED result, and none reports Running0/skip. ROOT's hand-built failed-body
   control is accepted only when that aggregate zero-pass summary is removed; SAME independently
   confirmed the discrepancy. Correct/calibrate a NEW runner version to accept genuine failed
   bodies while rejecting actual zero execution. Do not edit the old receipt or call this green.

Raw full evidence: `.codex/issue-4235-native-20261004/de2037b2-focused-baseline/`.
Actual invocation used `.codex/issue-4235-helper-cpu-rebind-20261004.ps1` with the full de203
H/P/T above, `-Cohort focused-baseline -ExecuteNativeProof`. SAME source-cleared wrapper SHA256
`fad217554c5e63ec4693a2efc0df3e8cdc2deaabc7fe503e483ed0beee64e72a` pins original runner0ab
and changes only four identity/evidence assignments; derived UTF8 source SHA256
`80bb83d6ab3ad267f0d805e81b4bed7d2dcad6a73100062ddcf8769e8222effb`.
The earlier392 wrapper/derived-source hashes remain in its retained invocation/history.
APP313/process0 were unchanged before/after, final native handle exit1/process census empty.
No app/game, Vulkan device/queue, SPIR-V SDK validation or production counterfactuals were run.
The passing packet cases include CPU VM execution; this is not GPU or full-helper validation.
The historical32 Kena-original frontier and unfinished whole-source review still apply.

## Canonical-EXEC correction: 50PASS/5FAIL, new cache negatives exposed

Commit `802b4f80051ee7cbfa53ff912a6b110ffed5fe17`, parent
`df70c42ea18ac9c54d4f2cbff57bb190fceba307`, tree
`a5ff03afa80fbfb2a12cfcdd8938801eb84df00a`, changes two files (+28/-1). The helper proof now
recognizes exactly `OperandKind::Special` value126 before resolving ordinary SGPR saved-mask
aliases. Noncanonical SGPR126 no longer grants incoming-EXEC authority. No general special-register
acceptance, fabricated mask, title condition or changes to alias expiry/whole-quad/live-export
requirements were added. Existing tests assert actual decoded source kinds and add VCC/M0/unknown/
null/inline-zero/undefined-pair and forged-IR negatives; the55-case population is unchanged.
SAME bounded source review reported **SCOPED SOURCE-CLEAR**, not whole-PR approval. Initial
changed-line formatting failed and was corrected by scoped formatting; exact-head final
formatting/architecture/GTest-policy/contribution/diff gates exited0.

ROOT actually configured0/inspected graph0/built0/strict-dirty certified0 all four named PEs.
Actual unique registration and execution both55: **50PASS/5ordinaryFAIL**, CTest8/helper1,
1.97s real test time. FragmentRasterLaunch18PASS/5FAIL; the other32 focused cases all PASS.
Nine previously failing cases now PASS, including the canonical registered PREFETCH positive
through modes1..3, both position-free cases, original helper plan/kernel and assembled-helper
execution. The passing position-free CPU case now reaches its count/assembly/original-VM/
whole-draw validation and upper32 corruption assertions. These are CPU VM/source assertions,
NOT physical helper pixels, SPIR-V SDK legality or full general Wave64/Kena acceptance.

Five failures remain; preserve their complete actual bodies and prioritize the newly exposed
authority negatives rather than weakening their expectations:

- `PendingEmptyFsCannotBorrowWarmCodeOrAnotherDrawObservation`: kind0 clears the draw's private
  `launch_source`; preparation retains no launch source, yet the cache returns the SAME warm
  successful plan with capacity. Cold raster compilation independently requires the private
  source, whereas `profile_key` checks it only when `input_free_layout` is false. Inspect this
  conditional admission before warm lookup and preserve valid identical-code reuse.
- `FreshRegisteredGenerationAndPhysicalLaunchCannotReuseOldPermission`: forging the old
  launch source onto a new captured draw likewise gets a successful cached plan despite missing
  prepared launch authority. Its transaction checks still run separately. The actual
  `instantiate_fragment_draw_transaction` raster branch checks private-source identity/match,
  so these are observed cache-admission failures, NOT demonstrated attachment publication.
- `SavedMaskNumericalExposureAndPartialAliasReplacementStayUnproved`: all newly added negatives
  produce no assertion failures, but the preexisting high-word replacement gets
  `fragment-raster-quad-local-program-unimplemented:pc=5`, not the expected mask refusal atpc6.
  The authored word `0xbe950080` decodes SOP1 opcode0; the actual `kSop1OpcodeMovB32` is0x03.
  Correct the intended s21 zero-write encoding (`0xbe950380`) and assert its actual decode before
  using it to test overlapping saved-mask expiry. This fixture correction is NOT yet made.
- `SameOriginalProgramCannotBorrowAnotherDrawsEntryObservation` and
  `WiderParameterProgramStillNeedsItsCompleteEntryAndExecutionRecipe` retain the previously
  recorded raw-owner/float-mode/warm-reuse failures. Causes remain unisolated; do not weaken
  genuine profile knownness or foreign-owner boundaries to make them pass.

The immutable old runner's `WrongGtestBodies=5` is still its separately documented zero-passes
classification defect. ROOT independently joined all55 full JUnit bodies to LastTest and found
one exact matching RUN/timed OK-or-FAILED per body, no Running0/skip markers or missing bodies.
There are no timeouts/XML errors/disabled/inverted/bad-state cases. The actual five ordinary
failures still make the baseline red regardless of the classifier bug. SAME independently read
all five failed bodies and checked the cache/transaction guard discrepancy and malformed opcode;
this is a bounded artifact/source audit, not execution or approval. No production-only
counterfactual/restoration campaign was executed, and the broader connected review is unfinished.

Raw receipts: `.codex/issue-4235-native-20261004/802b4f80-canonical-exec/`, including all four
strict PE certificates/hashes, actual graph/discovery, JUnit/LastTest and preserved observed.json.
Actual invocation used the unchanged reviewed rebind wrapper with full802 H/P/T above,
`-Cohort canonical-exec -ExecuteNativeProof`; derived UTF8 source SHA256
`63ff25ed5296c8fc21c4d5a4c3fcb51941073e3e21ca49aaa3b256946deb7080`.
Native handle finished exit1 and final native process census0. Source was exact/clean802 for the
cohort. APP313/process0 stayed unchanged before/after; no app/DLL/game/device/queue/SDK validation.
Do not transfer this failed cohort to a later documentation/source head or overwrite de203's
41PASS/14FAIL receipts. No whole-source APPROVED; draft #4384 remains unmerged.

## Cache-authority correction: 53PASS/2FAIL, still not acceptance

Commit `cae82d776749c8a9ad8cf02281e39ea896fb7a01`, parent
`78f4b6daa43ceaed45c6d5908654e6b610cada79`, tree
`e48185acc1c96b21231f186bb063a54d7163ef94`, changes two files (+36/-3). Raster-enabled profile
keys now include the actual checked private launch availability: a matching retained source and
prepared source, plus `matches(in)`. Both weak source aliases and copied-code cache keys use this
same profile. Missing/foreign authority cannot borrow a good warm helper plan or poison a later
valid cold query. No per-draw owner ID was added to the copied-code key; valid different draw
observations still reuse compiled SOURCE. Non-input-free early refusal and input-free/nonraster
cold recipe rules are preserved. Cold helper and independent transaction checks are unchanged.

The PendingEmpty regression now queries missing authority before the good compile, retains all
five forged variants, and restores the valid warm lookup after each one without extra compile
calls. The genuine high-word write is corrected to `0xbe950380`; the test explicitly checks its
actual decoded pc5/SOP1/op03/s21/inline-zero before the existing pc6 alias-expiry refusal.
SAME bounded source review reported **SCOPED SOURCE-CLEAR**, not whole-PR approval. Initial
changed-line formatting failed, then scoped formatting and exact-head source gates all exited0.

ROOT actual clean cae82 native cohort: configure0/graph0/build0/strict four-PE certificates0,
55 unique registrations and executions, **53PASS/2ordinaryFAIL**, CTest8/helper1,1.92s real time.
The two warm-cache negatives and corrected partial-mask test now PASS, including cold poisoning,
all forged variants and restored genuine warm/no-recompile checks. FragmentRasterLaunch21PASS/
2FAIL; the other32 focused cases all PASS. The remaining two failures are exactly the earlier
`SameOriginalProgramCannotBorrowAnotherDrawsEntryObservation` raw-owner inequality and
`WiderParameterProgramStillNeedsItsCompleteEntryAndExecutionRecipe` float-mode-unavailable/
warm-plan inequality. Their causes remain unisolated; retain assertions and authentic fixture
registration/profile authority while investigating, rather than changing return expectations.

Full evidence is in `.codex/issue-4235-native-20261004/cae82d77-launch-authority/`.
Actual rebind invocation used full cae82 H/P/T above and
`-Cohort launch-authority -ExecuteNativeProof`; derived UTF8 source SHA256
`a483ef2bb284cdf8fee30001836dcff0dfc82e98933dee55869ee0eaf5174278`.
ROOT independently checked all55 matching RUN/timed-result/full-LastTest body joins, with no
Running0, skips, timeouts, disabled/inverted cases, errors or bad states. Preserved
`WrongGtestBodies=2` remains the old runner's zero-passes false positive; the two genuine
failures independently keep the cohort red. Both complete failed bodies and three newly passing
bodies were read. Final native handle exit1/process census0; APP313/process0 unchanged.
No production-only controls/restorations, SPIR-V SDK validation, Vulkan device/queue/helper
pixels or game run. This is a verified bounded CPU correction, not general Wave64 acceptance;
the full-original Kena frontier, missing connected review and whole-PR approval still remain.
Historical41/14 and50/5 receipts are preserved, not replaced or relabeled.

## Authentic fixture correction: first focused CPU baseline55/55 PASS

Commit `8819e5c475d6cb73b818bd2cd00aa43215d60fcb`, parent
`a406039d0bbfe4b00c6ad865b6f70a04ca15b39c`, tree
`f1816ed4bc21a52841f162d8548346fb852cd513`, changes only three TEST paths (+44/-10).
No production code or acceptance/refusal assertion was weakened:

- The old `fragment_draw_fixture::realize` registered a fresh immutable program on every call.
  The same-original test therefore compared two genuine but DIFFERENT registered generations.
  `register_programs` retains the real HLE mapping/AGC registration and process lifetime;
  `realize_registered` observes each draw through the real state writes and normal realizer.
  The test now registers ONE authentic immutable program and realizes two draws with different
  genuine USER_DATA. Original raw-owner equality, unequal entry observations, cross-entry refusal
  and self-match assertions remain. Convenience `realize` still creates fresh generations for
  other callers; no copied owner bytes, rewritten registered words or post-realization stamping.
- The pull fixture never wrote physical `SPI_SHADER_PGM_RSRC1_PS`, so the wider-parameter test
  correctly stopped at float-mode-unavailable before its intended program refusal. Its optional
  RSRC1 argument now writes only a caller-supplied word BEFORE normal realization. Default stays
  genuinely absent, and supplied zero is present. The test adds the missing-profile negative with
  no capacity, then supplies the real IEEE profile for the original target. Exact
  `fragment-raster-vector-source-unproved:pc=0`, warm-plan identity, changed-device refusal and
  empty-entry/unready-transaction assertions remain intact.

The SAME independent reviewer inspected exact8819, complete fixture bodies, immutable ownership,
real registration/entry writes and optional presence semantics: **SCOPED SOURCE-CLEAR** only.
It performed no execution and gave no whole-PR approval. Proactive changed-range formatting,
architecture/GTest-policy/contribution-shape and diff gates all exited0.

ROOT's actual clean8819 cohort configured0, inspected the CPU-only Ninja graph0, built0 and
strict-dirty certified0 ALL FOUR explicitly requested PEs. Actual unique configured discovery55,
actual JUnit executions55, **55PASS/0FAIL; CTest0/helper0;4.17s real time**. Families: raster
launch23, plan/residency6, packet exports17, scalar reads9. No errors/skips/disabled/inverted/bad
states/timeouts/missing bodies. ROOT independently joined every full JUnit body to LastTest with
CRLF-to-LF normalization only, exactly one matching RUN and timed OK per case, no Running0/skip,
and rehashed all four current PEs against the retained certificate/observed hashes. Both newly
passing complete bodies were read. Prior41/14,50/5 and53/2 cohorts remain immutable failed history.

Raw evidence: `.codex/issue-4235-native-20261004/8819e5c4-fixture-owners-profile/`, including
calibration, configure/graph/build/precertificate/certification, actual registration, full CPU
log/JUnit/LastTest and observed.json. Actual invocation from ROOT's owned worktree:

```powershell
& '<REPO_ROOT>/.codex/issue-4235-helper-cpu-rebind-20261004.ps1' -SourceHead 8819e5c475d6cb73b818bd2cd00aa43215d60fcb -SourceParent a406039d0bbfe4b00c6ad865b6f70a04ca15b39c -SourceTree f1816ed4bc21a52841f162d8548346fb852cd513 -Cohort fixture-owners-profile -ExecuteNativeProof
```

Unchanged reviewed rebind wrapper SHA256
`fad217554c5e63ec4693a2efc0df3e8cdc2deaabc7fe503e483ed0beee64e72a` pins old runner0ab;
derived UTF8 source SHA256 `c214476bd54357e210c0686355f9cdd5462bd98f4dc8d2985d5009f5545f3187`.
Certified PE SHA256 identities:

| Target | SHA256 |
| --- | --- |
| `test_fragment_raster_launch` | `691f247aa8da23cb8c3fecff51ce621518c2efd2585b7de9850c72a5dd4545d2` |
| `test_fragment_draw_plan` | `55eb61e6c700dddba426e5fd4966ab3f2ecc64eca5092f545cd5a41855cf8cba` |
| `test_fragment_packet_exports` | `4cb4837642fb789d0a2353e063acfef80c8d9fecf14b81d529e661e4bcc477e0` |
| `test_fragment_scalar_reads` | `89e7f2013f6982ba0cc0b98ae8ceb236c07c70b1d4bb4d0babe79ab9236aed02` |

Native handle terminal0; final process census0. APP313/process0 unchanged before/after, no
app/DLL target, app/game launch, Vulkan device/queue, helper pixels or SPIR-V SDK validation.
Passing CPU VM/source assertions are not physical rendering. No production-only omission controls
or restorations, expanded full helper campaign or complete connected review was executed.
The old runner's zero-passes classification defect remains UNFIXED; wrong-body0 in this all-pass
cohort does not calibrate genuine FAILED bodies. Version/calibrate a NEW apparatus, retaining old
script/receipts. The historical Kena original frontier is still0/32 under the necessary-condition
assessment below, NOT a current dropped-shader measurement. General Wave64/Kena acceptance and
registered whole-source APPROVED remain outstanding; draft #4384 must not merge on this result.

## Integrated connected baseline: actual helper pixels PASS,96/99 overall

Normal merge `aab7ac2db1f3c0ba7ae211fc8878a8e40e9ac48e`, parents
`f7dc8ce43edf1eb423aec418b0a920fa9430dbb9` and
`9b2f0681772ad70972233244868010db0ffc8af8`, tree
`2fa2ca2fa0035b015558e57ae9cfb0c1eb91ba0f`, integrates main's #4386 decoder test/CMake
delta without conflicts. Startup instructions match, ahead23/behind0. The standalone new decoder
target is NOT in this cohort; no execution credit is assigned to its nine cases.

ROOT independently matched current source declarations to the NEW99-name/ten-target apparatus:
the earlier55 plus composition8, parameters5, quad-swizzle5, render-state9, capture9 and draw8.
This is92 CPU and7 Vulkan cases; the draw target also contains one CPU-only limits case despite
its CTest gpu label. Source inventory is distinct from the subsequent actual discovery/execution.

Actual clean aab7 configure0/inspected graph0/build0/strict-dirty certification0 for ALL TEN
explicit PEs; actual unique registration99 and JUnit execution99. **89CPU PASS/3ordinaryFAIL;
7/7Vulkan PASS;96/99 overall; CTest8/helper1;4.83s real time**. No errors/skips/disabled/
inverted/bad states/timeouts/missing bodies. ROOT independently joined all99 full JUnit bodies to
LastTest with CRLF-to-LF only, one exact matching RUN and timed OK/FAILED per case, and checked
all ten current PE hashes against observed.json. The complete three failed bodies, strict cert
log and physical helper body were read. Genuine FAILED bodies now classify correctly:Wrong0.

The actual helper test
`FragmentDrawExec.SavedLiveWqmReadsActualUncoveredHelperPositionThenRestoresExportMask` reports
NVIDIA GeForce RTX4090/discrete, successful enabled float-transport/device context, and three
normal-backend original transactions at16x12,15x12,16x12. Every asserted channel passes, including
live pixel14 selecting uncovered helper15's POS_X15.5 via the unchanged original saved-EXEC/WQM/
DS alias/LGKM/restored-live export. Six actual draw/pixel cases PASS with14 retained transaction
markers; the seventh Vulkan case checks object lifetime. No game/app launch or title image.
This is physical GPU evidence for this proved class, NOT full Wave64 or a Kena milestone.

The three CPU failures are tracked in [#4402](https://github.com/mattias800/prosper/issues/4402):
`PacketQuadSwizzle.CounterSpecificCompletionCannotBeBorrowedFromVmOrExport`,
`AllPathWaitAndOverwriteRemainSeparateLoadBearingGuards`, and
`TopologyModesAndLegacyPolicyRemainExplicitNotBroadDsAdmission`. Their returned rejection
strings contain the expected bare tags; the test wrongly expects appended`:pc=N`. The API
separately logs terminal`pc=N reason=TAG`. Actual wait/overwrite/legacy refusals are at6/7/4/2;
unknown topology refuses earlier at WQMpc1, not the assumed DSpc2. Preserve both reason and
actual-site assertions by checking the terminal diagnostic with a nonzero observation-only
identity, and retain empty artifacts/positive controls. Do not merely remove PC assertions or
change the established public reason-string contract. These corrections were NOT implemented in
this aab7 cohort; the subsequent exact9972 correction and new receipts are recorded below.

NEW reviewed apparatus `.codex/issue-4235-helper-connected-native-20261004.ps1` SHA256
`11c91d8f47ee35a924e40a4a65a57ca8352487072a4888e6410deb5aa58e9af9`, derived UTF8 source SHA256
`28de1128a77f5b2b9326b3dacf2d49287c11795af2ebf91d5d8b6dd75d2d44bb`, pins immutable old0ab.
It fixes genuine FAILED/zero-PASSED classification while refusing actual Running0/foreign/skip/
status-mismatch bodies. Apparatus review found an initially overbroad GPU-success flag; before
native execution it was corrected to require statusrun, no failure/error/skip nodes, matching
RUN+timedOK and transaction marker. Actual hand-built discriminator controls and bounded source
rereview passed. This is apparatus calibration, NOT production shader omission/restoration proof.

Actual invocation from ROOT's owned worktree:

```powershell
& '<REPO_ROOT>/.codex/issue-4235-helper-connected-native-20261004.ps1' -SourceHead aab7ac2db1f3c0ba7ae211fc8878a8e40e9ac48e -SourceParent 'f7dc8ce43edf1eb423aec418b0a920fa9430dbb9 9b2f0681772ad70972233244868010db0ffc8af8' -SourceTree 2fa2ca2fa0035b015558e57ae9cfb0c1eb91ba0f -Cohort connected-baseline -ExecuteNativeProof
```

Raw cohort `.codex/issue-4235-native-20261004/aab7ac2d-connected-baseline/` retains graph,
build/precertificate/certification, calibration, actual discovery, connected.log/JUnit/LastTest,
all ten exact PE hashes and observed.json. Native handle terminal1/process census0; APP313/
process0 unchanged before/after and no app/DLL target. No SPIR-V SDK validation or production-only
controls/restorations; additional old capture/diagnostic/bank/default-factory neighbors remain.
All earlier failed cohorts and8819's focused green receipt retain their own source identities.

### Connected review: three production blockers, NOT CLEAR

SAME reviewer completed the previously missing launch/GPU fixtures, eight bank-conflict contexts,
uniform DS phases, validator/replay bounds, completion retention and factory/discovery wiring at
exact aab7. It reports **SOURCE REVIEW NOT CLEAR**, with three concrete findings; ROOT checked the
relevant actual producer/collection/plan/pipeline/guard fields. No wrong pixels, VUID, race or title
FPS effect was measured for these untested variants:

- [#4397](https://github.com/mattias800/prosper/issues/4397): physical DB_DEPTH_CONTROL bit31 is
  admitted by the`&0xf` workitem guard but its color-suppression meaning is not implemented by
  resolved attachment checks/replay. [AMD PAL's field mask](https://raw.githubusercontent.com/GPUOpen-Drivers/pal/dev/src/core/hw/gfxip/gfx9/chip/gfx9_plus_merged_mask.h)
  identifies`DISABLE_COLOR_WRITES_ON_DEPTH_PASS=0x80000000`; a bit name alone is not a measured
  pixel oracle. Program the genuine bit before realization, require honest named refusal until
  supported, and ultimately implement the actual general color-write behavior.
- [#4398](https://github.com/mattias800/prosper/issues/4398): center interpolation plus packed
  POS_X can pass complete-original proof and require generated collector coefficient GS, but
  shipping`FragmentDrawCollectGpuProgram::acquire` binds exactly VS+FS. Connect the actual retained
  GS with same-device enabled feature/limits/interfaces/effects/lifetime checks and a genuinely
  registered center-plus-position GPU discriminator. Passing position-only pixels cannot clear it.
- [#4399](https://github.com/mattias800/prosper/issues/4399): the normal realizer creates a fresh
  immutable generated-GS owner every draw, while profile/generation checks require that exact
  identity. Repeated identical pull/linear draws, including refused plans, cannot remain warm.
  Cache actual derived GS by producing source/layout/transport/device shape; preserve private
  per-draw entry/launch authority. Add real repeated registrations with nonempty GS; canonical
  empty-GS17-key warm tests do not discriminate this path.

All three are in the UNMERGED helper slice, not newly verified main behavior. Full general
input/system/M0/SMEM/resources/control/export scope and the historical0/32 original frontier still
apply. Do not merge the draft on seven passing Vulkan cases or substitute a position-only goal.

## Latest verified correction and parked source branches

Exact executed source `997243e7800c4c1c059f64c2e0fae65638ca11a5`, parent
`8db1cb7fdd8387b360db01d5513011d71945442c`, tree
`cb3625611582f266d0a2e14f69d0c8a7a8e466b8`, changes only the quad-swizzle test (+38/-11).
It asserts bare reasons, freshly captured terminal tag/PC/payload, nonzero observation-only
identity and all three empty output artifacts. Genuine branch-through-wait and overwrite-after-
wait peers also compile. SAME reviewer reports scoped SOURCE-CLEAR/no findings; actual incoming
9b2f->0cf seven-path test/CMake range is applicability-clear, not executed neighbor coverage.

Normal merge8db integrates `0cf6760452106caac973e71cadeecdc316dfce7b`: #4390/#4396/#4401
test/CMake corrections, no conflicts,
instructions match. The exact source above subsequently configured/built/certified all ten PEs
and executed all99 unique names: **92CPU PASS +7Vulkan PASS; CTest0/helper0;4.65s**. ROOT independently
joined99 full UTF8 bodies to LastTest with CRLF-to-LF only, checked all ten current PE hashes,
and read the three corrected bodies plus actual16x12->15x12->16x12 helper transaction body.
No failures/errors/skips/timeouts/wrong bodies. APP313/process0 unchanged; no app/DLL/game build.
SDK legality and production omission/restoration campaigns remain UNEXECUTED.

```powershell
& '<REPO_ROOT>/.codex/issue-4235-helper-connected-native-20261004.ps1' -SourceHead 997243e7800c4c1c059f64c2e0fae65638ca11a5 -SourceParent 8db1cb7fdd8387b360db01d5513011d71945442c -SourceTree cb3625611582f266d0a2e14f69d0c8a7a8e466b8 -Cohort diagnostic-contracts -ExecuteNativeProof
```

Raw receipts: `.codex/issue-4235-native-20261004/997243e7-diagnostic-contracts/`.
Unchanged reviewed runner SHA11c91d8f above; exact derived UTF8 source SHA256
`9333d4846849ab3799fa0a8ddb477bc3b8f0a32e012b04d2c740f67f0fe96b39`.
Native handle is terminal0; no live process or pending native work. #4402 is corrected in the
unmerged draft, not closed on main. Documentation successors receive no new executable credit.

Two source agents committed separate public correction checkpoints at the maintainer's latest
quota checkpoint request. **Neither is integrated into this draft's executed source, independently
reviewed, compiled, discovered or executed.** Source gates alone are not acceptance:

- #4398: [8e0f2064](https://github.com/mattias800/prosper/commit/8e0f2064dcf21ab948d3a78cee32df15c09ded16),
  branch `fix/issue-4398-collector-geometry`, parent8db, tree
  `9b085eeadf699e67fa5c2e241e5a4dcfed749951`; six paths +374/-10. Attempts retained actual GS
  stage/device/limit/interface/effect/lifetime wiring plus separate
  `FragmentCollectorGeometry.ActualCenterIjProducerSurvivesHelperPackingAndOrderedReplay`
  in `test_fragment_collector_geometry`. Existing bounded diagnostic supplies a numerical
  center-IJ/POS witness, with normal16/15/16 attachment pixel assertions; no callback seam.
- #4399: [ef87c026](https://github.com/mattias800/prosper/commit/ef87c026a96666dc680eeea1845f82e9fae2f900),
  branch `fix/issue-4399-derived-geometry`, parent9972, tree
  `f3e976027391a331954c825caa76b67e34945ede`; five paths +279/-6. Attempts weak immutable
  producer/profile-keyed GS residence while keeping fresh per-draw launch authority. Two new
  `DerivedInterpolationGeometry` cases in `test_derived_interpolation_geometry` cover17 real
  nonempty-GS observations and changed source/layout/transport. All these cases are UNEXECUTED.

Latest ROOT-fetched main is `a524d233f9d8522add93dc7ead21e95509339a23`, NOT integrated;
the intervening15 commits include real decoder/ALU changes. Do not call9972 current-main evidence.
Resume by inspecting/integrating those deltas, independently reviewing both source attempts,
then extending actual registration/certification and running new numerical/production controls.
#4397 still has no correction. All three production issues remain open; CI remains waived, not green.

## Objective and acceptance boundary

Implement accurate, general guest Wave64 on hosts without native Wave64. Kena's title-screen 3D
is an important end-to-end target, not permission for a game allowlist, custom shader, full-mask
shortcut, fabricated input/resource or narrower permanent goal. A passing small shader recipe
does not establish general support. Remaining original instructions, system/parameter inputs,
control flow, helper/EXEC semantics, resources and complete attachment effects remain required.

This lane's native machine is **Windows / dedicated NVIDIA RTX 4090**, not historical Linux/AMD
unified-memory investigation hosts. Keep platform/source identities separate. Reuse authentic
dumped original shaders for offline work; rerun Kena when integrated admission has a meaningful
chance of improving the real picture. Future interactive runs should preserve the requested
input/audio and25% volume. A screenshot/pixel oracle must come from the normal Windows app path.

## Source identities and durable refs

| State | Exact identity | Evidence/status |
| --- | --- | --- |
| Latest fetched main, NOT integrated | `a524d233f9d8522add93dc7ead21e95509339a23` | Fifteen commits after the executed source's integrated0cf; decoder/ALU/diagnostic deltas require fresh applicability review and execution. |
| Latest integrated main | `0cf6760452106caac973e71cadeecdc316dfce7b` | #4390/#4396/#4401 test/CMake range, integrated at8db before9972's99/99. Additional standalone decoder targets are NOT in that cohort. |
| Earlier connected main | `9b2f0681772ad70972233244868010db0ffc8af8` | #4386 decoder tests/CMake only; integrated at aab7's96/99. Its standalone nine-case decoder target was NOT executed. |
| Earlier main integrated by8819 | `ee9dcb1e1e2c695426acb25575e7c6c320971d1d` | #4382 changed two scalar-bank/ordered-producer fixture files; integrated at4b4a8ba6. The8819 focused CPU receipt belongs to this earlier source, not aab7. |
| Accepted main integrated by helper | `3fe269ef0d035a46baee750af30b6a0004d4f7b0` | Includes merged #4374. |
| Frozen helper CODE head | `3fcb2d181367505ebb5672e8d3cfe2b98396baed` | Parent `59995895857475b9971658ba7bacacd526b6c444`; tree `a75d5db570e7a47e597de52d77c343417273b4c6`. Source-only, partial review. Later checkpoint-document commits do not provide native credit. |
| Frozen Windows artifact fix | `215ee6d29b29177c70a44ca6ae2abb11e0656d84` | Parent3fe; tree `e18dffd3803f551f20a3cd9079875205f0cb2da3`. Native baseline FAILED; [draft PR #4383](https://github.com/mattias800/prosper/pull/4383). |

Helper branch: `fix/issue-4235-fragment-launch-composition`.
Protected local helper ref: `proof/issue-4235-helper-source-3fcb2d18`.
Windows branch: `fix/issue-4378-native-refused-shader-paths`.
Do not rewrite these historical code checkpoints or failed receipts to conceal review findings.

Helper history remains separable: mechanical extraction
`815d3201961af13e3716fbf111464e3fa649a369`, semantic work
`bff64e068804f777dae4c7ffacfa639f4a9d4066`, normal accepted-bank merge59995895,
then factory/format3fcb2d18. The reviewer independently reconstructed the extraction body/hashes
and inverse; this is mechanical equality, not compilation/runtime proof.

## Already accepted and actually verified

[Merged PR #4374](https://github.com/mattias800/prosper/pull/4374) and
[issue #4322 checkpoint](https://github.com/mattias800/prosper/issues/4322#issuecomment-5978505672)
record the bounded scalar-bank prerequisite, independent exact-head review and failed history.
Actual executed source was `9e595b11f75fe2a3f05ad81e3fe1bf6c7d083bc3`, parent
`3c32c551ba6ff35883d988d931b244847724760f`, tree
`ba56de10bb401c1002754baf7728eba60c0871b9`:

- CPU60/60 PASS, eleven explicitly certified PEs.
- GPU17/17 PASS on RTX4090, three explicitly certified PEs,19 completed bounded observations.
- SPIR-V585 actual validated modules:6 genuine bank-plan modules plus579 preserved default
  outputs, emitter coverage29/29, six independent calibrated SDK revalidations0.
- APP SHA256 `313eb7fdfe7729c71bb6515a4bb10cf73e538518f2345da419240716c8d4e662`
  unchanged; no app/game execution in those proof cohorts. CI was waived, NOT green.

One immutable demanded-byte bank is uploaded once and shared across logical waves with original
code/profile/site/schema, descriptor/allocation/read-point authority and counter-specific
readiness. No per-wave payload copy, invented descriptor backing or title branch. Arbitrary guest
CPU byte-writer exclusion is NOT proved; submitted-input stability remains required.

Do not transfer these certificates to the helper draft, integrated main or Windows215 binaries.
The accepted first readonly x1/x2/x4 straight-line recipe admits **zero complete programs from
the historical joined32 Kena originals**. Main3fe was accepted via independently checked source
projection, not a fresh native execution of that merged hash.

Failed experiments and qualifications are retained. In particular: programmed ROP3 CLEAR, not
zero scalar data, caused the black bank-fixture pixels; a private-wire injector initially targeted
s2 rather than s3 because it omitted two prefix DWORDs; structural barrier omissions are not
measured hardware races/VUID claims; failed3c's wrong tool-role checks are not retroactively a
passing cohort. See `prosper/docs/gpu/RECOMPILER_REMAINING.md`'s Ruled out rows and #4374 review.

## Helper/launch source implemented, but not accepted

Frozen3fcb delta against3fe:60 paths,+3264/-150. Production44 paths includes the shipping backend
companion; tests/data11; contribution maps3; one registration path; one factory path. No new
native CPU, VM, validator, SPIR-V, GPU, app or game execution has been performed for this slice.
Source formatting/architecture/contribution/GTest/diff gates passed; they are not native proof.

The source connects a separately typed, privately draw-bound original-PS recipe: registered raw
owner/control-block/generation and physical launch; genuine full-quad/helper backing; saved
incoming live EXEC -> WQM -> basic-quad DS_SWIZZLE -> LGKM completion -> restored live EXEC ->
original EXP. The collector gathers quad values unconditionally before non-helper election;
helper SSBO stores are not relied upon. All-wave provenance/index/export validation precedes
whole-draw attachment replay. Existing old and scalar-bank recipes retain independent authority.
The initial physical raster class is single-sample with required depth/kill/coverage controls;
this restriction is a current proof frontier, not the full objective's final scope.

Principal seams:

- `prosper/src/gpu/execute/fragment_raster_launch.{hpp,cpp}`: private exact original/launch binding.
- `prosper/src/gpu/execute/fragment_raster_program.cpp`: complete cold helper-program proof.
- `prosper/src/gpu/execute/fragment_raster_contract.cpp`: actual present physical raster controls.
- `prosper/src/gpu/recompiler/fragment_packet_quad_swizzle.cpp`: quad mode and completion proof.
- `prosper/src/gpu/execute/fragment_draw_plan.cpp`: distinct original entry recipe and composition.
- Connected collector, capacity/assembly/validator/replay, realizer/preparer/backend/cache/factory
  files are enumerated in the frozen publication artifact below and in the public branch diff.

Source declares42 added names (41CPU+1GPU). Its minimum seven-target campaign is63 names
(56CPU+7Vulkan, including21 preexisting names), ALL UNEXECUTED for this slice. Preserve additional
changed-capture, accepted bank/cache/diagnostic and default-emitter neighbors, not just these63.
Derive exact current registrations from real configured discovery before claiming a population.

The designed GPU discriminator is
`FragmentDrawExec.SavedLiveWqmReadsActualUncoveredHelperPositionThenRestoresExportMask`:
widths16->15->16, destination/source alias, edge live pixel14 depends on uncovered helper15's
nonconstant POS_X=15.5, then restored live export. Native fallback/zero input/helper exports must
not silently pass. This oracle is authored, NOT executed. Optional retained swizzle SOURCE uses
`PROSPER_PACKET_QUAD_SWIZZLE_SPV_DIRECTORY`; connected draw SOURCE uses
`PROSPER_FRAGMENT_DRAW_SPV_DIRECTORY`. The exact source receipt lists five real helper factory
forms and six connected emitted names; none has a fresh validation result.

## Two confirmed helper findings at the original freeze

Both now have UNVERIFIED source-only attempts described above. The following records the original
findings and their required acceptance work; neither is an accepted fix yet.

1. **Cold proof and actual PREFETCH/CLAUSE dispatch disagree.**
   `fragment_raster_program.cpp:127` treats every SOPP0x20/0x21 as ignorable. Canonical PREFETCH
   requires SIMM16 exactly1..3; CLAUSE has its own length/type rules. Downstream mask/scalar/final
   packet gates reject these, so pending-empty-FS proof can approve a program which later fails
   complete plan compilation. No attachment corruption was demonstrated. Required successor:
   one common canonical PREFETCH predicate through proof/effects/mask/scalar facts and actual
   Architectural dispatch; retain LegacyRaw/default policy, and precisely refuse CLAUSE until
   its span/type proof exists. Add genuine registered/private-owner/final-compile PREFETCH3
   positive plus reserved0/high-field/CLAUSE negatives.
2. **Position-free originals lose their raster entry schema.**
   A valid saved-live/WQM/inline-constant -> DS/WAIT/restored-live/EXP program may have ENA=ADDR0
   and no position columns. Plan/capacity/assembly/whole-draw validation infer raster entry from
   nonempty `raster_inputs`, losing the typed contract. This causes compile/refusal/wrong old
   validation shape; unsafe publication was NOT shown. Required successor: explicit immutable
   entry recipe through plan/profile/cache/capacity/assembly/validator/factory, independent of
   column count. Prove a real zero-position raster positive and retain old absent-mask negatives.

The independent SAME reviewer was `/root/silksong_windows_boot`; author `/root/packet_entry_abi`.
At the original freeze only partial connected review existed, **no whole-source CLEAR or APPROVED**.
Those completed reads covered
private launch/program/contract, collector quad/election body, composition/parameters/swizzle,
resource/definedness/mask/scalar/VGPR changes, plan/cache/capacity/assembly/validator diffs and
realizer/preparer/backend changed hunks, state/coverage/capture diffs and mechanical extraction.
Relevant AMD EXEC/WQM/DS and Vulkan/SPIR-V helper contracts were actually read.

Remaining review at that freeze: remainder of launch tests/GPU producer fixtures; full connected backend
lifecycle and eight bank-conflict contexts; CFG/uniform barrier participation; full attachment
validation/replay bounds; actual PAL fixed-function masks; factory/CMake emission/discovery;
per-draw generated-geometry cache behavior; both future correction deltas and new tests. A new
reviewer must independently establish these, not inherit the earlier partial reads as approval.
The later aab7 connected review above completed these reads and found three additional blockers;
it is NOT CLEAR or approval, and later corrections must be re-reviewed.

## Historical Kena originals: actual next frontier

Input index `.codex/wave64-20261001/offline-corpus/root-rejected-census-20261002.json` joins32
complete original bins under the historical Windows36133651 compiler-source cohort. It records307
original compiler attempts,62 backend refusals/39 distinct source entries; **not current drops**.
The checkpoint map physically rehashed32/32 complete bins and matched the indexed hashes.
Historical complete decoder records were consulted, not rerun. No game bytes are committed here.

**0/32 complete originals satisfy the new helper recipe.** Necessary-condition source assessment,
not current compiler/game admission: first unmet classes18SMEM,12M0,1other VALU,1existing4096-owner
budget (real original4946DWORDs). Every retained opcode set contains SMEM/MIMG/VOP2 and branches;
none has the newly proved DS0x35 (two haveDS0xb1). Retained metadata has25VINTRP,21WQM and25wide
S_BUFFER originals. Do not truncate the4946-word program or make these frontiers permanent.

The next general connected work is not another Kena exemption:

1. Authentic direct/table S_LOAD origin chains and x8/x16 S_BUFFER demand/readiness support.
2. Actual packed post-user system/M0/primitive/virtual-LDS association, independent per-vertex
   P0/P10/P20 coefficients and real ENA/ADDR/BC-centroid packing. Final smooth interpolants or
   arbitrary user words cannot manufacture this authority.
3. Preserve incoming live EXEC/helper backing across original branches, WQM, numeric mask/VCC
   exposure and legal logical64 scheduling/primitive/region/realign/collision constraints.
4. Genuine image/resource operations, derivatives/quad cooperation and non-basic DS services.
5. Compressed/multiple/VM0/non-MRT0 architectural exports and whole-draw attachment composition;
   wider source/float/raster/depth/sample classes. No required output may be fabricated.

## Windows shader-evidence lane: native baseline FAILED

[Draft PR #4383](https://github.com/mattias800/prosper/pull/4383) preserves215 and the detailed
seven-path host-platform/dumper investigation. Held v2 source and70/12 CPU apparatus were
independently source-cleared, not natively approved. ROOT completed exact215 configure/graph/
build/strict certification0 with12 named PEs and70 actual unique registered/executed cases:
**67PASS,1ordinaryFAIL,2TIMEOUT; CTest8/helper1;242.29s**. No skips/disabled/missing bodies; the
timeouts are abnormal, not calibrated expected failures. Retained APP313 and process0 unchanged;
no app/GPU/game execution. Final native handles are terminal and process census was empty.

- Both `RefusedShaderDump.LongCaptureRootKeepsRawIndexAndOncePerOwnerWork` and
  `.RelativeLongRootIsResolvedBeforeNativeDirectoryAndFileOperations` time out around120s.
  Producer messages report successful raw writes; native `only_bin` enumeration finds0 .bin and
  asserts before timeout. Exact stuck operation/cleanup cause is NOT isolated.
- `FileStdio.ExtendedDriveUncAndAlreadyExtendedSpellings` fails ordinarily: UNC
  `\\server\share\leaf\file.bin` becomes `\\?\C:\server\share\leaf\file.bin`, expected
  `\\?\UNC\server\share\leaf\file.bin`. The actual toolchain's root parsing/normalization needs
  isolation; no network-share behavior was measured.
- Separate directory/raw/index extended-path omission controls and restorations are planned,
  **UNEXECUTED**. First repair the genuine baseline, preserving authentic byte/index/memo oracles.

## Resume procedure and artifact index

Read `CLAUDE.md`, `LOCAL.md` if present, `CONTRIBUTING.md` and applicable nested instructions;
run `python3 prosper/tools/session_start.py` at session start/after worktree changes. On this
machine the Store python3 alias may fail; real Python312 is available as `python`/`py -3.12`.
Fetch in your own worktree, inspect current-main instructions/deltas and preserve others' trees.

1. Continue from exact9972's passing99/10 baseline, retaining aab7's failures. #4402 is corrected;
   integrate/review the parked #4398/#4399 SOURCE-ONLY branches and newer main, then implement the real
   generated-GS producer/caching and color-control contracts in #4397/#4398/#4399 with genuine
   programmed/register-owned controls. Do not weaken ownership, substitute zero/default words
   or turn this temporary position-only proof frontier into permanent policy. SAME reviewer
   rechecks corrections/newer main; preserve successor H/P/T, calibrated NEW apparatus and exact
   discovery/certificates. Complete real SPIR-V SDK validation, old capture/bank/default neighbors
   and production-only omission/restoration controls before acceptance. Do not borrow aab7 or
   accepted-bank native credit for newer source or a broader test domain.
2. In parallel only if useful, isolate/fix the recorded Windows UNC/enumeration/timeout failures.
   Do NOT repeat the unchanged hanging baseline or convert its timeouts to expected controls.
3. ROOT alone owns heavy native Windows builds/4090 execution. Review agents remain SOURCE-ONLY.
   Inspect live process handles; never change ROOT source while a native cohort is live. Certify
   every exact requested PE, retain actual graph/registrations/full bodies/hashes, forbid app/DLL
   rebuild in offscreen proof, and keep the retained app unchanged unless intentionally rebuilding.
4. Preserve all old bank/source-window/cache/diagnostic/default-emitter neighbors; execute real
   helper-backed pixels and independent production-only controls/restorations before acceptance.
5. Extend the real full-original frontier above. Only then use a fresh normal Windows Kena
   title-screen run/drop census and image oracle to measure end-to-end improvement.
6. Drafts remain drafts until independent registered literal APPROVED on current public head.
   New main after approval requires SAME reviewer applicability check. CI is maintainer-waived,
   not green; only ROOT merges. Keep #4235/#2420/broader #4322 open while full work remains.

All paths below are workspace-relative. Preserve these local raw artifacts; they are NOT all
committed or downloaded by a fresh clone, whereas source and this document are in the draft PR.

| Artifact/worktree | What to trust it for |
| --- | --- |
| `.codex/issue-4235-helper-durable-handoff-3fcb2d18-20261004.md` | Clean code freeze; both repairs unstarted. |
| `.codex/issue-4235-launch-helper-source-3fcb2d18-20261004.md` | Full60paths/63names/source contract, factory/SOURCE names; UNEXECUTED. |
| `.codex/issue-4235-kena32-helper-frontier-3fcb2d18-20261004.md` | Exact32 filenames/hashes/prefix PCs/historical body flags and remaining obligations. |
| `.codex/issue-4235-native-20261004/154531e3-focused-baseline/` | Actual configure0/graph0/build1; private friend-signature failure, no certification or tests. |
| `.codex/issue-4235-helper-cpu-native-20261004.ps1` | Reviewed ROOT-only55/4 apparatus and discriminator calibration; exact154 failure retained, not a passing test receipt. |
| `.codex/issue-4235-native-20261004/f3773be0-private-friend-objects/`, `.codex/issue-4235-private-friend-objects-20261004.ps1` | Actual two requested translation units compile after private-friend fix; source/app/graph/exit receipts, no PE/test/SPIR-V/device credit. |
| `.codex/issue-4235-native-20261004/392f3c3a-focused-baseline/`, `.codex/issue-4235-helper-cpu-392f3c3a-20261004.ps1` | Actual second build1 on nonexistent RegisterFile API; no certificate/tests. |
| `.codex/issue-4235-native-20261004/de2037b2-focused-baseline/`, `.codex/issue-4235-helper-cpu-rebind-20261004.ps1` | Actual four-PE strict certificate and55 complete cases41PASS/14FAIL; original body-classifier false positive preserved and separately explained above. |
| `.codex/issue-4235-native-20261004/802b4f80-canonical-exec/` | Actual new four-PE strict certificate and55 cases50PASS/5FAIL after canonical-EXEC correction; warm-cache negatives and malformed high-word fixture exposed, no GPU/SDK/production-control acceptance. |
| `.codex/issue-4235-native-20261004/cae82d77-launch-authority/` | Actual four-PE strict certificate and55 cases53PASS/2FAIL after checked launch availability partitions both cache paths and real high-word fixture correction; remaining owner/profile failures, no GPU/SDK/production-control acceptance. |
| `.codex/issue-4235-native-20261004/8819e5c4-fixture-owners-profile/` | First actual focused55/55 CPU PASS and four strict PE certificates after authentic shared-registration/physical-profile fixture corrections; full graph/discovery/bodies/hashes retained. NOT full helper/GPU/SDK/production-control or Kena acceptance. |
| `.codex/issue-4235-native-20261004/aab7ac2d-connected-baseline/`, `.codex/issue-4235-helper-connected-native-20261004.ps1` | Actual99/10 strict native cohort96PASS/3FAIL;89/92CPU and7/7Vulkan with real4090 helper pixels. New classifier/cals source-reviewed; raw certs/hashes/full bodies/failures retained. No SDK/production-controls or full-source approval. |
| `.codex/issue-4235-native-20261004/997243e7-diagnostic-contracts/` | Actual99/99 PASS,92CPU+7Vulkan, ten strict certificates; corrected reason/site tests. New companion source branches and later documentation are NOT certified by it. |
| `.codex/issue-4378-native-20261004/215ee6d2-baseline/` | Actual failed native configure/graph/build/certs/registration/JUnit/LastTest/observed.json. |
| `.codex/issue-4378-cpu-native-20261004.ps1` | Reviewed ROOT-only70/12 helper, SHA256 `4df72940d6492ca983eb84996ada823686ce0f04c68fd1dc8196c5f765554695`; requires exact clean H/P/T and NEW cohort name. |
| `.codex/issue-4378-source-freeze-215ee6d2-20261004.txt`, held-scope v1/v2 and handoff | Windows source-gate/history, not a passing native result. |
| `.codex/issue-4322-native-20261004/9e595b11-final-tool/` | Accepted exact9e CPU60 receipts. |
| `.codex/issue-4322-observation-native-20261004/9e595b11-final-tool/` | Accepted exact9e GPU17/observations receipts. |
| `.codex/issue-4322-spv-native-20261004/9e595b11-final-corrected/` | Accepted exact9e585 actual modules/validator evidence. |
| `.codex/worktrees/wave64-followthrough-20261002` | ROOT own central worktree; sole retained `prosper/build-flags-native-clang`. Latest ten requested PEs are linked/certified9972; unrelated retained binaries keep their older identities. Native receipts do not certify later documentation/source successors. |
| `.codex/worktrees/fragment-launch-composition-20261004` | Source-only helper author tree, clean3fcb at freeze, no builds. |
| `.codex/worktrees/refused-shader-paths-20261004` | Removed after clean/ignored-path census, author acknowledgement and exact remote215 verification; no build or raw receipts deleted. Recreate an owned tree from draft #4383 if resuming. |

Safe cleanup: after draft remote refs and protected local refs/artifacts are verified, remove only
finished owned source-only worktrees after clean/ignored-path census and author acknowledgement,
using unforced `git worktree remove` on validated exact paths. Retain the one central build and
failed evidence. Never delete shared/user trees or recursively erase the workspace. The Windows
source-only tree was safely removed after the original freeze; the helper author tree remains.
Source-review agents perform no native builds. The helper follow-up has focused CPU and bounded
GPU evidence; the Windows-path follow-up has a failed CPU baseline and NO GPU evidence. Neither
establishes full accepted fixes or general Wave64/Kena rendering acceptance.
