# `tests/gpu/recompiler` — translation tests and the legacy compute execution fixture

`test_arbitrary_placeholder_execution` executes scalar relays, saved SDWA predicates, narrowed
old values, scalar/vector loop carries and a real supplied scalar-buffer load through the public
compiler and integer interpreter.
`test_arbitrary_placeholder_policy` keeps every non-unloaded root Hard, including a lost operand
beside an unloaded one, and checks strict-first retry/metadata selection. Missing launch and
spill twins retain their separate refusal contracts.

`test_placeholder_origins` checks unloaded, unsupplied launch, unknown launch and spill roots,
including relays, both merge predecessors and loop backedges. Public compiler refusals assert the
same appended details while retaining their refusal class.

`test_native_loop_exec_data` executes genuine cohort inputs across native SCC-loop EXEC
PHIs. Missing inputs and live back-edge dependencies remain refused. The certificate's saved
mask joins, physical-word overlap and original-profile checks are separate from route admission.

`test_uniform_cmpx_loop_data` separates early DATA CMPX exits from a uniform scalar-counter
cap. Its Wave64 interpreter observes different per-lane counts and untouched output; missing
inputs, live scalar results, unproved counter sources and READFIRST relays remain controls.

`test_native_image_results` pins the admitted native legacy sample's complete active words.
Its public fragment regression and compute transport observe the second dword, fresh VCC,
both Wave64 halves and inactive output. The interpreter owns a single exact-center nearest mip;
it makes no general derivative/filtering claim. Missing sources, expanded EXEC and stale
instruction/result publications retain their refusals; global effects stay conservative.

`test_saved_mask_phi` restores a loop-carried saved EXEC before a fresh DATA comparison.
Its native Wave64 interpreter witness observes three iterations, both cohort halves and inactive
output. Changed masks, unavailable relays, pending/changed PHIs and wrong source contexts remain
negative controls; recording ancestry never supplies physical scalar words.

`test_current_word_roots` checks root-view ownership, actual input dependencies and canonical
word lifetime. `test_native_atomic_active_scope` adds buffer ADD/EXCHANGE interpreter witnesses,
missing-input twins and a broader-cohort control. The oracle opts into the closed buffer atomic
profile explicitly; its existing NGG increment contract remains separate.

`test_accumulator_cohort` observes a narrower accumulating write and the untouched part
of its original cohort through the public compiler and Wave64 interpreter. Missing old values,
missing active operands and expanded readers retain their refusals. Unit cases preserve every
ordinary root/flag plane while checking the coverage owner.

`test_fresh_mask_operand_roots` exercises a native SCC loop whose fresh data predicate combines
a current complete rewrite with a pre-loop loaded operand. Its interpreter witness observes the
real loaded value and enabled lanes; missing resources and live back-edge dependencies remain
refusals. It complements the complete-writer and scalar-relay loop fixtures.

`test_local_index_waterfall` pins the fragment READFIRSTLANE cohort transfer and the closed
register-index proof. Its compute transport executes fragment ALU policy in the SPIR-V
interpreter, observing the selected bank, inactive old words, restored EXEC and preserved VCC;
the public fragment regression separately covers the genuine input/export shell. Missing
sources and index escapes remain refusals.

`test_relative_bank_source` adds the complete-cohort result proof: bank words exist only in
that cohort, the old destination is missing, and a fresh mask consumes the result. The interpreter
observes selected values, fresh VCC, restored EXEC and untouched inactive output. Missing and
unbounded banks, stale index/M0 generations and expanded readers remain negative controls.

`test_deferred_m0_retirement` pins the bounded retirement proof's literal read-back and
legacy scalar-memory/read boundaries. The deferred-tail cases in `test_relative_bank_source`
exercise the public fragment path and observe selected words, EXEC, VCC and restored M0
in both Wave64 halves; missing banks and observable M0/index reads remain refusals.

Most tests here ask a question of `src/gpu/recompiler` alone: does this instruction stream decode,
does it recompile, does the module it produces have the shape the contract promises, and — just as
often — does a stream that must NOT compile still reject, and reject *loudly*. No device is created,
so these run everywhere and are the cheapest place in the project to pin a translator contract.
The historical exception is `test_game_compute.cpp`: it executes synthetic shaders through the live
Vulkan backend, including storage writeback, cache authority, and real mapped-page write watches.
Its existing end-to-end fixtures remain here; new standalone execution tests belong in `execute/`.

**The boundary against its siblings.** `tests/gpu/execute/` runs modules on a real device and
asserts pixels or buffer contents; anything needing a queue belongs there, not here. Two large
recompiler tests also live one level up in `tests/gpu/` for historical reasons —
`test_rdna2_spirv_struct.cpp` (structural module validation) and `test_recompile_coverage.cpp` —
and new work usually belongs in this folder instead.

`test_saved_mask_implicit_reads` covers physical ballot demand from implicit SDST consumers.
Its high-word CMPK and low-word ADDK interpreter witnesses complement RMW compilation and
unknown/partial-launch refusals; an SDST storage field alone must never supply missing mask DATA.

SMEM's relocated route callers live in
[`smem_case_routes.hpp`](../../../src/gpu/recompiler/ir/lower/smem_case_routes.hpp).
The [SMEM index execution fixtures](ir/test_ir_smem_index_execution.cpp) exercise the real compiler
and its PC-relative/dynamic-index routes; owned backing and VCC lifetime coverage remains in
[`test_raw_wide_mask_lifetime.cpp`](test_raw_wide_mask_lifetime.cpp).
The provenance definitions retain their existing owners outside these relocated callers.

`test_native_loop_hard_active_roots` executes complete rewrites whose retained inactive lanes owe
an older loop root. Its missing-input, expanded-reader, merge, repeated-write and root-retirement
controls keep that global obligation distinct from the current writer's active sources.

Counted-loop detection and its shared branch helpers live in
[`rdna2_cfg_counted_loop.hpp`](../../../src/gpu/recompiler/rdna2_cfg_counted_loop.hpp) and
[`rdna2_cfg_branch_helpers.hpp`](../../../src/gpu/recompiler/rdna2_cfg_branch_helpers.hpp).
The existing [region-loop fixtures](ir/test_ir_cfg_region_loops.cpp) and
[structured-loop execution fixtures](ir/test_ir_cfg_structured_loop_execution.cpp) exercise their
admission and emitted behavior through the unchanged umbrella header.

FLAT access routes live in
[`flat_case_routes.hpp`](../../../src/gpu/recompiler/ir/lower/flat_case_routes.hpp).
The [window execution fixtures](ir/test_ir_flat_window_execution.cpp) exercise real compiler loads;
[`test_flat_active_word.cpp`](test_flat_active_word.cpp) covers access-source admission. Address
arithmetic remains covered by the existing window value and lowering fixtures.

Divergent-loop shape and validation helpers live in
[`rdna2_cfg_divergent_shapes.hpp`](../../../src/gpu/recompiler/rdna2_cfg_divergent_shapes.hpp).
The existing [region-loop fixtures](ir/test_ir_cfg_region_loops.cpp) and
[structured-loop execution fixtures](ir/test_ir_cfg_structured_loop_execution.cpp) exercise the
relocated helpers through the unchanged CFG umbrella.

The remaining VOP3 route chain lives in
[`vop3_case_routes.cpp`](../../../src/gpu/recompiler/ir/lower/vop3_case_routes.cpp).
The existing [structural shader fixtures](../test_rdna2_spirv_struct.cpp) cover emitted forms;
[numeric MBCNT fixtures](../execute/test_numeric_mbcnt.cpp) exercise the numeric-source route.
Modifier setup, CLAMP admission and destination predication retain their parent ownership.

Divergent-loop interior and final-shape admission lives in
[`rdna2_cfg_loop_admission.hpp`](../../../src/gpu/recompiler/rdna2_cfg_loop_admission.hpp).
The existing [region-loop fixtures](ir/test_ir_cfg_region_loops.cpp) and
[structured-loop execution fixtures](ir/test_ir_cfg_structured_loop_execution.cpp) cover its
refusals and admitted shapes through the unchanged CFG umbrella.

The remaining VOP1 direct routes live in
[`vop1_case_routes.cpp`](../../../src/gpu/recompiler/ir/lower/vop1_case_routes.cpp).
The existing [structural shader fixtures](../test_rdna2_spirv_struct.cpp) cover emitted forms;
[`test_readfirstlane_uniformity.cpp`](test_readfirstlane_uniformity.cpp) retains the separate
READFIRSTLANE uniformity-proof checks. Operand selection and destination predication stay with the
relocated VOP1 family.

Divergent-loop detection and its bounded rejection reporter live in
[`rdna2_cfg_divergent_detector.hpp`](../../../src/gpu/recompiler/rdna2_cfg_divergent_detector.hpp).
The existing [region-loop fixtures](ir/test_ir_cfg_region_loops.cpp) and
[structured-loop execution fixtures](ir/test_ir_cfg_structured_loop_execution.cpp) exercise the
relocated detector through the unchanged CFG umbrella.

Workgroup-uniform VCC/SCC branch proofs live in
[`rdna2_cfg_uniform_branch.hpp`](../../../src/gpu/recompiler/rdna2_cfg_uniform_branch.hpp), with their
operand and definition bodies in the companion source. The existing CPU-only
[workgroup-uniformity fixtures](../test_workgroup_uniformity.cpp) compile both proof sources and
the pure scalar instruction-read predicates directly and
retain their current admission/refusal expectations through the unchanged CFG umbrella.

Scalar merge-deadness and its VCC liveness scans live in
[`rdna2_cfg_scalar_merge.hpp`](../../../src/gpu/recompiler/rdna2_cfg_scalar_merge.hpp).
The existing [VINTRP scalar-liveness fixtures](test_vintrp_scalar_liveness.cpp) and
[compare-kill fixtures](test_fragment_compare_kill.cpp) exercise its reads, physical half-width
inputs and path termination through the original proof and compiler entry points.

Native subgroup row and BPERMUTE emission lives in
[`wave_lane_ops.cpp`](../../../src/gpu/recompiler/spirv/wave_lane_ops.cpp).
The existing [DPP source fixtures](ir/test_ir_dpp_source_execution.cpp),
[row-XOR fixtures](ir/test_ir_dpp_row_xor_execution.cpp) and
[BPERMUTE oracle](test_portable_bpermute.cpp) exercise lane selection through the real compiler.

SPIR-V buffer element selection and ordinary loads/stores live in
[`buffer_access.cpp`](../../../src/gpu/recompiler/spirv/buffer_access.cpp).
The existing [SMEM index fixtures](ir/test_ir_smem_index_execution.cpp) and
[buffer-store fixtures](ir/test_ir_buffer_store_execution.cpp) reach these methods through the
real compiler.

Interpolation geometry setup and vertex emission live in
[`geometry_setup.cpp`](../../../src/gpu/recompiler/spirv/geometry_setup.cpp) and
[`geometry_vertices.cpp`](../../../src/gpu/recompiler/spirv/geometry_vertices.cpp).
The existing [interpolation rendering fixtures](test_interp_render.cpp) and
[rectangle synthesis fixtures](test_rect_synthesis_render.cpp) exercise their emitted modules.

Storage-image declarations and read/write emission live in
[`storage_image_access.cpp`](../../../src/gpu/recompiler/spirv/storage_image_access.cpp).
The existing [storage-image copy fixtures](../test_storage_image_copy.cpp) exercise these methods
through the real compiler and Vulkan execution.

Fragment and vertex entry setup lives in
[`stage_entry.cpp`](../../../src/gpu/recompiler/spirv/stage_entry.cpp).
The existing [fragment front-face fixtures](test_fragment_front_face.cpp) and
[vertex VCC lifetime fixtures](test_vertex_vcc_scalar_lifetime.cpp) exercise the corresponding
stage shells through the real compiler.

Manual 2D comparison sampling lives in
[`image_compare_manual.cpp`](../../../src/gpu/recompiler/spirv/image_compare_manual.cpp).
The existing [shadow comparison fixtures](../test_shadow_compare_render.cpp) exercise nearest and
linear sampling through the real compiler and Vulkan execution.

Storage-buffer declarations live in
[`buffer_declarations.cpp`](../../../src/gpu/recompiler/spirv/buffer_declarations.cpp).
The existing [SMEM index execution fixtures](ir/test_ir_smem_index_execution.cpp) exercise their
table-indexed additional bindings through the real compiler; declarations and defaults remain
available on `SpirvCompute` through its umbrella header.

Compute entry setup lives in
[`spirv/compute_entry.cpp`](../../../src/gpu/recompiler/spirv/compute_entry.cpp).
The existing [native-launch fixtures](test_native_compute_launch.cpp) and
[structural module fixtures](../test_rdna2_spirv_struct.cpp) continue through the unchanged compiler
entry points. Storage declarations and invocation-coordinate setup retain their emission order.

The remaining DS case routes live in
[`ds_case_lowering.cpp`](../../../src/gpu/recompiler/ir/lower/ds_case_lowering.cpp).
Their [address execution fixtures](ir/test_ir_ds_address_execution.cpp) cover real LDS accesses;
[cross-lane fixtures](ir/test_ir_ds_shuffle_execution.cpp) retain the separate shuffle definition
coverage. The relocation changes the caller's home, not either operation contract.

The MIMG storage-image arm lives in
[`image_storage_routes.cpp`](../../../src/gpu/recompiler/ir/lower/image_storage_routes.cpp).
The [storage-image copy and selector fixtures](../test_storage_image_copy.cpp) run its load/store
paths on Vulkan. Coordinate source proofs and the shared result writer remain at the original
MIMG call site; sampled-texture routes retain their own fixtures.

The MUBUF/MTBUF access tail lives in
[`buffer_access_routes.cpp`](../../../src/gpu/recompiler/ir/lower/buffer_access_routes.cpp).
Its [format execution fixtures](ir/test_ir_buffer_format_execution.cpp) and
[store execution fixtures](ir/test_ir_buffer_store_execution.cpp) continue through the real compiler.
Descriptor/proof resolution and the diagnostic logger retain their original parent-case ownership.

Sampled-image type construction and texture declarations live in
[`texture_types.cpp`](../../../src/gpu/recompiler/spirv/texture_types.cpp).
The existing [structural shader fixtures](../test_rdna2_spirv_struct.cpp) retain their sampled-image
coverage; mechanical token read-back separately verifies the moved bodies and declaration defaults.

The remaining LDS/GDS storage and scratch-wave append bodies live in
[`lds_gds_access.cpp`](../../../src/gpu/recompiler/spirv/lds_gds_access.cpp).
The existing [structural shader fixtures](../test_rdna2_spirv_struct.cpp) and
[DS address execution fixtures](ir/test_ir_ds_address_execution.cpp) retain their operation coverage;
mechanical token read-back checks the relocation and retained declaration defaults separately.

**Prefer a small binary over another arm in a big one, whenever rejects are the subject.** Most
tests here return at their first `[FAIL]`, which is fine for a suite that should be all-green but
useless the moment you want to know *which* assertions moved: a mutation run reports the first
failure and nothing after it. A reject-contract test is exactly that case, because the way you
establish it is load-bearing at all is to break the mechanism deliberately and read which arms
redden. So a focused reject test gets its own executable and a `CHECK` macro that counts failures
and keeps going (`test_entry_m0_dispatcher.cpp`, `test_vertex_fetch_reject_diagnostic.cpp`).

**A reject arm without controls beside it is not evidence.** An empty result means the stream was
refused; it never says by whom or why, and a recompiler has many reasons to refuse. The arms that
have held up here carry a positive control with the same region shape and one instruction changed,
so a reject that came from a mis-encoded branch or an unrepresentable region cannot pass as the
contract being tested. Where the route matters, say so in the module: assert an opcode only one
emitter produces, or read `last_terminal_reject_reason()` (it needs a non-zero program address —
`record_terminal_reject_reason` early-returns on zero) and assert on the tag.

Register new cases in `prosper/CMakeLists.txt` under `if(TARGET prosper_core)`, which is
platform-independent — the file also has `if(WIN32)`/`if(UNIX)` branches, and a case registered in
the wrong one silently never runs while ctest stays green. Verify by name with `ctest -N`, never by
the total count.

`test_native_loop_carry_mask` executes compare-produced SGPR masks as carry/borrow inputs inside
native Wave64 EXEC-zero loops. Its missing-mask, physical-DATA and merge-poison controls retain
the boundary between a current Bool mask and fabricated scalar words; the missing-spill controls
remain in `test_native_cfg_saved_mask_data`.
The same carry/borrow kernels execute through the exact dispatcher beside the native regressions.

`test_native_loop_scalar_scope` exercises current-EXEC rewrites followed by READFIRSTLANE and
scalar compare/VCC relays in native Wave64 loops. Its observed missing-back-edge twin keeps the
loop assumption check load-bearing while the interpreter checks the defined overwrite's result.
Varying lane reads, wider EXEC, replacement generations and joins remain negative controls.
VCC preflight tests distinguish numeric scratch from actual mask reads and publication.

The direct SOPP opcode switch lives in
[`sopp_case_routes.cpp`](../../../src/gpu/recompiler/ir/lower/sopp_case_routes.cpp).
Existing structural shader and branch fixtures keep their compiler entry points when this switch
moves; the complete token read-back and corpus comparison qualify the mechanical relocation.

`test_raw_wide_mask_lifetime` separates a fresh emitted Bool mask from a possible surviving
physical high-word dependency of an earlier raw x4 load. Numeric negatives assert the actual raw
load's PC/op in addition to classification and an empty stage module; a later unrelated refusal
does not prove the backing gate worked. Keep per-compile diagnostic addresses unique across stages.
The no-descriptor-patch high-word arm also exercises the outer lifetime shortcut, while the patched
arm reaches derived provenance. The one-arm compare targets the MUST mask meet at a forward join.

## Ruled out

- A launch-provided VGPR can retain genuine inactive words after a partial overwrite. A cohort
  regression must first create an unavailable old generation, then overwrite only the intended
  cohort; a globally genuine word does not exercise the saved-mask ancestry boundary.

- Complete owned packet inputs do not supply live mask facts to unreachable dispatcher blocks.
  [PR #4114's d317 Linux CI](https://github.com/mattias800/prosper/actions/runs/37032364442/job/110922088436)
  rejected the unchanged all-ended `mbcnt_98` at dead PC9 BCNT (`be8c106a`) despite its entry
  S_BRANCH targeting S_ENDPGM. Omit only structurally entry-unreachable owned packet cases;
  conditional branches retain both arms and the caller still inventories every instruction/slot.
  The corrected source requires fresh exact-head execution; this is the observed failure, not a
  passing result for the correction.

- A nonempty comparison-sample module need not contain `OpImageSampleExplicitLod`:
  `ShaderResource` defaults to linear filtering. Both new VS/PS positive checks failed
  while their valid modules each contained four `OpImageFetch` instructions instead.
  Choose nearest filtering explicitly for the explicit-LOD route assertion; preserve
  the failed calibration and independently validated linear vectors (#3987).

- A supported `image_sample_c_lz` is not a guest-memory write. The exact immutable GTA V
  `0x205a95de00` compiler input's PC52 was forced into `rdna2_raw_wide_data_loads` by unrelated
  comparison samples. The paired classifier experiment preserved all lifetime facts and real
  numeric/store/atomic/unknown-op refusals; its residual PC30 still requires backing. This rules
  out PC52 in that input as evidence of a numeric or per-wave backing requirement, not the broader
  raw plan. See [PR #3987](https://github.com/mattias800/prosper/pull/3987) and the title's Ruled out row.

- Running the owned-input chain proof before every cache lookup is not a negligible warm-hit
  cost. A balanced synthetic unowned 1,024-VMOV chain measured about 0.566 ms per current hit
  versus 0.000817 ms with that private guard disabled. This is CPU cache-hit evidence, not title
  FPS. Memoize the code-only proof by both immutable analysis versions; live resource obligations
  and markers still require a check before every lookup. Preserve the uncached measurement.

- Stripping only the v62 marker tail did not form a valid v61 capture after the initial v63
  owned-wide count/width tail. The first current-base raw backing calibration failed this
  historical-layout control. The owned obligation is now v65 after official v64 FLOAT_MODE and v63 draw-width
  tails; remove each newer tail and separately pin retained v62 markers, v63 width and v64 mode.
  A future-version refusal fixture must use a version above the current writer (#3987).
- A successful current capture could not still be asserted as v62 after the initial owned-wide v63
  addition. The full Linux suite exposed this stale expectation in both existing indirect-pointer
  capture controls; keep their owned-byte, unused-candidate and round-trip assertions intact.
- A non-CMPX compare's SGPR-pair-shaped destination does not prove both physical words were
  overwritten in an unknown wave mode. A fresh Bool condition can be independent while an ordinary
  numeric high-word read still requires backing.
- A SAVEEXEC fixture that leaves EXEC narrowed cannot export a vertex position. Restore full EXEC
  before that export when testing independent mask transfer; otherwise an unrelated export refusal
  hides whether the mask proof worked.
- An NGG integer-row spill negative exporting its source VGPR can reject only at the later EXP,
  hiding that ADD published an unresolved scalar-spill placeholder through the row phase.
  Export an independent genuine VGPR and assert the first row's rejection PC; matched unrelated
  spills must still compile in native and portable profiles, with and without a barrier (#4959).
- An inventoried WRITELANE slot is historical after a complete ordinary overwrite under full EXEC.
  Test a genuine full overwrite and a both-arm overwrite against partial EXEC, one-arm and re-spill
  controls; only an all-path current-generation proof can expire the slot across an NGG row phase.
  Native partial-overwrite lifetime behavior is a separate existing contract (#4959).
- A row-source overwrite positive must export and check the row result itself; an independent
  sink can hide a later EXP failure caused by resurrected historical slot maps. Cover a genuine
  overwrite after a guest barrier too, without adding an artificial EXEC restore (#4959).
- A VGPR storage-write count does not prove a complete definition: plain f16 writes preserve half
  the old destination, and FMAC reads the old accumulator. These must not expire an unresolved
  scalar spill in the newly selected portable NGG row dispatcher (#4959).
- Restricting current-generation expiration to VMOV and row-operation opcodes loses genuine
  complete integer producers. The literal pre-fix portable NGG shell exports correct row values
  after MUL_U24, signed/unsigned MIN/MAX, shifts, AND/OR/XOR/XNOR and ADD/SUB/SUBREV overwrites;
  preserve that audited full-word family alongside the halfword/accumulator negatives (#4959).
- Whole-word generation semantics extend beyond the row opcodes or integer VOP2 producers.
  Genuine unary/F32/VOP3/packed-full-word producers need usable row-result exports too; their
  preserving-half and implicit-destination siblings remain first-row negatives. Follow actual
  lowering effects, since an architectural VOP3 opcode can still have no supported emitter (#4959).
- A complete destination word does not authenticate its inputs. DS services read the value plane
  directly; a live spill can reach a row through a fresh intermediate VGPR. Meet ordinary source
  ownership along every CFG edge, with direct, relayed and one-arm spill negatives and genuine
  full/both-arm replacement controls (#4959).
- Decoder storage spans can contain reserved fields. DS WRXCHG consumes ADDR/DATA0, never DATA1;
  verify its usable row result after an unrelated v0 spill. Memory-result generation must check
  every actual consecutive/NSA coordinate and both FLAT address words, with matched genuine and
  unrelated-spill controls. Uninventoried dimensional/packed MIMG forms cannot expire a spill;
  this source-lifetime check grants no memory initialization or resource authority (#4959).
- A16 does not prove a one-word MIMG address: the storage-image lowering reads ordinary 2D
  coordinates while the sampled-texture lowering unpacks them. A resource-independent lifetime
  effect must remain conservative for that packed form, with a live second-coordinate spill
  rejected at the first row and matched ordinary storage-image coordinate controls (#4959).
- A phase reload is not a genuine new source definition. A DS relay of a live scalar spill before
  a barrier remained unowned in the first phase but was re-seeded as ordinary in the next; replacing
  the original spill after the relay also emptied the alias map and disabled that local guard.
  Retain original per-PC ownership across all phases and full EXEC at each barrier boundary, with
  relay-before/after/no-barrier negatives and usable genuine row-result replacements (#4959).


CFG trip-bound selector parsing and operation-pin storage live in
[`rdna2_cfg_trip_settings.hpp`](../../../src/gpu/recompiler/rdna2_cfg_trip_settings.hpp).
The existing [operation-scope fixtures](../execute/test_trip_bound_operation.cpp) cover coherent sampling
and nested restoration; the [dispatcher diagnostic fixtures](../../shared/diagnostics/test_cfg_trip_bound.cpp)
retain the arm/compile/disarm/compile coverage.

The workgroup branch proofs' must-fact walk and full-EXEC predicate live in
[`rdna2_cfg_must_dataflow.hpp`](../../../src/gpu/recompiler/rdna2_cfg_must_dataflow.hpp).
The existing CPU-only [workgroup-uniformity fixtures](../test_workgroup_uniformity.cpp) compile the
predicate's source directly and exercise the proofs through the unchanged CFG umbrella.

Register-state data and shared mark helpers live in
[`rdna2_reg_state.hpp`](../../../src/gpu/recompiler/rdna2_reg_state.hpp).
The existing [portable mask-half fixtures](test_portable_mask_half.cpp) cover half replacement
and [launch-word fixtures](test_unloaded_launch_vgpr.cpp) cover definedness through the real compiler.
Their umbrella include remains available to existing callers.

The `rdna2_spirv_compute*.inl` sections retain the builder declaration behind the internal umbrella.
Existing structural and execution fixtures keep using that API; relocating the sections must retain
the original field order, initializers, method defaults and class scope.
The existing source-shape guard expands all member sections from the class entry fragment and keeps
the inline-body scan nonempty; no method body over 40 lines may return to those declarations.

`test_packed_scalar_width` executes packed FMA followed by a fresh predicate with an absent
unused scalar sibling. Missing actual operands and missing lane spills still refuse; the
physical-width inventory covers every known VOP3P source through the canonical effects table.
Partial-launch native and dispatcher witnesses keep unrelated packed reads from demanding
physical DATA for Bool-only mask logic; numeric VCC high words retain independent poison checks.

Hard-input controls use a launch word that is genuinely unknown (the possible scratch-offset
SGPR), or an untracked spill. A never-loaded ordinary SGPR is Arbitrary under ADR 0038 and
cannot stand in for a missing meaningful input. Keep the original consumer/branch positions
when correcting such a fixture.
