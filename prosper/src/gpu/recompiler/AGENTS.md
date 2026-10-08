# `recompiler` — RDNA2 machine code → SPIR-V

Takes a guest shader's instruction bytes and emits a SPIR-V module.

- `rdna2_decode` — instruction decode: formats, opcodes, operands. Pure and side-effect free, which
  makes it the cheapest thing in the stack to unit-test.
- `rdna2_to_spirv` (+ `_internal`, `emit_alu`, `emit_cfg`, `alu_support`, `cfg_support`) — the
  translator: register state, control-flow structurization, and per-instruction lowering.
- `ngg_subgroup_shell` / `ngg_subgroup_abi` / `ngg_export_record` — the merged ES+GS NGG subgroup
  shell (#3135 P2): one Vulkan workgroup per guest subgroup, launch values and an export record
  buffer in descriptor set 2, GS_ALLOC_REQ captured into a per-subgroup header. `ngg_subgroup_abi`
  is the static admission over the linked program (launch SGPR/VGPR reads, EXEC, side effects,
  messages, export shapes), each refusal named; `ngg_export_record` is the record layout the
  pass-through draw consumes. Compile and offline execution only; nothing live dispatches it.
- `ngg_raster_commit` — that pass-through draw (#3135 P3): a vertex stage that reads the export
  record buffer, rejects malformed connectivity as degenerate primitives (and counts it), rotates
  corners for PROVOKING_VTX_LAST, takes the layer from the provoking vertex, and routes it through
  the vertex stage, a forwarding geometry stage, or the interpolation geometry stage. Offline only.
- `param_ps_routing` — the one PARAM-to-fragment-input routing rule every vertex-side commit stage
  publishes through, so the owned-wave and NGG commits cannot drift from each other.
- `rdna2_cfg_registers` — shared register storage/effect inventory extracted from the capped CFG
  file. Native effects remain unchanged; an explicit owned-packet caller includes genuine VINTRP
  destinations for predicated preservation/P2. Storage reload never grants per-lane entry validity.
- `rdna2_exec_skip_region` — the GUEST-level verdict on the region a forward `s_cbranch_execz` skips:
  scalar live-out, scalar memory effect, wave-level side effect, foreign exit (ADR 0028 route 2). The
  structured fragment emitter publishes a clean verdict as `Prosper.FragmentExecSkipVote=<id>` module
  metadata on the vote; `lower_fragment_votes` treats it only as an additional necessary condition,
  because effects such as `s_sendmsg` lower to nothing and the SPIR-V cannot show them.
- `rdna2_counted_loop_guard` — the counted-loop lowering's proof that a wave-empty
  `s_cbranch_execz` guard around or inside the loop may be linearized (the `s_and_saveexec` form
  and UE4's `s_mov sN, exec … v_cmpx` form). A shape it does not prove is left to `emit_alu`,
  which refuses the branch under narrowed EXEC.
- `rdna2_smem_pointer_provenance` — the once-per-program SMEM pointer and raw-data provenance seed
  every compute and fragment shell runs before emission; it only fills `RegState` facts.
- `rdna2_dead_wave_masks` — the liveness proof that lets the CFG emitter elide a 64-bit VCC mask
  logical whose result no observable read can reach.
- `rdna2_spilled_mask_halves` — a CFG MUST fact the dispatcher's Wave64 mask analysis carries:
  which saved-mask instance each `v_writelane` slot and reloaded SGPR holds a half of, so a pair
  reassembled from both halves of one mask stays a mask across a block edge.
- `rdna2_mask_half_alias` — the companion MUST analysis for exact native Wave64: which physical
  half (LO/HI) of EXEC a spill slot or reloaded SGPR holds, so a reload can publish that ballot
  word as scalar data.
- `rdna2_spill_slot_domain` — whether each `v_writelane` slot holds data or a mask on every path,
  so the dispatcher types a `v_readlane` reload the way `emit_alu` did. The dispatcher's two
  Function variables per slot carry no runtime tag; without this a reload read the other domain's
  placeholder (#4600).
- `rdna2_loop_vcc_carry` — what the divergent-loop emitter does with VCC when a loop body recycles
  it as scalar scratch: the back-edge placeholder (only when VCC is provably dead at the header),
  the merge check, and the exit-state cleanup. Every refusal here logs a terminal reason.
- `rdna2_lane_slot_carry` — the STRUCTURED emitter's counterpart of the dispatcher's slot
  variables: `v_writelane` spill slots get loop-header phis, the exit edge's value at a loop merge,
  and phis at if merges. Without it a slot read in a loop saw its preheader value forever (Kena's
  spilled loop counter hung the GPU).
- `rdna2_recompile_fragment_packet` — an owned 64-slot guest-fragment executor in a physical
  64-worker compute workgroup. It directly uses the synchronized CFG services for whole-wave
  votes, saved-mask reductions, canonical-half mask MBCNT, READLANE and explicit logical-quad B64 WQM,
  alongside numeric source-word MBCNT at owned logical lane positions, then records raw EXP
  metadata/payload instead of killing physical workers or writing a framebuffer. It is NOT a
  raster fallback: missing
  slot/demanded-mask/scalar state, unsupported interpolation, FP arithmetic, image/memory effects and repeated exports
  refuse transactionally. `fragment_packet_contract` evaluates actual emitted uint sinks from
  project-owned packets; `spv_validate` emits this entry separately. No live DrawItem enters it.
  Quad consumers additionally require the supplied consecutive-logical-quad topology tag; it grants
  no raster packing/helper authority. B32 WQM and unsupported constant/forms remain named gaps.
- `raster_quad_collector` — an input-only fragment emitter and transactional raw-record decoder.
  Four unconditional QuadBroadcasts retain helpers before a nonhelper reserves a bounded scratch
  slot. These are host raster observations, not initialized guest registers or logical Wave64
  packets. The shipping renderer companion in `tests/fixtures/raster_quad_collection_gpu.h` owns
  the actual scratch pass; it declines unproved pre-raster effects, missing producing modules and
  unsupported domains. No framebuffer export, depth/blend commit or guest-wave packing is added.
- `fragment_resource_packet` — a separate owned-input packet compiler/transactional EXP consumer.
  `fragment_packet_resource_preflight` checks original read-PC/descriptor/parameter/M0 ownership
  and all-path counter-specific WAIT completion; `fragment_packet_resource_services` emits actual SMEM,
  P1/P2, explicit-LZ/L nearest sampling and integer finite F32 operations from those inputs.
  `fragment_packet_f32` preserves explicit input/output denorm and rounding modes in integer
  arithmetic; nonfinite/overflow and non-exact interpolation remain named runtime failures.
  All64 workers rendezvous and append sticky statuses; ANY failure prevents the consumer from
  publishing ANY raw EXP record. Integer packet ABI/native paths remain separate. Missing/partial
  VGPR columns activate the appended VGP1 per-logical-lane validity contract; full-dword masked
  writers establish scratch definitions, never launch input authority. Direct/implicit P2/wide
  image reads, EXEC-ignoring selected peers and even inactive enabled raw EXP payload must have
  genuine values. `fragment_packet_definedness` checks reads before overlapping writes and keeps
  the first failure; READLANE validity uses the existing uniform phase. Completed consumers must
  validate all64 records and the original-site whitelist before publishing any EXP. Fully supplied
  legacy packets keep their unextended wire format. This does not initialize inputs from host raster records,
  enable implicit/bias sampling, establish live resource epochs, or admit any real DrawItem.
  `fragment_packet_special_f32` supplies integer-backed RCP/SQRT/direct RSQ, choosing correctly
  rounded software results within the published approximation envelope, not AMD-unit bit identity.
  Opcode-specific sign-preserving denormal flushing is separate from ordinary mode controls.
  Retained producing PS RSRC2 must prove no handler, or disabled relevant floating exceptions and
  DEBUG; MODE/STATUS observation remains unsupported. NaN payloads and negative roots remain
  transactional named runtime failures, not guessed canonical values. This is not full special FP.
- `fragment_packet_vgpr_requirements` inventories exact immutable raw program storage/read facts,
  pinned by the existing ShaderCodeAnalysis owner and consumed by shipping draw preparation.
  A structural writer is not proof of a value on EXEC-off lanes; runtime validity and real entry
  mask/helper/system/composition/commit authority remain separate obligations. The CPU-only
  `PROSPER_VGPR_DEFINEDNESS_SPV_DIRECTORY` diagnostic retains actual emitted SOURCE for validation;
  unset writes no files and never changes guest lowering or admission.
- `fragment_packet_mask_requirements` is the same immutable analysis owner's instruction-order,
  all-path inventory of demanded initial EXEC/VCC/SCC. Kernel compilation and cached wave packing
  distinguish individual presence from raw storage; shipping preparation consumes the exact code
  facts without per-draw parsing. Complete original writers may make initial values unnecessary,
  never an actual supplied branch value or an internal zero placeholder. READLANE's selected peer,
  WQM/partial writes and the existing inactive raw EXP contract remain separate obligations.
- `rdna2_waitcnt` decodes the original RDNA2 split VM, EXP and six-bit LGKM fields. A zero
  threshold drains only its own pending set; a maximum grants no completion. Effects-only
  scalar-origin/mask inventory may preserve canonical partial WAITs without granting readiness,
  while execution still refuses intermediate thresholds until a count/order proof exists.
  Unmodeled bit7 is retained and named unsupported, not declared ISA-invalid.
- `fragment_packet_wave_data` separates cached original-program SOURCE/profile from dynamic owned
  logical64 wave regions. Checked per-workgroup bases load genuine scalar/M0/resource/VGPR words;
  all-wave status validation precedes any publication. Shared image bindings remain a bounded
  initial domain. Private readonly binding2 placement ownership guards mutable binding0 routes
  uniformly before guest/barriers/stores; that retained dispatcher authority is not a live P5 lease.
  It grants no raster scheduling, system-entry derivation or shipping admission.
- `fragment_draw_capacity` and `_emission` define a separate WAT2 immutable GPU-capacity/placement
  contract. It binds the complete collector shape before shader address generation; capacity and
  zero storage never authenticate guest entry state. The default WAT1 compiler/wire stay unchanged.
  `fragment_draw_gpu` emits count/indirect-dispatch and genuine-quad assembly; `_commit_gpu` checks
  every original worker status/site plus retained raster provenance and unique pixel indexing
  before publishing one device whole-draw gate. Replay writes through normal fixed-function
  attachments only after that gate. Its immutable original EXP inventory and explicit Architectural
  EXP14 policy must match emitted schema metadata; a raw12 or forged raw14 shape is not authority.
  Missing/inactive occupied exports refuse the whole draw before attachment mutation. Genuine
  helper/system/M0/coefficients/resources and observable
  cross-quad composition are separate future recipes, not padded values or waived requirements.
- `fragment_packet_exports` owns explicit architectural EXP observation/schema and transactional
  typed accumulation; `fragment_packet_exports_emit` emits EXP14 only when that immutable policy
  is selected. Original current EXEC, not host eligibility/final VM, authorizes payload reads.
  Unobserved sources remain optional/absent; compressed channels retain raw16 bits without guest
  format conversion. Last executed VM controls validity; DONE/control/null events remain visible.
  All existing default raw12 observations and inactive assertions remain unchanged. Completed
  resource and cached-wave consumers stage typed events until every status/wave succeeds. This
  is suitable input for a later ordered attachment transaction, NOT attachment/raster authority.
  `fragment_packet_export_timing` proves current EXEC and every enabled physical payload word
  stable through all forward paths until EXPCNT0 or ordinary END's implicit drain. Unsafe
  overwrites remain named gaps; DONE alone never drains. Eager private records do not claim bus
  timing, externally visible attachment writes or live resource epochs. RDNA2 STATUS.SKIP_EXPORT
  is VS-only; this PS contract does not invent a missing status input or infer a stored zero.
- `rdna2_mask_move` supplies the shared S_MOV_B64-to-EXEC lowering. Owned Wave64 packets consume
  two genuine instruction-order MUST scalar words at their logical lane position, independently
  of old EXEC, and preserve SCC. `rdna2_packet_raw_masks` materializes selected complete ordinary
  saved-SGPR masks through the uniform tagged workgroup phase before physical DATA reads/partial
  replacements; either physical overwrite expires the old complete alias. Actual absent sources
  remain named refusals; scalar initialization or the old complete Bool alias is not that proof. Native fragment lowering is
  separate and retains its existing exact-subgroup contract.
- `spirv_builder` — small hand-built SPIR-V modules. **These include shipped shaders**:
  `frontends/shared/live/live_compute.cpp`'s `prepare_compare_pipeline()` feeds
  `build_compute_compare_uvec4()` straight to `vkCreateShaderModule` on the live path. The GPU
  texture detiler and `packed_rtt_conversion.hpp` also use modules built here. Treating
  them as test fixtures is what let an invalid `OpAccessChain` reach real devices (#1711), and
  `tools/spv_validate` exists to stop exactly that — so a change here is a change to shipped
  shader code, not to a fixture. It is not a general emitter either.
- `gta5/`, `indirect/` — see their own AGENTS.md.

**An unsupported op is a FATAL gap, not an acceptable skip.** Reject paths exist as a fail-visible
backstop for genuinely unknown encodings — mark `CONFIDENCE: LOW`, log loudly, file an issue with the
exact opcode — but every one hit on a live boot is the next thing to implement. A silently skipped
instruction drops real rendered content and reads as "handled".

**No title id in a condition, and no title-named module outside `gta5/`.** The architecture
ratchet (`CLAUDE.md` § *Architecture and performance ratchets*) counts `PPSA#####` outside comments
and caps `gta5/`'s size, so a lowering one title needs either generalises into the translator or
lands in `gta5/` with its measurement — and shrinking `gta5/` by generalising is the direction.
Compile cost is a frame-time cost: the recompiler's output must be cacheable by program bytes so
that, after warm-up, nothing here runs on the submit thread (P3). `rdna2_emit_alu.cpp` and
`rdna2_emit_cfg.cpp` are past the 5,000-line cap; grow a new file, not them.

## `SignedZeroInfNanPreserve`: a correctness contract, and a device gate

RDNA2 float arithmetic defines Inf, NaN and signed zero exactly. Vulkan does **not** promise that by
default: without `SPV_KHR_float_controls`' execution mode a driver may compile the module assuming
those values never occur. Guest shaders do rely on them — a compiler-generated `sign()` idiom
synthesises `+Inf` with integer shifts and multiplies by it — and on NVIDIA that multiply returned 0,
which blacked out a whole title's world while its UI stayed perfect (#3479). RADV preserves Inf, so
the same build was correct on Linux and the defect looked like a title quirk.

`declare_float_controls()` in `rdna2_to_spirv_internal.hpp` emits it from every `begin*()` — but only
on a device that has been **measured** to accept it. Declaring a capability the device does not
satisfy makes the module invalid rather than merely unoptimised, and a driver is free to honour an
invalid module: the first revision declared it unconditionally and CI's Vulkan validation scan
reported two VUIDs x475 across four binaries while every one of 464 tests passed (#3561). The verdict
is published by whoever owns a Vulkan device, through
`gpu/execute/float_controls_probe.hpp`; the emitter only reads it, and reads "no" until somebody has
asked a real device. Publishers are ANDed, and nothing is declared until one publishes — an offline
caller therefore emits the neutral form, which is legal everywhere.

Three things follow. **Do not add a new stage entry point without calling it** —
`tests/gpu/recompiler/test_float_controls.cpp` asserts the declaration for all four stages and is the
cheapest place to notice. **Do not re-derive the gate away from a device list**: "every device we run
on has the property" is the claim that failed, and no green test run can contradict a property
nothing queries. And **an execution test cannot guard any of this**: an `Inf * x` kernel passes on
RADV and on the lavapipe CI runs whether or not the mode is declared, because those implementations
preserve Inf anyway, so the guard is structural on purpose. The hand-built modules in
`spirv_builder` are prosper's own code rather than translated guest code and are deliberately outside
this contract.

`FloatTransportConfig` is a separate, immutable producing profile. Explicit nonfinite F32 transport
requires both enabled `shaderFloatControls2` and the SZI32 property; device owners publish only after
successful device creation. Cache keys, retained draws/failures and replay pass the actual profile,
never an ambient capability guess. Its per-instruction `FPFastMathMode None` covers only scalar
cross-float Bitcasts and floating fragment Input loads. It does not promise signaling-NaN payload
identity, arithmetic denorm/rounding behavior, or any uniform/frozen/address authority. The vote
parser accepts only this exact typed envelope and removes its old implicit finite-back premise for
decorated transport; `FPFastMathDefault`, other flags and unrelated decorated operations refuse.

## `PROSPER_CFG_TRIP_BOUND` — is this a non-terminating loop?

A guest program whose control flow neither structured emitter accepts is lowered by
`emit_cfg_state_machine` into ONE SPIR-V loop over a switch of dispatch ordinals. A recompiled loop
that never ends hangs the GPU into a driver reset, and from outside that is indistinguishable from a
slow shader or from a defect elsewhere in the submit. `PROSPER_CFG_TRIP_BOUND=N` caps that back edge
so the question becomes one run. It is a **diagnostic**: truncating guest control flow produces wrong
results by construction.

    PROSPER_CFG_TRIP_BOUND=4096            # required: the cap
    PROSPER_CFG_TRIP_BOUND_PROGRAM=0xADDR  # one program, so other shaders stay byte-identical
    PROSPER_CFG_TRIP_BOUND_PHASE=0         # REQUIRED -- nothing is emitted without it
    PROSPER_CFG_TRIP_BOUND_ORDINAL=45      # optional: cap ONE loop inside a multi-loop program

Three things about it are load-bearing and each has cost someone a run:

- **It covers the CFG dispatcher only.** The two structured loop emitters never call it, so a null
  result from a structurizer-accepted program means *not measured*, never *does not run away*.
- **The witness is COMPUTE-only; the cap is not.** The device-side hit record lives in the internal
  GDS buffer, which only the compute executor binds, prepares and reads back. A graphics program is
  still capped — and says so, once, on stderr — but publishes nothing, so on a draw you must confirm
  the draw still ran (`PROSPER_DRAW_PROGRAM_CENSUS`) before reading a surviving device as a hit.
- **A plain cap says "some loop here", not "this loop".** `_ORDINAL=K` counts only the traversals
  about to re-enter ordinal K, so one arm per candidate header localizes the runaway; read the
  ordinal → guest-pc map the arm prints, never map one by hand. Astro Bot's world-map pixel program
  `0x5008f1400` has four guest loops behind one dispatcher and this is how the runaway among them was
  named (#3193).

- **The selectors are sampled ONCE per shader-cache operation, not read per site.** They are part
  of the cache key — the cache is keyed on code BYTES and never on the address, so a bounded target
  and an unbounded non-target with identical bodies would collide on one entry. That only works if
  the key and the module agree about which settings were in force, and before #3714 the key sampled
  them in `gpu_executor.cpp` while the emitter re-read the environment from **three** further sites
  (`rdna2_emit_cfg.cpp:1211/3582/5483`). A change in between keys an entry under settings A while
  the module in it was built under B, for good, with nothing to report it. `TripBoundOperation` pins
  one owned value across both halves. Inside a submit the pinned value is sampled once per submit,
  and outside every submit each operation parses afresh — which is what keeps
  `test_cfg_trip_bound`'s arm/compile/disarm/compile working in one process.
  **`compute_trip_witness_active()` is a fourth reader and is deliberately NOT pinned**: its only
  callers are `gpu_executor.cpp:8573` and `:8880`, which ask at DISPATCH time — after the module
  exists — to decide whether to bind the GDS witness buffer. It is outside any operation, reads
  live, and is the ~283k-call residual #3714 left behind. So "the settings are pinned" is a
  statement about the key and the emitter, not about every reader in the tree.

Every SPIR-V emitter path is `spirv-val`-gated in CI (`tools/spv_validate`) with one representative
module per path, not one per game shader.

Primary reference: *"RDNA 2" Instruction Set Architecture Reference Guide* (AMD doc 70648). PS5
extensions and AGC behaviour still require live title evidence.
