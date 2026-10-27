#ifndef PROSPER_GPU_RECOMPILER_SPIRV_COMPUTE_MEMBER_SCOPE
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"   // IWYU pragma: export
#else
#ifndef PROSPER_GPU_RECOMPILER_RDNA2_SPIRV_COMPUTE_STATE_INL
#define PROSPER_GPU_RECOMPILER_RDNA2_SPIRV_COMPUTE_STATE_INL
// Class-scope fragment: preserve the mechanically moved member bytes.
// clang-format off
    std::shared_ptr<const PortableMaskHalfProgram> portable_mask_half_program;
    // Producing exact-extent configuration may pad a workgroup with absent guest invocations.
    // A common symbolic-half read cannot derive their scalar bits from physical workers.
    bool portable_mask_common_partial_entry = false;
    // Guest semantics and physical synchronization ownership are separate. The legacy is_*
    // booleans select the existing physical stage shells; a packet is guest Fragment while its
    // physical shell is GLCompute. Never set both legacy compute and fragment flags.
    // Some legacy numeric/raster helpers still branch on those booleans. This packet slice refuses
    // FP/raster instructions; extending it must select genuine guest semantics, not inherit LOD0,
    // compute clamp/float mode or native fragment helper assumptions from the physical shell.
    GuestShaderStage guest_stage = GuestShaderStage::Standalone;
    PhysicalExecutionDomain physical_domain = PhysicalExecutionDomain::Invocation;
    GuestWaveEventDomain guest_wave_event_domain = GuestWaveEventDomain::None;
    // Set only after complete packet window admission. Raw loads address one immutable interval
    // appended to binding0, not a per-invocation source pointer or the descriptor fallback bank.
    std::map<uint32_t, std::pair<RawWaveWideCertificate, uint32_t>> packet_raw_windows;
    FragmentPacketQuadTopology packet_quad_topology = FragmentPacketQuadTopology::Unknown;
    bool has_workgroup_execution() const {
        return physical_domain == PhysicalExecutionDomain::Workgroup;
    }
    bool is_fragment_packet() const {
        return guest_stage == GuestShaderStage::Fragment && has_workgroup_execution();
    }
    // `exts` holds OpExtension. SPIR-V requires it AFTER every OpCapability and BEFORE
    // OpExtInstImport, which is why it is a separate section rather than appended to `caps` or
    // prepended to `extimp` -- see the module assembly order below. Getting that order wrong fails
    // spirv-val with a LAYOUT complaint that says nothing about descriptors.
    std::vector<uint32_t> caps, exts, extimp, mem, entry, exec, debug, deco, types, code;
    RecompileDiagnosticContext diagnostic{};
    FragmentFloatMode fragment_float_mode{};
    // Explicit packet-owned launch facts only; native fragment arithmetic remains unchanged.
    FragmentFloatFlags fragment_float_flags{};
    GraphicsFloatMode vertex_float_mode{};
    FragmentArithmeticObservation* fragment_arithmetic_observation = nullptr;
    void observe_fragment_arithmetic_site(uint32_t pc, FragmentArithmeticFamily family) {
        if (is_fragment && fragment_arithmetic_observation)
            fragment_arithmetic_observation->record(pc, family);
    }
    FloatTransportConfig float_transport{};
    bool float_transport_declared = false;
    bool float_transport_warning = false;
    void decorate_float_transport(uint32_t result);
    uint64_t fragment_program_hash = 0;
    bool fragment_float_mode_warning = false;
    void warn_fragment_float_mode_unavailable(uint32_t pc);
    // Ordinal of the next CFG dispatcher emitted for THIS module. A barrier-phased compute program
    // emits one dispatcher per barrier-free phase, and they are not interchangeable — each covers a
    // different guest pc range. Numbering them lets a diagnostic name which one it means.
    uint32_t cfg_phase_ordinal = 0;
    bool descriptor_indexing_declared = false;
    // Declare the capability set and extension for an indexed descriptor array, once per module and
    // only when one is actually declared, so every module that does not use one is byte-identical to
    // what this emitter produced before stage 4b.
    void declare_descriptor_indexing();
    // GUEST FLOAT SEMANTICS ARE INF/NAN-EXACT; VULKAN'S DEFAULT IS NOT (#3479).
    // RDNA2 VALU arithmetic defines Inf, NaN and signed zero exactly, and real guest programs rely
    // on it. Worked example, the one that found this: Unity's `sign()` idiom synthesises +Inf with
    // integer shifts (((1<<8)-1) << 23 == 0x7F800000), multiplies the operand by it, then clamps to
    // [0,1] and truncates. A module that does not declare SignedZeroInfNanPreserve leaves the host
    // driver free to compile under no-Inf/no-NaN assumptions. Measured on Windows/NVIDIA: that
    // multiply yields 0 rather than +Inf, so sign() answers 0 everywhere, PPSA02664's colour-grading
    // LUT builder writes an all-black 1024x32 LUT, and every graded pixel of the scene composites to
    // black -- while the UI, composited after the grade, stays pixel-perfect. RADV preserves Inf,
    // which is why the same build renders the same frame correctly on Linux/AMD. A vendor split, not
    // a title quirk, and invisible to any test whose only device preserves Inf anyway.
    // Declared for EVERY guest module the device can take it on, not only ones observed to need it:
    // a driver may exploit the assumption at any float op, so "which shaders touch Inf" is not a
    // property this emitter can decide.
    // DEVICE-GATED, and the gate is not optional. An earlier revision of this comment asserted that
    // shaderSignedZeroInfNanPreserveFloat32 is VK_TRUE everywhere prosper runs, so no gate was
    // needed, "and if one ever reports VK_FALSE, pipeline creation fails loudly". Both halves were
    // false: CI's Vulkan validation scan reported VUID-...-08740 and -08742 x475 across four
    // binaries WHILE ALL 464 TESTS PASSED -- an invalid module a driver honoured anyway, which is an
    // undefined contract rather than a loud failure (#3561). The device-side query and the whole
    // contract live at rdna2_to_spirv.hpp's publish_float_controls_support(); this call site only
    // reads the answer, and reads it as "no" until some device owner has measured a device.
    bool float_controls_declared = false;
    void declare_float_controls(uint32_t entry);
    bool float32_rte_declared = false;
    void declare_float32_rte();
    // Binary64 (float64.cpp, float64_support.hpp). t_f64() declares Float64, DenormPreserve 64 and
    // the float-controls extension once, on first use; callers check float64_admitted() first.
    uint32_t t_f64_cache = 0;
    bool float_controls_extension_declared = false;
    void declare_float_controls_extension();
    uint32_t t_f64();
    uint32_t fcmp64(uint32_t cmpop, uint32_t a64, uint32_t b64);   // U64 words -> Bool
    std::unordered_map<uint32_t, uint32_t> fconst_cache, uconst_cache;
    std::shared_ptr<const PlaceholderOriginProgram> placeholder_origin_program;
    uint32_t next_id = 1;
    uint32_t stride = 1;
    bool raw_output_words = false;
    bool raw_input_words = false;
    uint32_t packet_input_base = 0, packet_output_base = 0;   // optional dynamic wave ABI SSA ids
    // Descriptor set of begin()'s binding 0 (input) / binding 1 (output) buffers. The merged-NGG
    // subgroup shell moves them to set 2 so the guest's set-0 bindings keep their numbers.
    uint32_t shell_io_descriptor_set = 0;
    uint32_t load_packet_word(uint32_t index);
    // fixed ids (set in begin()):
    uint32_t t_void=0, t_fn=0, t_f32=0, t_u32=0, t_i32=0, t_v3u=0, t_bool=0, t_ptr_sb_f32=0;
    uint32_t v_gid=0, v_groupid=0, v_in=0, v_out=0, gidx=0, f_main=0, glsl=0, bconst_false=0;
    uint32_t globalid_comp[3] = {0, 0, 0}, groupid[3] = {0, 0, 0}, localid_comp[3] = {0, 0, 0};
    uint32_t invocation_guard_merge = 0;
    // Set only by the barrier-phase emitter when a partial workgroup remains in uniform control
    // flow and uses the dispatcher's separate ACTIVE bit to suppress padded guest invocations.
    // recompile_compute uses this proof bit to distinguish that route from the ordinary divergent
    // entry guard, which is never legal around a workgroup barrier.
    bool partial_barrier_phases_emitted = false;
    uint32_t v_push_constants = 0, t_ptr_push_u32 = 0;
    uint32_t push_constant_dword_count = 0;
    uint32_t linear_localid = 0, local_count = 64, wave_size = 64;
    std::array<uint32_t, 3> guest_local_size{64, 1, 1};
    uint32_t v_cbuf=0, v_cbuf1=0, t_ptr_sb_u32=0;   // scalar-memory constant buffers (bindings 2 and 3)
    uint32_t t_ptr_sb_struct_u=0;                    // shared runtime-u32 Block pointer type
    uint32_t t_ptr_sb_u64=0, t_ptr_sb_struct_u64=0; // lazy alias view for 64-bit buffer atomics
    uint32_t t_ptr_img_u32=0;                       // OpImageTexelPointer result for R32_UINT atomics
    uint32_t guest_scratch=0, t_ptr_guest_scratch_u32=0;
    int32_t guest_scratch_min_byte=0, guest_scratch_saddr=-1;
    uint32_t guest_scratch_dwords=0;
    std::map<uint32_t, uint32_t> cbuf_var;          // binding -> storage-buffer var (N-buffer model; 2/3 map to v_cbuf/v_cbuf1)
    std::map<uint32_t, uint32_t> cbuf_u64_var;      // binding -> runtime-u64 alias variable
    // binding -> declared array length for a TABLE-INDEXED binding (#2412). Absent means an ordinary
    // single descriptor, so an access chain for it takes no leading index. Kept beside `cbuf_var`
    // because every consumer of the variable also needs to know whether it is an array.
    std::map<uint32_t, uint32_t> cbuf_table_arity;
    // #2481: a DynamicSbufferByteOffset table's element index is not in user data — it is whatever
    // the producing `s_buffer_load_dwordx4` had in its SOFFSET at run time. The emitter stores that
    // already-divided index here when it lowers the producer, so `cbuf_element_ptr` can select with
    // a live SSA value instead of a push constant. Absent means no producer has executed on this
    // path yet, which must keep failing visibly rather than silently selecting element zero.
    std::map<uint32_t, uint32_t> cbuf_table_selector_value;
    // Label id of the block that emitted each selector above. A raw SSA id is only usable where the
    // producer DOMINATES the consumer, and nothing here guarantees that: the CFG dispatcher emits
    // one switch case per guest block, and sibling switch cases do not dominate one another (a
    // structured if/endif has the same problem). Consuming across that boundary would emit an
    // OpAccessChain against an id outside its dominance region -- malformed SPIR-V that `spirv-val`
    // does not catch here for want of a representative module, surfacing instead as a dropped
    // dispatch. Requiring producer and consumer to be the SAME block is the conservative subset
    // that is always sound; carrying the value through a Function variable would lift the
    // restriction and is the right long-term shape.
    std::map<uint32_t, uint32_t> cbuf_table_selector_block;
    // binding -> user SGPR holding the descriptor index, when one is available.
    std::map<uint32_t, uint32_t> cbuf_table_index_sgpr;
    // Set when an array access cannot be represented exactly. The backend already refuses writable
    // arrays; keeping the same boundary in the emitter prevents a caller that bypasses reflection
    // from receiving a module that redirects an invalid store/atomic to descriptor zero.
    bool invalid_cbuf_array_access = false;
    // A two-byte guest V# can only back our u32 SSBO ABI when every use of that binding is the exact
    // one-record Uint16/Float16 path. Ordinary load/store/atomic helpers blacklist their bindings;
    // finish() emits the explicit typed zero-pad contract only for candidates with no competing use.
    std::map<uint32_t, StorageBufferTailSemantic> cbuf_zero_pad_candidates;
    std::set<uint32_t> cbuf_ordinary_accesses;
    std::set<uint32_t> cbuf_coherent_vars;
    bool     is_fragment=0;                          // true in the fragment shell (gates VINTRP interp)
    bool     is_vertex=0;                            // true in every vertex shell
    // Why an instruction the emitter DECODED could not be lowered in THIS stage. `mode` on the
    // reject line separates "no lowering exists" (`unknown-encoding`) from "the lowering exists and
    // an operand/descriptor did not resolve" (`unresolved-operand`), and a cross-lane operation in a
    // stage with no guest wave is neither: the encoding is known, every operand resolves, and the
    // stage simply has no lane set to reduce over. Reported as a third mode so a census does not
    // read it as a descriptor defect and go looking for the resource table (#2412, #3135). Set by
    // the emitter at the deciding instruction; consumed by the reject site, which appends it to the
    // one terminal line so the recorded reason names the cause rather than the effect.
    uint32_t stage_reject_pc = UINT32_MAX;
    std::string stage_reject_reason;
    // Private attempt result, never an admission fact: the original closed store loop may retry
    // its lost header DATA through a fresh exact dispatcher, keeping every dispatcher proof.
    bool compute_store_loop_data_retry = false;
    // The loop-header assumptions (blanket_root keys) some lane-bit read was admitted on, through a
    // word whose only mark was a header's blanket mark (sreg_loop_blanket). Each loop checks its own
    // at its back edge (LoopBlanketCheck). Mutable: the admitting predicate takes the builder const.
    mutable std::set<uint64_t> loop_blanket_roots_read;
    // The subset of those roots some read needed DEFINED, not merely free of hard lanes: an
    // unbounded or writing memory address (rdna2_loaded_arbitrary.cpp). Such a root cannot be held
    // weakly by an arbitrary back edge (rdna2_loop_arbitrary_carry.cpp, #5193).
    mutable std::set<uint64_t> loop_blanket_roots_strong;
    // Set per instruction by note_memory_address: this LDS read bounds its index (#5193).
    bool lds_bound_reads = false;
    // The exit condition [first, second) of the top-tested compute EXEC loop being emitted, when
    // it has no narrowing certificate. A fresh predicate there may not rest on an ARBITRARY lane:
    // its structured invocations leave one by one, so a lane outside the preheader EXEC reaching
    // the condition means the loop is not provably the wave's, and refusing there is what routes
    // it to the exact dispatcher (closed_compute_store_loop_data_retry, #5193).
    std::pair<uint32_t, uint32_t> strict_loop_condition{UINT32_MAX, 0};
    // The first V_MBCNT whose src0 is a general SGPR mask, when the program ALSO builds the
    // canonical all-ones lane-index pair. That mixture disqualifies the flattened-lane vertex model
    // for the whole program, so the reject surfaces at the all-ones instruction -- which is
    // lowerable on its own -- and names a PC that is not the cause. Kept so the reject can say which
    // instruction actually disqualified it (#3135).
    uint32_t vertex_general_mask_mbcnt_pc = UINT32_MAX;
    bool     allow_b32_masks=0;                      // proven Wave32 or byte-exact graphics exception
    bool     ngg_one_lane=0;                         // exact GS_ALLOC_REQ wrapper: one guest lane/invocation
    bool ngg_workgroup_shell = false;   // merged-NGG workgroup shell: guest waves share LDS
    // Retain original per-PC ownership across phase reloads, which are not fresh entry values.
    bool ngg_row_source_analysis_done = false;
    std::map<uint32_t, std::set<int>> ngg_ordinary_row_sources;
    std::set<uint32_t> ngg_full_exec_pcs;
    // The merged-NGG subgroup shell (#3135 P2) captures GS_ALLOC_REQ (`s_sendmsg 9`) through this
    // hook. Unset, `s_sendmsg` stays the historical no-op every other stage relies on.
    std::function<bool(struct RegState&, const Rdna2Inst&)> ngg_alloc_request;
    uint32_t ngg_probe_trace_pc = UINT32_MAX;        // optional offline VGPR milestone
    uint32_t ngg_probe_trace_vgpr = UINT32_MAX;
    bool     ngg_probe_trace_seen = false;
    std::unordered_set<uint32_t> ngg_uniform_wave_reduction_pcs;
    bool     ngg_logical_lane=0;                     // proven wave64 no-GS producer uses flattened guest lane
    bool     ngg_private_lds=0;                      // exact captured wrapper whose LDS projection is known
    uint32_t ngg_vertex_index_read_pc = UINT32_MAX;   // NGG wave/LDS prologue handoff -> host VertexIndex
    uint32_t ngg_vertex_index_value = 0;
    uint32_t ngg_instance_index_value = 0;
    bool     is_compute=0;                            // true in the compute shell (gates LDS / s_barrier)
    // Whole-original-shader authority, carried unchanged through recursive phases. A recognized
    // waterfall anywhere keeps its existing per-invocation linearization; testing a phase alone
    // would silently change the readfirstlane/removed-backedge contract in the other phase.
    bool     portable_readfirstlane_shader=false;
    bool     uses_barrier=0;                          // guest or synthesized workgroup barrier emitted
    // Ordinary LDS writes that feed a synthesized float-atomic publication boundary are emitted as
    // atomic exchanges. RDNA serializes indexed same-bank conflicts, while Vulkan ordinary stores
    // from separate invocations would otherwise form a data race before the common-phase barrier.
    // The whole-stream synchronization proof populates exact guest PCs; unrelated LDS stores retain
    // their ordinary lowering.
    std::unordered_set<uint32_t> atomicized_lds_store_pcs;
    // GTA V 0x413ce6000's exact pc153 scalar descriptor-table read is admitted only after the
    // complete dispatch/source/table proof has been repeated at the final compiler boundary.
    // Keeping that authority on the builder, rather than inferring it from a resource field inside
    // emit_alu, prevents a hand-built or stale marker from enabling instruction-local shortcuts.
    bool gta5_selected_sbuffer_dispatch_validated = false;
    uint32_t gta5_selected_sbuffer_soffset = UINT32_MAX;
    bool gta5_cf9200_no_backing_dispatch_validated = false;
    bool indirect_buffer_dispatch_validated = false;
    uint32_t indirect_buffer_binding = UINT32_MAX;
    uint32_t indirect_buffer_source_bytes = 0;
    uint32_t indirect_buffer_slot_count = 0;
    uint32_t indirect_buffer_contract_tag = 0;
    uint32_t indirect_buffer_header_bytes = 0;
    uint32_t indirect_buffer_slot_bytes = 0;
    uint32_t indirect_buffer_atomic_binding = UINT32_MAX;
    uint32_t indirect_buffer_atomic_byte_offset = 0;
    // Dispatch/compiler-boundary authority for generic pointer relocation. The proof identifies
    // exact GLOBAL consumers; the remaining fields describe the already-validated carrier bound at
    // `indirect_pointer_binding`. Version 2's StaticFootprint path needs only the segment directory.
    // Version 3's DescriptorRange path additionally preserves the dynamic source-record identity
    // and Base48 root in per-invocation Function variables across CFG dispatcher cases.
    // Every field deliberately defaults to no authority so a caller which has not repeated the
    // complete proof cannot activate either lowering.
    const IndirectPointerRelocationProof* indirect_pointer_proof = nullptr;
    uint32_t indirect_pointer_binding = UINT32_MAX;
    uint32_t indirect_pointer_source_bytes = 0;
    uint32_t indirect_pointer_record_count = 0;
    uint32_t indirect_pointer_record_directory_byte_offset = 0;
    uint32_t indirect_pointer_segment_count = 0;
    uint32_t indirect_pointer_segment_directory_byte_offset = 0;
    uint32_t indirect_pointer_payload_byte_offset = 0;
    uint32_t indirect_pointer_carrier_bytes = 0;
    uint32_t indirect_pointer_source_stride = 0;
    uint32_t indirect_pointer_source_pointer_byte_offset = 0;
    uint32_t indirect_pointer_source_record_var = 0;
    uint32_t indirect_pointer_source_root_lo_var = 0;
    uint32_t indirect_pointer_source_root_hi_var = 0;
    // S_CSELECT_B64 normally needs both scalar source dwords. A captured GTA V kernel consumes
    // only the selected VCC_LO word and leaves VCC_HI dead on every successor path; the whole-stream
    // liveness proof records that exact exception before emission. Keep it builder-local so recursive
    // phase/CFG emitters see the proof derived from the original complete instruction stream.
    bool cselect_b64_low_only_analysis_done = false;
    std::unordered_set<uint32_t> cselect_b64_low_only_pcs;
    // GTA V also selects one B32 scalar directly into Wave64 VCC_LO and consumes only that low word.
    // The untouched VCC_HI mask is unrepresentable once the pair becomes a mixed data/mask lifetime;
    // retain only sites whose whole-stream proof shows that high half dead before any mask read.
    bool vcc_b32_low_only_analysis_done = false;
    std::unordered_set<uint32_t> vcc_b32_low_only_pcs;
    // A Wave64 B32 scalar write to one physical VCC word can also retain a complete architectural
    // predicate when the dispatcher MUST analysis proves the untouched sibling is scalar data at
    // that exact PC.  The proof is separate from Function-variable presence: dispatcher variables
    // have zero placeholders on paths where no scalar definition reaches the block.
    std::unordered_set<uint32_t> vcc_b32_scalar_pair_pcs;
    // The same dispatcher analysis separately proves that both encoded inputs to a B32 VCC write
    // are scalar words on every path reaching that exact PC. Dispatcher Function variables exist
    // for mask-only lifetimes too, so map membership alone must never select the scalar lowering.
    std::unordered_set<uint32_t> vcc_b32_scalar_result_pcs;
    // SOP2 writes into VCC whose scalar sources are all written earlier in the same basic block
    // (rdna2_local_vcc_data.hpp). Recomputed per emit_body; no record pass erases it.
    std::unordered_set<uint32_t> vcc_local_scalar_write_pcs;
    // Predicate-only partial VCC replacements: actual complete compare and constant-word MUST
    // producers, bound to the exact original consumer. No initial placeholder or ballot authority.
    bool partial_vcc_word_analysis_done = false;
    std::vector<uint32_t> partial_vcc_original_words;
    std::map<uint32_t, Rdna2Inst> partial_vcc_word_pcs;
    // S_BFE_U64 may write VCC as both a scalar pair and a per-lane predicate. In a CFG
    // dispatcher, the sreg variables can contain fabricated zero placeholders, so only the
    // MUST-scalar source proof permits retaining its complete scalar result.
    std::unordered_set<uint32_t> vcc_bfe_u64_scalar_result_pcs;
    // A scalar halfword pack into VCC_LO restores its complete predicate only when both
    // sources and the untouched VCC_HI word are MUST-scalar at this dispatcher PC.
    std::unordered_set<uint32_t> vcc_pack_scalar_pair_pcs;
    // Exact structured-CFG consumers whose source pair is a saved Wave64 VCC mask on every
    // incoming path. S_MOV_B64 also materializes ballot words, so emit_alu needs this separate
    // lifetime fact to select the Bool-domain value at the consumer without guessing from maps.
    // Original-program entry participation, never re-seeded from a sliced recursive phase.
    bool saved_mask_word_analysis_done = false;
    std::vector<uint32_t> saved_mask_original_words;
    std::set<uint32_t> saved_mask_entry_prefix_pcs;
    std::set<uint32_t> saved_mask_reconverged_entry_pcs;
    // Separate routing-uniform/genuine-DATA proof for original complete native single-wave loops.
    std::set<uint32_t> native_participating_mask_pcs;
    // Current numeric MOV-to-VCC/EXEC sources are uniform on every original incoming path.
    // This is independent of physical word availability and never re-seeded by a reload.
    std::set<uint32_t> native_uniform_numeric_mask_move_pcs;
    std::vector<uint64_t> native_participation_entry_state;
    uint32_t native_mask_runtime_write_pc = UINT32_MAX;
    bool saved_mask_in_dispatcher = false;
    std::set<int> saved_mask_word_roots;
    std::set<int> native_logical_word_roots;
    std::set<uint32_t> native_logical_data_demand_pcs;
    // The subset of native_logical_data_demand_pcs whose DATA is read by something other than a
    // spill of a saved-mask half (native_logical_data_demanded).
    std::set<uint32_t> native_logical_data_beyond_mask_spill_pcs;
    // Original native MOV-to-VCC generations whose proved Bool has no physical DATA live-out.
    std::set<uint32_t> native_bool_only_vcc_copy_pcs;
    // Invocation-domain fragment copies retain only their existing predicate, never DATA words.
    std::set<uint32_t> fragment_bool_only_vcc_copy_pcs;
    // Original invocation-domain fragment mask generations with no physical DATA live-out.
    // Register-wide demand from another generation cannot require these producers' words.
    std::set<uint32_t> fragment_bool_only_saved_mask_pcs;
    // Native NGG copies preserve only the same current whole-mask predicate generation.
    std::set<uint32_t> ngg_bool_only_vcc_copy_pcs;
    std::array<uint32_t, 10> native_vcc_copy_entry_domain{};
    std::shared_ptr<SavedMaskCopyFrame> saved_mask_copy_frame;
    uint64_t saved_mask_copy_revocation_epoch = 0;
    std::set<int> native_vcc_copy_entry_masks;
    std::unordered_set<int> native_vcc_copy_entry_b32_masks;
    std::set<uint32_t> native_logical_uniform_entry_pcs;
    // Only the ordinary compute shell seeds SGPR DATA from push constants/group IDs.
    bool native_logical_uniform_launch = false;
    // Checked original dispatch words prove complete guest groups, independently of host routing.
    bool guest_compute_complete_groups = false;
    // Actual ordinary-compute dispatch profile, separate from host routing uniformity.
    // Legacy/NGG/standalone builders retain unknown raw launch words by default.
    ComputeDispatchLaunchProfile guest_compute_launch;
    ComputeDispatchLaunchProfile saved_mask_original_compute_launch;
    bool saved_mask_original_fragment_entry = false;
    bool native_compute_physical_entry_expired = false;
    std::set<uint32_t> native_uniform_exec_mov_pcs;
    bool structured_wave64_mask_reduction_analysis_done = false;
    std::unordered_set<uint32_t> structured_wave64_mask_reduction_pcs;
    // Original bounded terminal tails complete scalar liveness at structured joins. The emitter
    // still consumes its original stream and branch policy; these facts seed no register value.
    std::vector<Rdna2Inst> saved_mask_liveness_instructions;
    // Exact Wave64 mask spills retain a Bool view in vgpr_lane_mask_slots. When a CFG MUST proof
    // also identifies which physical ballot half the slot represents, V_READLANE can publish the
    // corresponding scalar dword without guessing from the SGPR number or spill lane.
    std::unordered_map<uint32_t, uint32_t> wave64_mask_readlane_half_for_pc;
    // A mask half restored to an ordinary SGPR may be written back to a spill lane after crossing
    // a dispatcher edge. Keep the exact proven V_WRITELANE sites separate from the SGPR number:
    // physical SGPRs are routinely recycled as scalar data elsewhere in the same shader.
    std::unordered_set<uint32_t> wave64_mask_writelane_alias_pcs;
    // V_WRITELANE sites that spill the HIGH dword of a MUST saved-mask pair whose low SGPR the
    // dispatcher persists as a Bool (SpillSlotDomain::MaskHi). The slot stores the pair's Bool, so
    // a reload proven to be that half (wave64_mask_readlane_half_for_pc) rebuilds it exactly.
    std::unordered_set<uint32_t> wave64_mask_high_half_writelane_pcs;
    // recompile_compute's #5175 retry: spill each half of a saved-mask instance as the pair's
    // Bool and reload it as a proven half, so a producer without physical words need not have them.
    bool spilled_mask_bool_halves = false;
    bool     declared_subgroup=0, declared_subgroup_vote=0, declared_subgroup_arithmetic=0;
    // When non-zero, the backend promises to create this compute pipeline with an exact required
    // subgroup size equal to the PS5 wave. Native votes/scans are then architecture-exact.
    uint32_t native_subgroup_size=0;
    uint32_t native_storage_format_support=0;
    bool storage_buffer_int64_atomics=false;
    bool packed_r11_storage=true;
    uint32_t compute_pgm_rsrc1=kDefaultComputePgmRsrc1;
    Fp16OverflowMode compute_fp16_overflow{};
    uint32_t compute_min_subgroup_size=0;             // non-semantic backend contract (4/16/32/64)
    // ADR 0028: this module was compiled through the exact exchange dispatcher (a second chance
    // taken when the ordinary lowering needed a wider subgroup than the host has). Recorded as
    // `Prosper.ComputeWave64Exchange=1` so the backend can apply the exchange's own launch limits.
    bool wave64_exchange_dispatcher = false;
    uint32_t fragment_required_subgroup_size=0;       // exact guest-wave contract (32 or 64)
    // WHY that width was required, as a bitmask (#2147). The size alone is not actionable: a
    // shader needing 64 for lane IDENTITY can never run at 32, while one needing it only for a
    // width-agnostic branch vote might. The skip diagnostic reported neither, because its
    // `required-ops` field scans for Vote/Arithmetic/Shuffle CAPABILITIES and the lane-id path
    // declares none of them -- so the two cases printed identically.
    uint32_t fragment_wave_reasons=0;
    // Result ids of fragment WaveAny votes that guard a forward `s_cbranch_execz` over a region the
    // GUEST-level classifier (`classify_exec_skip_region`) found free of scalar, memory, wave-level
    // and exit effects. finish() publishes them as `Prosper.FragmentExecSkipVote=<id>` module
    // metadata. It is evidence the SPIR-V cannot carry (an s_sendmsg is lowered to nothing), and it
    // is only ever an additional NECESSARY condition for `lower_fragment_votes`, never sufficient.
    std::vector<uint32_t> fragment_exec_skip_votes;
    // SSA provenance for fragment WaveAny results. A vote needs the exact guest-wave width when
    // that particular bool reaches a guest scalar-data consumer. Tracking result ids avoids the
    // false whole-module inference "this shader contains both a vote and S_CSELECT".
    std::unordered_set<uint32_t> fragment_wave_vote_values;
    std::unordered_set<uint32_t> fragment_wave_vote_scalar_consumers;
    std::unordered_map<uint32_t, std::vector<uint32_t>> fragment_wave_vote_dependents;
    std::unordered_map<size_t, uint32_t> fragment_wave_vote_phi_patch_results;
    uint32_t v_subgroupid=0, v_subgroup_localid=0, t_ptr_in_u32=0;
    uint32_t v_helper_invocation=0, t_ptr_in_bool=0;
    uint32_t v_internal_gds=0, t_ptr_gds_u32=0;
    // Descriptor set for this stage's resources. VS and PS share ONE Vulkan pipeline, so they must NOT
    // reuse binding numbers within one set (both stages number their cbuf/texture from binding 2 -> a
    // set-0 collision made the descriptor layout invalid, corrupting the VS's reads -> degenerate
    // geometry). Each stage owns its own set: VS=set 0, PS=set 1 (mirrors the PS5's per-stage resource
    // tables). Set in begin_fragment(); the host binds one descriptor set per stage.
    uint32_t desc_set=0;
    size_t function_var_insert = 0;                   // first-function-block OpVariable insertion point
    uint32_t exec_model=0;                           // deferred EntryPoint (emitted in finish() so lazily-
    std::vector<uint32_t> iface;                     // declared I/O varyings can join the interface list)
    // Geometry-probe (PROSPER_GEOM_PROBE): when set before begin_vertex(), decorate gl_Position for
    // transform-feedback capture. Inert to the shader's computation; only marks the output for readback.
    bool capture_position = false;
    // Shader I/O value tap (PROSPER_SHADER_TAP=pc): after the instruction at `tap_pc`, snapshot its
    // destination VGPR (and the next 3) as a vec4; the vertex position export is then redirected to it,
    // so the geometry-probe capture reports the intermediate value AT that PC (e.g. a MUBUF fetch result)
    // instead of gl_Position. Inert unless the env var is set. Values are bitcast-as-float in the output.
    uint32_t tap_pc = 0xFFFFFFFFu;
    uint32_t tap_vec = 0;   // 0 = no tap captured; else the vec4 SPIR-V id export_position emits
    void set_tap(uint32_t a, uint32_t bb, uint32_t c, uint32_t d);
// clang-format on
#endif
#endif
