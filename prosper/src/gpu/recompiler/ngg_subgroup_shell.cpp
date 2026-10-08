// ngg_subgroup_shell.cpp -- see ngg_subgroup_shell.hpp.
#include "gpu/recompiler/ngg_subgroup_shell.hpp"

#include "gpu/recompiler/ngg_subgroup_abi.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_recompile_shared.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

namespace prosper::gpu {
namespace {

constexpr uint32_t kGuestWaveLanes = 64;
constexpr uint32_t kMaxShellWaves = 4;
// Vulkan guarantees 128 bytes of push constants.
constexpr uint32_t kMaxPushWords = 32;

bool fail(std::string* refusal, const RecompileDiagnosticContext& diagnostic, const char* reason,
          const std::string& detail = {}) {
    std::string text = std::string("reason=") + reason;
    if (!detail.empty()) text += " " + detail;
    log_recompile_diagnostic(diagnostic, "ngg-subgroup-reject", "terminal", "%s", text.c_str());
    if (refusal) *refusal = text;
    return false;
}

// One raw word of the export buffer, by absolute index.
uint32_t output_word_pointer(SpirvCompute& b, uint32_t index) {
    const uint32_t pointer = b.id();
    b.putv(b.code, Op_AccessChain, {b.t_ptr_sb_u32, pointer, b.v_out, b.uconst(0), index});
    return pointer;
}

// `if (condition) body();` as a structured selection inside the current block.
template <typename Body>
void emit_if(SpirvCompute& b, uint32_t condition, Body&& body) {
    const uint32_t then_label = b.id(), merge_label = b.id();
    b.put(b.code, Op_SelectionMerge, {merge_label, 0});
    b.put(b.code, Op_BranchConditional, {condition, then_label, merge_label});
    b.put(b.code, Op_Label, {then_label});
    b.cur_block = then_label;
    body();
    b.put(b.code, Op_Branch, {merge_label});
    b.put(b.code, Op_Label, {merge_label});
    b.cur_block = merge_label;
}

void atomic_increment(SpirvCompute& b, uint32_t index) {
    const uint32_t pointer = output_word_pointer(b, index);
    const uint32_t result = b.id();
    b.put(b.code, Op_AtomicIAdd,
          {b.t_u32, result, pointer, b.uconst(Scope_Device), b.uconst(0), b.uconst(1)});
}

}   // namespace

std::vector<uint32_t> recompile_ngg_subgroup(const uint32_t* linked_code, size_t dwords,
                                             const ShaderResourceTable* resources,
                                             const NggSubgroupShellConfig& config,
                                             NggExportRecordLayout* layout_out,
                                             RecompileDiagnosticContext diagnostic,
                                             std::string* refusal) {
    if (refusal) refusal->clear();
    if (layout_out) *layout_out = {};
    const uint32_t push_words = config.user_sgprs + (config.user_data_address_known ? 2u : 0u);
    if (!linked_code || !dwords || config.waves == 0 || config.waves > kMaxShellWaves ||
        config.rsrc2_gs_lds_size > kNggMaxLdsGranules || config.user_sgprs > 98u ||
        push_words > kMaxPushWords) {
        fail(refusal, diagnostic, "ngg-shell-config");
        return {};
    }
    std::vector<Rdna2Inst> ins;
    rdna2_walk(linked_code, dwords, ins);
    NggSubgroupAbiLaunch launch;
    launch.user_sgprs = config.user_sgprs;
    launch.user_data_address_known = config.user_data_address_known;
    const NggSubgroupAbiFacts facts = analyze_ngg_subgroup_abi(ins, launch);
    if (!facts.ok()) {
        if (refusal) *refusal = facts.refusal;
        log_recompile_diagnostic(diagnostic, "ngg-subgroup-reject", "terminal", "%s",
                                 facts.refusal.c_str());
        return {};
    }
    // Private scratch is refused with the other partition-visible effects (design section 4).
    const StaticScratchLayout scratch = analyze_static_scratch(ins);
    if (scratch.used || !scratch.valid) {
        fail(refusal, diagnostic, "ngg-side-effect", "scratch");
        return {};
    }
    // LDS float min/max needs recompile_compute's store/atomic synchronization proof, which this
    // shell does not run; refuse rather than emit an unsynchronized reduction.
    if (std::any_of(ins.begin(), ins.end(), [](const Rdna2Inst& in) {
            return in.fmt == Rdna2Format::DS &&
                   (in.opcode == kDsOpcodeMinF32 || in.opcode == kDsOpcodeMaxF32 ||
                    in.opcode == 0x32u || in.opcode == 0x33u);
        })) {
        fail(refusal, diagnostic, "ngg-lds-float-minmax-unsupported");
        return {};
    }
    // LDS without a size would silently get the builder's 16 KiB default, and an address past
    // a too-small allocation is undefined behaviour in Vulkan, not zeros.
    const bool uses_lds = std::any_of(ins.begin(), ins.end(), [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::DS && !in.ds_gds;
    });
    if (uses_lds && config.rsrc2_gs_lds_size == 0) {
        fail(refusal, diagnostic, "ngg-shell-config", "cause=lds-unsized");
        return {};
    }
    const bool has_barrier = std::any_of(ins.begin(), ins.end(), [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::SOPP && in.opcode == kSoppOpcodeBarrier;
    });
    if (has_barrier) {
        // s3 differs between the waves of one workgroup, so a scalar terminal guard ahead of a
        // barrier could let some waves skip a Vulkan workgroup barrier the others wait on.
        const BarrierPhasedCompute phases = analyze_barrier_phased_compute(ins);
        if (!phases.found || phases.guarded) {
            fail(refusal, diagnostic, "ngg-barrier-not-uniform",
                 phases.guarded ? "per-wave-terminal-guard" : "barrier-phase-proof");
            return {};
        }
    }
    const NggExportRecordLayout& layout = facts.layout;
    const uint32_t local = kGuestWaveLanes * config.waves;
    const uint32_t block_words = layout.block_words(config.waves);

    const bool original_has_waterfall = !waterfall_branches(ins).empty();
    SpirvCompute b;
    b.diagnostic = diagnostic;
    b.ngg_workgroup_shell = true;
    b.shell_io_descriptor_set = kNggShellDescriptorSet;
    if (config.rsrc2_gs_lds_size) b.lds_dwords = config.rsrc2_gs_lds_size * kNggLdsGranuleDwords;
    b.native_subgroup_size = config.native_wave64 ? kGuestWaveLanes : 0u;
    b.begin(kNggLaunchWordsPerLane, resources, local, 1, 1, kGuestWaveLanes, push_words,
            /*raw_word_output*/ true, /*raw_word_input*/ true);
    b.portable_readfirstlane_shader = !config.native_wave64 && !original_has_waterfall;
    {
        std::vector<uint32_t> marker;
        char text[96];
        std::snprintf(text, sizeof(text), "Prosper.NggSubgroupShell.Waves=%u.Wave64=%s",
                      config.waves, config.native_wave64 ? "native" : "portable");
        b.pstr(marker, text);
        b.putv(b.debug, Op_ModuleProcessed, marker);
    }
    b.declare_guest_scratch(scratch);

    RegState rs;
    rs.vcc = b.bfalse();
    rs.scc = b.bfalse();
    rs.exec = b.btrue();
    seed_smem_pointer_provenance(rs, ins);
    // The whole-program raw scalar-data proofs recompile_compute seeds, so an immediate raw load
    // (Kena's ES prolog reads two words at pc 6) resolves through its own fetch-PC resource.
    if (!retain_original_owned_raw_x2_proof(rs, rdna2_owned_raw_x2_chains(ins)) ||
        !retain_original_owned_raw_wide_proof(rs, rdna2_owned_raw_wide_data_loads(ins))) {
        fail(refusal, diagnostic, "ngg-compile-rejected", "owned-raw-data-proof");
        return {};
    }
    for (uint32_t pc : rdna2_proven_raw_x2_data_loads(ins)) rs.smem_raw_x2_data_loads.insert(pc);
    for (uint32_t pc : rdna2_raw_wide_data_loads(ins)) rs.smem_raw_wide_data_loads.insert(pc);
    for (uint32_t pc : rdna2_proven_raw_immediate_wide_data_loads(ins))
        rs.smem_raw_immediate_wide_data_loads.insert(pc);
    for (uint32_t pc : rdna2_proven_raw_nested_wide_data_loads(ins))
        rs.smem_raw_nested_wide_data_loads.insert(pc);
    for (uint32_t reg = 0; reg < 9u; ++reg) rs.vreg[static_cast<int>(reg)] = b.load_input(reg);
    rs.sreg[3] = b.load_input(kNggLaunchS3Word);
    // An untouched ABI index read by a vertex fetch is VertexID (v5) / InstanceID (v8).
    b.ngg_vertex_index_value = rs.vreg[5];
    b.ngg_instance_index_value = rs.vreg[8];
    // Direct descriptors live in sreg_input, as in recompile_compute, so a fetch through an untouched
    // user-data V# still resolves through the resource table.
    std::set<uint32_t> descriptor_sgprs;
    if (resources) {
        for (const auto& resource : resources->resources) {
            if (resource.srt_offset != 0xFFFFFFFFu || resource.sgpr_base == 0xFFFFFFFFu) continue;
            const uint32_t words = (resource.cls == ResourceClass::Texture ||
                                    resource.cls == ResourceClass::StorageImage)
                                       ? 8u
                                       : 4u;
            for (uint32_t word = 0; word < words; ++word)
                descriptor_sgprs.insert(resource.sgpr_base + word);
        }
    }
    for (uint32_t k = 0; k < config.user_sgprs; ++k) {
        const uint32_t reg = 8u + k;
        const uint32_t value = b.load_push_constant(k);
        if (descriptor_sgprs.contains(reg))
            rs.sreg_input[static_cast<int>(reg)] = value;
        else
            rs.sreg[static_cast<int>(reg)] = value;
    }
    if (config.user_data_address_known) {
        rs.sreg[0] = b.load_push_constant(config.user_sgprs);
        rs.sreg[1] = b.load_push_constant(config.user_sgprs + 1u);
    }

    // store_output_word indexes gidx * words_per_lane + word, where gidx = group * local + thread.
    // Adding group * header + header places thread t of workgroup g at its block's record t.
    b.packet_output_base =
        b.ibin(Op_IAdd, b.ibin(Op_IMul, b.groupid[0], b.uconst(kNggSubgroupHeaderWords)),
               b.uconst(kNggSubgroupHeaderWords));
    const uint32_t block_base = b.ibin(Op_IMul, b.groupid[0], b.uconst(block_words));
    const uint32_t guest_lane = b.ibin(Op_BitwiseAnd, b.linear_localid, b.uconst(63));
    const uint32_t guest_wave = b.ibin(Op_ShiftRightLogical, b.linear_localid, b.uconst(6));
    // The launch must describe this shell: a wave whose s3 disagrees on W or on its own index
    // marks the block invalid rather than producing triangles sized for another subgroup.
    {
        const uint32_t s3 = rs.sreg[3];
        const uint32_t waves_field = b.ibin(Op_ShiftRightLogical, s3, b.uconst(28));
        const uint32_t index_field =
            b.ibin(Op_BitwiseAnd, b.ibin(Op_ShiftRightLogical, s3, b.uconst(24)), b.uconst(0xfu));
        const uint32_t mismatch = b.lor(b.ucmp(Op_INotEqual, waves_field, b.uconst(config.waves)),
                                        b.ucmp(Op_INotEqual, index_field, guest_wave));
        emit_if(b, b.land(b.ucmp(Op_IEqual, guest_lane, b.uconst(0)), mismatch), [&] {
            atomic_increment(b, b.ibin(Op_IAdd, block_base, b.uconst(kNggHeaderLaunchMismatches)));
        });
    }

    std::string emit_refusal;
    bool saw_export = false;
    const auto export_record = [&](RegState& state, const Rdna2Inst& in) -> bool {
        uint32_t base = 0, flag = 0;
        if (in.exp_target == kExpTargetPrim) {
            base = kNggRecordPrimWord;
            flag = kNggFlagPrim;
        } else if (in.exp_target == kExpTargetPos0) {
            base = kNggRecordPos0Word;
            flag = kNggFlagPos0;
        } else if (in.exp_target == kExpTargetPos1 && layout.pos1_word != kNggRecordAbsent) {
            base = layout.pos1_word;
            flag = kNggFlagPos1;
        } else {
            const auto it =
                std::find(layout.param_targets.begin(), layout.param_targets.end(), in.exp_target);
            if (it == layout.param_targets.end()) {
                emit_refusal = "ngg-export-target-unsupported";
                return false;
            }
            const uint32_t index = static_cast<uint32_t>(it - layout.param_targets.begin());
            base = layout.param_word(index);
            flag = 1u << (kNggFlagParamShift + index);
        }
        const uint32_t exec = state.exec_narrowed ? state.exec : 0u;
        for (uint32_t channel = 0; channel < 4; ++channel) {
            if (!(in.exp_en & (1u << channel))) continue;
            bool resolved = true;
            const uint32_t bits = operand_bits(b, state, in, in.src[channel], &resolved);
            if (!resolved) {
                emit_refusal = "ngg-export-operand-unresolved";
                return false;
            }
            b.store_output_word(bits, layout.words_per_lane, base + channel, exec);
        }
        // The flags word belongs to this invocation alone; OR in the target's bit.
        const uint32_t flags_index =
            b.ibin(Op_IAdd, b.packet_output_base,
                   b.ibin(Op_IMul, b.gidx, b.uconst(layout.words_per_lane)));
        const uint32_t pointer = output_word_pointer(b, flags_index);
        const uint32_t old = b.id();
        b.put(b.code, Op_Load, {b.t_u32, old, pointer});
        uint32_t updated = b.ibin(Op_BitwiseOr, old, b.uconst(flag));
        if (exec) {
            const uint32_t selected = b.id();
            b.put(b.code, Op_Select, {b.t_u32, selected, exec, updated, old});
            updated = selected;
        }
        b.put(b.code, Op_Store, {pointer, updated});
        saw_export = true;
        return true;
    };
    b.ngg_alloc_request = [&](RegState& state, const Rdna2Inst& in) -> bool {
        bool resolved = true;
        Operand m0;
        m0.kind = OperandKind::Special;
        m0.value = 124;
        const uint32_t value = operand_bits(b, state, in, m0, &resolved);
        if (!resolved) {
            emit_refusal = "ngg-sendmsg-m0-unproven";
            return false;
        }
        const uint32_t lane0 = b.ucmp(Op_IEqual, guest_lane, b.uconst(0));
        const uint32_t wave0 = b.ucmp(Op_IEqual, guest_wave, b.uconst(0));
        emit_if(b, b.land(lane0, wave0), [&] {
            b.put(b.code, Op_Store,
                  {output_word_pointer(b,
                                       b.ibin(Op_IAdd, block_base, b.uconst(kNggHeaderVertsAlloc))),
                   b.ibin(Op_BitwiseAnd, value, b.uconst(0xfffu))});
            b.put(b.code, Op_Store,
                  {output_word_pointer(b,
                                       b.ibin(Op_IAdd, block_base, b.uconst(kNggHeaderPrimsAlloc))),
                   b.ibin(Op_BitwiseAnd, b.ibin(Op_ShiftRightLogical, value, b.uconst(12)),
                          b.uconst(0xfffu))});
            atomic_increment(b, b.ibin(Op_IAdd, block_base, b.uconst(kNggHeaderAllocRequests)));
        });
        emit_if(b, b.land(lane0, b.logical_not(wave0)), [&] {
            atomic_increment(b, b.ibin(Op_IAdd, block_base, b.uconst(kNggHeaderStrayRequests)));
        });
        return true;
    };

    auto safe_branches = safe_execz_branches(ins);
    for (uint32_t pc : waterfall_branches(ins)) safe_branches.insert(pc);
    const bool force_phases_for_dpp = (config.native_wave64 && has_barrier) ||
                                      std::any_of(ins.begin(), ins.end(), is_dpp_row_shr_bounded);
    const bool emitted =
        emit_body(b, rs, ins, safe_branches, resources,
                  /*allow_exec_update*/ true, /*allow_smem*/ resources != nullptr, export_record,
                  linked_code, dwords, nullptr, true, 0, force_phases_for_dpp);
    if (!emitted || !saw_export) {
        std::string detail = emit_refusal.empty() ? std::string() : "cause=" + emit_refusal;
        if (diagnostic.program_address) {
            const std::string inner = last_terminal_reject_reason(diagnostic.program_address);
            if (!inner.empty())
                detail += (detail.empty() ? "" : " ") + std::string("inner=") + inner;
        }
        fail(refusal, diagnostic,
             emit_refusal.empty() ? "ngg-compile-rejected" : emit_refusal.c_str(), detail);
        return {};
    }
    if (layout_out) *layout_out = layout;
    return b.finish();
}

}   // namespace prosper::gpu
