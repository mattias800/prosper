#include "gpu/recompiler/rdna2_dpp_row_shr.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_vgpr_word_definitions.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <algorithm>
#include <bitset>
#include <deque>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <vector>

namespace prosper::gpu {
namespace {

bool is_inplace_integer_row_shr(const Rdna2Inst& in, uint32_t opcode) {
    return in.fmt == Rdna2Format::VOP2 && in.opcode == opcode && in.has_dpp && !in.has_modifier &&
           !in.has_sdwa && !in.dpp_bound_ctrl && in.dpp_ctrl >= 0x111u && in.dpp_ctrl <= 0x11fu &&
           in.dpp_row_mask == 0xfu && in.dpp_bank_mask == 0xfu && in.n_src == 2 && !in.src_neg[0] &&
           !in.src_neg[1] && !in.src_abs[0] && !in.src_abs[1] && !in.clamp && !in.omod &&
           in.dst.kind == OperandKind::VGPR && in.src[0].kind == OperandKind::VGPR &&
           in.src[1].kind == OperandKind::VGPR && in.dst.value == in.src[0].value &&
           in.dst.value == in.src[1].value;
}
}  // namespace

bool is_inplace_vmax_u32_dpp_row_shr(const Rdna2Inst& in) {
    return is_inplace_integer_row_shr(in, 0x14u);
}

bool is_inplace_vor_b32_dpp_row_shr(const Rdna2Inst& in) {
    return is_inplace_integer_row_shr(in, 0x1cu);
}

bool is_compute_dpp_row_shr(const Rdna2Inst& in, bool ngg_workgroup_shell) {
    return is_inplace_vadd_nc_u32_dpp_row_shr(in) || is_inplace_vmax_u32_dpp_row_shr(in) ||
           is_inplace_vor_b32_dpp_row_shr(in) ||
           (ngg_workgroup_shell && is_dpp_row_shr_bounded(in));
}

bool needs_compute_dpp_row_shr_dispatcher(const std::vector<Rdna2Inst>& instructions,
                                          bool ngg_workgroup_shell, uint32_t native_subgroup_size) {
    // A portable NGG workgroup owns guest lanes independently of host subgroup width. Even a
    // branch-free integer row operation must rendezvous through the tagged workgroup phase.
    // Ordinary compute retains its established minimum-subgroup contract and generated graph.
    const bool portable_ngg = ngg_workgroup_shell && !native_subgroup_size;
    if (!std::any_of(instructions.begin(), instructions.end(), [portable_ngg](const Rdna2Inst& in) {
            return is_inplace_vmax_u32_dpp_row_shr(in) || is_inplace_vor_b32_dpp_row_shr(in) ||
                   (portable_ngg && is_compute_dpp_row_shr(in, true));
        }))
        return false;
    return portable_ngg ||
           std::any_of(instructions.begin(), instructions.end(), [](const Rdna2Inst& in) {
               return in.fmt == Rdna2Format::SOPP && in.opcode >= 0x02 && in.opcode <= 0x09 &&
                      in.opcode != 0x03;
           });
}

std::map<uint32_t, std::set<int>>
ngg_row_sources_without_live_spills(const std::vector<Rdna2Inst>& instructions,
                                    const RegState& entry, bool* terminal_full_exec,
                                    std::set<uint32_t>* full_exec_pcs) {
    std::set<int> sources;
    for (const auto& slot : entry.vgpr_lane_slots) sources.insert(slot.first);
    for (const auto& slot : entry.vgpr_lane_mask_slots) sources.insert(slot.first);
    for (const auto& in : instructions) {
        if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361u) sources.insert(in.dst.value);
    }
    std::map<uint32_t, std::set<int>> owned;
    const auto full_exec =
        std::none_of(instructions.begin(), instructions.end(), rdna2_instruction_may_change_exec)
            ? std::vector<uint8_t>(instructions.size(), !entry.exec_narrowed)
            : must_fact_at(instructions, exec_write_sets_full_mask,
                           rdna2_instruction_may_change_exec, !entry.exec_narrowed);
    if (terminal_full_exec) *terminal_full_exec = !full_exec.empty() && full_exec.back();
    if (full_exec_pcs) {
        full_exec_pcs->clear();
        for (size_t i = 0; i < full_exec.size(); ++i)
            if (full_exec[i]) full_exec_pcs->insert(instructions[i].pc);
    }
    if (sources.empty()) return owned;
    if (full_exec.empty()) {
        for (const auto& in : instructions) owned.try_emplace(in.pc);
        return owned;
    }
    // Carry ordinary ownership through complete definitions, including sources. Some services
    // read the VGPR value plane directly, so a fresh full word must not inherit a live spill's
    // unresolved placeholder through an intermediate destination. Initialization remains the
    // caller's entry contract; its supplied entry values seed this MUST analysis.
    using Words = std::bitset<256>;
    Words entry_words;
    for (const auto& [reg, value] : entry.vreg)
        if (reg >= 0 && reg < 256 && value && !entry.vgpr_lane_slots.contains(reg) &&
            !entry.vgpr_lane_mask_slots.contains(reg))
            entry_words.set(static_cast<size_t>(reg));
    std::map<uint32_t, size_t> index;
    for (size_t i = 0; i < instructions.size(); ++i) index.emplace(instructions[i].pc, i);
    std::vector<std::vector<size_t>> successors(instructions.size());
    for (size_t i = 0; i < instructions.size(); ++i) {
        const auto& in = instructions[i];
        if (in.is_end) continue;
        const bool branch = in.fmt == Rdna2Format::SOPP && in.opcode >= 0x02u &&
                            in.opcode <= 0x09u && in.opcode != 0x03u;
        if (branch) {
            const uint32_t target = branch_target(in);
            if (const auto at = index.find(target); at != index.end())
                successors[i].push_back(at->second);
            else if (target <= instructions.back().pc)
                return {};
        }
        if (!(branch && in.opcode == 0x02u) && i + 1u < instructions.size())
            successors[i].push_back(i + 1u);
    }
    Words all;
    all.set();
    std::vector<Words> ordinary(instructions.size(), all);
    ordinary.front() = entry_words;
    std::deque<size_t> pending;
    std::vector<bool> queued(instructions.size(), true);
    for (size_t i = 0; i < instructions.size(); ++i) pending.push_back(i);
    while (!pending.empty()) {
        const size_t i = pending.front();
        pending.pop_front();
        queued[i] = false;
        const auto& in = instructions[i];
        Words out = ordinary[i];
        bool ordinary_sources = true;
        for (uint32_t k = 0; k < in.n_src; ++k) {
            if (in.src[k].kind != OperandKind::VGPR) continue;
            const uint32_t span = ordinary_vgpr_consumed_source_words(in, k);
            if (span > 256u) {
                ordinary_sources = false;
                break;
            }
            for (uint32_t word = 0; word < span; ++word) {
                const int reg = ordinary_vgpr_consumed_source_register(in, k, word);
                if (reg < 0 || reg >= 256 || !ordinary[i].test(static_cast<size_t>(reg)))
                    ordinary_sources = false;
            }
        }
        // M0-relative reads and writes, and v_writelane's single selected lane, are the effects
        // table's (every encoding of v_movrel*/v_swaprel, not only the VOP1 movreld/movrels).
        const Rdna2InstructionEffects& effects = rdna2_instruction_effects(in);
        const bool valu = effects.describes_vector_result();
        const bool relative_write = rdna2_writes_m0_relative_vgpr(in);
        const bool relative_read = valu && (effects.m0_relative == Rdna2M0Relative::Source ||
                                            effects.m0_relative == Rdna2M0Relative::Both);
        const bool lane_write = valu && effects.lane_flow == Rdna2LaneFlow::WriteLane;
        const uint32_t clobbers = rdna2_vgpr_write_count(in);
        for (uint32_t word = 0; word < clobbers; ++word) {
            const int reg = in.dst.value + static_cast<int>(word);
            if (reg >= 0 && reg < 256 && (!ordinary_sources || lane_write || relative_read))
                out.reset(static_cast<size_t>(reg));
        }
        // A swap also writes SRC0's VGPR (relative to M0 for v_swaprel_b32).
        const int swapped =
            effects.writes_src0 && in.src[0].kind == OperandKind::VGPR ? in.src[0].value : -1;
        if (relative_write)
            for (int reg =
                     std::max(0, swapped >= 0 ? std::min(swapped, in.dst.value) : in.dst.value);
                 reg < 256; ++reg)
                out.reset(static_cast<size_t>(reg));
        else if (swapped >= 0 && swapped < 256)
            out.reset(static_cast<size_t>(swapped));
        if (ordinary_sources && full_exec[i]) {
            const uint32_t complete = ordinary_vgpr_complete_results(in);
            for (uint32_t word = 0; word < complete; ++word) {
                const int reg = in.dst.value + static_cast<int>(word);
                if (reg >= 0 && reg < 256) out.set(static_cast<size_t>(reg));
            }
        }
        for (size_t successor : successors[i]) {
            const Words meet = ordinary[successor] & out;
            if (meet == ordinary[successor]) continue;
            ordinary[successor] = meet;
            if (!queued[successor]) {
                pending.push_back(successor);
                queued[successor] = true;
            }
        }
    }
    for (size_t i = 0; i < instructions.size(); ++i) {
        auto& at = owned[instructions[i].pc];
        for (int reg = 0; reg < 256; ++reg)
            if (ordinary[i].test(static_cast<size_t>(reg))) at.insert(reg);
    }
    return owned;
}

void prepare_ngg_row_source_proof(SpirvCompute& b, const RegState& entry,
                                  const std::vector<Rdna2Inst>& instructions) {
    if (!b.ngg_workgroup_shell || b.ngg_row_source_analysis_done || instructions.empty()) return;
    b.ngg_row_source_analysis_done = true;
    b.ngg_ordinary_row_sources =
        ngg_row_sources_without_live_spills(instructions, entry, nullptr, &b.ngg_full_exec_pcs);
}

void restore_ngg_row_source_state(SpirvCompute& b, RegState& state,
                                  const std::map<uint32_t, std::set<int>>& ordinary_sources,
                                  const std::vector<uint32_t>& starts,
                                  const std::vector<std::vector<uint32_t>>& dispatch_blocks,
                                  uint32_t dispatch, uint32_t end_pc, bool terminal_full_exec) {
    const auto ordinary_here = [&](int reg) {
        const auto at = [&](uint32_t pc) {
            const auto found = ordinary_sources.find(pc);
            return found != ordinary_sources.end() && found->second.contains(reg);
        };
        return dispatch == UINT32_MAX
                   ? at(end_pc)
                   : std::all_of(dispatch_blocks[dispatch].begin(), dispatch_blocks[dispatch].end(),
                                 [&](uint32_t block) { return at(starts[block]); });
    };
    std::erase_if(state.vgpr_lane_slots,
                  [&](const auto& slot) { return ordinary_here(slot.first); });
    std::erase_if(state.vgpr_lane_mask_slots,
                  [&](const auto& slot) { return ordinary_here(slot.first); });
    if (dispatch == UINT32_MAX && terminal_full_exec) {
        state.exec = b.btrue();
        state.exec_narrowed = false;
    }
}

bool resolve_compute_dpp_row_source(SpirvCompute& b, RegState& state, const Rdna2Inst& in,
                                    const std::map<uint32_t, std::set<int>>& ordinary_sources,
                                    uint32_t& source) {
    bool resolved = true;
    if (b.ngg_workgroup_shell && !ordinary_sources.empty()) {
        const auto ordinary = ordinary_sources.find(in.pc);
        resolved = ordinary != ordinary_sources.end() && ordinary->second.contains(in.src[0].value);
    }
    if (resolved) source = operand_bits(b, state, in, in.src[0], &resolved);
    return resolved;
}

bool emit_compute_inplace_dpp_row_shr(SpirvCompute& b, const RegState& rs, const Rdna2Inst& in,
                                      uint32_t source, uint32_t old_destination, bool wave_uniform,
                                      uint32_t& result) {
    if (!b.is_compute || (b.ngg_workgroup_shell && !b.native_subgroup_size)) return false;
    // Retain the established ADD ladder and its generated graph. Other ADD strides belong to
    // the existing event-isolated CFG service rather than the ordinary instruction emitter.
    const bool add =
        is_inplace_vadd_nc_u32_dpp_row_shr(in) && (in.dpp_ctrl == 0x111u || in.dpp_ctrl == 0x112u ||
                                                   in.dpp_ctrl == 0x114u || in.dpp_ctrl == 0x118u);
    const bool maximum = wave_uniform && is_inplace_vmax_u32_dpp_row_shr(in);
    const bool bitwise_or = wave_uniform && is_inplace_vor_b32_dpp_row_shr(in);
    if (!add && !maximum && !bitwise_or) return false;

    // DPP16 ROW_SHR selects lane-N inside its 16-lane row. BC0 keeps VDST at a row edge.
    // Preserve VDST for an invalid source under this encoded BC0/FI0 contract. For an
    // in-place unsigned maximum, max(own, 0) == own as well. OR has the same identity:
    // CONFIDENCE: HIGH -- AMD RDNA2 70648 tables 90/91 define FI0's inactive source as zero
    // and BC0's out-of-row destination as unwritten; own | 0 == own. Preserve the old
    // destination for either unavailable source. The caller separately gates destination EXEC.
    uint32_t valid_source = 0;
    const uint32_t shifted =
        b.subgroup_row_shr(source, rs.exec, in.dpp_ctrl - 0x110u, &valid_source);
    const uint32_t combined = maximum
                                  ? b.uext2(Glsl_UMax, source, shifted)
                                  : b.ibin(bitwise_or ? Op_BitwiseOr : Op_IAdd, source, shifted);
    result = b.sel(valid_source, combined, old_destination);
    return true;
}

// Publish event/EXEC metadata for every worker, including nonpending and completed waves.
// Two barriers bound reuse of the value plane by later wave services and loop iterations.
bool emit_portable_compute_dpp_row_shr_phase(SpirvCompute& b,
                                             const ComputeDppRowShrPhaseVariables& variables,
                                             uint32_t value_base, uint32_t metadata_base,
                                             const std::set<int>& destinations,
                                             const std::map<int, uint32_t>& vv,
                                             const std::map<std::pair<int, int>, uint32_t>& lv,
                                             const std::map<std::pair<int, int>, uint32_t>& lmv) {
    const uint32_t zero = b.uconst(0), no = b.bfalse();
    const uint32_t dpp_pending = b.load_function(b.t_bool, variables.pending);
    const uint32_t dpp_active = b.load_function(b.t_bool, variables.active);
    const uint32_t dpp_source = b.load_function(b.t_u32, variables.source);
    const uint32_t dpp_event = b.load_function(b.t_u32, variables.event);
    b.cfg_scratch_store(b.ibin(Op_IAdd, b.uconst(value_base), b.linear_localid), dpp_source);
    const uint32_t dpp_metadata =
        b.sel(dpp_pending,
              b.ibin(Op_BitwiseOr, b.ibin(Op_ShiftLeftLogical, dpp_event, b.uconst(1)),
                     b.sel(dpp_active, b.uconst(1), zero)),
              zero);
    b.cfg_scratch_store(b.ibin(Op_IAdd, b.uconst(metadata_base), b.linear_localid), dpp_metadata);
    b.barrier();

    const uint32_t dpp_control = b.load_function(b.t_u32, variables.amount);
    // The bounded form belongs to the merged-NGG workgroup shell. Preserve the established GTA
    // unbounded reduction's generated graph and invalid-source write rule outside that shell.
    const uint32_t dpp_bounded =
        b.ngg_workgroup_shell
            ? b.ucmp(Op_INotEqual, b.ibin(Op_BitwiseAnd, dpp_control, b.uconst(0x100)), zero)
            : no;
    const uint32_t dpp_amount =
        b.ngg_workgroup_shell ? b.ibin(Op_BitwiseAnd, dpp_control, b.uconst(0xf)) : dpp_control;
    const uint32_t dpp_row_lane = b.ibin(Op_BitwiseAnd, b.linear_localid, b.uconst(15));
    const uint32_t dpp_in_bounds = b.ucmp(Op_UGreaterThanEqual, dpp_row_lane, dpp_amount);
    // Keep even the disabled lane's scratch address valid. BOUND_CTRL=0 uses the validity gate
    // below to preserve old VDST rather than consuming this self-addressed placeholder.
    const uint32_t dpp_source_index =
        b.sel(dpp_in_bounds, b.ibin(Op_ISub, b.linear_localid, dpp_amount), b.linear_localid);
    const uint32_t dpp_shifted =
        b.cfg_scratch_load(b.ibin(Op_IAdd, b.uconst(value_base), dpp_source_index));
    const uint32_t dpp_source_metadata =
        b.cfg_scratch_load(b.ibin(Op_IAdd, b.uconst(metadata_base), dpp_source_index));
    const uint32_t dpp_source_event =
        b.ibin(Op_ShiftRightLogical, dpp_source_metadata, b.uconst(1));
    const uint32_t dpp_source_active =
        b.ucmp(Op_INotEqual, b.ibin(Op_BitwiseAnd, dpp_source_metadata, b.uconst(1)), zero);
    uint32_t dpp_valid_source = b.land(dpp_in_bounds, dpp_source_active);
    dpp_valid_source = b.land(dpp_valid_source, b.ucmp(Op_IEqual, dpp_source_event, dpp_event));
    uint32_t dpp_result =
        b.ibin(Op_IAdd, dpp_source,
               b.ngg_workgroup_shell ? b.sel(dpp_valid_source, dpp_shifted, zero) : dpp_shifted);
    // Allocate and consume an operation field only for streams containing MAX. An ADD-only
    // stream retains its previous IDs and graph exactly. Static events isolate different sites.
    if (variables.maximum) {
        const uint32_t maximum = b.load_function(b.t_bool, variables.maximum);
        dpp_result = b.sel(maximum, b.uext2(Glsl_UMax, dpp_source, dpp_shifted), dpp_result);
    }
    // OR shares the unavailable-source identity. BC0 keeps VDST; bounded NGG OR writes own|0.
    if (variables.bitwise_or) {
        const uint32_t bitwise_or = b.load_function(b.t_bool, variables.bitwise_or);
        dpp_result =
            b.sel(bitwise_or,
                  b.ibin(Op_BitwiseOr, dpp_source, b.sel(dpp_valid_source, dpp_shifted, zero)),
                  dpp_result);
    }
    const uint32_t dpp_write =
        b.land(b.land(dpp_pending, dpp_active),
               b.ngg_workgroup_shell ? b.lor(dpp_bounded, dpp_valid_source) : dpp_valid_source);
    const uint32_t dpp_dst = b.load_function(b.t_u32, variables.destination);
    for (int reg : destinations) {
        const auto kv = vv.find(reg);
        if (kv == vv.end()) return false;
        const uint32_t selected =
            b.land(dpp_write, b.ucmp(Op_IEqual, dpp_dst, b.uconst(static_cast<uint32_t>(reg))));
        const uint32_t old = b.load_function(b.t_u32, kv->second);
        b.store_function(kv->second, b.sel(selected, dpp_result, old));
    }
    // The physical VGPR definition invalidates scalar lane-spill aliases even when EXEC or the
    // shifted source suppresses this invocation's data write.
    for (const auto& kv : lv) {
        if (!destinations.contains(kv.first.first)) continue;
        const uint32_t selected =
            b.land(dpp_pending,
                   b.ucmp(Op_IEqual, dpp_dst, b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_u32, kv.second);
        b.store_function(kv.second, b.sel(selected, zero, old));
    }
    for (const auto& kv : lmv) {
        if (!destinations.contains(kv.first.first)) continue;
        const uint32_t selected =
            b.land(dpp_pending,
                   b.ucmp(Op_IEqual, dpp_dst, b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_bool, kv.second);
        b.store_function(kv.second, b.bsel(selected, no, old));
    }
    b.barrier();
    return true;
}

}   // namespace prosper::gpu
