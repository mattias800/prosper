#include "gpu/recompiler/rdna2_scalar_exec_scopes.hpp"

#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_cfg_branch_helpers.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"

#include <algorithm>
#include <functional>
#include <cstdint>
#include <map>
#include <tuple>
#include <vector>

namespace prosper::gpu {
namespace {
// Native READFIRSTLANE currently lowers to THIS lane's value. A new definedness view must
// therefore prove that expression agrees in the active cohort; it cannot admit a missing
// broadcast. Restrict this proof to complete lane-wise definitions in the same basic block,
// from constants and still-current launch scalars. Cross-lane lowering remains separate.
bool first_lane_source_agrees(const RegState& rs, const std::vector<Rdna2Inst>& instructions,
                              const Rdna2Inst& read) {
    const auto at = std::find(instructions.begin(), instructions.end(), read);
    if (at == instructions.end()) return false;
    uint32_t first = 0;
    for (const auto& in : instructions) {
        if (rdna2_escapes_decoded_effects(in)) return false;
        if (in.pc < read.pc && (in.is_end || rdna2_escapes_decoded_effects(in) ||
                                (in.fmt == Rdna2Format::SOPP && !sopp_is_noop(in))))
            first = std::max(first, in.pc + in.len_dwords);
        if (in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode)) {
            const uint32_t target = scalar_branch_target(in);
            if (target <= read.pc) first = std::max(first, target);
        }
    }
    std::function<bool(const Operand&, uint32_t, unsigned)> agrees;
    auto prove = [&](const Operand& operand, uint32_t before, unsigned depth) {
        if (depth > 32) return false;
        if (operand.kind == OperandKind::InlineInt || operand.kind == OperandKind::InlineFloat ||
            operand.kind == OperandKind::Literal)
            return true;
        if (operand.kind == OperandKind::SGPR) {
            bool written = false;
            for (const auto& in : instructions)
                for_each_scalar_write(in, [&](int first_word, uint32_t width) {
                    written |= operand.value >= first_word &&
                               operand.value < first_word + static_cast<int>(width);
                });
            return !written && !rs.sreg_merge_placeholder.contains(operand.value) &&
                   (rs.sreg.contains(operand.value) || rs.sreg_input.contains(operand.value));
        }
        if (operand.kind != OperandKind::VGPR) return false;
        for (auto in = at; in != instructions.begin();) {
            --in;
            if (in->pc >= before) continue;
            if (in->pc < first) return false;
            const uint32_t width = rdna2_vgpr_write_count(*in);
            if (operand.value < in->dst.value ||
                operand.value >= in->dst.value + static_cast<int>(width))
                continue;
            const auto& effects = rdna2_instruction_effects(*in);
            // Lane-mapped does not imply agreement: MBCNT also consumes the lane index,
            // and mask operands select a different bit in each lane even from one SGPR.
            if (in->has_modifier || in->has_dpp || !effects.describes_vector_result() ||
                effects.lane_flow != Rdna2LaneFlow::LaneWise || effects.implicit_reads ||
                effects.lane_mask_src2 || !vector_write_completes_active_word(*in))
                return false;
            for (const auto& between : instructions)
                if (between.pc > in->pc && between.pc < before &&
                    rdna2_instruction_may_change_exec(between))
                    return false;
            for (uint32_t k = 0; k < in->n_src; ++k) {
                auto source = in->src[k];
                const bool reg =
                    source.kind == OperandKind::SGPR || source.kind == OperandKind::VGPR;
                const uint32_t words = reg ? effects.ssrc_dwords[k] : 1u;
                if (!words) return false;
                for (uint32_t word = 0; word < words; ++word) {
                    if (!agrees(source, in->pc, depth + 1)) return false;
                    if (reg) ++source.value;
                }
            }
            return in->n_src != 0;
        }
        return false;
    };
    // Repeated operands share the same proof. Depth is part of the key so caching cannot
    // turn a query beyond the existing recursion limit into a successful shallower one.
    std::map<std::tuple<OperandKind, int, uint32_t, unsigned>, bool> memo;
    agrees = [&](const Operand& operand, uint32_t before, unsigned depth) {
        const auto key = std::tuple{operand.kind, operand.value, before, depth};
        if (const auto known = memo.find(key); known != memo.end()) return known->second;
        const bool result = prove(operand, before, depth);
        memo.emplace(key, result);
        return result;
    };
    return agrees(read.src[0], read.pc, 0);
}
}   // namespace

bool scalar_scope_first_lane_agrees(const SpirvCompute& b, const RegState& rs,
                                    const Rdna2Inst& read) {
    const auto& words = b.saved_mask_original_words;
    if (!b.is_compute || b.native_subgroup_size != b.wave_size ||
        (b.wave_size != 32 && b.wave_size != 64) || read.pc >= words.size() ||
        read.fmt != Rdna2Format::VOP1 || !rdna2_lane_read_writes_sgpr(read) || read.has_modifier ||
        read.has_dpp || read.src[0].kind != OperandKind::VGPR)
        return false;
    std::vector<Rdna2Inst> instructions;
    for (uint32_t pc = 0; pc < words.size();) {
        auto in = rdna2_decode_one(words.data() + pc, words.size() - pc);
        if (!in.len_dwords) return false;
        in.pc = pc;
        instructions.push_back(in);
        pc += in.len_dwords;
        if (in.is_end) break;
    }
    return first_lane_source_agrees(rs, instructions, read);
}

const ScalarSourceMarks* current_scalar_exec_scope(const RegState& rs, int reg) {
    const auto scope = rs.sreg_exec_scopes.find(reg);
    const auto value = rs.sreg.find(reg);
    if ((rs.vector_wave_size != 32 && rs.vector_wave_size != 64) ||
        scope == rs.sreg_exec_scopes.end() || value == rs.sreg.end() || !value->second ||
        value->second != scope->second.value || !rs.sreg_merge_placeholder.contains(reg) ||
        rs.unavailable_saved_mask_words.contains(reg))
        return nullptr;
    // A current value ID alone is insufficient: an interned zero can also be a missing edge.
    // Joins discard these views, and an intervening change to the global marks invalidates one.
    const auto roots = rs.sreg_loop_blanket.find(reg);
    const auto& source = scope->second.source;
    if (!source.placeholder || source.hard || !source.scoped_complete_word ||
        !source.outside_exec || roots == rs.sreg_loop_blanket.end() ||
        roots->second != source.blanket_roots ||
        rs.sreg_memory_pattern.contains(reg) != source.memory)
        return nullptr;
    return &source;
}

void record_scalar_exec_scope(RegState& rs, const Rdna2Inst& in, int reg, bool ballot,
                              const ScalarSourceMarks& source, bool first_agrees) {
    // READFIRSTLANE consumes the active cohort. READLANE's selected peer can lie outside it;
    // spills and mask outputs retain their existing independent lifetime rules.
    const bool first_lane = in.fmt == Rdna2Format::VOP1 && rdna2_lane_read_writes_sgpr(in);
    const bool scalar_alu =
        in.fmt == Rdna2Format::SOP1 || in.fmt == Rdna2Format::SOP2 || in.fmt == Rdna2Format::SOPK;
    const auto value = rs.sreg.find(reg);
    if ((rs.vector_wave_size != 32 && rs.vector_wave_size != 64) || ballot ||
        (!first_lane && !scalar_alu) || (first_lane && !first_agrees) || value == rs.sreg.end() ||
        !value->second || !source.placeholder || source.hard || !source.scoped_complete_word ||
        source.blanket_roots.empty() || !source.outside_exec || !rs.exec ||
        !exec_within(rs, rs.exec, source.outside_exec)) {
        rs.sreg_exec_scopes.erase(reg);
        return;
    }
    // The global assumption remains in the original maps. This view covers only lanes executing
    // this definition; expanding EXEC still needs the global assumption and its back-edge check.
    auto current = source;
    current.outside_exec = rs.exec;
    rs.sreg_exec_scopes.insert_or_assign(reg, RegState::ScalarExecScope{value->second, current});
}
}   // namespace prosper::gpu
