// Remaining lanes share a counter ordinal; DATA CMPX may leave early. Scalar results die at exit.
#include "gpu/recompiler/rdna2_uniform_counter_data.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <vector>

namespace prosper::gpu {
namespace {
bool writes_word(const Rdna2Inst& in, int reg) {
    bool writes = false;
    for_each_scalar_write(in, [&](int base, uint32_t width) {
        writes |= base <= reg && reg - base < static_cast<int>(width);
    });
    return writes;
}

bool counter_entry(const SpirvCompute& b, const std::vector<Rdna2Inst>& ins, const DivLoop& loop,
                   const RegState& entry, int counter) {
    const auto value = entry.sreg.find(counter);
    if (value == entry.sreg.end() || !value->second ||
        sreg_word_may_be_fabricated(entry, counter) || scalar_is_lane_local(entry, counter))
        return false;
    const Rdna2Inst* init = nullptr;
    for (const auto& in : ins)
        if (in.pc < loop.header_pc && writes_word(in, counter)) init = &in;
    if (!init || init->fmt != Rdna2Format::SOP1 || init->opcode != kSop1OpcodeMovB32 ||
        init->dst.kind != OperandKind::SGPR || init->dst.value != counter || init->n_src != 1 ||
        init->src[0].kind != OperandKind::InlineInt || init->src[0].value ||
        init->len_dwords != 1 || init->pc >= b.saved_mask_original_words.size())
        return false;
    auto decoded = rdna2_decode_one(b.saved_mask_original_words.data() + init->pc,
                                    b.saved_mask_original_words.size() - init->pc);
    decoded.pc = init->pc;
    if (decoded != *init) return false;
    for (const auto& in : ins) {
        const bool branch = in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode);
        // Every incoming preheader path must execute the literal initialization.
        if (in.pc > init->pc && in.pc < loop.header_pc && branch) return false;
        if (branch && (in.pc < loop.header_pc || in.pc > loop.backedge_pc)) {
            const auto target = branch_target(in);
            if (target > init->pc && target <= loop.header_pc) return false;
        }
    }
    return true;
}

bool counter_step(const Rdna2Inst& in, int counter) {
    return in.fmt == Rdna2Format::SOP2 && (in.opcode == 0x00u || in.opcode == 0x02u) &&
           in.dst.kind == OperandKind::SGPR && in.dst.value == counter && in.n_src == 2 &&
           in.src[0].kind == OperandKind::SGPR && in.src[0].value == counter &&
           in.src[1].kind == OperandKind::InlineInt && in.src[1].value == 1;
}
} // namespace

bool uniform_counter_loop_data_boundary(const SpirvCompute& b, const std::vector<Rdna2Inst>& ins,
                                        const DivLoop& loop, const RegState& entry) {
    if (loop.condition != DivLoop::Condition::Exec || loop.bottom_tested || !loop.continue_on_set ||
        loop.direct_exec_breaks || loop.direct_wave_breaks || !loop.break_pcs.empty() ||
        loop.header_pc >= loop.exit_branch_pc || loop.exit_branch_pc >= loop.backedge_pc ||
        loop_live_wave_scalar(ins, loop) >= 0)
        return false;
    const Rdna2Inst* compare = nullptr;
    for (const auto& in : ins)
        if (in.pc < loop.exit_branch_pc && in.pc + in.len_dwords == loop.exit_branch_pc)
            compare = &in;
    // Full-word scalar SDWA is intentional: both operands broadcast identically in the cohort.
    if (!compare || compare->fmt != Rdna2Format::VOPC || compare->opcode != 0xd1u ||
        compare->n_src != 2 || compare->has_dpp || compare->has_modifier ||
        compare->src[0].kind != OperandKind::SGPR || compare->src[0].value < 0 ||
        compare->src[0].value > 105 || compare->src[1].kind != OperandKind::InlineInt ||
        compare->src[1].value < 1 || compare->src[1].value > 64 || compare->sdwa_src0_sel != 6 ||
        compare->sdwa_src1_sel != 6 || compare->sdwa_src0_sext || compare->sdwa_src1_sext)
        return false;
    const int counter = compare->src[0].value;
    if (!counter_entry(b, ins, loop, entry, counter)) return false;
    unsigned steps = 0;
    for (const auto& in : ins) {
        if (in.pc < loop.header_pc || in.pc > loop.backedge_pc) continue;
        const auto& effects = rdna2_instruction_effects(in);
        if (effects.m0_relative != Rdna2M0Relative::None || in.has_dpp ||
            (effects.describes_sources() && effects.lane_flow != Rdna2LaneFlow::LaneWise &&
             effects.lane_flow != Rdna2LaneFlow::None) ||
            in.fmt == Rdna2Format::DS)
            return false;
        if (in.pc == loop.exit_branch_pc || in.pc == loop.backedge_pc) {
            const uint32_t opcode =
                in.pc == loop.exit_branch_pc ? kSoppOpcodeCbranchExecz : kSoppOpcodeBranch;
            const uint32_t target = in.pc == loop.exit_branch_pc ? loop.exit_pc : loop.header_pc;
            if (in.fmt != Rdna2Format::SOPP || in.opcode != opcode || branch_target(in) != target)
                return false;
        } else if (in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode)) {
            return false;
        }
        // Earlier condition-region DATA CMPX instructions also only remove lanes. Remaining
        // lanes still execute exactly one scalar step per trip; the scalar result dies at exit.
        const bool data_cmpx =
            in.pc < compare->pc && in.fmt == Rdna2Format::VOPC && vopc_is_cmpx(in.opcode);
        if (in.pc != compare->pc && !data_cmpx &&
            (rdna2_instruction_may_change_exec(in) || writes_word(in, 126) || writes_word(in, 127)))
            return false;
        if (writes_word(in, counter)) {
            if (in.pc <= loop.exit_branch_pc || !counter_step(in, counter)) return false;
            ++steps;
        }
    }
    return steps == 1;
}
}   // namespace prosper::gpu
