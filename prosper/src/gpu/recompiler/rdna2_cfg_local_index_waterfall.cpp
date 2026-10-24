#include "gpu/recompiler/rdna2_cfg_local_index_waterfall.hpp"

#include "gpu/recompiler/rdna2_cfg_branch_helpers.hpp"
#include "gpu/recompiler/rdna2_cfg_scalar_merge.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>

namespace prosper::gpu {
namespace {
bool scalar(const Operand& operand, int reg) {
    return (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special) &&
           operand.value == reg;
}

bool move(const Rdna2Inst& in, int destination, int source, bool pair) {
    return in.fmt == Rdna2Format::SOP1 &&
           in.opcode == (pair ? kSop1OpcodeMovB64 : kSop1OpcodeMovB32) &&
           scalar(in.dst, destination) && scalar(in.src[0], source);
}

bool ordinary_pair(const Operand& operand) {
    return operand.kind == OperandKind::SGPR && operand.value >= 0 && operand.value <= 104;
}

bool writes_word(const Rdna2Inst& in, int reg) {
    bool writes = false;
    for_each_scalar_write(in, [&](int first, uint32_t width) {
        writes |= reg >= first && reg < first + static_cast<int>(width);
    });
    return writes;
}

// Both mask pairs must still be copies of the same EXEC cohort. A branch into the
// middle of these definitions, an unknown effect or an EXEC change invalidates it.
bool saved_cohort_prefix(const std::vector<Rdna2Inst>& ins, size_t read, int remain, int saved,
                         size_t& first) {
    size_t remain_at = ins.size(), saved_at = ins.size();
    for (size_t at = read; at-- > 0;) {
        const auto& in = ins[at];
        if (remain_at == ins.size() && (writes_word(in, remain) || writes_word(in, remain + 1))) {
            if (!move(in, remain, 126, true)) return false;
            remain_at = at;
        }
        if (saved_at == ins.size() && (writes_word(in, saved) || writes_word(in, saved + 1))) {
            if (!move(in, saved, 126, true)) return false;
            saved_at = at;
        }
        if (remain_at != ins.size() && saved_at != ins.size()) break;
    }
    if (remain_at == ins.size() || saved_at == ins.size()) return false;
    first = std::min(remain_at, saved_at);
    for (size_t at = first; at < read; ++at) {
        const auto& in = ins[at];
        if (in.is_end || rdna2_escapes_decoded_effects(in) ||
            rdna2_instruction_may_change_exec(in) ||
            (in.fmt == Rdna2Format::SOPP && !sopp_is_noop(in)))
            return false;
    }
    return true;
}

bool unobservable_after(const std::vector<Rdna2Inst>& ins, size_t tail, int index) {
    const auto& restore_m0 = ins[tail + 4];
    if (restore_m0.fmt != Rdna2Format::SOP1 || restore_m0.opcode != kSop1OpcodeMovB32 ||
        !scalar(restore_m0.dst, 124) || scalar(restore_m0.src[0], index))
        return false;
    const uint32_t after = ins[tail + 3].pc;
    // CMPX on gfx10 writes EXEC only; the allowed body and tail also preserve VCC.
    // An outside VCC read therefore observes the same pre-waterfall generation.
    return sgpr_dead_at_merge(ins, after, index);
}

size_t relative_moves_end(const std::vector<Rdna2Inst>& ins, size_t first) {
    bool moved = false;
    size_t tail = first;
    while (tail < ins.size()) {
        const auto& in = ins[tail];
        if (sopp_is_noop(in)) {
            ++tail;
            continue;
        }
        if (in.fmt != Rdna2Format::VOP1 || in.opcode != 0x43 ||
            in.src[0].kind != OperandKind::VGPR || in.has_modifier || in.has_dpp)
            break;
        moved = true;
        ++tail;
    }
    return moved ? tail : ins.size();
}
} // namespace

uint32_t local_index_waterfall_latch(const std::vector<Rdna2Inst>& ins, uint32_t read_pc) {
    const auto read =
        std::find_if(ins.begin(), ins.end(), [&](const auto& in) { return in.pc == read_pc; });
    if (read == ins.end() || ins.empty() || !ins.back().is_end) return UINT32_MAX;
    const size_t at = static_cast<size_t>(read - ins.begin());
    if (at + 8 >= ins.size() || read->fmt != Rdna2Format::VOP1 ||
        !rdna2_lane_read_writes_sgpr(*read) || read->dst.value < 0 || read->dst.value > 105 ||
        read->src[0].kind != OperandKind::VGPR || read->has_modifier || read->has_dpp)
        return UINT32_MAX;
    const int index = read->dst.value;
    const auto& compare = ins[at + 1];
    if (compare.fmt != Rdna2Format::VOPC || compare.opcode != kVopcOpcodeCmpxEqU32 ||
        compare.has_modifier || compare.has_dpp || !scalar(compare.src[0], index) ||
        compare.src[1] != read->src[0] || !move(ins[at + 2], 124, index, false))
        return UINT32_MAX;
    const size_t tail = relative_moves_end(ins, at + 3);
    if (tail + 5 >= ins.size()) return UINT32_MAX;
    const auto& andn2 = ins[tail];
    if (andn2.fmt != Rdna2Format::SOP2 || andn2.opcode != 0x15 || !ordinary_pair(andn2.dst) ||
        andn2.src[0] != andn2.dst || !scalar(andn2.src[1], 126))
        return UINT32_MAX;
    const int remain = andn2.dst.value;
    const auto& latch = ins[tail + 2];
    const auto& restore = ins[tail + 3];
    if (!move(ins[tail + 1], 126, remain, true) || latch.fmt != Rdna2Format::SOPP ||
        latch.opcode != 0x05 || latch.simm16 >= 0 || branch_target(latch) != read_pc ||
        restore.fmt != Rdna2Format::SOP1 || restore.opcode != kSop1OpcodeMovB64 ||
        !scalar(restore.dst, 126) || !ordinary_pair(restore.src[0]))
        return UINT32_MAX;
    const int saved = restore.src[0].value;
    if (std::abs(remain - saved) < 2 || (index >= remain && index <= remain + 1) ||
        (index >= saved && index <= saved + 1))
        return UINT32_MAX;
    size_t first = 0;
    if (!saved_cohort_prefix(ins, at, remain, saved, first) ||
        !unobservable_after(ins, tail, index))
        return UINT32_MAX;
    for (const auto& in : ins) {
        if (rdna2_escapes_decoded_effects(in)) return UINT32_MAX;
        if (in.pc == latch.pc || in.fmt != Rdna2Format::SOPP ||
            !sopp_opcode_is_direct_branch(in.opcode))
            continue;
        const uint32_t target = branch_target(in);
        if (target > ins[first].pc && target <= ins[tail + 4].pc) return UINT32_MAX;
    }
    return waterfall_branches(ins).contains(latch.pc) ? latch.pc : UINT32_MAX;
}
} // namespace prosper::gpu
