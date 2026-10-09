#pragma once

// INTERNAL recompiler companion. The two unchanged native entry-VCC proofs were promoted with
// tools/refactor/promote_internal.py before demanded packet-state behavior was introduced.
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include <vector>

namespace prosper::gpu {

// #3231 — is the CFG region's ENTRY-BLOCK VCC value dead?
//
// The dispatcher stores one value into `vcc_var` before its loop, then dispatches block 0 first,
// exactly once, with every invocation active (`selector = active ? pc : UINT32_MAX`, and
// `active_var` is seeded true when the caller has no partial-workgroup extent). So that stored
// value is observable only until block 0 overwrites it: if block 0 DEFINES the complete VCC pair
// before any instruction in it can read VCC, nothing anywhere in the region can see the entry
// value, and persisting `false` for it invents nothing. Block 0's own `save_state` publishes the
// real definition before the iteration's common phases run, so the two direct `vcc_var` readers
// (portable readlane into 106, and the vote-to-VCC merge) see it too.
//
// This is deliberately narrow, because the failure the caller's gate prevents is silent-wrong
// rather than a crash. What it does NOT admit:
//   * anything but a `v_cmp_*` (VOPC, never `v_cmpx_*`) whose destination is the VCC pair. That is
//     the one encoding that defines both words for every lane in a single instruction, and it is
//     the form the live evidence uses. A VOP3B carry-out into VCC, a 64-bit scalar write of the
//     pair, and `v_cmpx_*`'s EXEC write are all left rejected.
//   * a b32 write of vcc_lo or vcc_hi alone — half the pair would still carry the entry value, so
//     the scan stops there rather than continuing to a later full define.
//   * an entry block that reads VCC first, in ANY form. "Reads" is over-approximated: the implicit
//     consumers this file already enumerates for the mask-domain analyses, plus any operand that
//     can name a word of the pair — including a wide scalar read rooted low enough to reach s106.
//     An operand the decoder left stale is read too (all four source slots, not `n_src`), because
//     over-reading only ever moves the answer to "not dead".
//
// `lo`/`hi` are the entry block's half-open pc range as the dispatcher itself partitions it
// (`starts[0]` and `starts[1]`), so a block split by a branch target, or by one of the synchronized
// cross-lane events that each get their own block, shortens the window rather than widening it.
inline bool entry_block_defines_vcc_before_any_read(const std::vector<Rdna2Inst>& ins, uint32_t lo,
                                                    uint32_t hi) {
    auto may_name_vcc = [](const Operand& operand) {
        if (operand.kind == OperandKind::Special)
            return operand.value == 106 || operand.value == 107;
        // The widest scalar operand is an eight-word T#, so a root as low as s99 still covers s106.
        // Treat every scalar operand at or above that root as touching the pair.
        if (operand.kind == OperandKind::SGPR) return operand.value >= 99;
        return false;
    };
    for (const auto& in : ins) {
        if (in.pc < lo) continue;
        if (in.pc >= hi || in.is_end) return false;
        // Sources before the define: the defining compare may itself read a VCC word as scalar data.
        for (const Operand& source : in.src)
            if (may_name_vcc(source)) return false;
        if (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x06 || in.opcode == 0x07))
            return false;   // s_cbranch_vccz / s_cbranch_vccnz
        if (in.fmt == Rdna2Format::VOP2 &&
            (in.opcode == 0x01 || (in.opcode >= 0x28 && in.opcode <= 0x2a)))
            return false;   // e32 cndmask and the carry-in/out forms
        if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
            !(in.dst.kind == OperandKind::SGPR && in.dst.value <= 105))
            return true;   // the define: v_cmp_* into VCC
        if (may_name_vcc(in.dst) || may_name_vcc(in.sdst)) return false;
    }
    return false;
}

// #2952 — the same question as above, answered over the WHOLE region instead of block 0.
//
// A barrier phase often recycles the physical VCC words as scalar scratch a few blocks in, after
// an EXEC-narrowing loop head that defines no VCC at all. Metaphor: ReFantazio's UI-composite
// kernel 0x2281c74100 is the observed case: its second phase opens `v_cmpx_lt_u32` (GFX10's CMPX
// writes EXEC only) plus `s_cbranch_execz`, and only the loop body's division idiom defines VCC --
// `s_sub_i32 vcc_lo, 0, s26` and then `v_readfirstlane_b32 vcc_hi, v5` -- before its first read at
// `s_mul_i32 vcc_lo, vcc_lo, vcc_hi`. Block 0 holds neither write, so the narrow proof above
// declines, and the region was refused with `missing-entry-vcc` although no path can observe the
// entry value.
//
// The proof here is a forward MUST dataflow (`must_fact_at`) per VCC word: "this word has been
// written since region entry on every path to here". The entry value is dead when every
// instruction that may read either word runs where BOTH words hold that fact. Deliberately
// conservative in the same directions as the block-0 proof:
//   * reads are over-approximated exactly as above (every source slot, the implicit VCC consumers,
//     the implicit destination reads, and any scalar operand rooted at s99 or higher), and any read
//     demands the complete pair, even a b32 read of one half;
//   * a write defines a word only when it is unconditional: a `v_cmp_*` into VCC (both words), or
//     an explicit scalar write of that word through the shared writer inventory, excluding the
//     conditional moves -- s_cmov_b32/b64 and s_cmovk_i32 keep the old value on one SCC outcome;
//   * any control transfer the branch list cannot describe fails closed (#3719): s_setpc/swappc,
//     s_call and the subvector-loop SOPKs.
// CONFIDENCE: HIGH for soundness (a missed write only rejects; an over-reported read only rejects).
inline bool region_defines_vcc_before_any_read(const std::vector<Rdna2Inst>& ins) {
    auto may_name_vcc = [](const Operand& operand) {
        if (operand.kind == OperandKind::Special)
            return operand.value == 106 || operand.value == 107;
        if (operand.kind == OperandKind::SGPR) return operand.value >= 99;
        return false;
    };
    for (const auto& in : ins) {
        if (in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x20u && in.opcode <= 0x22u) return false;
        if (in.fmt == Rdna2Format::SOPK &&
            (in.opcode == kSopkOpcodeCallB64 || in.opcode == kSopkOpcodeSubvectorLoopBegin ||
             in.opcode == kSopkOpcodeSubvectorLoopEnd))
            return false;
    }
    auto defines_word = [](const Rdna2Inst& in, int word) {
        if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
            !(in.dst.kind == OperandKind::SGPR && in.dst.value <= 105))
            return true;   // v_cmp_* into VCC: the whole pair, every lane
        if ((in.fmt == Rdna2Format::SOP1 &&
             (in.opcode == kSop1OpcodeCmovB32 || in.opcode == kSop1OpcodeCmovB64)) ||
            (in.fmt == Rdna2Format::SOPK && in.opcode == kSopkOpcodeCmovkI32))
            return false;   // conditional: may keep the entry value
        bool defines = false;
        for_each_scalar_write(in, [&](int base, uint32_t width) {
            defines |= word >= base && word < base + static_cast<int>(width);
        });
        return defines;
    };
    auto never = [](const Rdna2Inst&) { return false; };
    const std::vector<uint8_t> lo_defined =
        must_fact_at(ins, [&](const Rdna2Inst& in) { return defines_word(in, 106); }, never, false);
    const std::vector<uint8_t> hi_defined =
        must_fact_at(ins, [&](const Rdna2Inst& in) { return defines_word(in, 107); }, never, false);
    if (lo_defined.size() != ins.size() || hi_defined.size() != ins.size()) return false;
    for (size_t i = 0; i < ins.size(); ++i) {
        const Rdna2Inst& in = ins[i];
        if (in.is_end) continue;
        bool reads = false;
        for (const Operand& source : in.src) reads |= may_name_vcc(source);
        reads |= in.fmt == Rdna2Format::SOPP && (in.opcode == 0x06 || in.opcode == 0x07);
        reads |= in.fmt == Rdna2Format::VOP2 &&
                 (in.opcode == 0x01 || (in.opcode >= 0x28 && in.opcode <= 0x2a));
        // Keep this implicit-read inventory in step with entry_block_defines_vcc_before_any_read:
        // when V_DIV_FMAS (which reads VCC implicitly) is lowered, it belongs in BOTH lists.
        reads |= scalar_implicit_destination_read_width(in) != 0 && may_name_vcc(in.dst);
        // A conditional move into VCC keeps the old value on one outcome: that is a read.
        reads |= ((in.fmt == Rdna2Format::SOP1 &&
                   (in.opcode == kSop1OpcodeCmovB32 || in.opcode == kSop1OpcodeCmovB64)) ||
                  (in.fmt == Rdna2Format::SOPK && in.opcode == kSopkOpcodeCmovkI32)) &&
                 may_name_vcc(in.dst);
        if (reads && !(lo_defined[i] && hi_defined[i])) return false;
    }
    return true;
}

}   // namespace prosper::gpu
