#pragma once

// Lifted out of rdna2_emit_cfg.cpp's anonymous namespaces so the code that operates on them
// can live in its own translation units. Declared in prosper/src/gpu/recompiler; who may include it is
// a decision this banner does not make -- say so with --note if it is restricted.

// rdna2_emit_cfg.cpp — the divergent-control-flow state machine and emit_body, split out of
// rdna2_to_spirv.cpp. Shared state lives in gpu/recompiler/rdna2_to_spirv_internal.hpp.
#include <atomic>
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/fragment_packet_definedness.hpp"
#include "gpu/recompiler/rdna2_packet_raw_masks.hpp"
#include "gpu/diagnostics/diagnostic_selectors.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_cfg_registers.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_cf9200_contract.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_compute_contracts.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_packed_pointer.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_buffer_shadow.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_pointer_analysis.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <algorithm>
#include <bit>
#include <cstdarg>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/fragment_loop_mask.hpp"

namespace prosper::gpu {

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
