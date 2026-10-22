// Existing operand, definition and condition proofs behind the original closures.
#include "gpu/recompiler/rdna2_cfg_uniform_branch_parts.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <vector>

namespace prosper::gpu {

bool cfg_uniform_writes_reg(const Rdna2Inst& in, int& reg, uint32_t& words) {
    bool hit = false;
    for_each_scalar_write(in, [&](int base, uint32_t width) {
        if (reg < base + static_cast<int>(width) && base < reg + static_cast<int>(words))
            hit = true;
    });
    // A VOPC without an explicit scalar destination writes architectural VCC implicitly; the
    // writer inventory only reports the explicit SGPR form.
    if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
        in.dst.kind == OperandKind::Special && in.dst.value == 106 && reg < 108 &&
        106 < reg + static_cast<int>(words))
        hit = true;
    return hit;
}

bool cfg_uniform_entry_value(std::map<int, bool>& entry_value_cache,
                             const std::vector<Rdna2Inst>& ins,
                             const std::function<bool(const Rdna2Inst&, int, uint32_t)>& writes_reg,
                             size_t& branch_index, int& reg) {
    auto cached = entry_value_cache.find(reg);
    if (cached != entry_value_cache.end()) return cached->second;
    const std::vector<uint8_t> fact = must_fact_at(
        ins, [](const Rdna2Inst&) { return false; },
        [&](const Rdna2Inst& in) { return writes_reg(in, reg, 1); }, true);
    const bool proved = !fact.empty() && fact[branch_index] != 0;
    entry_value_cache[reg] = proved;
    return proved;
}

bool cfg_uniform_operand(std::function<bool(int, size_t, uint32_t)>& uniform_scalar,
                         const Operand& op, uint32_t& words, size_t& before, uint32_t& depth) {
    switch (op.kind) {
        case OperandKind::InlineInt:
        case OperandKind::InlineFloat:
        case OperandKind::Literal: return true;
        case OperandKind::Special:
            if (op.value == 125) return true;   // SGPR_NULL
            if (op.value == 126 || op.value == 127 || op.value == 253)
                return false;   // EXEC and SCC are not data
            [[fallthrough]];
        case OperandKind::SGPR:
            for (uint32_t word = 0; word < words; ++word)
                if (!uniform_scalar(op.value + static_cast<int>(word), before, depth + 1))
                    return false;
            return true;
        default: return false;   // VGPR and everything else
    }
}

    // A scalar load outside the branch's block can still be traced through a straight-line
    // entry prefix. Require a sole load definition and no edge that can re-enter that prefix:
    // one static writer does not mean one execution, and a later backedge may bring different
    // descriptor values or bypass an earlier definition. Indirect SOP1 edges fail closed; the
    // remaining SOPK control-flow inventory gap is tracked in #3719.
    //
    // Resolve descriptor/address inputs at each load, not globally. This admits constant-buffer
    // indirection and later scratch reuse (Sonic #2790), without confusing that later overwrite
    // with the value actually read. Every contributing scalar operand must be uniform, including
    // both words of a B64 source even when the instruction writes only one word.
bool cfg_uniform_entry_scalar(
    const std::vector<Rdna2Inst>& ins,
    const std::function<bool(const Rdna2Inst&, int, uint32_t)>& writes_reg,
    std::function<bool(int, size_t, uint32_t)>& entry_uniform_scalar, int& reg, size_t& before,
    uint32_t& depth) {
    if (depth > 16 || reg < 0 || reg > 105) return false;
    size_t w = ins.size();
    for (size_t i = before; i-- > 0;) {
        if (writes_reg(ins[i], reg, 1)) {
            w = i;
            break;
        }
    }
    if (w == ins.size()) return true;   // No writer in [0, before); holds wave-entry launch data.
    const Rdna2Inst& in = ins[w];
    switch (in.fmt) {
        case Rdna2Format::SMEM: {
            // Only a load's result is a function of its address operands.
            if (!rdna2_smem_load_dwords(in.opcode) || in.n_src < 1) return false;
            const uint32_t base_words = rdna2_smem_address_dwords(in.opcode);
            auto op_uniform = [&](const Operand& op, uint32_t words) {
                return cfg_uniform_entry_scalar_operand(entry_uniform_scalar, w, depth, op, words);
            };
            if (!op_uniform(in.src[0], base_words)) return false;
            return in.n_src < 2 || op_uniform(in.src[1], 1);
        }
        case Rdna2Format::SOP1: {
            if (in.opcode == 0x1f) return true;   // s_getpc_b64
            // Bitset and conditional-move forms read the old destination (and cmov also
            // reads SCC). The explicit operand alone cannot prove their result uniform.
            if (rdna2_instruction_may_change_exec(in) || in.n_src != 1 ||
                scalar_implicit_destination_read_width(in))
                return false;
            // Which register a relative move reads depends on M0, which no operand names.
            if (in.opcode == kSop1OpcodeMovrelsB32 || in.opcode == kSop1OpcodeMovrelsB64)
                return false;
            const Operand& op = in.src[0];
            const uint32_t words = scalar_alu_source_words(in, 0);
            if (op.kind == OperandKind::Special && op.value == 125) return true;
            if (op.kind == OperandKind::InlineInt || op.kind == OperandKind::Literal) return true;
            if (op.kind != OperandKind::SGPR && op.kind != OperandKind::Special) return false;
            for (uint32_t word = 0; word < words; ++word) {
                const int r = op.value + static_cast<int>(word);
                if (!entry_uniform_scalar(r, w, depth + 1)) return false;
            }
            return true;
        }
        case Rdna2Format::SOP2: {
            if (in.opcode == 0x04 || in.opcode == 0x05 || in.opcode == 0x0a || in.opcode == 0x0b)
                return false;
            if (in.n_src != 2) return false;
            const uint32_t words = scalar_write_width(in) == 2 ? 2u : 1u;
            auto op_uniform = [&](const Operand& op) {
                return cfg_uniform_entry_scalar_address(words, entry_uniform_scalar, w, depth, op);
            };
            return op_uniform(in.src[0]) && op_uniform(in.src[1]);
        }
        case Rdna2Format::SOPK: return in.opcode == 0x00;   // s_movk_i32 is a literal
        default: return false;
    }
}

bool cfg_uniform_entry_scalar_operand(
    std::function<bool(int, size_t, uint32_t)>& entry_uniform_scalar, size_t& w, uint32_t& depth,
    const Operand& op, uint32_t& words) {
    if (op.kind == OperandKind::Special && op.value == 125) return true;   // SGPR_NULL
    if (op.kind == OperandKind::InlineInt || op.kind == OperandKind::Literal) return true;
    if (op.kind != OperandKind::SGPR && op.kind != OperandKind::Special) return false;
    for (uint32_t word = 0; word < words; ++word) {
        const int r = op.value + static_cast<int>(word);
        if (!entry_uniform_scalar(r, w, depth + 1)) return false;
    }
    return true;
}

bool cfg_uniform_entry_scalar_address(
    const uint32_t& words, std::function<bool(int, size_t, uint32_t)>& entry_uniform_scalar,
    size_t& w, uint32_t& depth, const Operand& op) {
    if (op.kind == OperandKind::Special && op.value == 125) return true;
    if (op.kind == OperandKind::InlineInt || op.kind == OperandKind::Literal) return true;
    if (op.kind != OperandKind::SGPR && op.kind != OperandKind::Special) return false;
    for (uint32_t word = 0; word < words; ++word) {
        const int r = op.value + static_cast<int>(word);
        if (!entry_uniform_scalar(r, w, depth + 1)) return false;
    }
    return true;
}

bool cfg_uniform_sole_launch(const std::vector<Rdna2Inst>& ins,
                             const std::function<bool(const Rdna2Inst&, int, uint32_t)>& writes_reg,
                             std::set<uint32_t>& targets,
                             std::function<bool(int, size_t, uint32_t)>& entry_uniform_scalar,
                             int& reg) {
    if (reg < 0 || reg > 105) return false;   // EXEC/VCC/SCC are not launch data
    size_t only = ins.size();
    for (size_t i = 0; i < ins.size(); ++i) {
        if (!writes_reg(ins[i], reg, 1)) continue;
        if (only != ins.size()) return false;   // more than one definition: not this shape
        only = i;
    }
    if (only == ins.size()) return false;
    // The textual backward slice is valid only if no other edge can enter through the load.
    // This includes an edge to pc0: the second visit need not carry launch-time register data.
    if (!targets.empty() && *targets.begin() <= ins[only].pc) return false;
    for (const Rdna2Inst& in : ins)
        if (in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x20u && in.opcode <= 0x22u)
            return false;   // indirect PC change could enter anywhere in the prefix
    for (size_t i = 0; i < only; ++i) {
        const Rdna2Inst& before_def = ins[i];
        if (before_def.is_end) return false;
        if (before_def.fmt == Rdna2Format::SOPP && !sopp_is_noop(before_def)) return false;
        if (before_def.fmt == Rdna2Format::SOP1 && before_def.opcode >= 0x20u &&
            before_def.opcode <= 0x22u)
            return false;   // indirect PC change
    }
    const Rdna2Inst& def = ins[only];
    if (def.fmt != Rdna2Format::SMEM || !scalar_write_width(def) || def.n_src < 1) return false;
    // Scalar-buffer loads carry a four-dword descriptor; scalar-memory loads a base pair.
    const uint32_t base_words = def.opcode >= 0x8 ? 4u : 2u;
    auto address_operand_is_launch = [&](const Operand& op, uint32_t count) {
        return cfg_uniform_sole_launch_operand(entry_uniform_scalar, only, op, count);
    };
    if (!address_operand_is_launch(def.src[0], base_words)) return false;
    return def.n_src < 2 || address_operand_is_launch(def.src[1], 1);
}

bool cfg_uniform_sole_launch_operand(
    std::function<bool(int, size_t, uint32_t)>& entry_uniform_scalar, size_t& only,
    const Operand& op, uint32_t& count) {
    if (op.kind == OperandKind::Special && op.value == 125) return true;   // SGPR_NULL
    if (op.kind == OperandKind::InlineInt || op.kind == OperandKind::Literal) return true;
    if (op.kind != OperandKind::SGPR && op.kind != OperandKind::Special) return false;
    for (uint32_t word = 0; word < count; ++word) {
        const int r = op.value + static_cast<int>(word);
        if (!entry_uniform_scalar(r, only, 0)) return false;
    }
    return true;
}

bool cfg_uniform_scalar(
    const std::function<size_t(int, uint32_t, size_t)>& last_writer,
    const std::vector<Rdna2Inst>& ins, const std::function<bool(int)>& still_entry_value,
    const std::function<bool(int)>& sole_launch_derived_definition,
    const std::function<bool(const Operand&, uint32_t, size_t, uint32_t)>& uniform_operand,
    int& reg, size_t& before, uint32_t& depth) {
    if (depth > 64) return false;
    const size_t w = last_writer(reg, 1, before);
    if (w == ins.size())
        return (reg >= 0 && reg <= 105 && still_entry_value(reg)) ||
               sole_launch_derived_definition(reg);
    const Rdna2Inst& in = ins[w];
    switch (in.fmt) {
        case Rdna2Format::SOP1:
            if (in.opcode == 0x1f) return true;   // s_getpc_b64
            // Bitset and conditional-move forms read the old destination (and cmov also
            // reads SCC). The explicit operand alone cannot prove their result uniform.
            if (rdna2_instruction_may_change_exec(in) || in.n_src != 1 ||
                scalar_implicit_destination_read_width(in))
                return false;
            // Which register a relative move reads depends on M0, which no operand names.
            if (in.opcode == kSop1OpcodeMovrelsB32 || in.opcode == kSop1OpcodeMovrelsB64)
                return false;
            return uniform_operand(in.src[0], scalar_alu_source_words(in, 0), w, depth);
        case Rdna2Format::SOP2: {
            // Carry and cselect forms consume SCC as well as their decoded operands; the mask
            // slice models the cselect shape explicitly and nothing else may use them as data.
            if (in.opcode == 0x04 || in.opcode == 0x05 || in.opcode == 0x0a || in.opcode == 0x0b)
                return false;
            const uint32_t words = scalar_write_width(in) == 2 ? 2u : 1u;
            return in.n_src == 2 && uniform_operand(in.src[0], words, w, depth) &&
                   uniform_operand(in.src[1], words, w, depth);
        }
        case Rdna2Format::SMEM: {
            // Scalar-buffer loads use a four-dword descriptor; scalar-memory loads a base pair.
            // Only a load's result is a function of its address operands.
            if (!rdna2_smem_load_dwords(in.opcode) || in.n_src < 1) return false;
            const uint32_t base_words = rdna2_smem_address_dwords(in.opcode);
            if (!uniform_operand(in.src[0], base_words, w, depth)) return false;
            return in.n_src < 2 || uniform_operand(in.src[1], 1, w, depth);
        }
        case Rdna2Format::SOPK: return in.opcode == 0x00;   // s_movk_i32 is a literal
        default: return false;   // VOPC masks, v_readfirstlane, v_readlane, everything else
    }
}

bool cfg_uniform_scc(
    size_t& block_start, const std::vector<Rdna2Inst>& ins,
    const std::function<bool(const Operand&, uint32_t, size_t, uint32_t)>& uniform_operand,
    size_t& before, uint32_t& depth) {
    if (depth > 64) return false;
    for (size_t i = before; i-- > block_start;) {
        const Rdna2Inst& in = ins[i];
        if (sopp_is_noop(in)) continue;
        if (in.fmt == Rdna2Format::SOPC) {
            if (in.n_src == 0) return false;
            const uint32_t words = in.opcode == 0x12 || in.opcode == 0x13 ? 2u : 1u;
            for (uint32_t s = 0; s < in.n_src; ++s)
                if (!uniform_operand(in.src[s], words, i, depth)) return false;
            return true;
        }
        // A SOP1 that provably leaves SCC alone is stepped OVER rather than stopping the walk:
        // the SCC the branch consumes is still the one the compare below it produced. This is
        // not a relaxation of the proof, it is the proof reaching its subject -- the guest puts
        // `s_mov_b64 exec, -1` between the compare and the branch (#3713, `0x2005713000` at
        // pc 199/200/201), so stopping at every SOP1 made the compare unreachable and the whole
        // uber-shader family unprovable. `sop1_opcode_leaves_scc_unmodified` is derived for THIS
        // question and is deliberately NOT the SCC-liveness list in rdna2_emit_cfg.cpp -- that
        // one asks whether SCC is still scalar-valued, which an SCC WRITER can satisfy. See its
        // definition for the `s_bcnt1_i32_b64` case that separates them.
        //
        // The SOPK waitcnt family (s_waitcnt_vscnt/vmcnt/expcnt/lgkmcnt, 0x17..0x1a) only
        // waits on counters and never writes SCC, so it is stepped over like the SOP1s above.
        //
        // Everything else in SOP1/SOP2/SOPK may write SCC and still stops the walk: consuming an
        // older compare through an unmodeled writer would prove uniformity of a condition the
        // branch never saw. Vector, memory, export and SOPP instructions do not write SCC
        // (VOPC writes VCC/EXEC, not SCC) and fall through, as before.
        if (in.fmt == Rdna2Format::SOP1 && sop1_opcode_leaves_scc_unmodified(in.opcode)) continue;
        if (in.fmt == Rdna2Format::SOPK && in.opcode >= kSopkOpcodeWaitcntVscnt &&
            in.opcode <= kSopkOpcodeWaitcntLgkmcnt)
            continue;
        if (in.fmt == Rdna2Format::SOP1 || in.fmt == Rdna2Format::SOP2 ||
            in.fmt == Rdna2Format::SOPK)
            return false;
    }
    return false;   // SCC entering the region from outside is not proved
}

bool cfg_uniform_mask(const std::function<size_t(int, uint32_t, size_t)>& last_writer,
                      const std::vector<Rdna2Inst>& ins,
                      std::function<bool(size_t, uint32_t)>& uniform_scc,
                      std::function<bool(int, size_t, uint32_t)>& uniform_mask, int& reg,
                      size_t& before, uint32_t& depth) {
    if (depth > 64) return false;
    const size_t w = last_writer(reg, 2, before);
    if (w == ins.size()) return false;   // a mask entering from outside is not proved
    const Rdna2Inst& in = ins[w];
    if (in.fmt != Rdna2Format::SOP2 || in.dst.value != reg || scalar_write_width(in) != 2 ||
        in.n_src != 2)
        return false;
    auto is_exec = [](const Operand& op) {
        return (op.kind == OperandKind::SGPR || op.kind == OperandKind::Special) && op.value == 126;
    };
    auto is_zero = [](const Operand& op) {
        return op.kind == OperandKind::InlineInt && op.value == 0;
    };
    if (in.opcode == 0x0b) {   // s_cselect_b64 dst, EXEC, 0  (either polarity)
        const bool shape = (is_exec(in.src[0]) && is_zero(in.src[1])) ||
                           (is_zero(in.src[0]) && is_exec(in.src[1]));
        return shape && uniform_scc(w, depth + 1);
    }
    if (in.opcode == 0x0f || in.opcode == 0x11) {   // s_and_b64 / s_or_b64
        for (uint32_t s = 0; s < 2; ++s) {
            const Operand& op = in.src[s];
            if (op.kind != OperandKind::SGPR && op.kind != OperandKind::Special) return false;
            if (op.value == 126 || op.value == 127) return false;   // raw EXEC is per-wave
            if (!uniform_mask(op.value, w, depth + 1)) return false;
        }
        return true;
    }
    return false;
}

void cfg_uniform_block_start(size_t& block_start, std::set<uint32_t>& targets,
                             const std::vector<Rdna2Inst>& ins,
                             const std::function<bool(const Rdna2Inst&)>& is_branch) {
    while (block_start > 0) {
        if (targets.count(ins[block_start].pc)) break;   // another edge enters here
        const Rdna2Inst& prev = ins[block_start - 1];
        if (prev.is_end || is_branch(prev)) break;
        --block_start;
    }
}

bool cfg_uniform_exec_full(UniformBranchCondition& condition, const std::vector<Rdna2Inst>& ins,
                           size_t& branch_index) {

    // Obligation (1): EXEC is full at the branch along every path. Required for VCC branches
    // because VCC masks are constructed via `s_cselect_b64 vcc, exec, 0`, so an empty or narrowed
    // EXEC causes VCCZ to differ per-wave. SCC branches consume the scalar condition code directly,
    // which is independent of EXEC.
    if (condition == UniformBranchCondition::Vcc) {
        const std::vector<uint8_t> exec_full = must_fact_at(
            ins, exec_write_sets_full_mask,
            [](const Rdna2Inst& in) { return rdna2_instruction_may_change_exec(in); }, true);
        if (exec_full.empty() || !exec_full[branch_index]) return false;
    }
    return true;
}

}   // namespace prosper::gpu
