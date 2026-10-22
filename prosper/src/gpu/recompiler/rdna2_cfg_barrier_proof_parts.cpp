// Existing terminal-SCC recursive slice and scalar-prefix check.
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_cfg_barrier_proof_parts.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace prosper::gpu {
bool cfg_terminal_uniform_operand(std::function<bool(int, uint32_t, uint32_t)>& uniform_reg_at,
                                  const Rdna2Inst& writer, uint32_t& depth, const Operand& operand,
                                  uint32_t& words) {
    switch (operand.kind) {
        case OperandKind::InlineInt:
        case OperandKind::InlineFloat:
        case OperandKind::Literal: return true;
        case OperandKind::Special:
            if (operand.value == 125) return true;
            if (operand.value == 126 || operand.value == 127 || operand.value == 253) return false;
            [[fallthrough]];
        case OperandKind::SGPR:
            for (uint32_t word = 0; word < words; ++word)
                if (!uniform_reg_at(operand.value + static_cast<int>(word), writer.pc, depth + 1))
                    return false;
            return true;
        default: return false;
    }
}

bool cfg_terminal_uniform_register(const std::vector<Rdna2Inst>& ins,
                                   std::function<bool(int, uint32_t, uint32_t)>& uniform_reg_at,
                                   int& reg, uint32_t& use_pc, uint32_t& depth) {
    if (depth > 64) return false;
    for (size_t i = ins.size(); i-- > 0;) {
        const Rdna2Inst& writer = ins[i];
        if (writer.pc >= use_pc) continue;
        bool writes = false;
        for_each_scalar_write(writer, [&](int base, uint32_t width) {
            writes |= reg >= base && reg < base + static_cast<int>(width);
        });
        // Implicit VOPC writes target architectural VCC even when the decoder exposes it as a
        // Special operand rather than an ordinary SGPR destination.
        if ((reg == 106 || reg == 107) && writer.fmt == Rdna2Format::VOPC &&
            !vopc_is_cmpx(writer.opcode))
            writes = true;
        if (!writes) continue;

        auto uniform_operand = [&](const Operand& operand, uint32_t words) {
            return cfg_terminal_uniform_operand(uniform_reg_at, writer, depth, operand, words);
        };

        switch (writer.fmt) {
            case Rdna2Format::SOP1: {
                // GETPC is identical for every wave. Mask/saveexec operations consume EXEC and
                // are rejected by rdna2_instruction_may_change_exec; ordinary scalar operations trace
                // their source at the destination's architectural width.
                if (writer.opcode == 0x1f) return true;
                if (rdna2_instruction_may_change_exec(writer) || writer.n_src != 1) return false;
                return uniform_operand(writer.src[0], scalar_write_width(writer) == 2 ? 2u : 1u);
            }
            case Rdna2Format::SOP2: {
                // Carry/cselect forms consume SCC in addition to their decoded operands. They are
                // outside this narrow slice until SCC itself is represented as a recursive value.
                if (writer.opcode == 0x04 || writer.opcode == 0x05 || writer.opcode == 0x0a ||
                    writer.opcode == 0x0b)
                    return false;
                const uint32_t words = scalar_write_width(writer) == 2 ? 2u : 1u;
                return writer.n_src == 2 && uniform_operand(writer.src[0], words) &&
                       uniform_operand(writer.src[1], words);
            }
            case Rdna2Format::SMEM: {
                // Scalar-buffer loads use a four-dword descriptor; scalar-memory loads use a
                // base pair. The scalar offset is one dword in both forms. Only a load's
                // result is a function of its address operands.
                if (!rdna2_smem_load_dwords(writer.opcode) || writer.n_src < 1) return false;
                const uint32_t base_words = rdna2_smem_address_dwords(writer.opcode);
                if (!uniform_operand(writer.src[0], base_words)) return false;
                return writer.n_src < 2 || uniform_operand(writer.src[1], 1);
            }
            case Rdna2Format::SOPK:
                // S_MOVK is a literal definition. Other SOPK data writers are read/modify/write
                // or hardware-register reads and stay outside this proof.
                return writer.opcode == 0x00;
            default: return false;
        }
    }
    // Ordinary compute launch SGPRs (user data and enabled system SGPRs) are uniform within one
    // workgroup. Special registers are numbered above this range and were handled explicitly.
    return reg >= 0 && reg <= 105;
}

void cfg_barrier_scalar_prefix(bool& scalar_prefix, BarrierPhasedCompute& result,
                               const std::vector<Rdna2Inst>& ins) {
    for (size_t i = 0; scalar_prefix && i < result.guard_index; ++i) {
        const Rdna2Inst& in = ins[i];
        // Scalar ALU/loads retain workgroup-uniform values when entered from compute user data and
        // TGIDs. EXEC is lane state, so any explicit access to either half invalidates this proof.
        // VCC physical words may be scalar scratch provided no vector instruction enters the prefix.
        const bool scalar = in.fmt == Rdna2Format::SOP1 || in.fmt == Rdna2Format::SOP2 ||
                            in.fmt == Rdna2Format::SOPK || in.fmt == Rdna2Format::SOPC ||
                            in.fmt == Rdna2Format::SMEM ||
                            (in.fmt == Rdna2Format::SOPP && sopp_is_noop(in));
        if (!scalar) {
            scalar_prefix = false;
            break;
        }
        auto is_exec = [](const Operand& operand) {
            return (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special) &&
                   (operand.value == 126 || operand.value == 127);
        };
        if (is_exec(in.dst)) scalar_prefix = false;
        for (uint32_t source = 0; source < in.n_src; ++source)
            if (is_exec(in.src[source])) scalar_prefix = false;
    }
}

}   // namespace prosper::gpu
