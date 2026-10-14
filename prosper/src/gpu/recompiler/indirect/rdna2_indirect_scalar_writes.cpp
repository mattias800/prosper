// See rdna2_indirect_scalar_writes.hpp. The one copy of these checks; the descriptor-range
// proof used to keep its own, which disagreed about VOPC (#5146).
#include "gpu/recompiler/indirect/rdna2_indirect_scalar_writes.hpp"

#include "gpu/recompiler/rdna2_instruction_effects.hpp"

#include <algorithm>

namespace prosper::gpu {

uint32_t conservative_scalar_write_width(const Rdna2Inst& instruction, IndirectWave wave) {
    // A lane read writes one SGPR through a destination the decoder files as a VGPR operand, so it
    // must be counted before the operand-kind filter below (both old copies returned 0 for it).
    if (rdna2_lane_read_writes_sgpr(instruction)) return 1u;
    if (instruction.dst.kind != OperandKind::SGPR && instruction.dst.kind != OperandKind::Special)
        return 0u;
    if (instruction.fmt == Rdna2Format::SMEM) {
        switch (instruction.opcode) {
            case kSmemOpcodeLoadDword:
            case kSmemOpcodeBufferLoadDword: return 1u;
            case kSmemOpcodeLoadDwordX2:
            case kSmemOpcodeBufferLoadDwordX2: return 2u;
            case kSmemOpcodeLoadDwordX4:
            case kSmemOpcodeBufferLoadDwordX4: return 4u;
            case kSmemOpcodeLoadDwordX8:
            case kSmemOpcodeBufferLoadDwordX8: return 8u;
            case kSmemOpcodeLoadDwordX16:
            case kSmemOpcodeBufferLoadDwordX16: return 16u;
            default: return 1u;
        }
    }
    // The analyzer deliberately admits unrelated same-length scalar packets. Count every SOP1/SOP2
    // destination as a pair rather than maintaining a second, inevitably partial B64 opcode table:
    // under-counting an unbound B64 mutation could hide a write that starts one SGPR before a saved
    // mask. The false-positive edge is harmless here; rejecting an otherwise unrelated mutation is
    // preferable to granting relocation authority after a partially hidden mask clobber.
    if (instruction.fmt == Rdna2Format::SOP1 || instruction.fmt == Rdna2Format::SOP2) return 2u;
    // A compare's lane mask: one SGPR in Wave32, a pair in Wave64. The two proofs used to keep
    // separate copies of this function, one answering 1 and one 2 -- each right only for the wave
    // size its own proof happens to require (#5146).
    if (instruction.fmt == Rdna2Format::VOPC) return wave == IndirectWave::Wave32 ? 1u : 2u;
    // SOPK: s_call_b64 writes its return address to a pair. Every other SOPK form is charged one
    // word, including those whose SDST field is a source (s_cmpk_*), which over-counts harmlessly.
    if (instruction.fmt == Rdna2Format::SOPK) {
        const Rdna2InstructionEffects& effects = rdna2_instruction_effects(instruction);
        return std::max<uint32_t>(1u, effects.known() ? effects.sdst_dwords : 2u);
    }
    return 1u;
}

bool scalar_range_unchanged(const std::vector<Rdna2Inst>& instructions, uint32_t begin_pc,
                            uint32_t end_pc, uint32_t first_sgpr, uint32_t count,
                            IndirectWave wave) {
    const uint64_t wanted_end = static_cast<uint64_t>(first_sgpr) + count;
    return std::none_of(instructions.begin(), instructions.end(), [&](const Rdna2Inst& in) {
        if (in.pc <= begin_pc || in.pc >= end_pc) return false;
        const uint32_t width = conservative_scalar_write_width(in, wave);
        if (width && in.dst.value >= 0) {
            const uint32_t first = static_cast<uint32_t>(in.dst.value);
            const uint64_t end = static_cast<uint64_t>(first) + width;
            if (first < wanted_end && end > first_sgpr) return true;
        }
        if ((in.sdst.kind == OperandKind::SGPR || in.sdst.kind == OperandKind::Special) &&
            in.sdst.value >= 0) {
            // A VOP3B carry-out is a lane mask too.
            const uint32_t first = static_cast<uint32_t>(in.sdst.value);
            const uint64_t end =
                static_cast<uint64_t>(first) + (wave == IndirectWave::Wave32 ? 1u : 2u);
            return first < wanted_end && end > first_sgpr;
        }
        return false;
    });
}

}   // namespace prosper::gpu
