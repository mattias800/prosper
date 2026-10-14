// See rdna2_indirect_scalar_writes.hpp. Moved unchanged out of rdna2_indirect_pointer_analysis.cpp.
#include "gpu/recompiler/indirect/rdna2_indirect_scalar_writes.hpp"

#include <algorithm>

namespace prosper::gpu {

uint32_t conservative_scalar_write_width(const Rdna2Inst& instruction) {
    if (instruction.dst.kind != OperandKind::SGPR &&
        instruction.dst.kind != OperandKind::Special)
        return 0u;
    if (instruction.fmt == Rdna2Format::SMEM) {
        switch (instruction.opcode) {
            case kSmemOpcodeLoadDword: case kSmemOpcodeBufferLoadDword: return 1u;
            case kSmemOpcodeLoadDwordX2: case kSmemOpcodeBufferLoadDwordX2: return 2u;
            case kSmemOpcodeLoadDwordX4: case kSmemOpcodeBufferLoadDwordX4: return 4u;
            case kSmemOpcodeLoadDwordX8: case kSmemOpcodeBufferLoadDwordX8: return 8u;
            case kSmemOpcodeLoadDwordX16: case kSmemOpcodeBufferLoadDwordX16: return 16u;
            default: return 1u;
        }
    }
    // The analyzer deliberately admits unrelated same-length scalar packets. Count every SOP1/SOP2
    // destination as a pair rather than maintaining a second, inevitably partial B64 opcode table:
    // under-counting an unbound B64 mutation could hide a write that starts one SGPR before a saved
    // mask. The false-positive edge is harmless here; rejecting an otherwise unrelated mutation is
    // preferable to granting relocation authority after a partially hidden mask clobber.
    if (instruction.fmt == Rdna2Format::SOP1 || instruction.fmt == Rdna2Format::SOP2)
        return 2u;
    if (instruction.fmt == Rdna2Format::VOPC) return 2u;
    return 1u;
}

bool scalar_range_unchanged(const std::vector<Rdna2Inst>& instructions,
                            uint32_t begin_pc, uint32_t end_pc,
                            uint32_t first_sgpr, uint32_t count) {
    const uint64_t wanted_end = static_cast<uint64_t>(first_sgpr) + count;
    return std::none_of(instructions.begin(), instructions.end(), [&](const Rdna2Inst& in) {
        if (in.pc <= begin_pc || in.pc >= end_pc) return false;
        const uint32_t width = conservative_scalar_write_width(in);
        if (width && in.dst.value >= 0) {
            const uint32_t first = static_cast<uint32_t>(in.dst.value);
            if (first < wanted_end && static_cast<uint64_t>(first) + width > first_sgpr)
                return true;
        }
        if ((in.sdst.kind == OperandKind::SGPR || in.sdst.kind == OperandKind::Special) &&
            in.sdst.value >= 0) {
            const uint32_t first = static_cast<uint32_t>(in.sdst.value);
            return first < wanted_end && static_cast<uint64_t>(first) + 1u > first_sgpr;
        }
        return false;
    });
}

}   // namespace prosper::gpu
