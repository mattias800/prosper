// carry_mask_write.cpp — the lane-mask half of every carry op (ADR 0012).
//
// A carry op's D and carry Bool are opcode-specific and come from ir/ops/vector_carry.cpp. What the
// instruction then does with the carry is the same for all of them: mask it with EXEC and write it to
// VCC (VOP2) or to the VOP3B SDST, with that destination's mask-domain bookkeeping. The VOP3B write
// was emit_alu's write_vop3b_carry_output lambda, moved verbatim; the VOP2 write was the assignment
// at the end of lower/vop2.cpp's carry case.
#include "gpu/recompiler/ir/lower/carry_mask_write.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu::ir {

uint32_t exec_masked_carry(SpirvCompute& b, RegState& rs, uint32_t carry) {
    return rs.exec_narrowed ? b.land(rs.exec, carry) : carry;
}

void write_vop2_carry_mask(SpirvCompute& b, RegState& rs, uint32_t carry) {
    // Carry-out masks follow ISA 3.9 like compare results: an EXEC-inactive lane's
    // VCC bit is written 0, not its raw carry (wave votes must not see phantom bits).
    rs.vcc = exec_masked_carry(b, rs, carry);
}

void write_vop3b_carry_mask(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                            uint32_t carry) {
    // carry-out -> sdst mask (VCC or a saved SGPR-pair bool). ISA 3.9: an
    // EXEC-inactive lane's mask bit is written 0, never its raw carry.
    const uint32_t carry_mask = exec_masked_carry(b, rs, carry);
    if (in.sdst.kind == OperandKind::SGPR && b.allow_b32_masks &&
        (b.is_fragment || (b.is_compute && b.wave_size == 32))) {
        // A Wave32 carry destination is one physical word. Preserve an independently-live
        // adjacent SGPR and register the new lifetime in the B32 mask domain used by CFG joins.
        rs.sreg_bool[in.sdst.value] = carry_mask;
        rs.sreg_bool_narrowed[in.sdst.value] = true;
        rs.sreg_bool_b32.insert(in.sdst.value);
        rs.sreg.erase(in.sdst.value);
        rs.sreg_srt.erase(in.sdst.value);
        if (in.sdst.value == 106) rs.vcc = carry_mask;
    } else if (in.sdst.value == 106 || in.sdst.value == 107) {
        rs.vcc = carry_mask;
    } else if (in.sdst.kind == OperandKind::SGPR) {
        rs.sreg_bool[in.sdst.value] = carry_mask;
    } else {
        ok = false;
    }
}

}   // namespace prosper::gpu::ir
