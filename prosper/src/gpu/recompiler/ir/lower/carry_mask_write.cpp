// carry_mask_write.cpp — the lane-mask half of every carry op (ADR 0012).
//
// A carry op's D and carry Bool are opcode-specific and come from ir/ops/vector_carry.cpp. What the
// instruction then does with the carry is the same for all of them: mask it with EXEC and write it to
// VCC (VOP2) or to the VOP3B SDST, with that destination's mask-domain bookkeeping. The VOP3B write
// was emit_alu's write_vop3b_carry_output lambda, moved verbatim; the VOP2 write was the assignment
// at the end of lower/vop2.cpp's carry case, and the v_mad_u64_u32 / v_mad_i64_i32 write was
// emit_alu's. All three write VCC_LO's Wave32 B32 lifetime together with `rs.vcc` (#5348).
#include "gpu/recompiler/ir/lower/carry_mask_write.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu::ir {

uint32_t exec_masked_carry(SpirvCompute& b, RegState& rs, uint32_t carry) {
    return rs.exec_narrowed ? b.land(rs.exec, carry) : carry;
}

namespace {

// A Wave32 B32-mask stage: one physical SGPR word holds a whole lane mask.
bool wave32_b32_masks(const SpirvCompute& b) {
    return b.allow_b32_masks && (b.is_fragment || (b.is_compute && b.wave_size == 32));
}

// A Wave32 carry destination is one physical word. Preserve an independently-live adjacent SGPR
// and register the new lifetime in the B32 mask domain used by CFG joins. VCC_LO's lifetime is
// also the architectural VCC, so the two are written together: the e64 readers that name vcc_lo
// (v_cndmask_b32_e64, a VOP3B carry-in) prefer the lifetime, the e32 forms read `rs.vcc` (#5348).
void write_b32_carry_word(RegState& rs, int reg, uint32_t carry_mask) {
    rs.sreg_bool[reg] = carry_mask;
    rs.sreg_bool_narrowed[reg] = true;
    rs.sreg_bool_b32.insert(reg);
    rs.sreg.erase(reg);
    rs.sreg_srt.erase(reg);
    if (reg == 106) rs.vcc = carry_mask;
}

}   // namespace

void write_vop2_carry_mask(SpirvCompute& b, RegState& rs, uint32_t carry) {
    // Carry-out masks follow ISA 3.9 like compare results: an EXEC-inactive lane's
    // VCC bit is written 0, not its raw carry (wave votes must not see phantom bits).
    const uint32_t carry_mask = exec_masked_carry(b, rs, carry);
    // In Wave32 the implicit VCC is VCC_LO, the same destination as the VOP3B form's
    // `sdst = vcc_lo`, so it gets the same word write. Setting `rs.vcc` alone left the VCC_LO B32
    // lifetime holding the previous mask (#5348). Wave64 has no B32 lifetime and is unchanged.
    if (wave32_b32_masks(b))
        write_b32_carry_word(rs, 106, carry_mask);
    else
        rs.vcc = carry_mask;
}

void write_vop3b_carry_mask(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                            uint32_t carry) {
    // carry-out -> sdst mask (VCC or a saved SGPR-pair bool). ISA 3.9: an
    // EXEC-inactive lane's mask bit is written 0, never its raw carry.
    const uint32_t carry_mask = exec_masked_carry(b, rs, carry);
    if (in.sdst.kind == OperandKind::SGPR && wave32_b32_masks(b)) {
        write_b32_carry_word(rs, in.sdst.value, carry_mask);
    } else if (in.sdst.value == 106 || in.sdst.value == 107) {
        rs.vcc = carry_mask;
    } else if (in.sdst.kind == OperandKind::SGPR) {
        rs.sreg_bool[in.sdst.value] = carry_mask;
    } else {
        ok = false;
    }
}

void write_mad64_carry_mask(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                            uint32_t carry) {
    // ISA 3.9 carry-mask rule: an EXEC-inactive lane's bit is written 0, never the raw carry
    // (wave votes/spills must not see phantom bits from inactive lanes).
    const uint32_t carry_masked = exec_masked_carry(b, rs, carry);
    // `sdst = vcc_lo` in Wave32 is the same word write as the carry ops' (#5348).
    if (in.sdst.value == 106 && in.sdst.kind == OperandKind::SGPR && wave32_b32_masks(b))
        write_b32_carry_word(rs, 106, carry_masked);
    else if (in.sdst.value == 106 || in.sdst.value == 107) rs.vcc = carry_masked;
    else if (in.sdst.kind == OperandKind::SGPR) rs.sreg_bool[in.sdst.value] = carry_masked;
    else ok = false;
}

}   // namespace prosper::gpu::ir
