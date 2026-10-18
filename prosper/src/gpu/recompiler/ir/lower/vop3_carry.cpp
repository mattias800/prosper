// vop3_carry.cpp — the VOP3B carry ops: v_add/sub/subrev_co_u32 (carry-out to SDST) and
// v_add/sub/subrev_co_ci_u32 (carry-in from SRC2, carry-out to SDST). The value is the IR
// definition (ir/ops/vector_carry.cpp); the carry-out goes through the shared mask write
// (carry_mask_write.cpp); the carry-in mask read is emit_alu's, unchanged.
#include "gpu/recompiler/ir/lower/vop3_mask_lowering_internal.hpp"
#include "gpu/recompiler/ir/lower/carry_mask_write.hpp"
#include "gpu/recompiler/ir/ops/vector_carry.hpp"

namespace prosper::gpu::ir {

std::optional<bool> Vop3MaskLowering::route_carry_in_out() {
    const VectorCarryDef* def = find_vector_carry(in.fmt, in.opcode);
    if (!def || !def->reads_carry_in) return std::nullopt;
    // Add/sub with carry-in + carry-out (VOP3B): v_add_co_ci_u32 (0x128), v_sub_co_ci (0x129),
    // v_subrev_co_ci (0x12A). carry-in = src2 mask (VCC or an SGPR bool); carry-out -> sdst mask.
    // Carry-in from an UNTRACKED mask must reject (like cndmask), not silently
    // default to 0 — a wrong carry produces 64-bit address math off by one in the low
    // word with no diagnostic.
    const Operand& s2 = in.src[2];
    uint32_t cin_mask = 0;
    if (s2.value == 106 || s2.value == 107) {
        // As v_cndmask_b32_e64 reads SRC2 (vop3_cndmask.cpp): a Wave32 form may name either VCC
        // word as its own one-dword mask, so prefer that typed lifetime. Only VCC_LO may fall back
        // to the architectural predicate. VCC_HI is a distinct word in Wave32: read through
        // rs.vcc it would take VCC_LO's lanes for a register that may hold scalar data. That was
        // unreachable while the fresh-mask proof charged a Wave32 mask the pair and refused on
        // the never-written word above VCC_HI (#5202).
        auto it = rs.sreg_bool.find(s2.value);
        if (rs.sreg_bool_b32.contains(s2.value) && it != rs.sreg_bool.end())
            cin_mask = it->second;
        else if (s2.value == 106)
            cin_mask = rs.vcc;
    } else if (s2.kind == OperandKind::SGPR) {
        auto it = rs.sreg_bool.find(s2.value);
        if (it != rs.sreg_bool.end()) cin_mask = it->second;
    }
    if (!cin_mask)
        ok = false;
    else {
        uint32_t a = val(in.src[0]), c = val(in.src[1]);
        const CarryIds out = emit_vector_carry(b, *def, a, c, cin_mask);
        vreg[in.dst.value] = out.d;
        write_vop3b_carry_mask(b, rs, in, ok, out.carry);
    }
    return true;
}

std::optional<bool> Vop3MaskLowering::route_carry_out() {
    const VectorCarryDef* def = find_vector_carry(in.fmt, in.opcode);
    if (!def || def->reads_carry_in) return std::nullopt;
    // v_add_co_u32 (0x30F) / v_sub_co_u32 (0x310) / v_subrev_co_u32 (0x319): 32-bit add/
    // subtract that writes a carry/borrow-out to the VOP3B sdst mask (VCC or an SGPR pair).
    // No carry-IN — that is the _co_ci_ family (VOP2 0x28-0x2A / VOP3B 0x128-0x12A).
    // GTA V (PPSA04263) UI/content compute shaders use these; the missing op dropped the whole
    // shader, so its output texture rendered as bare untextured triangles (#1163/#1165).
    // CONFIDENCE: HIGH (RDNA2 ISA 6.5/3.9; coverage test in test_recompile_coverage).
    const uint32_t a = val(in.src[0]), c = val(in.src[1]);
    const CarryIds out = emit_vector_carry(b, *def, a, c);
    vreg[in.dst.value] = out.d;
    write_vop3b_carry_mask(b, rs, in, ok, out.carry);
    return true;
}

bool lower_vop3_carry_out(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                          bool allow_wave) {
    return Vop3MaskLowering({b, rs, in, ok, allow_wave}).route_carry_out().has_value();
}

bool lower_vop3_carry_in_out(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                             bool allow_wave) {
    return Vop3MaskLowering({b, rs, in, ok, allow_wave}).route_carry_in_out().has_value();
}

}   // namespace prosper::gpu::ir
