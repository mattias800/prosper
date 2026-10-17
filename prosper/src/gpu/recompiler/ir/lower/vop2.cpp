// vop2.cpp — VOP2 lowering: DPP routes, source modifiers, the operation, destination modifiers.
#include "gpu/recompiler/ir/lower/vector_lowering_internal.hpp"

namespace prosper::gpu::ir {

bool lower_vop2(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok, bool allow_wave) {
    return Vop2Lowering({b, rs, in, ok, allow_wave}).lower();
}

bool Vop2Lowering::lower() {
    a = val(in.src[0]);
    c = val(in.src[1]);
    old_d = vreg_old(b, rs, in.dst.value);
    if (auto routed = route_dpp_row_shr_in_place()) return *routed;
    if (auto routed = route_vadd_dpp_partial_row()) return *routed;
    if (auto routed = stage_dpp_source()) return *routed;
    // SDWA float source modifiers (only ever set on float ops by the assembler): abs then neg.
    // Packed-f16 ops apply these after selecting/unpacking the half below.
    packed_f16 = in.opcode == 0x32 || in.opcode == 0x33 || in.opcode == 0x35 ||
                 in.opcode == 0x39 || in.opcode == 0x3A;
    integer_sdwa = in.has_sdwa &&
        (in.opcode == 0x0B || (in.opcode >= 0x11 && in.opcode <= 0x14) ||
         in.opcode == 0x16 || in.opcode == 0x18 ||
         (in.opcode >= 0x1A && in.opcode <= 0x1E) ||
         (in.opcode >= 0x25 && in.opcode <= 0x2A));
    if (auto routed = stage_source_modifiers()) return *routed;
    uint32_t& d = vreg[in.dst.value];
    if (auto computed = compute(d)) return *computed;
    if (auto routed = stage_sdwa_destination(d)) return *routed;
    if (auto routed = stage_output_modifiers(d)) return *routed;
    if (ok) {
        if (dpp_active && !in.dpp_bound_ctrl) d = b.sel(dpp_active, d, old_d);
        predicate_write(b, rs, in.dst.value, old_d);
    }
    return true;
}

// The operation itself, on the already-transformed sources.
std::optional<bool> Vop2Lowering::compute(uint32_t& d) {
    // The integer and bitwise group goes through its IR definition (ir/ops/vector_alu.cpp).
    if (const VectorOpDef* def = find_vector_op(in.fmt, in.opcode)) {
        d = emit_vector_op(b, *def, a, c);
        return std::nullopt;
    }
    if (compute_float(d)) return std::nullopt;
    if (compute_f16(d)) return std::nullopt;
    // The lane-mask readers and writers: the value is an IR definition, the VCC read and write
    // are this lowering's (vop2_mask.cpp).
    if (const VectorCndmaskDef* def = find_vector_cndmask(in.fmt, in.opcode))
        return compute_cndmask(*def, d);
    if (const VectorCarryDef* def = find_vector_carry(in.fmt, in.opcode))
        return compute_carry(*def, d);
    ok = false;
    return std::nullopt;
}

} // namespace prosper::gpu::ir
