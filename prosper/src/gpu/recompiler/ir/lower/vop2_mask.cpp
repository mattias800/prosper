// vop2_mask.cpp — the VOP2 (e32) lane-mask ops: v_cndmask_b32 reads VCC, and
// v_add/sub/subrev_co_ci_u32 read their carry-in from VCC and write their carry-out to it. The
// per-lane values are IR definitions (ir/ops/vector_cndmask.cpp, ir/ops/vector_carry.cpp); the VCC
// write is the shared carry mask write (carry_mask_write.cpp). Both used to be cases of
// Vop2Lowering::compute's switch.
#include "gpu/recompiler/ir/lower/carry_mask_write.hpp"
#include "gpu/recompiler/ir/lower/vector_lowering_internal.hpp"

namespace prosper::gpu::ir {

// v_cndmask_b32: dst = vcc ? src1 : src0. A sub-dword SDWA destination selects the source words
// first and merges the result into the other half of the old VDST.
std::optional<bool> Vop2Lowering::compute_cndmask(const VectorCndmaskDef& def, uint32_t& d) {
    if (!vcc) { ok = false; return true; }
    if (in.sdwa_dst_sel == 6) { d = emit_vector_cndmask(b, def, vcc, c, a); return std::nullopt; }
    auto word = [&](uint32_t raw, uint8_t sel) {
        if (sel == 5) raw = b.ibin(Op_ShiftRightLogical, raw, b.uconst(16));
        return b.ibin(Op_BitwiseAnd, raw, b.uconst(0xFFFFu));
    };
    uint32_t selected = emit_vector_cndmask(b, def, vcc, word(c, in.sdwa_src1_sel),
                                            word(a, in.sdwa_src0_sel));
    d = in.sdwa_dst_sel == 5
        ? b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                 b.ibin(Op_ShiftLeftLogical, selected, b.uconst(16)))
        : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), selected);
    return std::nullopt;
}

// Carry ops (VOP2 e32 form): carry-in + carry-out are VCC. v_add_co_ci(0x28)/
// v_sub_co_ci(0x29)/v_subrev_co_ci(0x2a), the same definitions as VOP3B 0x128-0x12A.
std::optional<bool> Vop2Lowering::compute_carry(const VectorCarryDef& def, uint32_t& d) {
    if (!vcc) { ok = false; return true; }
    const CarryIds out = emit_vector_carry(b, def, a, c, vcc);
    d = out.d;
    write_vop2_carry_mask(b, rs, out.carry);
    return std::nullopt;
}

}   // namespace prosper::gpu::ir
