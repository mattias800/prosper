// Mixed VOP3P source preparation and result publication, moved by extract_case_routes.py.
// F32 delegates to the shared IR Fma; mixed-half rounding remains the legacy #5225 gap.
#include "gpu/recompiler/ir/lower/vop3p_mixed_float.hpp"
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/ir/ops/vector_fma_float.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <cstdint>

namespace prosper::gpu::ir {
namespace {
struct Vop3pMixedFloatLowering : LoweringContext {
    bool lower();
};
bool Vop3pMixedFloatLowering::lower() {
    // ISA 70648 12.10: F32 FMA after independent full/half source expansion. GLSL Fma has
    // MED hosted precision (#5225), as for the standard F32 forms; it is not a strict FMA
    // feature or a correction of other guest MODE/nonfinite contracts. Mixed-half results
    // retain their old arithmetic and preserve the other half of VDST; they need one final
    // half rounding, which an F32 Fma followed by packing does not establish.
    if (in.opcode != 0x20 && in.opcode != 0x21 && in.opcode != 0x22) {
        ok = false;
        return true;
    }
    uint32_t old_d = vreg_old(b, rs, in.dst.value);
    // Per-source mix resolve (#273): OPSEL_HI[k] -> f16 half (OPSEL[k]: 0=lo,1=hi) converted
    // to f32; else full f32. An inline constant read at f16 width is the dword its width
    // produces (see inline_16bit_operand_dword), so OPSEL selects a half of THAT — a float
    // reads 0.0h from the high half, not the constant again. This used to treat the select
    // as a no-op for inline floats (#2119, kernel 32r14b).
    // NEG_HI = abs, NEG = negate (abs first, hardware order).
    auto mixv = [&](int k) -> uint32_t {
        uint32_t v = val(in.src[k]);
        const bool half = (in.vop3p_opsel_hi >> k) & 1u;
        if (half) {
            const uint32_t sel = (in.vop3p_opsel >> k) & 1u;
            uint32_t inline_dword;
            if (inline_16bit_operand_dword(in.src[k], inline_dword))
                v = b.unpack_half(b.uconst(inline_dword), sel);
            else
                v = b.unpack_half(v, sel);
        }
        if (in.src_abs[k]) v = b.fext1(Glsl_FAbs, v);
        if (in.src_neg[k]) v = b.fneg(v);
        return v;
    };
    uint32_t r = in.opcode == 0x20 ? emit_fused_mad_values(b, mixv(0), mixv(1), mixv(2))
                                   : b.fbin(Op_FAdd, b.fbin(Op_FMul, mixv(0), mixv(1)), mixv(2));
    if (in.clamp) r = b.clamp01(r);
    uint32_t& d = rs.vreg[in.dst.value];
    if (in.opcode == 0x20)
        d = r;
    else if (in.opcode == 0x21)
        d = b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)),
                   b.pack_half_lo(r));
    else
        d = b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                   b.ibin(Op_ShiftLeftLogical, b.pack_half_lo(r), b.uconst(16)));
    if (ok) predicate_write(b, rs, in.dst.value, old_d);
    return true;
}
}   // namespace

bool lower_vop3p_mixed_float(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                             bool allow_wave) {
    return Vop3pMixedFloatLowering{{b, rs, in, ok, allow_wave}}.lower();
}
}   // namespace prosper::gpu::ir
