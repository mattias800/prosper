// Mixed VOP3P source preparation and result publication, moved by extract_case_routes.py.
// The mixed-half arithmetic is retained verbatim; this relocation adds no value definition.
#include "gpu/recompiler/ir/lower/vop3p_mixed_float.hpp"
#include "gpu/recompiler/ir/lower/lowering_context.hpp"

namespace prosper::gpu::ir {
namespace {
struct Vop3pMixedFloatLowering : LoweringContext {
    bool lower();
};
bool Vop3pMixedFloatLowering::lower() {
    // Mixed-precision FMA family, trivial form only (all sources full f32 — the decoder set
    // has_modifier for any opsel/neg/clamp bits, rejected above). v_fma_mix_f32 (0x20):
    // d = s0*s1+s2. v_fma_mixlo/hi_f16 (0x21/0x22): the f32 result converts to f16 into the
    // LOW/HIGH half of d, PRESERVING the other half (DOLL's box-blur PS packs its result
    // this way, #273). VERIFIED(round-trip llvm-mc gfx1010: 0xcc210000 0x041600f2 ->
    // v_fma_mixlo_f16 v0, 1.0, v0, v5 — the live blur bytes).
    if (in.opcode != 0x20 && in.opcode != 0x21 && in.opcode != 0x22) { ok = false; return true; }
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
    uint32_t r = b.fbin(Op_FAdd, b.fbin(Op_FMul, mixv(0), mixv(1)), mixv(2));
    if (in.clamp) r = b.clamp01(r);
    uint32_t& d = rs.vreg[in.dst.value];
    if (in.opcode == 0x20) d = r;
    else if (in.opcode == 0x21)
        d = b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), b.pack_half_lo(r));
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
