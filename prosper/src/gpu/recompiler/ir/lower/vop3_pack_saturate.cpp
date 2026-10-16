// Existing unsigned saturating halfword pack, mechanically extracted without changing reads.
#include "gpu/recompiler/ir/lower/vop3_pack_saturate_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_pack_saturate_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
bool Vop3PackSaturateLowering::lower() {
    if (false) {
    } else if (in.opcode == 0x36A) {   // v_cvt_pk_u16_u32
        // Pack two u32 into u16 halves with UNSIGNED SATURATION: lo = min(s0,0xFFFF),
        // hi = min(s1,0xFFFF); dst = lo | hi<<16. VERIFIED(round-trip llvm-mc gfx1010:
        // VOP3 0x36a — 0xd76a…). CONFIDENCE: HIGH.
        uint32_t lo = b.uext2(Glsl_UMin, val(in.src[0]), b.uconst(0xFFFFu));
        uint32_t hi = b.uext2(Glsl_UMin, val(in.src[1]), b.uconst(0xFFFFu));
        vreg[in.dst.value] =
            b.ibin(Op_BitwiseOr, lo, b.ibin(Op_ShiftLeftLogical, hi, b.uconst(16)));
    } else {
        return false;
    }
    return true;
}
bool lower_vop3_pack_saturate(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                              bool allow_wave) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3PackSaturateLowering({b, rs, in, ok, allow_wave}).lower();
}
}   // namespace prosper::gpu::ir
