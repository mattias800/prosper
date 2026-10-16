// Original unsigned 24-bit multiply-add value leaf, extracted without changing its reads.
#include "gpu/recompiler/ir/lower/vop3_mad24_lowering_internal.hpp"
#include "gpu/recompiler/ir/lower/vop3_mad24_lowering.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
bool Vop3Mad24Lowering::lower() {
    // Temporary sentinel preserves the extracted else-if chain verbatim.
    if (false) {
    } else if (in.opcode == 0x143) {   // v_mad_u32_u24 = (s0&0xFFFFFF)*(s1&0xFFFFFF)+s2
        uint32_t m24 = b.uconst(0xFFFFFF);
        uint32_t p = b.ibin(Op_IMul, b.ibin(Op_BitwiseAnd, val(in.src[0]), m24),
                            b.ibin(Op_BitwiseAnd, val(in.src[1]), m24));
        vreg[in.dst.value] = b.ibin(Op_IAdd, p, val(in.src[2]));
    } else {
        return false;
    }
    return true;
}
bool lower_vop3_mad24(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                      bool allow_wave) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3Mad24Lowering({b, rs, in, ok, allow_wave}).lower();
}
}   // namespace prosper::gpu::ir
