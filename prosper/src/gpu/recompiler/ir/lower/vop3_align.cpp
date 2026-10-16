// Original VOP3 bit/byte alignment value cases, extracted with their modifier guard.
#include "gpu/recompiler/ir/lower/vop3_align_lowering_internal.hpp"
#include "gpu/recompiler/ir/lower/vop3_align_lowering.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/ir/ops/vector_align.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
bool Vop3AlignLowering::lower() {
    const auto* def = find_vector_align_op(in.fmt, in.opcode);
    if (!def) return false;
    const bool modified = in.src_abs[0] || in.src_abs[1] || in.src_abs[2] || in.src_neg[0] ||
                          in.src_neg[1] || in.src_neg[2] || in.clamp || in.omod;
    if (modified) {
        ok = false;
    } else {
        vreg[in.dst.value] = emit_ternary_integer_op(
            b, *def, [this](uint32_t source) { return val(in.src[source]); });
    }
    return true;
}
bool lower_vop3_align(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                      bool allow_wave) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3AlignLowering({b, rs, in, ok, allow_wave}).lower();
}
}   // namespace prosper::gpu::ir
