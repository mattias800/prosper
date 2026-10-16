// Preserve source/result modifier callbacks and caller publication order.
#include "gpu/recompiler/ir/lower/vop3_div_fixup_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_div_fixup_lowering_internal.hpp"
#include "gpu/recompiler/ir/ops/vector_div_fixup.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
namespace prosper::gpu::ir {
bool Vop3DivFixupLowering::lower() {
    const auto* def = find_vector_div_fixup_op(in.fmt, in.opcode);
    if (!def) return false;
    if (!fv || !fresult) {
        std::fprintf(stderr, "[ir] fixup callbacks missing\n");
        std::abort();
    }
    vreg[in.dst.value] = fresult(emit_ternary_integer_op(
        b, *def, [this](uint32_t source) { return fv(static_cast<int>(source)); }));
    return true;
}
bool lower_vop3_div_fixup_erased(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                                 bool allow_wave, const DivFixupSource& fv,
                                 const DivFixupResult& fresult) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3DivFixupLowering({b, rs, in, ok, allow_wave}, fv, fresult).lower();
}
}   // namespace prosper::gpu::ir
