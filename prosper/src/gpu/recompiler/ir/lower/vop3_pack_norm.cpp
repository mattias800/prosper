// Preserve source float modifiers and leave packed-result clamp/predication to the caller.
#include "gpu/recompiler/ir/lower/vop3_pack_norm_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_pack_norm_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/ir/ops/vector_pack_norm.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
namespace prosper::gpu::ir {
bool Vop3PackNormLowering::lower() {
    const auto* def = find_vector_pack_norm_op(in.fmt, in.opcode);
    if (!def) return false;
    if (!fv) {
        std::fprintf(stderr, "[ir] normalized-pack source callback missing\n");
        std::abort();
    }
    vreg[in.dst.value] = emit_ternary_integer_op(
        b, *def, [this](uint32_t source) { return fv(static_cast<int>(source)); });
    return true;
}
bool lower_vop3_pack_norm(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                          bool allow_wave, const Vop3PackNormSource& fv) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3PackNormLowering({b, rs, in, ok, allow_wave}, fv).lower();
}
}   // namespace prosper::gpu::ir
