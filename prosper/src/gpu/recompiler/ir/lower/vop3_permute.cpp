// Original byte permutation value case and helper, extracted with their modifier guard.
#include "gpu/recompiler/ir/lower/vop3_permute_lowering_internal.hpp"
#include "gpu/recompiler/ir/lower/vop3_permute_lowering.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/ir/ops/vector_permute.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
namespace {
uint32_t emit_v_perm_b32(SpirvCompute& b, uint32_t high, uint32_t low, uint32_t sel_dword,
                         const TernaryIntegerDef& def) {
    const uint32_t sources[] = {high, low, sel_dword};
    return emit_ternary_integer_op(b, def, [&](uint32_t source) { return sources[source]; });
}
}   // namespace
bool Vop3PermuteLowering::lower() {
    const auto* def = find_vector_permute_op(in.fmt, in.opcode);
    if (!def) return false;
    const bool modified = in.src_abs[0] || in.src_abs[1] || in.src_abs[2] || in.src_neg[0] ||
                          in.src_neg[1] || in.src_neg[2] || in.clamp || in.omod;
    if (modified) {
        ok = false;
    } else {
        vreg[in.dst.value] =
            emit_v_perm_b32(b, val(in.src[0]), val(in.src[1]), val(in.src[2]), *def);
    }
    return true;
}
bool lower_vop3_permute(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                        bool allow_wave) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3PermuteLowering({b, rs, in, ok, allow_wave}).lower();
}
}   // namespace prosper::gpu::ir
