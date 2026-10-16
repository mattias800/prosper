// Existing DX9 legacy multiply body mechanically moved, preserving all old call positions.
#include "gpu/recompiler/ir/lower/vop3_legacy_mul_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_legacy_mul_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/ir/ops/vector_legacy_mul.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
namespace prosper::gpu::ir {
bool Vop3LegacyMulLowering::lower() {
    const auto* def = find_vector_legacy_mul_op(in.fmt, in.opcode);
    if (!def) return false;
    if (!fv || !fresult) {
        std::fprintf(stderr, "[ir] legacy multiply callbacks missing\n");
        std::abort();
    }
    vreg[in.dst.value] = fresult(emit_ternary_integer_op(
        b, *def, [&](uint32_t source) { return fv(static_cast<int>(source)); }));
    return true;
}
bool lower_vop3_legacy_mul_erased(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                                  bool allow_wave, const LegacyMulSource& fv,
                                  const LegacyMulResult& fresult) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3LegacyMulLowering({b, rs, in, ok, allow_wave}, fv, fresult).lower();
}
}   // namespace prosper::gpu::ir
