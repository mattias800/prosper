// Existing non-fused accumulator body mechanically relocated; oldD is not an explicit FV source.
#include "gpu/recompiler/ir/lower/vop3_mad_float_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_mac_float_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/ir/ops/vector_mac_float.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
namespace prosper::gpu::ir {
bool Vop3MacFloatLowering::lower() {
    const auto* def = find_vector_mac_float_op(in.fmt, in.opcode);
    if (!def) return false;
    if (!fv || !fresult) {
        std::fprintf(stderr, "[ir] MAC float callbacks missing\n");
        std::abort();
    }
    vreg[in.dst.value] = fresult(emit_ternary_integer_op(
        b, *def, [&](uint32_t slot) { return slot == 2 ? old_d : fv(static_cast<int>(slot)); }));
    return true;
}
bool lower_vop3_mac_float(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                          bool allow_wave, const Vop3FloatSource& fv,
                          const Vop3FloatResult& fresult, uint32_t old_d) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3MacFloatLowering({b, rs, in, ok, allow_wave}, fv, fresult, old_d).lower();
}
}   // namespace prosper::gpu::ir
