// Existing non-fused VOP3 MAD body, mechanically moved; fused FMA stays in the legacy route.
#include "gpu/recompiler/ir/lower/vop3_mad_float_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_mad_float_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/ir/ops/vector_mad_float.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
namespace prosper::gpu::ir {
bool Vop3MadFloatLowering::lower() {
    const auto* def = find_vector_mad_float_op(in.fmt, in.opcode);
    if (!def) return false;
    if (!fv || !fresult) {
        std::fprintf(stderr, "[ir] VOP3 float callbacks missing\n");
        std::abort();
    }
    const uint32_t value =
        emit_mad_float_op(b, *def, [&](uint32_t slot) { return fv(static_cast<int>(slot)); });
    vreg[in.dst.value] = fresult(value);
    return true;
}

bool lower_vop3_mad_float(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                          bool allow_wave, const Vop3FloatSource& fv,
                          const Vop3FloatResult& fresult) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3MadFloatLowering({b, rs, in, ok, allow_wave}, fv, fresult).lower();
}
}   // namespace prosper::gpu::ir
