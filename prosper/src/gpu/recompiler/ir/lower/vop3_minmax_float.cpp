// Existing float extrema keep root-owned modifiers and final publication.
#include "gpu/recompiler/ir/lower/vop3_minmax_float_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_minmax_float_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/ir/ops/vector_minmax_float.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
namespace prosper::gpu::ir {
bool Vop3MinmaxFloatLowering::lower() {
    const auto* def = find_vector_minmax_float_op(in.fmt, in.opcode);
    if (!def) return false;
    if (!fv || !fresult) {
        std::fprintf(stderr, "[ir] float callbacks missing\n");
        std::abort();
    }
    vreg[in.dst.value] = fresult(emit_minmax_float_op(
        b, *def, [this](uint32_t source) { return fv(static_cast<int>(source)); }));
    return true;
}
bool lower_vop3_minmax_float(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                             bool allow_wave, const Vop3MinmaxFloatSource& fv,
                             const Vop3MinmaxFloatResult& fresult) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3MinmaxFloatLowering({b, rs, in, ok, allow_wave}, fv, fresult).lower();
}
}   // namespace prosper::gpu::ir
