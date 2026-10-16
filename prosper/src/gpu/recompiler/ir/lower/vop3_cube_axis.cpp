// Mechanically extracted existing cubemap face and signed major-axis cases.
#include "gpu/recompiler/ir/lower/vop3_cube_axis_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_cube_axis_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/ir/ops/vector_cube_axis.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
namespace prosper::gpu::ir {
bool Vop3CubeAxisLowering::lower() {
    const auto* def = find_vector_cube_axis_op(in.fmt, in.opcode);
    if (!def) return false;
    if (!fv || !fresult) {
        std::fprintf(stderr, "[ir] cube callbacks missing\n");
        std::abort();
    }
    vreg[in.dst.value] = fresult(emit_ternary_integer_op(
        b, *def, [this](uint32_t source) { return fv(static_cast<int>(source)); }));
    return true;
}
bool lower_vop3_cube_axis(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                          bool allow_wave, const CubeAxisSource& fv,
                          const CubeAxisResult& fresult) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3CubeAxisLowering({b, rs, in, ok, allow_wave}, fv, fresult).lower();
}
}   // namespace prosper::gpu::ir
