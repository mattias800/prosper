// CUBE stacked-face coordinate values; mechanically relocated before the IR migration.
#include "gpu/recompiler/ir/lower/image_cube_coordinates_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu::ir {

std::array<uint32_t, 4> lower_image_cube_coordinates(SpirvCompute& b, uint32_t ci,
                                                     const ImageAddressSlotReader& read) {
    const auto& vread = read;
    const auto cvg = [](uint32_t slot) { return slot; };
    uint32_t x = vread(cvg(ci)), y = vread(cvg(ci + 1)), fid = vread(cvg(ci + 2));
    const uint32_t one = b.uconst(fbits(1.0f)), zero = b.uconst(fbits(0.0f));
    uint32_t uf = b.fbin(Op_FSub, x, one);
    uint32_t vf = b.fext2(Glsl_FMax, b.fext2(Glsl_FMin, b.fbin(Op_FSub, y, one), one), zero);
    uint32_t layer = b.fext1(
        Glsl_RoundEven, b.fext2(Glsl_FMax, b.fext2(Glsl_FMin, fid, b.uconst(fbits(5.0f))), zero));
    uint32_t v6 = b.fbin(Op_FMul, b.fbin(Op_FAdd, layer, vf), b.uconst(fbits(1.0f / 6.0f)));
    return {uf, vf, layer, v6};
}

} // namespace prosper::gpu::ir
