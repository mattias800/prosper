// Per-word MIMG D16 packing. The caller owns access, admission and each masked write.
#include "gpu/recompiler/ir/lower/image_d16_lowering.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
void lower_image_d16_results(SpirvCompute& b, const Rdna2Inst& in, const uint32_t out[4],
                             const ImageD16Writer& write_packed) {
    uint32_t packed = b.uconst(0);
    int component_index = 0;
    for (uint32_t component = 0; component < 4; ++component) {
        if (!(in.mimg_dmask & (1u << component))) continue;
        const uint32_t half = b.pack_half_lo(out[component]);
        if ((component_index & 1) == 0) {
            packed = half;
        } else {
            packed = b.ibin(Op_BitwiseOr, packed, b.ibin(Op_ShiftLeftLogical, half, b.uconst(16)));
            write_packed(packed);
        }
        ++component_index;
    }
    if (component_index & 1) write_packed(packed); // odd dmask: high half is zero.
}
uint32_t lower_image_d16_gather_word(SpirvCompute& b, const uint32_t out[4], uint32_t word) {
    return b.pack_half2x16(out[word * 2], out[word * 2 + 1]);
}
} // namespace prosper::gpu::ir
