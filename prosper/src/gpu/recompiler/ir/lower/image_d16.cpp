// MIMG value-only packing. The caller keeps every original old-value read and masked write.
#include "gpu/recompiler/ir/lower/image_d16_lowering.hpp"
#include "gpu/recompiler/ir/ops/image_d16.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
namespace prosper::gpu::ir {
void lower_image_d16_results(SpirvCompute& b, const Rdna2Inst& in, const uint32_t out[4],
                             const ImageD16Writer& write_packed) {
    // Preserve the pre-IR initialization's cached constant allocation, including empty masks.
    b.uconst(0);
    std::array<uint32_t, 4> selected{};
    uint32_t count = 0;
    for (uint32_t component = 0; component < 4; ++component)
        if (in.mimg_dmask & (1u << component)) selected[count++] = out[component];
    for (uint32_t at = 0; at < count; at += 2) {
        const bool pair = at + 1 < count;
        write_packed(
            emit_image_d16_word(b, pair ? ImageD16Form::SelectedPair : ImageD16Form::SelectedLow,
                                selected[at], pair ? selected[at + 1] : 0));
    }
}
uint32_t lower_image_d16_gather_word(SpirvCompute& b, const uint32_t out[4], uint32_t word) {
    return emit_image_d16_word(b, ImageD16Form::GatherPair, out[size_t{word} * 2],
                               out[size_t{word} * 2 + 1]);
}
} // namespace prosper::gpu::ir
