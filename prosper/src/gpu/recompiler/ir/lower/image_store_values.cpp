// MIMG storage values. Access, eligibility, source proof and masked writes stay caller-owned.
#include "gpu/recompiler/ir/lower/image_store_values_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
uint32_t lower_image_store_d16_word(SpirvCompute& builder, uint32_t packed, uint32_t half,
                                    bool unsigned_data) {
    if (!unsigned_data) return builder.unpack_half(packed, half);
    // UINT is an unsigned 16-bit numeric value, so extraction zero-extends it to the uvec4
    // storage interface. CONFIDENCE: MED: RDNA2 format-selected integers + later AMD's explicit
    // UINT16 table; production backend value tests cover sign-bit/high-half values.
    const uint32_t shifted = builder.ibin(Op_ShiftRightLogical, packed, builder.uconst(half * 16u));
    return builder.ibin(Op_BitwiseAnd, shifted, builder.uconst(0xffffu));
}
void lower_image_storage_load_selects(SpirvCompute& b, const uint32_t out[4], const int channel[4],
                                      const uint32_t constant[4], uint32_t routed_out[4]) {
    for (uint32_t k = 0; k < 4u; ++k)
        routed_out[k] = channel[k] >= 0 ? out[channel[k]] : b.uconst(constant[k]);
}
void lower_image_storage_store_selects(SpirvCompute& b, const uint32_t vals[4], const int source[4],
                                       uint32_t routed[4]) {
    for (uint32_t c = 0; c < 4u; ++c) routed[c] = source[c] >= 0 ? vals[source[c]] : b.uconst(0);
}
}   // namespace prosper::gpu::ir
