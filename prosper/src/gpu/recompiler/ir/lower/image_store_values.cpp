// Per-component storage values. Access, eligibility, source proof and masked writes stay caller-owned.
#include "gpu/recompiler/ir/lower/image_store_values_lowering.hpp"
#include "gpu/recompiler/ir/ops/image_store_values.hpp"
#include <array>
#include <span>
#include <cstdint>
namespace prosper::gpu::ir {
uint32_t lower_image_store_raw_word(SpirvCompute& builder, uint32_t word) {
    return emit_image_storage_value(builder, build_image_storage_word(ImageStorageWordKind::Raw, 0),
                                    std::array{word});
}
uint32_t lower_image_store_d16_word(SpirvCompute& builder, uint32_t packed, uint32_t half,
                                    bool unsigned_data) {
    const auto kind =
        unsigned_data ? ImageStorageWordKind::UnsignedHalf : ImageStorageWordKind::FloatHalf;
    return emit_image_storage_value(builder, build_image_storage_word(kind, half),
                                    std::array{packed});
}
void lower_image_storage_load_selects(SpirvCompute& b, const uint32_t out[4], const int channel[4],
                                      const uint32_t constant[4], uint32_t routed_out[4]) {
    for (uint32_t k = 0; k < 4u; ++k)
        routed_out[k] =
            emit_image_storage_value(b, build_image_storage_select(channel[k], constant[k]),
                                     std::span<const uint32_t>(out, 4));
}
void lower_image_storage_store_selects(SpirvCompute& b, const uint32_t vals[4], const int source[4],
                                       uint32_t routed[4]) {
    for (uint32_t c = 0; c < 4u; ++c)
        routed[c] = emit_image_storage_value(b, build_image_storage_select(source[c], 0),
                                             std::span<const uint32_t>(vals, 4));
}
}   // namespace prosper::gpu::ir
