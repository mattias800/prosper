#include "gpu/recompiler/storage_dst_sel.hpp"

namespace prosper::gpu {

bool data_format_is_integer(DataFormat f) {
    return f == DataFormat::Uint8 || f == DataFormat::Uint16 || f == DataFormat::Uint32 ||
           f == DataFormat::Sint8 || f == DataFormat::Sint16 || f == DataFormat::Sint32 ||
           f == DataFormat::Uint2_10_10_10 || f == DataFormat::Sint2_10_10_10;
}

bool storage_store_sources(const uint32_t (&swizzle)[4], uint32_t components, int (&source)[4]) {
    for (int& s : source) s = kDstSelNoSource;
    for (uint32_t k = 0; k < 4u; ++k) {
        const uint32_t selector = swizzle[k];
        if (selector == 2u || selector == 3u || selector > 7u) return false;
        if (selector < 4u) continue;
        const uint32_t c = selector - 4u;
        if (source[c] != kDstSelNoSource) return false;
        source[c] = static_cast<int>(k);
    }
    for (uint32_t c = 0; c < components && c < 4u; ++c)
        if (source[c] == kDstSelNoSource) return false;
    return true;
}

bool storage_load_selects(const uint32_t (&swizzle)[4], DataFormat format, int (&channel)[4],
                          uint32_t (&constant)[4]) {
    const uint32_t one = data_format_is_integer(format) ? 1u : 0x3f800000u;
    for (uint32_t k = 0; k < 4u; ++k) {
        const uint32_t selector = swizzle[k];
        if (selector == 2u || selector == 3u || selector > 7u) return false;
        channel[k] = selector >= 4u ? static_cast<int>(selector - 4u) : kDstSelNoSource;
        constant[k] = selector == 1u ? one : 0u;
    }
    return true;
}

}  // namespace prosper::gpu
