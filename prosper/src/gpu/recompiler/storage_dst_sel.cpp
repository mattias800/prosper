#include "gpu/recompiler/storage_dst_sel.hpp"

#include <atomic>
#include <cstdio>

namespace prosper::gpu {
namespace {
// A refusal drops the whole program, so say which selector caused it. Bounded: one line per
// distinct (direction, selector word), capped, because the same descriptor recurs every dispatch.
void report_refusal(const char* direction, const uint32_t (&swizzle)[4], uint32_t components,
                    const char* why) {
    static std::atomic<unsigned> reported{0};
    if (reported.fetch_add(1, std::memory_order_relaxed) >= 32) return;
    std::fprintf(stderr, "[storage-dst-sel] refused image_%s: DST_SEL=(%u,%u,%u,%u) components=%u: %s\n",
                 direction, swizzle[0], swizzle[1], swizzle[2], swizzle[3], components, why);
}
}  // namespace

bool data_format_is_integer(DataFormat f) {
    return f == DataFormat::Uint8 || f == DataFormat::Uint16 || f == DataFormat::Uint32 ||
           f == DataFormat::Sint8 || f == DataFormat::Sint16 || f == DataFormat::Sint32 ||
           f == DataFormat::Uint2_10_10_10 || f == DataFormat::Sint2_10_10_10;
}

bool storage_store_sources(const uint32_t (&swizzle)[4], uint32_t components, int (&source)[4]) {
    for (int& s : source) s = kDstSelNoSource;
    for (uint32_t k = 0; k < 4u; ++k) {
        const uint32_t selector = swizzle[k];
        if (selector == 2u || selector == 3u || selector > 7u) {
            report_refusal("store", swizzle, components, "reserved selector");
            return false;
        }
        if (selector < 4u) continue;
        const uint32_t c = selector - 4u;
        if (source[c] != kDstSelNoSource) {
            report_refusal("store", swizzle, components, "two selectors name one stored channel");
            return false;
        }
        source[c] = static_cast<int>(k);
    }
    for (uint32_t c = 0; c < components && c < 4u; ++c)
        if (source[c] == kDstSelNoSource) {
            report_refusal("store", swizzle, components, "a stored channel no selector names");
            return false;
        }
    return true;
}

bool storage_load_selects(const uint32_t (&swizzle)[4], DataFormat format, int (&channel)[4],
                          uint32_t (&constant)[4]) {
    const uint32_t one = data_format_is_integer(format) ? 1u : 0x3f800000u;
    for (uint32_t k = 0; k < 4u; ++k) {
        const uint32_t selector = swizzle[k];
        if (selector == 2u || selector == 3u || selector > 7u) {
            report_refusal("load", swizzle, 0, "reserved selector");
            return false;
        }
        channel[k] = selector >= 4u ? static_cast<int>(selector - 4u) : kDstSelNoSource;
        constant[k] = selector == 1u ? one : 0u;
    }
    return true;
}

}  // namespace prosper::gpu
