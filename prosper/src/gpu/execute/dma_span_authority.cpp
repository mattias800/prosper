#include "gpu/execute/dma_span_authority.hpp"
#include <cstdint>

namespace prosper::gpu {
namespace {
// [a, a + a_bytes) and [b, b + b_bytes) share at least one byte. Saturates rather than wraps.
bool ranges_overlap(uint64_t a, uint64_t a_bytes, uint64_t b, uint64_t b_bytes) {
    if (!a_bytes || !b_bytes) return false;
    const uint64_t a_end = a > UINT64_MAX - a_bytes ? UINT64_MAX : a + a_bytes;
    const uint64_t b_end = b > UINT64_MAX - b_bytes ? UINT64_MAX : b + b_bytes;
    return a < b_end && b < a_end;
}
} // namespace

bool dma_needs_authoritative_span(const std::vector<DrawItem>& span, uint64_t dst, uint64_t src,
                                  uint32_t bytes, uint32_t sels) {
    const bool source_gds = ((sels >> 8u) & 0xffu) == 1u;
    const bool destination_gds = (sels & 0xffu) == 1u;
    const auto touches = [&](uint64_t base, uint64_t extent) {
        return (!source_gds && ranges_overlap(src, bytes, base, extent)) ||
               (!destination_gds && ranges_overlap(dst, bytes, base, extent));
    };
    for (const DrawItem& draw : span) {
        for (uint32_t slot = 0; slot < draw.color_targets.size(); ++slot) {
            const auto& target = draw.color_targets[slot];
            if (!target.base) continue;
            if (!target.raw_snapshot_footprint_bytes) return true;   // unproved extent
            if (touches(target.base, target.raw_snapshot_footprint_bytes)) return true;
        }
        // The named slots are what the renderer reads back; a disagreeing binding is unproved.
        if ((draw.color0_base && draw.color0_base != draw.color_targets[0].base) ||
            (draw.color1_base && draw.color1_base != draw.color_targets[1].base))
            return true;
    }
    return false;
}
} // namespace prosper::gpu
