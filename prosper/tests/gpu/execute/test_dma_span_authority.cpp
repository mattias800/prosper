// #4439: an ordered DMA copy asks for the authoritative span readback only when it can observe
// or overwrite a colour target the pending span renders. Unproved extents stay conservative.
#include "gpu/execute/dma_span_authority.hpp"
#include <gtest/gtest.h>
#include <vector>

namespace {
using prosper::gpu::DrawItem;
using prosper::gpu::GpuState;
bool dma_needs_authoritative_span(const std::vector<DrawItem>& span, const GpuState::DmaCopy& c) {
    return prosper::gpu::dma_needs_authoritative_span(span, c.dst, c.src, c.bytes, c.sels);
}

DrawItem draw_with_target(uint64_t base, uint64_t bytes, uint32_t slot = 0) {
    DrawItem draw;
    draw.color_targets[slot].base = base;
    draw.color_targets[slot].raw_snapshot_footprint_bytes = bytes;
    if (slot == 0) draw.color0_base = base;
    if (slot == 1) draw.color1_base = base;
    return draw;
}
GpuState::DmaCopy copy(uint64_t dst, uint64_t src, uint32_t bytes, uint32_t sels = 0) {
    GpuState::DmaCopy c;
    c.dst = dst; c.src = src; c.bytes = bytes; c.sels = sels;
    return c;
}
constexpr uint64_t kTarget = 0x40000000, kTargetBytes = 0x100000;
} // namespace

TEST(DmaSpanAuthority, EmptySpanNeverNeedsIt) {
    EXPECT_FALSE(dma_needs_authoritative_span({}, copy(kTarget, kTarget + 16, 64)));
}

TEST(DmaSpanAuthority, CopyAwayFromEveryTargetSkipsIt) {
    const std::vector<DrawItem> span{draw_with_target(kTarget, kTargetBytes)};
    EXPECT_FALSE(dma_needs_authoritative_span(span, copy(0x10000000, 0x20000000, 4096)));
    // Touching the target's end exactly is not an overlap; one byte in is.
    EXPECT_FALSE(dma_needs_authoritative_span(span, copy(kTarget + kTargetBytes, 0x20000000, 64)));
    EXPECT_FALSE(dma_needs_authoritative_span(span, copy(0x20000000, kTarget - 64, 64)));
    EXPECT_TRUE(dma_needs_authoritative_span(span, copy(kTarget + kTargetBytes - 1, 0x20000000, 64)));
}

TEST(DmaSpanAuthority, SourceOrDestinationOverlapNeedsIt) {
    const std::vector<DrawItem> span{draw_with_target(0x30000000, 0x1000),
                                     draw_with_target(kTarget, kTargetBytes, 3)};
    EXPECT_TRUE(dma_needs_authoritative_span(span, copy(kTarget + 0x800, 0x20000000, 64)))
        << "the destination lands under a span target";
    EXPECT_TRUE(dma_needs_authoritative_span(span, copy(0x20000000, kTarget + 0x800, 64)))
        << "the source reads a span target";
}

TEST(DmaSpanAuthority, UnprovedExtentStaysConservative) {
    const std::vector<DrawItem> span{draw_with_target(kTarget, 0)};
    EXPECT_TRUE(dma_needs_authoritative_span(span, copy(0x10000000, 0x20000000, 64)));
    DrawItem disagreeing = draw_with_target(kTarget, kTargetBytes);
    disagreeing.color0_base = kTarget + 0x10000;   // named slot not proven by its binding
    EXPECT_TRUE(dma_needs_authoritative_span({disagreeing}, copy(0x10000000, 0x20000000, 64)));
}

TEST(DmaSpanAuthority, GdsOperandsAreNotGuestRanges) {
    const std::vector<DrawItem> span{draw_with_target(0x24, 0x100)};
    // Destination selector 1 = GDS offset 0x24: numerically inside the "target" but not guest memory.
    EXPECT_FALSE(dma_needs_authoritative_span(span, copy(0x24, 0x20000000, 4, 0x1)));
    // The guest source of the same copy still counts.
    EXPECT_TRUE(dma_needs_authoritative_span(span, copy(0x24, 0x40, 4, 0x1)));
    // Source selector 1 = GDS read; the guest destination still counts.
    EXPECT_FALSE(dma_needs_authoritative_span(span, copy(0x20000000, 0x24, 4, 0x100)));
    EXPECT_TRUE(dma_needs_authoritative_span(span, copy(0x30, 0x24, 4, 0x100)));
}

// #4457: a target whose exact extent is unproved (an MSAA or unusual-swizzle view) still has a
// conservative bound, and the predicate decides against it -- disjoint copies skip the readback,
// overlapping ones keep it. PROSPER_DMA_EXACT_EXTENT_ONLY is read once per process, so its
// control arm is exercised by the live A/B, not here.
TEST(DmaSpanAuthority, BoundedUnprovedTargetIsDecidedByItsBound) {
    DrawItem draw = draw_with_target(kTarget, 0);
    draw.color_targets[0].footprint_bound_bytes = kTargetBytes;
    const std::vector<DrawItem> span{draw};
    EXPECT_FALSE(dma_needs_authoritative_span(span, copy(0x10000000, 0x20000000, 4096)));
    EXPECT_FALSE(dma_needs_authoritative_span(span, copy(kTarget + kTargetBytes, 0x20000000, 64)));
    EXPECT_TRUE(
        dma_needs_authoritative_span(span, copy(0x10000000, kTarget + kTargetBytes - 1, 2)));
    EXPECT_TRUE(dma_needs_authoritative_span(span, copy(kTarget, 0x20000000, 64)));
    // Without a bound the same target is unproved, so even the far-away copy keeps the readback.
    const std::vector<DrawItem> unbounded{draw_with_target(kTarget, 0)};
    EXPECT_TRUE(dma_needs_authoritative_span(unbounded, copy(0x10000000, 0x20000000, 4096)));
}
