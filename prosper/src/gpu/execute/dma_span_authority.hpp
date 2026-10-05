// Whether an ordered DMA_DATA copy needs the pending render span published to the CPU first.
#pragma once
#include "gpu/execute/gpu_execute.hpp"
#include "diagnostics/env_cache.hpp"
#include <cstdint>
#include <vector>

namespace prosper::gpu {
// What the authoritative flush actually does (#4459 review): the renderer synchronously reads every
// colour target the pending span rendered back into its own CPU cache (g_rtt[...].rgba) -- not into
// guest memory -- and declines any volume (3D) producer pass in that span. A DMA copy does not
// depend on either for correctness: its source is served by the live-target reader, which reads a
// GPU-only target back on demand, and its destination write calls notify_guest_gpu_write, which
// invalidates an overlapped renderer image and its CPU copy whatever the flush did. The eager
// readback only spares the reader that on-demand readback, and makes a failed one impossible.
//
// So keep it, conservatively, only for a copy that touches a colour target the span renders --
// there the old ordering is preserved exactly -- and skip it for every other copy (#4439: ~48 MB
// per flip on Alex Kidd for copies that touch none). An unproved target extent, or a named
// color0/color1 base its slot binding does not prove, counts as touched. A GDS operand (selector
// 1) is an offset, not guest memory, and is ignored.
bool dma_needs_authoritative_span(const std::vector<DrawItem>& span, uint64_t dst, uint64_t src,
                                  uint32_t bytes, uint32_t sels);

// The executor's decision. PROSPER_DMA_ALWAYS_AUTHORITATIVE (set, any value) restores the
// unconditional readback as the same-binary A/B control. It is a guest-behaviour selector because
// authoritative mode declines volume producer passes in the span, which changes what renders.
// Any copy record with dst/src/bytes/sels (the live GpuState::DmaCopy and the replay copy).
template <class Copy>
bool dma_flush_authoritative(const std::vector<DrawItem>& span, const Copy& copy) {
    return PROSPER_ENV_ON("PROSPER_DMA_ALWAYS_AUTHORITATIVE") ||
           dma_needs_authoritative_span(span, copy.dst, copy.src, copy.bytes, copy.sels);
}
} // namespace prosper::gpu
