// Whether an ordered DMA_DATA copy needs the pending render span published to the CPU first.
#pragma once
#include "gpu/execute/gpu_execute.hpp"
#include "diagnostics/env_cache.hpp"
#include <cstdint>
#include <vector>

namespace prosper::gpu {
// A DMA copy reads its source and writes its destination in guest memory. Before it runs, the
// executor flushes the pending render span; an AUTHORITATIVE flush also reads every colour target
// that span rendered back to guest memory. That readback is what a copy needs when one of its
// ranges overlaps such a target -- the source would otherwise be read stale, or the destination
// would land under a target whose other bytes are still only on the GPU -- and nothing else.
// A copy that touches no span target gains nothing from it (#4439: ~48 MB per flip on Alex Kidd).
//
// Conservative by construction: a colour target whose physical extent is unproved counts as
// overlapping, and so does a named color0/color1 base that disagrees with its slot binding. A GDS
// operand (selector 1) names an offset in GDS, not guest memory, and is ignored.
bool dma_needs_authoritative_span(const std::vector<DrawItem>& span, uint64_t dst, uint64_t src,
                                  uint32_t bytes, uint32_t sels);

// The executor's decision. PROSPER_DMA_ALWAYS_AUTHORITATIVE=1 restores the unconditional readback
// as the same-binary A/B control (guest-behaviour selector: it moves when span bytes reach guest
// memory, not what a correct copy reads or writes).
// Any copy record with dst/src/bytes/sels (the live GpuState::DmaCopy and the replay copy).
template <class Copy>
bool dma_flush_authoritative(const std::vector<DrawItem>& span, const Copy& copy) {
    return PROSPER_ENV_ON("PROSPER_DMA_ALWAYS_AUTHORITATIVE") ||
           dma_needs_authoritative_span(span, copy.dst, copy.src, copy.bytes, copy.sels);
}
} // namespace prosper::gpu
