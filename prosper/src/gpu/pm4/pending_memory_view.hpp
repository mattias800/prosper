#pragma once
#include "gpu/pm4/pm4_decode.hpp"
#include <cstdint>
#include <span>

namespace prosper::gpu {
struct PendingMemorySpan {
    uint64_t address = 0, bytes = 0;
};
PendingMemorySpan pending_memory_span(const Pm4Command& command);
bool pending_memory_overlaps(PendingMemorySpan first, PendingMemorySpan second);
bool pending_memory_completion(const Pm4Command& command);
enum class PendingMemoryOverlay { Disjoint, Applied, Unresolved };
// Apply only captured/fixed bytes, in command order. A timestamp or an uncaptured copy cannot
// be guessed from stale guest memory. GDS offsets occupy a separate address domain.
PendingMemoryOverlay overlay_pending_memory(const Pm4Command& command,
                                            std::span<const uint8_t> captured_dma, uint64_t address,
                                            std::span<uint8_t> destination);
} // namespace prosper::gpu
