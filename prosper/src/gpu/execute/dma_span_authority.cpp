#include "gpu/execute/dma_span_authority.hpp"
#include "diagnostics/exit_census.hpp"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>

namespace prosper::gpu {
namespace {
// [a, a + a_bytes) and [b, b + b_bytes) share at least one byte. Saturates rather than wraps.
bool ranges_overlap(uint64_t a, uint64_t a_bytes, uint64_t b, uint64_t b_bytes) {
    if (!a_bytes || !b_bytes) return false;
    const uint64_t a_end = a > UINT64_MAX - a_bytes ? UINT64_MAX : a + a_bytes;
    const uint64_t b_end = b > UINT64_MAX - b_bytes ? UINT64_MAX : b + b_bytes;
    return a < b_end && b < a_end;
}

enum class Reason { Plain, SourceOverlap, DestinationOverlap, UnprovedExtent, NamedBase, Count };

// Why each copy did or did not take the readback, printed once at exit (opt-out
// PROSPER_NO_DMA_AUTHORITY_CENSUS). Without it a title still paying for the readback cannot say
// whether its copies really touch rendered targets or only look like they might.
struct Census {
    std::atomic<uint64_t> counts[static_cast<size_t>(Reason::Count)]{};
};
Census& census() {
    static Census* value = new Census();   // never destroyed: read by the exit report
    static std::once_flag registered;
    std::call_once(registered, [] {
        prosper::diagnostics::register_census("PROSPER_NO_DMA_AUTHORITY_CENSUS", [] {
            const auto at = [](Reason r) {
                return value->counts[static_cast<size_t>(r)].load(std::memory_order_relaxed);
            };
            uint64_t total = 0;
            for (const auto& c : value->counts) total += c.load(std::memory_order_relaxed);
            if (!total) return false;
            std::fprintf(stderr,
                         "[dma-authority] RUN TOTAL copies=%llu plain=%llu authoritative: "
                         "src-overlap=%llu dst-overlap=%llu unproved-extent=%llu named-base=%llu\n",
                         (unsigned long long)total, (unsigned long long)at(Reason::Plain),
                         (unsigned long long)at(Reason::SourceOverlap),
                         (unsigned long long)at(Reason::DestinationOverlap),
                         (unsigned long long)at(Reason::UnprovedExtent),
                         (unsigned long long)at(Reason::NamedBase));
            return true;
        });
    });
    return *value;
}

Reason classify(const std::vector<DrawItem>& span, uint64_t dst, uint64_t src, uint32_t bytes,
                uint32_t sels) {
    const bool source_gds = ((sels >> 8u) & 0xffu) == 1u;
    const bool destination_gds = (sels & 0xffu) == 1u;
    for (const DrawItem& draw : span) {
        for (uint32_t slot = 0; slot < draw.color_targets.size(); ++slot) {
            const auto& target = draw.color_targets[slot];
            if (!target.base) continue;
            const uint64_t extent = target.raw_snapshot_footprint_bytes;
            if (!extent) return Reason::UnprovedExtent;
            if (!source_gds && ranges_overlap(src, bytes, target.base, extent))
                return Reason::SourceOverlap;
            if (!destination_gds && ranges_overlap(dst, bytes, target.base, extent))
                return Reason::DestinationOverlap;
        }
        // The named slots are what the renderer reads back; a disagreeing binding is unproved.
        if ((draw.color0_base && draw.color0_base != draw.color_targets[0].base) ||
            (draw.color1_base && draw.color1_base != draw.color_targets[1].base))
            return Reason::NamedBase;
    }
    return Reason::Plain;
}
} // namespace

bool dma_needs_authoritative_span(const std::vector<DrawItem>& span, uint64_t dst, uint64_t src,
                                  uint32_t bytes, uint32_t sels) {
    const Reason reason = classify(span, dst, src, bytes, sels);
    census().counts[static_cast<size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
    return reason != Reason::Plain;
}
} // namespace prosper::gpu
