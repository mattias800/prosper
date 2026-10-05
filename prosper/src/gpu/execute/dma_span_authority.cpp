#include "gpu/execute/dma_span_authority.hpp"
#include "diagnostics/exit_census.hpp"
#include <algorithm>
#include <array>
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
// PROSPER_NO_DMA_AUTHORITY_CENSUS, a diagnostic). Without it a title still paying for the readback
// cannot say whether its copies really touch rendered targets or only look like they might.
// `plain` means "no readback requested", which includes copies with no pending span at all.
struct Census {
    std::atomic<uint64_t> counts[static_cast<size_t>(Reason::Count)]{};
    // The copies with no bound on a target's extent, by that target's exact-extent refusal (#4457),
    // and by tile mode for the tile-mode refusals. A binding mirrored from a named alias alone
    // carries None.
    std::array<std::atomic<uint64_t>, static_cast<size_t>(ColorExtentRefusal::Count)> unproved{};
    std::array<std::atomic<uint64_t>, 32> unproved_tile_mode{};
};

void print_unproved_breakdown(const Census& c) {
    std::array<char, 512> line{};
    int at = std::snprintf(line.data(), line.size(), "[dma-authority] unproved-extent by refusal:");
    for (size_t r = 0; r < c.unproved.size(); ++r) {
        const uint64_t n = c.unproved[r].load(std::memory_order_relaxed);
        if (!n || at <= 0 || static_cast<size_t>(at) >= line.size()) continue;
        const auto reason = static_cast<ColorExtentRefusal>(r);
        at += std::snprintf(line.data() + at, line.size() - static_cast<size_t>(at), " %s=%llu",
                            reason == ColorExtentRefusal::None ? "not-derived"
                                                               : color_extent_refusal_name(reason),
                            (unsigned long long)n);
    }
    for (size_t mode = 0; mode < c.unproved_tile_mode.size(); ++mode) {
        const uint64_t n = c.unproved_tile_mode[mode].load(std::memory_order_relaxed);
        if (!n || at <= 0 || static_cast<size_t>(at) >= line.size()) continue;
        at += std::snprintf(line.data() + at, line.size() - static_cast<size_t>(at),
                            " sw_mode%zu=%llu", mode, (unsigned long long)n);
    }
    std::fprintf(stderr, "%s\n", line.data());
}
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
            if (at(Reason::UnprovedExtent)) print_unproved_breakdown(*value);
            return true;
        });
    });
    return *value;
}

Reason count_unproved(const DrawItem::ColorTargetBinding& target) {
    Census& c = census();
    c.unproved[static_cast<size_t>(target.footprint_refusal)].fetch_add(1,
                                                                        std::memory_order_relaxed);
    if (target.footprint_refusal == ColorExtentRefusal::TileMode)
        c.unproved_tile_mode[target.tile_mode & 31u].fetch_add(1, std::memory_order_relaxed);
    return Reason::UnprovedExtent;
}

Reason classify(const std::vector<DrawItem>& span, uint64_t dst, uint64_t src, uint32_t bytes,
                uint32_t sels) {
    // PROSPER_DMA_EXACT_EXTENT_ONLY: the #4457 A/B control. It ignores the conservative bound, so
    // every target without a proved exact extent counts as touched, as before.
    // NOLINTNEXTLINE(concurrency-mt-unsafe): one process-lifetime read; nothing sets it mid-run.
    static const bool exact_only = PROSPER_ENV_ON("PROSPER_DMA_EXACT_EXTENT_ONLY");
    const bool source_gds = ((sels >> 8u) & 0xffu) == 1u;
    const bool destination_gds = (sels & 0xffu) == 1u;
    for (const DrawItem& draw : span) {
        for (uint32_t slot = 0; slot < draw.color_targets.size(); ++slot) {
            const auto& target = draw.color_targets[slot];
            if (!target.base) continue;
            // The bound is never below the exact extent; taking both keeps a binding that carries
            // only the exact one (a hand-built DrawItem) proved.
            const uint64_t extent = exact_only ? target.raw_snapshot_footprint_bytes
                                               : std::max(target.footprint_bound_bytes,
                                                          target.raw_snapshot_footprint_bytes);
            if (!extent) return count_unproved(target);
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
