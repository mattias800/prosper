// buffer_arena_registry.hpp -- the arenas the compute buffer cache has decided to keep.
//
// The registry only remembers extents (and the recent windows that are evidence of overlap); the
// resident copy lives in the buffer cache under a key made from the extent, so an evicted arena is
// simply rebuilt on its next acquisition and the registry never has to hear about eviction. Both
// lists are bounded: the newest entries win. See buffer_arena_plan.hpp for what this is and is not.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "shared/compute/buffer_arena_plan.hpp"

namespace prosper::frontend {

class BufferArenaRegistry {
public:
    static constexpr size_t kMaxExtents = 64;
    static constexpr size_t kMaxRecentWindows = 32;

    // The cheap path: is the window already inside a registered arena? No guest probing needed.
    BufferArenaDecision find(uint64_t addr, uint64_t bytes, uint64_t offset_alignment) const {
        BufferArenaDecision decision;
        if (bytes < kBufferArenaMinWindowBytes || !offset_alignment) return decision;
        for (const BufferArenaExtent& arena : extents_) {
            if (!arena.contains(addr, bytes)) continue;
            if ((addr - arena.base) % offset_alignment) return decision;
            decision.action = BufferArenaAction::Reuse;
            decision.arena = arena;
            decision.offset = addr - arena.base;
            return decision;
        }
        return decision;
    }

    // Is there evidence this window belongs to a ring already seen? Without it no arena is built and
    // no guest memory is probed, so an isolated window costs nothing extra.
    bool has_evidence(uint64_t addr, uint64_t bytes) const {
        return bytes >= kBufferArenaMinWindowBytes &&
               buffer_arena_has_evidence(extents_, recent_, addr, bytes);
    }

    // The extent an arena for this window would have to cover: the window plus everything it shares a
    // ring with. The caller proves THIS range readable (not just the neighbourhood of the window), or
    // the arena would be clipped short of windows it should serve and a second arena would appear.
    BufferArenaExtent evidence_hull(uint64_t addr, uint64_t bytes) const {
        uint64_t lo = addr, hi = addr + bytes;
        for (const BufferArenaExtent& e : extents_)
            if (buffer_windows_share_a_ring(e, addr, bytes)) { lo = std::min(lo, e.base); hi = std::max(hi, e.end()); }
        for (const BufferArenaExtent& e : recent_)
            if (buffer_windows_share_a_ring(e, addr, bytes)) { lo = std::min(lo, e.base); hi = std::max(hi, e.end()); }
        return {lo, hi - lo};
    }

    // Remember a window that stayed private, so a later overlapping one is recognised.
    void remember(uint64_t addr, uint64_t bytes) {
        if (bytes < kBufferArenaMinWindowBytes) return;
        for (BufferArenaExtent& r : recent_)
            if (r.base == addr && r.bytes == bytes) return;
        if (recent_.size() >= kMaxRecentWindows) recent_.erase(recent_.begin());
        recent_.push_back({addr, bytes});
    }

    BufferArenaDecision plan(uint64_t addr, uint64_t bytes, uint64_t readable_lo,
                             uint64_t readable_hi, uint64_t offset_alignment) const {
        return plan_buffer_arena(extents_, recent_, addr, bytes, readable_lo, readable_hi,
                                 offset_alignment);
    }

    // Record a Create decision: the new arena replaces every extent it contains, and the recent
    // windows it now serves are forgotten.
    void commit(const BufferArenaDecision& decision) {
        if (decision.action != BufferArenaAction::Create) return;
        for (const BufferArenaExtent& gone : decision.replaces) erase(gone);
        for (size_t i = 0; i < recent_.size();)
            if (decision.arena.contains(recent_[i].base, recent_[i].bytes))
                recent_.erase(recent_.begin() + static_cast<std::ptrdiff_t>(i));
            else
                ++i;
        if (extents_.size() >= kMaxExtents) extents_.erase(extents_.begin());
        extents_.push_back(decision.arena);
    }

    void erase(const BufferArenaExtent& extent) {
        for (size_t i = 0; i < extents_.size(); ++i)
            if (extents_[i].base == extent.base && extents_[i].bytes == extent.bytes) {
                extents_.erase(extents_.begin() + static_cast<std::ptrdiff_t>(i));
                return;
            }
    }

    size_t size() const { return extents_.size(); }
    size_t recent_size() const { return recent_.size(); }

private:
    std::vector<BufferArenaExtent> extents_;
    std::vector<BufferArenaExtent> recent_;
};

// Decide how to bind one window. `readable(addr, bytes)` answers whether guest memory is readable;
// it is not called for a window with no overlap evidence. The returned decision's `replaces` lists
// the arenas a new one supersedes, so the caller can drop their cache entries.
template <class Readable>
BufferArenaDecision select_buffer_arena(BufferArenaRegistry& registry, uint64_t window, uint64_t span,
                                        uint64_t offset_alignment, Readable&& readable) {
    BufferArenaDecision found = registry.find(window, span, offset_alignment);
    if (found.action == BufferArenaAction::Reuse) {
        // A reused arena is re-probed: the slack around the window was proven readable when the arena
        // was created, and a level transition may have released it since.
        if (readable(found.arena.base, static_cast<uint32_t>(found.arena.bytes))) return found;
        registry.erase(found.arena);
    }
    if (!registry.has_evidence(window, span)) {
        registry.remember(window, span);
        return {};
    }
    const auto readable_slack = [&](uint64_t at, bool down) {
        for (uint64_t slack = kBufferArenaMaxSlackBytes; slack; slack >>= 1) {
            if (down ? at < slack : at > UINT64_MAX - slack) continue;
            if (readable(down ? at - slack : at, static_cast<uint32_t>(slack))) return slack;
        }
        return uint64_t{0};
    };
    // Prove the whole hull readable (not just the window's neighbourhood), then the headroom beyond it.
    const BufferArenaExtent hull = registry.evidence_hull(window, span);
    if (hull.bytes > kBufferArenaMaxBytes || !readable(hull.base, static_cast<uint32_t>(hull.bytes))) {
        registry.remember(window, span);
        return {};
    }
    const uint64_t down = readable_slack(hull.base, true);
    const uint64_t up = readable_slack(hull.end(), false);
    BufferArenaDecision plan = registry.plan(window, span, hull.base - down, hull.end() + up,
                                             offset_alignment);
    registry.commit(plan);
    if (plan.action == BufferArenaAction::Private) registry.remember(window, span);
    return plan;
}

// The cache key of an arena: its extent, with the materialization discriminator resized to match.
template <class Key>
Key buffer_arena_cache_key(const Key& window_key, const BufferArenaExtent& arena) {
    Key key = window_key;
    key.materialization.logical_bytes = key.materialization.binding_bytes = arena.bytes;
    key.gpu_addr = arena.base;
    key.host_data = 0;
    key.bytes = static_cast<uint32_t>(arena.bytes);
    return key;
}

}  // namespace prosper::frontend
