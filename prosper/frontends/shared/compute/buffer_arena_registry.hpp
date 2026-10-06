// buffer_arena_registry.hpp -- the arenas the compute buffer cache has decided to keep (ADR 0010).
//
// The registry only remembers extents; the resident copy lives in the buffer cache under a key made
// from the extent, so an evicted arena is simply rebuilt on its next acquisition and the registry
// never has to hear about eviction. It is bounded: the newest extents win.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "shared/compute/buffer_arena_plan.hpp"

namespace prosper::frontend {

class BufferArenaRegistry {
public:
    static constexpr size_t kMaxExtents = 64;

    BufferArenaDecision plan(uint64_t addr, uint64_t bytes, uint64_t readable_lo,
                             uint64_t readable_hi, uint64_t offset_alignment) const {
        return plan_buffer_arena(extents_, addr, bytes, readable_lo, readable_hi, offset_alignment);
    }

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

    // Record a Create decision: the new arena replaces every extent it subsumes.
    void commit(const BufferArenaDecision& decision) {
        if (decision.action != BufferArenaAction::Create) return;
        for (const BufferArenaExtent& gone : decision.replaces)
            for (size_t i = 0; i < extents_.size(); ++i)
                if (extents_[i].base == gone.base && extents_[i].bytes == gone.bytes) {
                    extents_.erase(extents_.begin() + static_cast<std::ptrdiff_t>(i));
                    break;
                }
        if (extents_.size() >= kMaxExtents) extents_.erase(extents_.begin());
        extents_.push_back(decision.arena);
    }

    size_t size() const { return extents_.size(); }

private:
    std::vector<BufferArenaExtent> extents_;
};

// Decide where a read-only window lives: find it in a known arena, else probe the readable slack
// around it (`readable(addr, bytes)`), plan and commit a new arena. Private means "own copy".
template <class Readable>
BufferArenaDecision select_buffer_arena(BufferArenaRegistry& registry, uint64_t window, uint64_t span,
                                        uint64_t offset_alignment, Readable&& readable) {
    BufferArenaDecision plan = registry.find(window, span, offset_alignment);
    if (plan.action != BufferArenaAction::Private) return plan;
    const auto slack_at = [&](uint64_t at, bool down) {
        for (uint64_t slack = kBufferArenaSlackBytes; slack; slack >>= 1) {
            if (down ? at < slack : at > UINT64_MAX - slack) continue;
            if (readable(down ? at - slack : at, static_cast<uint32_t>(slack))) return slack;
        }
        return uint64_t{0};
    };
    const uint64_t down = slack_at(window, true);
    const uint64_t up = slack_at(window + span, false);
    plan = registry.plan(window, span, window - down, window + span + up, offset_alignment);
    registry.commit(plan);
    return plan;
}

// The cache key of an arena: its extent, with the materialization discriminator resized to match.
template <class Key>
Key buffer_arena_cache_key(const Key& window_key, const BufferArenaDecision& plan) {
    Key key = window_key;
    key.materialization.logical_bytes = key.materialization.binding_bytes = plan.arena.bytes;
    key.gpu_addr = plan.arena.base;
    key.host_data = 0;
    key.bytes = static_cast<uint32_t>(plan.arena.bytes);
    return key;
}

}  // namespace prosper::frontend
