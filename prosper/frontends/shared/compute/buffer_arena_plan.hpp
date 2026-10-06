// buffer_arena_plan.hpp -- which guest windows of a read-only buffer share one resident copy?
//
// ADR 0010 (canonical resource identity): a guest allocation has one host resource, and a window of
// it is a view. The compute buffer cache was keyed by the V#'s own start address, so a guest that
// binds a constant ring through many windows at shifting bases (Black Flag's compute passes bind a
// 43 MiB window whose base advances a few hundred bytes per dispatch) got a separate resident copy
// per window, nearly identical to each other. Nine of them overflowed the 512 MiB cache, every
// acquisition missed, and each miss compared and re-uploaded ~43 MiB.
//
// This file decides the shape of the shared copy ("arena") as a pure function over plain numbers, so
// the rule is unit-tested without a device or a guest. The caller owns the proof that the arena is
// readable guest memory and that the binding is read-only.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace prosper::frontend {

struct BufferArenaExtent {
    uint64_t base = 0;
    uint64_t bytes = 0;
    uint64_t end() const { return base + bytes; }
    bool contains(uint64_t addr, uint64_t size) const {
        return addr >= base && size <= bytes && addr - base <= bytes - size;
    }
    bool overlaps(uint64_t addr, uint64_t size) const {
        return addr < end() && base < addr + size;
    }
};

// A window smaller than this keeps its own entry: its upload is cheap and sharing buys nothing.
inline constexpr uint64_t kBufferArenaMinWindowBytes = 1ull << 20;
// Headroom added on both sides of a new arena so windows drifting by a few MiB still fall inside it.
inline constexpr uint64_t kBufferArenaSlackBytes = 8ull << 20;
inline constexpr uint64_t kBufferArenaChunk = 1ull << 20;

enum class BufferArenaAction : uint8_t {
    Private,   // not shareable: use the window's own entry
    Reuse,     // `arena` already contains the window; bind it at `offset`
    Create,    // build `arena` (it replaces every extent in `replaces`), then bind at `offset`
};

struct BufferArenaDecision {
    BufferArenaAction action = BufferArenaAction::Private;
    BufferArenaExtent arena;
    uint64_t offset = 0;
    std::vector<BufferArenaExtent> replaces;
};

// `readable_*` bound the arena to guest memory the caller proved readable around the window (the
// window itself must be inside it). `offset_alignment` is the device's storage-buffer offset
// alignment: a window whose offset is not a multiple stays private.
inline BufferArenaDecision plan_buffer_arena(const std::vector<BufferArenaExtent>& existing,
                                             uint64_t window_addr, uint64_t window_bytes,
                                             uint64_t readable_lo, uint64_t readable_hi,
                                             uint64_t offset_alignment) {
    BufferArenaDecision decision;
    if (window_bytes < kBufferArenaMinWindowBytes || window_addr < readable_lo ||
        window_addr > readable_hi || window_bytes > readable_hi - window_addr || !offset_alignment)
        return decision;
    for (const BufferArenaExtent& arena : existing) {
        if (!arena.contains(window_addr, window_bytes)) continue;
        const uint64_t offset = window_addr - arena.base;
        if (offset % offset_alignment) return decision;
        decision.action = BufferArenaAction::Reuse;
        decision.arena = arena;
        decision.offset = offset;
        return decision;
    }
    // Build one arena over the window and every existing extent it touches, plus slack, clipped to
    // what is readable and rounded out to whole chunks.
    uint64_t lo = window_addr, hi = window_addr + window_bytes;
    for (const BufferArenaExtent& arena : existing)
        if (arena.overlaps(window_addr, window_bytes)) {
            lo = std::min(lo, arena.base);
            hi = std::max(hi, arena.end());
            decision.replaces.push_back(arena);
        }
    lo = lo > readable_lo + kBufferArenaSlackBytes ? lo - kBufferArenaSlackBytes : readable_lo;
    hi = readable_hi - hi > kBufferArenaSlackBytes ? hi + kBufferArenaSlackBytes : readable_hi;
    // Rounding to a chunk never leaves readable memory: round the base up only if it stays at or
    // below the window, and the end down only if it stays at or above it.
    const uint64_t lo_rounded = lo / kBufferArenaChunk * kBufferArenaChunk;
    if (lo_rounded >= readable_lo) lo = lo_rounded;
    const uint64_t hi_rounded = (hi + kBufferArenaChunk - 1) / kBufferArenaChunk * kBufferArenaChunk;
    if (hi_rounded <= readable_hi) hi = hi_rounded;
    const uint64_t offset = window_addr - lo;
    if (offset % offset_alignment) {
        decision.replaces.clear();
        return decision;
    }
    decision.action = BufferArenaAction::Create;
    decision.arena = {lo, hi - lo};
    decision.offset = offset;
    return decision;
}

}  // namespace prosper::frontend
