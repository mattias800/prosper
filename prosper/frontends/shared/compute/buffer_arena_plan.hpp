// buffer_arena_plan.hpp -- which guest windows of a read-only buffer share one resident copy?
//
// Interim mechanism, not ADR 0010. ADR 0010 (proposed) wants canonical identity per guest allocation,
// tracked at page granularity in guest/memory. This is a smaller step inside the compute buffer cache
// that fixes one measured case: a guest that binds a constant ring through many windows at shifting
// bases (Black Flag's compute passes bind a 43 MiB window whose base advances a few hundred bytes per
// dispatch) got a separate resident copy per window, nearly identical to each other. Nine of them
// overflowed the 512 MiB cache, every acquisition missed, and each miss compared and re-uploaded
// ~43 MiB. It is keyed by address proximity, so it can span unrelated allocations; it still validates
// by byte comparison. It does not implement PERF-P9.
//
// The plan is a pure function over plain numbers, so the rule is unit-tested without a device or a
// guest. The caller owns the proofs that the arena is readable guest memory and that the binding is
// read-only AND that no binding of the same dispatch that aliases it is writable.
//
// Cost rule (PERF-P5): per-dispatch cost grows with the arena, so an arena is created only on
// evidence of overlap (a second window overlapping a window already seen), its headroom is a fraction
// of what it covers rather than a constant, and it never grows past twice the largest window it serves.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace prosper::frontend {

struct BufferArenaExtent {
    uint64_t base = 0;
    uint64_t bytes = 0;
    // The largest single window this extent serves (0 = it is itself one window). Bounds how far a
    // chain of sliding windows may grow an arena.
    uint64_t largest_window = 0;
    uint64_t serves() const { return largest_window ? largest_window : bytes; }
    uint64_t end() const { return base + bytes; }
    bool contains(uint64_t addr, uint64_t size) const {
        return addr >= base && size <= bytes && addr - base <= bytes - size;
    }
    bool overlaps(uint64_t addr, uint64_t size) const {
        return addr < end() && base < addr + size;
    }
    // Bytes shared with [addr, addr+size).
    uint64_t overlap_bytes(uint64_t addr, uint64_t size) const {
        if (!overlaps(addr, size)) return 0;
        return std::min(end(), addr + size) - std::max(base, addr);
    }
};

// A window smaller than this keeps its own entry: its upload is cheap and sharing buys nothing.
inline constexpr uint64_t kBufferArenaMinWindowBytes = 1ull << 20;
// Upper bound of the headroom added on each side; the actual headroom is hull/16 up to this.
inline constexpr uint64_t kBufferArenaMaxSlackBytes = 8ull << 20;
inline constexpr uint64_t kBufferArenaChunk = 64ull << 10;
// The cache key stores an arena's size in 32 bits.
inline constexpr uint64_t kBufferArenaMaxBytes = 0xffffffffull;

enum class BufferArenaAction : uint8_t {
    Private,   // not shareable (or no evidence yet): use the window's own entry
    Reuse,     // `arena` already contains the window; bind it at `offset`
    Create,    // build `arena` (it replaces every extent in `replaces`), then bind at `offset`
};

struct BufferArenaDecision {
    BufferArenaAction action = BufferArenaAction::Private;
    BufferArenaExtent arena;
    uint64_t offset = 0;
    std::vector<BufferArenaExtent> replaces;   // extents the new arena fully contains
};

// Two windows of one ring overlap by at least half of the smaller one (a sliding window, not a
// neighbour that merely touches).
inline bool buffer_windows_share_a_ring(const BufferArenaExtent& a, uint64_t addr, uint64_t bytes) {
    return a.overlap_bytes(addr, bytes) * 2 >= std::min(a.bytes, bytes) && a.overlap_bytes(addr, bytes) != 0;
}

// Is there evidence that `window` belongs to a ring already seen (an arena or a recent window that
// it overlaps by at least half)?
inline bool buffer_arena_has_evidence(const std::vector<BufferArenaExtent>& arenas,
                                      const std::vector<BufferArenaExtent>& recent,
                                      uint64_t addr, uint64_t bytes) {
    for (const BufferArenaExtent& a : arenas)
        if (buffer_windows_share_a_ring(a, addr, bytes)) return true;
    for (const BufferArenaExtent& r : recent)
        if (buffer_windows_share_a_ring(r, addr, bytes)) return true;
    return false;
}

// `readable_*` bound the arena to guest memory the caller proved readable around the window (the
// window itself must be inside it). `offset_alignment` is the device's storage-buffer offset
// alignment: a window whose offset is not a multiple stays private.
inline BufferArenaDecision plan_buffer_arena(const std::vector<BufferArenaExtent>& arenas,
                                             const std::vector<BufferArenaExtent>& recent,
                                             uint64_t window_addr, uint64_t window_bytes,
                                             uint64_t readable_lo, uint64_t readable_hi,
                                             uint64_t offset_alignment) {
    BufferArenaDecision decision;
    if (window_bytes < kBufferArenaMinWindowBytes || window_addr < readable_lo ||
        window_addr > readable_hi || window_bytes > readable_hi - window_addr || !offset_alignment)
        return decision;
    for (const BufferArenaExtent& arena : arenas) {
        if (!arena.contains(window_addr, window_bytes)) continue;
        const uint64_t offset = window_addr - arena.base;
        if (offset % offset_alignment) return decision;
        decision.action = BufferArenaAction::Reuse;
        decision.arena = arena;
        decision.offset = offset;
        return decision;
    }
    // The hull of the window and everything it shares a ring with.
    uint64_t lo = window_addr, hi = window_addr + window_bytes;
    uint64_t largest = window_bytes;
    bool evidence = false;
    for (const BufferArenaExtent& e : arenas)
        if (buffer_windows_share_a_ring(e, window_addr, window_bytes)) {
            lo = std::min(lo, e.base); hi = std::max(hi, e.end());
            largest = std::max(largest, e.serves());
            evidence = true;
        }
    for (const BufferArenaExtent& e : recent)
        if (buffer_windows_share_a_ring(e, window_addr, window_bytes)) {
            lo = std::min(lo, e.base); hi = std::max(hi, e.end());
            largest = std::max(largest, e.serves());
            evidence = true;
        }
    if (!evidence) return decision;
    uint64_t hull = hi - lo;
    // Never serve a ring through an arena more than twice the largest window it covers.
    if (hull > 2 * largest || hull > kBufferArenaMaxBytes) return decision;
    const uint64_t slack = std::min(hull / 16, kBufferArenaMaxSlackBytes);
    // Headroom, then clip BOTH ends to what is proven readable. Every comparison is written so an
    // end beyond the readable range cannot wrap: an extent that reached past readable memory is
    // clipped back to it, never extended.
    lo = lo > readable_lo && lo - readable_lo > slack ? lo - slack : readable_lo;
    if (lo < readable_lo) lo = readable_lo;
    hi = readable_hi > hi && readable_hi - hi > slack ? hi + slack : readable_hi;
    if (hi > readable_hi) hi = readable_hi;
    // Whole chunks, but only while that stays inside the readable range.
    const uint64_t lo_rounded = lo / kBufferArenaChunk * kBufferArenaChunk;
    if (lo_rounded >= readable_lo) lo = lo_rounded;
    const uint64_t hi_rounded = (hi + kBufferArenaChunk - 1) / kBufferArenaChunk * kBufferArenaChunk;
    if (hi_rounded <= readable_hi) hi = hi_rounded;
    if (hi <= lo || hi - lo > kBufferArenaMaxBytes) return decision;
    // The window is inside the readable range, so it is inside [lo, hi) by construction.
    const uint64_t offset = window_addr - lo;
    if (offset % offset_alignment) return decision;
    decision.action = BufferArenaAction::Create;
    decision.arena = {lo, hi - lo, largest};
    decision.offset = offset;
    // Replace only extents the new arena actually contains: one clipped by the readable range stays.
    for (const BufferArenaExtent& e : arenas)
        if (decision.arena.contains(e.base, e.bytes)) decision.replaces.push_back(e);
    return decision;
}

// Which bindings of a dispatch may be served from an arena? A binding that aliases (same guest
// range) any WRITABLE binding of the dispatch may not: the shader would write through the shared
// arena while the writeback and result paths address the window at offset zero. Pure over plain keys
// so the aliasing rule is tested without a device.
struct BufferAliasKey {
    uint64_t gpu_addr = 0;
    uint64_t size = 0;
    uintptr_t host_data = 0;
    uint64_t host_data_size = 0;
    bool writable = false;
};
inline bool buffer_alias_group_has_writer(const std::vector<BufferAliasKey>& bindings, size_t i) {
    for (const BufferAliasKey& other : bindings)
        if (other.writable && other.gpu_addr == bindings[i].gpu_addr &&
            other.size == bindings[i].size && other.host_data == bindings[i].host_data &&
            other.host_data_size == bindings[i].host_data_size)
            return true;
    return false;
}

}  // namespace prosper::frontend
