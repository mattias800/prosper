// oversize_buffer_window.hpp -- how large is the part of an over-cap buffer window that is MAPPED?
//
// A buffer V# can declare a window far larger than anything the renderer will copy (the cap is
// 256 MiB): a "view of guest memory" descriptor whose NUM_RECORDS spans a whole aperture. What the GPU
// can touch inside such a window is only what is mapped; an access to an unmapped page faults on the
// console and cannot have been issued by a working guest. So when the mapped part is one contiguous
// run from the base and NOTHING else in the window is mapped, the window is exactly that run, and the
// descriptor can be published with its NUM_RECORDS clamped to it.
//
// This is a proof about the mapping table at one instant, not a size heuristic: any mapped page after
// the leading run (an island), or a leading run that is itself over the cap, returns "no clamp" and the
// descriptor stays refused as before.
#pragma once

#include <cstdint>
#include <functional>

namespace prosper::gpu {

// The two mapping queries the proof needs, injectable so the logic is testable without a live
// guest address space.
struct MappedMemoryProbe {
    // Longest leading range of [addr, addr + bytes) that is committed and CPU-readable.
    std::function<uint64_t(uint64_t addr, uint64_t bytes)> readable_prefix;
    // Is any byte of the page-sized range [addr, addr + bytes) tracked as a guest mapping?
    std::function<bool(uint64_t addr, uint64_t bytes)> any_mapped;
};

// Smallest guest mapping granule: the step at which the tail is scanned for islands.
constexpr uint64_t kMappingGranule = 0x4000;

// On success returns true and sets `clamped_bytes` to the mapped leading run, rounded down to a multiple
// of `stride` (at least one record). Returns false when the window needs no clamp (already within `cap`),
// has no mapped bytes at all, has a leading run over `cap`, or maps anything after the leading run.
bool clamp_oversized_buffer_window(uint64_t base, uint64_t window_bytes, uint64_t stride,
                                   uint64_t cap, const MappedMemoryProbe& probe,
                                   uint64_t& clamped_bytes);

// The same proof against the live mapping table, cached per (base, window) until the table changes.
bool clamp_oversized_buffer_window_live(uint64_t base, uint64_t window_bytes, uint64_t stride,
                                        uint64_t cap, uint64_t& clamped_bytes);

} // namespace prosper::gpu
