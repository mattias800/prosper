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
//
// WHERE IT RUNS. Not inside the descriptor fold. `resolve_dynamic_fetch`'s fold routes every
// environmental read through a FoldReader so a `.prfold` capture replays offline to the same outputs;
// the mapping table is not one of those recorded reads. So the fold only MARKS an over-cap raw use
// (`SrtUse::oversize_window`, with its V# unchanged), and `resolve_dynamic_fetch` resolves the marks
// against the live table after the fold returns: clamp, or drop the use exactly as it was dropped
// before this existed.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace prosper::gpu {

struct SrtUse;

// The two mapping queries the proof needs, injectable so the logic is testable without a live
// guest address space.
struct MappedMemoryProbe {
    // Longest leading range of [addr, addr + bytes) that is committed and CPU-readable.
    std::function<uint64_t(uint64_t addr, uint64_t bytes)> readable_prefix;
    // Is any byte of the page-sized range [addr, addr + bytes) tracked as a guest mapping?
    std::function<bool(uint64_t addr, uint64_t bytes)> any_mapped;
};

// Smallest guest mapping granule: the step at which the tail is scanned for islands. The scan probes
// both ends of every granule, so it assumes every tracked mapping is at least one granule long and
// granule aligned (guest mappings are 16 KiB pages). A mapping smaller than a granule lying strictly
// inside one would touch neither probe.
constexpr uint64_t kMappingGranule = 0x4000;

// The largest window the island scan will walk. The scan is linear in the window (two probes per
// granule, each a locked table lookup on the live path), so a window past this is refused rather
// than scanned. 4 GiB covers every window measured so far (2 GiB and 1000 MiB).
constexpr uint64_t kMaxScannedWindow = 0x100000000ull;

// A V#'s true window in bytes, computed in 64 bits from the raw words. `decode_buffer_descriptor`
// saturates `size_bytes` at 0xFFFFFFFF, so a whole-aperture descriptor (NUM_RECORDS 0xFFFFFFFF,
// stride 16: 64 GiB) reads there as a 4 GiB window; the proof must never use that value.
uint64_t buffer_descriptor_window_bytes(const uint32_t v[4]);

// On success returns true and sets `run_bytes` to the mapped leading run of [base, base + window).
// Returns false when the window needs no clamp (already within `cap`), is larger than
// kMaxScannedWindow, has no mapped bytes at all, has a leading run over `cap`, or maps anything after
// the leading run.
bool oversized_window_mapped_run(uint64_t base, uint64_t window_bytes, uint64_t cap,
                                 const MappedMemoryProbe& probe, uint64_t& run_bytes);

// The same proof, with the run rounded down to a multiple of `stride` (at least one record).
bool clamp_oversized_buffer_window(uint64_t base, uint64_t window_bytes, uint64_t stride,
                                   uint64_t cap, const MappedMemoryProbe& probe,
                                   uint64_t& clamped_bytes);

// The proof for a raw V#: its true 64-bit window, clamped to whole records. On success sets
// `clamped_records` to the NUM_RECORDS the descriptor should be published with.
bool clamp_oversized_buffer_descriptor(const uint32_t v[4], uint64_t cap,
                                       const MappedMemoryProbe& probe, uint32_t& clamped_records);

// The window proof against the live mapping table, cached per (base, window) until the table
// changes. The cached value is the mapped run, so descriptors with different strides share it.
bool clamp_oversized_buffer_window_live(uint64_t base, uint64_t window_bytes, uint64_t stride,
                                        uint64_t cap, uint64_t& clamped_bytes);

// Resolve the uses the fold marked `oversize_window` in uses[first..]: each is re-published with
// NUM_RECORDS clamped to its mapped run under `probe` (the live mapping table when omitted), or
// removed. Order of the surviving uses is preserved. Returns the number of uses clamped.
size_t resolve_oversized_buffer_windows(std::vector<SrtUse>& uses, size_t first);
size_t resolve_oversized_buffer_windows(std::vector<SrtUse>& uses, size_t first,
                                        const MappedMemoryProbe& probe);

}   // namespace prosper::gpu
