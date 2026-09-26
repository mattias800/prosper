#pragma once
// Host-side bytes moved per second, reported every run, loudly when it is absurd.
//
// WHY THIS EXISTS. This class of defect was found four separate times in one session, and every
// time it cost a hunt that should have taken seconds:
//
//   * a CPU render-target snapshot pool that matched on exact size instead of capacity, so it
//     missed 61.5% of the time and re-allocated -- 56,543 MiB of allocate-and-copy per two
//     minutes on one title;
//   * compute buffer bindings below a 1 MiB residency threshold, every one reporting
//     `cache=ineligible total-watch-chunks=0`, so a full compare AND a full upload on every
//     dispatch even when nothing changed;
//   * storage-image bindings copied host-side into an upload buffer on every dispatch -- 37.6 GiB
//     in 50 seconds on a screen that was not changing;
//   * guest buffer residency answered by `memcmp` against a CPU snapshot, so a cache HIT still
//     read 2x bytes to conclude nothing had changed.
//
// Each was invisible because the cost is DIFFUSE. No single call is slow, no profile leaf is
// damning -- `__memmove_avx512` at 11% looks like ordinary work -- and a frame-time breakdown
// attributes it to whatever phase happens to contain it. What makes it obvious is the aggregate
// nobody was computing: **bytes per second**. 770 MiB/s of host copying on a static splash screen
// is self-evidently wrong the moment anyone sees the number, and nobody had to see it.
//
// So this is deliberately NOT another opt-in census. It runs by default, costs two relaxed atomic
// adds per site, and prints one line at end of run. A number nobody is looking at is not a
// measurement, and the four defects above are what that costs.
//
// WHAT IT IS NOT. It does not measure GPU traffic, DMA, or anything the guest does -- only bytes
// this process moves with the CPU on behalf of the guest, at sites that opted in by calling
// `note_transfer`. A site that never reports is invisible here, so a LOW total means "the sites
// that report are quiet", never "nothing is moving". Add a site when you find one that matters;
// the categories are deliberately coarse so the line stays readable.
//
// `PROSPER_NO_TRANSFER_PRESSURE=1` silences it. `PROSPER_TRANSFER_PRESSURE_WARN_MIBPS=<n>`
// changes the threshold above which the line is prefixed `HIGH` (default 256 MiB/s, which every
// healthy title measured so far sits far below and every defect above sits far above).

#include <cstdint>

namespace prosper::diagnostics {

// Coarse on purpose. A category exists when a reader would act differently on seeing it.
enum class Transfer : unsigned char {
    StorageMaterialize = 0,  // guest storage images copied/unpacked host-side for a dispatch
    BufferUpload,            // compute buffer bindings uploaded to the device
    BufferCompare,           // bytes read only to decide whether something changed
    RenderTargetSnapshot,    // CPU copies of render targets for later sampling
    Detile,                  // CPU detiling of tiled guest surfaces
    Count
};

const char* transfer_name(Transfer category);

// Two relaxed atomic adds. Safe from any thread and during static initialisation.
void note_transfer(Transfer category, uint64_t bytes);

// Process-lifetime totals, for tests and tools.
uint64_t transfer_bytes(Transfer category);

}  // namespace prosper::diagnostics
