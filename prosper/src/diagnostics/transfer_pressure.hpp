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
// So this is deliberately NOT another opt-in census. It runs by default and prints one line at end
// of run. Its per-site cost is two relaxed `fetch_add`s plus a `clock_gettime` and two more relaxed
// operations to maintain the window -- the clock read dominates, so call it once per transfer, not
// once per row or per texel. A number nobody is looking at is not a
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
    StorageMaterialize = 0,   // guest storage images copied/unpacked host-side for a dispatch
    BufferUpload,   // compute buffer bindings uploaded to the device
    BufferCompare,   // bytes read only to decide whether something changed
    RenderTargetSnapshot,   // CPU copies of render targets for later sampling
    Detile,   // CPU detiling of tiled guest surfaces
    // The guest's front (scanout) buffer read back to the CPU: the copy out of guest memory plus
    // its de-swizzle (videoout_read_front_linear). Its own site so a host-copy alarm on the CPU
    // present fallback names the present path instead of reading as texture detiling (#3891).
    GuestScanout,
    // A draw's buffer binding (vertex, constant or storage data) copied into a zero-filled host
    // vector because no view of the guest's own memory could be borrowed for it, and then uploaded
    // from there. The whole vector is charged: it is allocated and zeroed whatever part of it
    // the guest's bytes fill (and uploaded unless they fill none). Per DRAW: MOUSE: P.I. For Hire's
    // menus staged 9.4 MiB for each of three vertex attributes of every draw and no site showed it.
    DrawBufferStage,
    Count
};

const char* transfer_name(Transfer category);

// Two relaxed atomic adds, a monotonic clock read, and two more relaxed operations for the window.
// Safe from any thread and during static initialisation.
void note_transfer(Transfer category, uint64_t bytes);

// Process-lifetime totals, for tests and tools.
uint64_t transfer_bytes(Transfer category);
// Process-lifetime count of note_transfer calls that carried bytes, per category: with
// transfer_bytes it gives bytes per call, which separates "more copies" from "bigger copies".
uint64_t transfer_calls(Transfer category);

// While one of these is alive, every note_transfer on THIS thread is charged to `category`,
// whatever category the reporting site passed. For a caller that reuses a shared helper which
// reports under its own generic category (detile_surface reports Detile) but whose copy belongs to
// a different reader: the guest-scanout read wraps its de-swizzle in one. Nests; the innermost wins.
class TransferAttributionScope {
public:
    explicit TransferAttributionScope(Transfer category);
    ~TransferAttributionScope();
    TransferAttributionScope(const TransferAttributionScope&) = delete;
    TransferAttributionScope& operator=(const TransferAttributionScope&) = delete;

private:
    Transfer previous_;
};

}  // namespace prosper::diagnostics
