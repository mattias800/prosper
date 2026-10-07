// unorm10_snapshot.hpp -- unpack a guest R10G10B10A2 UNORM linear image into the RGBA8 bytes the
// renderer keeps for the same render target.
//
// A compute pass may write a display buffer as 2_10_10_10 (R in bits [9:0], A in bits [31:30]).
// The renderer stores a 10:10:10:2 colour target as RGBA8 (backend_color_format falls back to it),
// so a result in the packed layout cannot be copied in; it is narrowed the same way a sampled
// upload of this format already is.
//
// COST. A sampled Black Flag capture put this conversion at 10.5% of the submit thread: 84% in the
// per-channel `(v * 255 + 511) / 1023` loop over every texel of a 1920x1080 target, and 15% zeroing
// the freshly allocated 8 MB output vector (the sampled stacks do not separate the page faults of that
// allocation, which on Windows come with `VirtualAlloc`). The output is therefore written into a pooled
// buffer that is neither reallocated nor zero-filled on a steady-state submit -- that is where most of the
// gain comes from -- and the conversion is a lookup table built from the original expression (exact by
// construction), which speeds the loop itself modestly: on an optimised (-O3) build the compiler already
// vectorises the constant division, so the table is worth roughly 1.2x on the loop, more at -O2.
#pragma once
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "diagnostics/env_cache.hpp"
#include "diagnostics/env_numeric.hpp"
#include "diagnostics/exit_reports.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "shared/live/cpu_rtt_snapshot_pool.hpp"

namespace prosper::frontend {

// One 10-bit channel to 8 bits, rounded to nearest. This is the original per-channel expression;
// the tables below are computed from it, so there is exactly one definition of the rounding.
constexpr uint8_t unorm10_channel_to_8(uint32_t v) { return static_cast<uint8_t>((v * 255U + 511U) / 1023U); }
// The 2-bit alpha to 8 bits (0, 85, 170, 255).
constexpr uint8_t unorm2_alpha_to_8(uint32_t a) { return static_cast<uint8_t>((a * 255U + 1U) / 3U); }

namespace detail {
constexpr std::array<uint8_t, 1024> make_unorm10_table() {
    std::array<uint8_t, 1024> table{};
    for (uint32_t v = 0; v < 1024; ++v) table[v] = unorm10_channel_to_8(v);
    return table;
}
constexpr std::array<uint8_t, 4> make_unorm2_table() {
    std::array<uint8_t, 4> table{};
    for (uint32_t a = 0; a < 4; ++a) table[a] = unorm2_alpha_to_8(a);
    return table;
}
}  // namespace detail

inline constexpr std::array<uint8_t, 1024> kUnorm10To8 = detail::make_unorm10_table();
inline constexpr std::array<uint8_t, 4> kUnorm2To8 = detail::make_unorm2_table();

// The one validation both entry points share: a non-empty, non-overflowing extent whose packed size is
// exactly width*height words. Returns false (and leaves `texels` alone) otherwise.
inline bool unorm10_texel_count(const uint8_t* packed, size_t packed_bytes, uint64_t width, uint64_t height,
                                size_t& texels) {
    if (!packed || !width || !height || width > SIZE_MAX / height) return false;
    const size_t count = static_cast<size_t>(width * height);
    if (count > SIZE_MAX / 4 || packed_bytes != count * 4) return false;
    texels = count;
    return true;
}

// Unpack `texels` packed words into `texels * 4` RGBA8 bytes. Neither buffer is checked; the caller
// owns both sizes. `out` is fully overwritten (every byte), so it need not be zeroed first.
inline void unpack_unorm10_words_to_rgba8(const uint8_t* packed, size_t texels, uint8_t* out) {
    for (size_t t = 0; t < texels; ++t) {
        uint32_t word;
        std::memcpy(&word, packed + t * 4, sizeof(word));
        const uint32_t r = kUnorm10To8[word & 0x3ffU];
        const uint32_t g = kUnorm10To8[(word >> 10) & 0x3ffU];
        const uint32_t b = kUnorm10To8[(word >> 20) & 0x3ffU];
        const uint32_t a = kUnorm2To8[word >> 30];
        if constexpr (std::endian::native == std::endian::little) {
            const uint32_t rgba = r | (g << 8) | (b << 16) | (a << 24);
            std::memcpy(out + t * 4, &rgba, sizeof(rgba));
        } else {
            out[t * 4 + 0] = static_cast<uint8_t>(r);
            out[t * 4 + 1] = static_cast<uint8_t>(g);
            out[t * 4 + 2] = static_cast<uint8_t>(b);
            out[t * 4 + 3] = static_cast<uint8_t>(a);
        }
    }
}

// Returns false when `bytes` is not exactly width*height packed words.
inline bool unpack_unorm10_to_rgba8(const uint8_t* packed, size_t packed_bytes, uint64_t width,
                                    uint64_t height, std::vector<uint8_t>& rgba8) {
    size_t texels = 0;
    if (!unorm10_texel_count(packed, packed_bytes, width, height, texels)) return false;
    rgba8.resize(texels * 4);
    unpack_unorm10_words_to_rgba8(packed, texels, rgba8.data());
    return true;
}

// This pool's accounting, reported on its own line so it is not merged into the `[rtt-pool]` figure of
// the sibling linear-storage pool.
inline CpuRttSnapshotPoolStats& unorm10_pool_stats() {
    static CpuRttSnapshotPoolStats stats;
    static const bool once = [] {
        prosper::diagnostics::register_exit_report(
            [] { print_cpu_rtt_snapshot_pool_stats("rtt-pool-unorm10", unorm10_pool_stats()); });
        return true;
    }();
    (void)once;
    return stats;
}

// Publish an exact RGB10A2 compute result into the renderer target at `addr` as RGBA8. No renderer
// pixel format carries 10:10:10:2, so without this the target kept its old pixels under a picture the
// compute chain had composited (a flipped display buffer stayed black).
//
// The output buffer comes from a pool: the renderer holds the pixels until its last consumer releases
// them, and only then does the buffer return, so a retained older version is never overwritten. A
// steady-state submit reuses a buffer of the same size, so it neither allocates nor zero-fills. The pool
// honours the switches that govern the sibling snapshot pool (PROSPER_NO_CPU_RTT_SNAPSHOT_POOL and
// PROSPER_CPU_RTT_SNAPSHOT_POOL_MB), so disabling "the snapshot pool" disables this one too.
inline void publish_unorm10_as_rgba8(uint64_t addr, uint32_t width, uint32_t height,
                                     const uint8_t* packed, size_t packed_bytes) {
    size_t texels = 0;
    if (!unorm10_texel_count(packed, packed_bytes, width, height, texels)) return;
    static const bool pooled = !PROSPER_ENV_ON("PROSPER_NO_CPU_RTT_SNAPSHOT_POOL");
    static CpuRttSnapshotPool pool(
        static_cast<size_t>(prosper::diag::env_u64_or_default_capped(
            "PROSPER_CPU_RTT_SNAPSHOT_POOL_MB", PROSPER_ENV_VALUE("PROSPER_CPU_RTT_SNAPSHOT_POOL_MB"), 128ULL,
            SIZE_MAX / (1024ULL * 1024ULL), "MiB") * 1024ULL * 1024ULL),
        4, &unorm10_pool_stats());
    CpuRttSnapshot snapshot = build_cpu_rtt_snapshot(pool, pooled, texels * 4, [&](uint8_t* out) {
        unpack_unorm10_words_to_rgba8(packed, texels, out);
    });
    if (!snapshot.pixels) return;
    prosper::gpu::notify_live_render_target_image_written(
        {addr, width, height, prosper::gpu::LiveTargetPixelFormat::Rgba8Unorm, std::move(snapshot.pixels)});
}

} // namespace prosper::frontend
