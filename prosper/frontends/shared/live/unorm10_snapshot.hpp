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
// the freshly allocated 8 MB output vector. So the conversion is a lookup table built from that very
// expression (exact by construction, not by a separate derivation), and the output is written into
// a pooled buffer that is neither reallocated nor zero-filled on a steady-state submit.
#pragma once
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

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
    if (!packed || !width || !height || width > SIZE_MAX / height) return false;
    const size_t texels = static_cast<size_t>(width * height);
    if (texels > SIZE_MAX / 4 || packed_bytes != texels * 4) return false;
    rgba8.resize(texels * 4);
    unpack_unorm10_words_to_rgba8(packed, texels, rgba8.data());
    return true;
}

// Publish an exact RGB10A2 compute result into the renderer target at `addr` as RGBA8. No renderer
// pixel format carries 10:10:10:2, so without this the target kept its old pixels under a picture the
// compute chain had composited (a flipped display buffer stayed black).
//
// The output buffer comes from a pool: the renderer holds the pixels until its last consumer releases
// them, and only then does the buffer return, so a retained older version is never overwritten. A
// steady-state submit reuses a buffer of the same size, so it neither allocates nor zero-fills.
inline void publish_unorm10_as_rgba8(uint64_t addr, uint32_t width, uint32_t height,
                                     const uint8_t* packed, size_t packed_bytes) {
    if (!packed || !width || !height) return;
    const size_t texels = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (texels > SIZE_MAX / 4 || packed_bytes != texels * 4) return;
    // Four retained buffers cover the targets a title cycles through; the byte budget bounds the pool.
    static CpuRttSnapshotPool pool(128ULL * 1024 * 1024, 4);
    CpuRttSnapshot snapshot = pool.build(texels * 4, [&](uint8_t* out) {
        unpack_unorm10_words_to_rgba8(packed, texels, out);
    });
    if (!snapshot.pixels) return;
    prosper::gpu::notify_live_render_target_image_written(
        {addr, width, height, prosper::gpu::LiveTargetPixelFormat::Rgba8Unorm, std::move(snapshot.pixels)});
}

} // namespace prosper::frontend
