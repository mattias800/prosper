// Native-typed readback of a small live render target, for values a picture cannot carry.
//
// `inspect_rtt_seed` converts a target to RGBA8 so it can be written as an image. That is the
// right thing for anything you look at, and the wrong thing for anything you need the NUMBER of:
// clamping to 0..1 and scaling to 0..255 makes an exposure of 10 and an exposure of 100 both
// read as 255, and it drops sign, NaN and infinity entirely. #3765.
//
// The case this was written for (#2790): Sonic Frontiers' tonemap reads its exposure from a
// 2x1 R32_SFLOAT target written earlier in the same submit. Discriminating "the HDR values are
// too high" from "the exposure is too high" needs that scalar, and no RGBA8 rendering of it can
// answer the question.
//
// Deliberately NOT a general format-conversion layer. It formats the first `max_texels` texels of
// a readback in the target's own type and says so when it cannot, which is the whole contract.
#pragma once

#include "../../src/gpu/execute/gpu_execute.hpp"
#include "../../src/gpu/resources/shader_resources.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace prosper::gpu::replay_tool {

// Bytes each texel occupies in the readback, and how many components it decodes to.
// 0 bytes means "this enumerator is not handled here" -- reported, never silently skipped.
struct NativeTexelLayout {
    const char* name = "unknown";
    uint32_t bytes = 0;
    uint32_t components = 0;
};

inline NativeTexelLayout native_texel_layout(prosper::gpu::LiveTargetPixelFormat format) {
    using F = prosper::gpu::LiveTargetPixelFormat;
    switch (format) {
        case F::Rgba8Unorm:     return {"Rgba8Unorm", 4, 4};
        case F::Rgba16Float:    return {"Rgba16Float", 8, 4};
        case F::R11G11B10Float: return {"R11G11B10Float", 4, 3};
        case F::R8Unorm:        return {"R8Unorm", 1, 1};
        case F::R32Uint:        return {"R32Uint", 4, 1};
        case F::R32Float:       return {"R32Float", 4, 1};
        case F::Rg8Unorm:       return {"Rg8Unorm", 2, 2};
        case F::Rgba32Float:    return {"Rgba32Float", 16, 4};
        case F::Rg16Float:      return {"Rg16Float", 4, 2};
        case F::R16Float:       return {"R16Float", 2, 1};
    }
    return {};
}

// Render one float the way an investigator needs to read it: full precision, and the three
// non-finite cases named rather than printed as some platform's spelling of them.
inline std::string native_float_text(float value) {
    if (std::isnan(value)) return "nan";
    if (std::isinf(value)) return value > 0.0f ? "+inf" : "-inf";
    char text[40];
    std::snprintf(text, sizeof text, "%.9g", static_cast<double>(value));
    return text;
}

// One texel's report line. `p` points at the texel's first byte; `t` is its linear index and
// (x, y) its coordinate, printed so a line can be matched back to the image.
inline std::string native_texel_line(prosper::gpu::LiveTargetPixelFormat format,
                                     const NativeTexelLayout& layout, const uint8_t* p,
                                     uint64_t t, uint64_t x, uint64_t y) {
    std::string raw, decoded;
    for (uint32_t c = 0; c < layout.components; ++c) {
        char piece[64];
        using F = prosper::gpu::LiveTargetPixelFormat;
        switch (format) {
            case F::Rgba8Unorm: case F::R8Unorm: case F::Rg8Unorm: {
                std::snprintf(piece, sizeof piece, "%02x", p[c]);
                raw += (c ? "," : ""); raw += piece;
                decoded += (c ? "," : "");
                decoded += native_float_text(static_cast<float>(p[c]) / 255.0f);
                break;
            }
            case F::Rgba16Float: case F::Rg16Float: case F::R16Float: {
                uint16_t h = 0; std::memcpy(&h, p + c * 2, sizeof h);
                std::snprintf(piece, sizeof piece, "%04x", h);
                raw += (c ? "," : ""); raw += piece;
                decoded += (c ? "," : "");
                decoded += native_float_text(prosper::gpu::half_to_float(h));
                break;
            }
            case F::R32Float: case F::Rgba32Float: {
                uint32_t u = 0; std::memcpy(&u, p + c * 4, sizeof u);
                float f = 0.0f; std::memcpy(&f, &u, sizeof f);
                std::snprintf(piece, sizeof piece, "%08x", u);
                raw += (c ? "," : ""); raw += piece;
                decoded += (c ? "," : ""); decoded += native_float_text(f);
                break;
            }
            case F::R32Uint: {
                uint32_t u = 0; std::memcpy(&u, p, sizeof u);
                std::snprintf(piece, sizeof piece, "%08x", u);
                raw += piece;
                decoded += std::to_string(u);
                break;
            }
            case F::R11G11B10Float: {
                // One `raw=` for the whole packed word, deliberately: the three channels
                // share a single 32-bit texel, so per-component raw fields would invent a
                // boundary the format does not have. R is the low 11 bits, G the middle 11,
                // B the high 10 -- the same convention as inspect_rtt_seed.
                uint32_t packed = 0; std::memcpy(&packed, p, sizeof packed);
                if (!c) { std::snprintf(piece, sizeof piece, "%08x", packed); raw += piece; }
                const float v = c == 0 ? prosper::gpu::f11_to_float(static_cast<uint16_t>(packed))
                              : c == 1 ? prosper::gpu::f11_to_float(static_cast<uint16_t>(packed >> 11))
                                       : prosper::gpu::f10_to_float(static_cast<uint16_t>(packed >> 22));
                decoded += (c ? "," : ""); decoded += native_float_text(v);
                break;
            }
        }
    }
    char line[512];
    std::snprintf(line, sizeof line, "    texel[%llu] (%llu,%llu) raw=%s value=%s\n",
                  static_cast<unsigned long long>(t),
                  static_cast<unsigned long long>(x),
                  static_cast<unsigned long long>(y),
                  raw.c_str(), decoded.c_str());
    return line;
}

// Format the first `max_texels` texels of `bytes` in the target's native type.
//
// Returns a human-readable block, ALWAYS non-empty: when the format is unhandled or the byte
// count does not match the extent, it says which, because a silent empty result here is
// indistinguishable from "the target was all zeros".
inline std::string format_native_texels(prosper::gpu::LiveTargetPixelFormat format,
                                        const uint8_t* bytes, size_t size,
                                        uint32_t width, uint32_t height,
                                        size_t max_texels) {
    const NativeTexelLayout layout = native_texel_layout(format);
    if (!layout.bytes)
        return std::string("  native readback: format enumerator ") +
               std::to_string(static_cast<int>(format)) + " is not handled here\n";
    const uint64_t texels = static_cast<uint64_t>(width) * height;
    const uint64_t expected = texels * layout.bytes;
    if (!bytes || size != expected) {
        char line[192];
        std::snprintf(line, sizeof line,
                      "  native readback: %s %ux%u wants %llu bytes, readback has %llu\n",
                      layout.name, width, height,
                      static_cast<unsigned long long>(expected),
                      static_cast<unsigned long long>(size));
        return line;
    }

    std::string out;
    char head[160];
    std::snprintf(head, sizeof head, "  native readback: %s %ux%u (%llu texel%s)\n",
                  layout.name, width, height, static_cast<unsigned long long>(texels),
                  texels == 1 ? "" : "s");
    out += head;

    const uint64_t shown = texels < max_texels ? texels : max_texels;
    for (uint64_t t = 0; t < shown; ++t)
        out += native_texel_line(format, layout, bytes + t * layout.bytes, t, t % width, t / width);
    if (shown < texels) {
        char line[128];
        std::snprintf(line, sizeof line, "    ... %llu more texel(s) not shown\n",
                      static_cast<unsigned long long>(texels - shown));
        out += line;
    }
    return out;
}

// Native-typed values at CHOSEN coordinates, for a target too large for the first-N report.
//
// The first-N report covers a scalar or a tiny reduction; it says nothing about a 3840x2160 HDR
// buffer, where the question is "what is the value inside the defect, and what is it just outside".
// Each requested point is printed with its own coordinate, and a point outside the extent is
// reported as such rather than dropped, for the same reason the report above is never empty.
struct NativeTexelPoint {
    uint32_t x = 0;
    uint32_t y = 0;
};

inline std::string format_native_texels_at(prosper::gpu::LiveTargetPixelFormat format,
                                           const uint8_t* bytes, size_t size,
                                           uint32_t width, uint32_t height,
                                           const std::vector<NativeTexelPoint>& points) {
    const NativeTexelLayout layout = native_texel_layout(format);
    if (!layout.bytes)
        return std::string("  native texels at points: format enumerator ") +
               std::to_string(static_cast<int>(format)) + " is not handled here\n";
    const uint64_t texels = static_cast<uint64_t>(width) * height;
    const uint64_t expected = texels * layout.bytes;
    if (!bytes || size != expected) {
        char line[192];
        std::snprintf(line, sizeof line,
                      "  native texels at points: %s %ux%u wants %llu bytes, readback has %llu\n",
                      layout.name, width, height,
                      static_cast<unsigned long long>(expected),
                      static_cast<unsigned long long>(size));
        return line;
    }
    char head[160];
    std::snprintf(head, sizeof head, "  native texels at %zu point(s): %s %ux%u\n",
                  points.size(), layout.name, width, height);
    std::string out = head;
    for (const NativeTexelPoint& point : points) {
        if (point.x >= width || point.y >= height) {
            char line[128];
            std::snprintf(line, sizeof line, "    (%u,%u) is outside the %ux%u extent\n",
                          point.x, point.y, width, height);
            out += line;
            continue;
        }
        const uint64_t t = static_cast<uint64_t>(point.y) * width + point.x;
        out += native_texel_line(format, layout, bytes + t * layout.bytes, t, point.x, point.y);
    }
    return out;
}

}  // namespace prosper::gpu::replay_tool
