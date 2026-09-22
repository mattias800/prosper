// RGBA8 rendering of a captured render-target seed or live readback, for writing it as an image.
//
// This is the picture path. It clamps every channel to 0..1 and scales to 0..255, which is right
// for anything you LOOK at and wrong for anything you need the number of -- native_texels.hpp is
// the path for numbers. Both paths exist, and the picture path must cover every format the
// capture can carry: it used to handle five of the ten and return an empty vector for the rest,
// so `--output-target-after` armed successfully on an R32Float / Rgba32Float / Rg16Float /
// R16Float / Rg8Unorm target and then failed with "conversion unavailable ... actual=0" (#3765).
//
// THE SWITCH IS EXHAUSTIVE AND HAS NO `default`, deliberately. gpu_replay compiles with
// -Werror=switch, so a GpuCaptureColorFormat enumerator added later without a case here is a
// BUILD failure rather than another silent empty image. The trailing `return {}` after the switch
// is reachable only for an out-of-range value read from a capture file, never for a real member.
//
// Channel conventions, chosen to match what already existed rather than invented per format:
//   * one-channel formats replicate into gray (as R8Unorm and R32Uint always did);
//   * two-channel formats write R and G, with B = 0;
//   * formats with no alpha channel write A = 255; four-channel formats convert their own alpha,
//     as Rgba16Float always did.
#pragma once

#include "../../src/gpu/capture/gpu_capture.hpp"
#include "../../src/gpu/resources/shader_resources.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace prosper::gpu::replay_tool {

// Bytes per texel in a seed's `rgba` payload for each capture format. 0 = not a known member.
inline uint32_t rtt_seed_texel_bytes(prosper::gpu::GpuCaptureColorFormat format) {
    using F = prosper::gpu::GpuCaptureColorFormat;
    switch (format) {
        case F::Rgba8Unorm:     return 4;
        case F::Rgba16Float:    return 8;
        case F::R11G11B10Float: return 4;
        case F::R8Unorm:        return 1;
        case F::R32Uint:        return 4;
        case F::R32Float:       return 4;
        case F::Rg8Unorm:       return 2;
        case F::Rgba32Float:    return 16;
        case F::Rg16Float:      return 4;
        case F::R16Float:       return 2;
    }
    return 0;
}

// A short name for log lines. Also exhaustive: the old inline ternary chain named five formats
// and printed "rgba8" for every other one, which mislabels the target being dumped.
inline const char* rtt_seed_format_name(prosper::gpu::GpuCaptureColorFormat format) {
    using F = prosper::gpu::GpuCaptureColorFormat;
    switch (format) {
        case F::Rgba8Unorm:     return "rgba8";
        case F::Rgba16Float:    return "rgba16f";
        case F::R11G11B10Float: return "r11g11b10f";
        case F::R8Unorm:        return "r8";
        case F::R32Uint:        return "r32ui";
        case F::R32Float:       return "r32f";
        case F::Rg8Unorm:       return "rg8";
        case F::Rgba32Float:    return "rgba32f";
        case F::Rg16Float:      return "rg16f";
        case F::R16Float:       return "r16f";
    }
    return "unknown";
}

// 0..1 float -> 0..255. EVERY non-finite value (NaN and both infinities) maps to 0, as do values
// <= 0; values >= 1 map to 255. That is exactly the rule the R11G11B10Float and Rgba16Float paths
// already used, kept byte-for-byte so no existing dump changes -- +inf reading as black is odd,
// and it is why native_texels.hpp exists for anything whose value matters.
inline uint8_t rtt_unit_to_byte(float value) {
    return !std::isfinite(value) || value <= 0.0f ? 0
         : value >= 1.0f ? 255
         : static_cast<uint8_t>(value * 255.0f + 0.5f);
}

// The seed's texels as tightly packed RGBA8, or an empty vector when the payload size does not
// match its extent and format (or the format is not a known member).
inline std::vector<uint8_t> rtt_seed_to_rgba8(const prosper::gpu::GpuCaptureRttSeed& seed) {
    using F = prosper::gpu::GpuCaptureColorFormat;
    const size_t texels = static_cast<size_t>(seed.width) * seed.height;
    const uint32_t bytes = rtt_seed_texel_bytes(seed.format);
    if (!bytes || seed.rgba.size() != texels * bytes) return {};
    const uint8_t* src = seed.rgba.data();
    std::vector<uint8_t> rgba(texels * 4, 0);

    auto half_at = [&](size_t texel, uint32_t channel) {
        uint16_t half = 0;
        std::memcpy(&half, src + texel * bytes + channel * 2, sizeof(half));
        return prosper::gpu::half_to_float(half);
    };
    auto f32_at = [&](size_t texel, uint32_t channel) {
        float value = 0.0f;
        std::memcpy(&value, src + texel * bytes + channel * 4, sizeof(value));
        return value;
    };
    auto gray = [&](size_t texel, uint8_t value) {
        rgba[texel * 4] = rgba[texel * 4 + 1] = rgba[texel * 4 + 2] = value;
        rgba[texel * 4 + 3] = 255;
    };

    switch (seed.format) {
        case F::Rgba8Unorm:
            return seed.rgba;
        case F::R11G11B10Float:
            for (size_t texel = 0; texel < texels; ++texel) {
                uint32_t packed = 0;
                std::memcpy(&packed, src + texel * 4, sizeof(packed));
                rgba[texel * 4] = rtt_unit_to_byte(
                    prosper::gpu::f11_to_float(static_cast<uint16_t>(packed)));
                rgba[texel * 4 + 1] = rtt_unit_to_byte(
                    prosper::gpu::f11_to_float(static_cast<uint16_t>(packed >> 11)));
                rgba[texel * 4 + 2] = rtt_unit_to_byte(
                    prosper::gpu::f10_to_float(static_cast<uint16_t>(packed >> 22)));
                rgba[texel * 4 + 3] = 255;
            }
            return rgba;
        case F::R8Unorm:
            for (size_t texel = 0; texel < texels; ++texel) gray(texel, src[texel]);
            return rgba;
        case F::R32Uint:
            for (size_t texel = 0; texel < texels; ++texel) {
                uint32_t value = 0;
                std::memcpy(&value, src + texel * 4, sizeof(value));
                gray(texel, static_cast<uint8_t>(std::min(value, 255u)));
            }
            return rgba;
        case F::Rgba16Float:
            for (size_t texel = 0; texel < texels; ++texel)
                for (uint32_t channel = 0; channel < 4; ++channel)
                    rgba[texel * 4 + channel] = rtt_unit_to_byte(half_at(texel, channel));
            return rgba;
        case F::R32Float:
            for (size_t texel = 0; texel < texels; ++texel)
                gray(texel, rtt_unit_to_byte(f32_at(texel, 0)));
            return rgba;
        case F::Rgba32Float:
            for (size_t texel = 0; texel < texels; ++texel)
                for (uint32_t channel = 0; channel < 4; ++channel)
                    rgba[texel * 4 + channel] = rtt_unit_to_byte(f32_at(texel, channel));
            return rgba;
        case F::Rg16Float:
            for (size_t texel = 0; texel < texels; ++texel) {
                rgba[texel * 4] = rtt_unit_to_byte(half_at(texel, 0));
                rgba[texel * 4 + 1] = rtt_unit_to_byte(half_at(texel, 1));
                rgba[texel * 4 + 3] = 255;
            }
            return rgba;
        case F::R16Float:
            for (size_t texel = 0; texel < texels; ++texel)
                gray(texel, rtt_unit_to_byte(half_at(texel, 0)));
            return rgba;
        case F::Rg8Unorm:
            for (size_t texel = 0; texel < texels; ++texel) {
                rgba[texel * 4] = src[texel * 2];
                rgba[texel * 4 + 1] = src[texel * 2 + 1];
                rgba[texel * 4 + 3] = 255;
            }
            return rgba;
    }
    return {};
}

}  // namespace prosper::gpu::replay_tool
