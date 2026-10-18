// Programs for the MIMG spatial-coordinate definition (ir/ops/image_coordinates.cpp), shared by
// the Vulkan execution test and its witness export. Each arm is `vector_lane_harness.hpp`'s
// program -- v2 = S0, v3 = S1, v5 = S2, v1 = old VDST, EXEC narrowed by `active` -- with one image
// instruction writing v1, against a linear W x H (x D) texture whose T# sits at s[16:23].
#pragma once

#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <bit>
#include <cstdint>
#include <vector>

namespace prosper::test::ir::image_coordinates {

inline constexpr uint32_t kLanes = 64, kW = 64, kH = 4, kD = 3;

enum class Arm : uint32_t {
    Load2d,   // image_load v1, v[2:3], s[16:23] dmask:0x1 dim:2D         (Separate, 2 axes)
    Load2dNsa,   // image_load v1, [v2, v5], s[16:23] dmask:0x1 dim:2D       (Separate, NSA)
    Load2dA16,   // image_load v1, v2, s[16:23] dmask:0x1 dim:2D a16         (Packed16)
    Load3dNsa,   // image_load v1, [v2, v3, v5], s[16:23] dmask:0x1 dim:3D   (Separate, 3 axes)
    SampleLz2dTexel,   // image_sample_lz v1, v[2:3], s[16:23], s[24:27] dmask:0x1 dim:2D, on a
    // FORCE_UNNORMALIZED S#                                    (normalize)
    SampleLz3dTexel,   // image_sample_lz v1, [v2, v3, v5], s[16:23], s[24:27] dmask:0x1 dim:3D,
    // on a FORCE_UNNORMALIZED S#                               (normalize, 3 axes)
};
inline constexpr Arm kArms[] = {Arm::Load2d,    Arm::Load2dNsa,       Arm::Load2dA16,
                                Arm::Load3dNsa, Arm::SampleLz2dTexel, Arm::SampleLz3dTexel};
inline const char* name(Arm a) {
    constexpr const char* names[] = {"Load2d",    "Load2dNsa",       "Load2dA16",
                                     "Load3dNsa", "SampleLz2dTexel", "SampleLz3dTexel"};
    return names[static_cast<uint32_t>(a)];
}
inline bool three_d(Arm a) {
    return a == Arm::Load3dNsa || a == Arm::SampleLz3dTexel;
}
inline bool sampled(Arm a) {
    return a == Arm::SampleLz2dTexel || a == Arm::SampleLz3dTexel;
}

// The instruction words. MIMG word0: 0xf0000000 | op[6:0]<<18 | dmask<<8 | dim<<3 | nsa<<1;
// word1: SSAMP/4 <<21 | SRSRC/4 <<16 | VDATA<<8 | VADDR, A16 = bit 30; NSA dword: address
// VGPRs 1..4, one per byte.
inline std::vector<uint32_t> words(Arm a) {
    switch (a) {
        case Arm::Load2d: return {0xf0000108u, 0x00040102u};
        case Arm::Load2dNsa: return {0xf000010au, 0x00040102u, 0x00000005u};
        case Arm::Load2dA16: return {0xf0000108u, 0x40040102u};
        case Arm::Load3dNsa: return {0xf0000112u, 0x00040102u, 0x00000503u};
        case Arm::SampleLz2dTexel: return {0xf09c0108u, 0x00c40102u};
        case Arm::SampleLz3dTexel: return {0xf09c0112u, 0x00c40102u, 0x00000503u};
    }
    return {};
}

// Texel (x, y, z): never 0, so a robust out-of-range read (0) cannot pass as data.
inline uint32_t texel(uint32_t x, uint32_t y, uint32_t z = 0) {
    return 0x1000u + (z * kH + y) * kW + x;
}

// Per lane: its own coordinates, some of them past the image (x >= W, y == H, z == D).
struct Coordinates {
    uint32_t x, y, z;
};
inline Coordinates coordinates(uint32_t lane) {
    return {(lane * 37u + 3u) % (kW + 8u), (lane * 5u + lane / 7u) % (kH + 1u),
            (lane * 3u + 1u) % (kD + 1u)};
}
// The sampled arm keeps every lane inside the image (a sample clamps or wraps rather than reading
// 0), one column per lane.
inline Coordinates sampled_coordinates(uint32_t lane) {
    return {lane % kW, (lane * 5u + lane / 7u) % kH, (lane * 3u + 1u) % kD};
}
inline bool in_range(Arm a, const Coordinates& c) {
    return c.x < kW && c.y < kH && (!three_d(a) || c.z < kD);
}

// The harness inputs for an arm: what S0/S1/S2 hold per lane. The sampled arm keeps every
// coordinate inside the image and samples texel centres; the integer arms read raw texels.
inline LaneInputs inputs(Arm a) {
    LaneInputs in = lane_inputs(0x1a6e0000u + static_cast<uint32_t>(a));
    in.s2.assign(kLanes, 0);
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        Coordinates c = sampled(a) ? sampled_coordinates(lane) : coordinates(lane);
        if (sampled(a)) {
            in.s0[lane] = std::bit_cast<uint32_t>(static_cast<float>(c.x) + 0.5f);
            in.s1[lane] = std::bit_cast<uint32_t>(static_cast<float>(c.y) + 0.5f);
            in.s2[lane] = std::bit_cast<uint32_t>(static_cast<float>(c.z) + 0.5f);
        } else if (a == Arm::Load2dA16) {
            in.s0[lane] = c.x | (c.y << 16);
            in.s1[lane] = 0xdead0000u | lane;   // unread: A16 packs both axes into v2
        } else if (a == Arm::Load2dNsa) {
            in.s0[lane] = c.x;
            in.s1[lane] = 0xdead0000u | lane;   // unread: NSA names v5 for y
            in.s2[lane] = c.y;
        } else {
            in.s0[lane] = c.x;
            in.s1[lane] = c.y;
            in.s2[lane] = c.z;
        }
    }
    return in;
}

// The value lane `lane` must store: its old VDST when inactive, 0 past the image (robust image
// access, #3531), otherwise the texel (the sampled arm reads its texel centre with point filtering).
inline uint32_t expected(Arm a, const LaneInputs& in, uint32_t lane) {
    if (!in.active[lane]) return in.old_d[lane];
    const Coordinates c = sampled(a) ? sampled_coordinates(lane) : coordinates(lane);
    if (sampled(a)) {
        return std::bit_cast<uint32_t>(static_cast<float>(texel(c.x, c.y, three_d(a) ? c.z : 0u)));
    }
    if (!in_range(a, c)) return 0u;
    return texel(c.x, c.y, three_d(a) ? c.z : 0u);
}

// The texture's bytes: Uint32 texels for the loads, Float32 texels (of the same integers) for the
// sampled arm.
inline std::vector<uint32_t> image_words(Arm a) {
    std::vector<uint32_t> out;
    const uint32_t depth = three_d(a) ? kD : 1u;
    for (uint32_t z = 0; z < depth; ++z)
        for (uint32_t y = 0; y < kH; ++y)
            for (uint32_t x = 0; x < kW; ++x)
                out.push_back(sampled(a)
                                  ? std::bit_cast<uint32_t>(static_cast<float>(texel(x, y, z)))
                                  : texel(x, y, z));
    return out;
}

// The arm's harness program with the texture added to its resource table. `buffer` and `image`
// are the guest addresses the live backend reads (0 for a witness, which never executes).
inline LaneProgram program(Arm a, uint32_t wave, uint64_t buffer = 0, uint64_t image = 0) {
    using namespace prosper::gpu;
    LaneProgram p = lane_program(words(a), kLanes, wave, /*third_source=*/true);
    p.table.resources[0].gpu_addr = buffer;
    ShaderResource t{};
    t.cls = ResourceClass::Texture;
    t.binding = 5;
    t.sgpr_base = 16;
    t.img_dim = three_d(a) ? 2u : 1u;
    t.format = sampled(a) ? DataFormat::Float32 : DataFormat::Uint32;
    t.num_components = 1;
    t.width = kW;
    t.height = kH;
    t.depth = three_d(a) ? kD : 1u;
    t.tile_mode = 0;
    t.declared_mip_levels = 1;
    t.gpu_addr = image;
    t.size = static_cast<uint32_t>(image_words(a).size() * sizeof(uint32_t));
    for (uint32_t c = 0; c < 4; ++c) t.swizzle[c] = 4u + c;
    if (sampled(a)) {
        t.unnormalized = 1;   // S# FORCE_UNNORMALIZED: texel-space coordinates
        t.mag_filter = t.min_filter = 0;   // point
    }
    p.table.resources.push_back(t);
    p.config.user_sgprs.resize(32);
    return p;
}

}   // namespace prosper::test::ir::image_coordinates
