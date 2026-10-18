// CUBE's existing stacked-face sampling contract. Each lane supplies its own AMD-processed
// x/y/face coordinates; every face and texel has a distinct value. This pins the current renderer
// convention, not a new cube-array or mip contract.
#pragma once

#include "gpu/recompiler/ir/vector_lane_harness.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <vector>

namespace prosper::test::ir::image_cube {

inline constexpr uint32_t kLanes = 64, kW = 16, kH = 8;
struct Arm {
    uint32_t opcode;
    bool nsa;
};
inline constexpr std::array<uint32_t, 5> kOpcodes{0x20, 0x24, 0x25, 0x27, 0x2f};
inline std::vector<Arm> arms() {
    std::vector<Arm> out;
    for (uint32_t opcode : kOpcodes)
        for (bool nsa : {false, true}) out.push_back({opcode, nsa});
    return out;
}
inline bool modifier_first(Arm arm) {
    return arm.opcode == 0x25 || arm.opcode == 0x2f;
}
inline uint32_t bits(float value) {
    return std::bit_cast<uint32_t>(value);
}

// The modifier is v7; x/y/face are v8/v9/v10; explicit LOD is v11. NSA deliberately names
// these separately. The ordinary form uses exactly the same contiguous address tuple.
inline std::vector<uint32_t> image_words(Arm arm) {
    const uint32_t start = modifier_first(arm) ? 7u : 8u;
    std::vector<uint32_t> out{0xf0000000u | (arm.opcode << 18u) | 0x118u | (arm.nsa ? 2u : 0u),
                              0x00c40100u | start};
    if (arm.nsa) out.push_back(modifier_first(arm) ? 0x000a0908u : 0x000b0a09u);
    return out;
}

// Exact texel centres in each in-face axis, with independently varying face ids. Some face ids
// lie outside [0,5], and fractional ids straddle every round-to-even tie.
inline uint32_t x(uint32_t lane) {
    return lane % kW;
}
inline uint32_t y(uint32_t lane) {
    return (lane * 3u + lane / kW) % kH;
}
inline constexpr std::array<float, 16> kFace{-2.0f, -0.5f, 0.0f,  0.5f, 0.75f, 1.5f, 1.75f, 2.5f,
                                             2.75f, 3.5f,  3.75f, 4.5f, 4.75f, 5.0f, 5.5f,  8.0f};
inline constexpr std::array<uint32_t, 16> kLayer{0, 0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 5, 5, 5};
inline uint32_t texel(uint32_t column, uint32_t row, uint32_t face) {
    return 1000u + (face * kH + row) * kW + column;
}
inline LaneInputs inputs(uint32_t mask) {
    LaneInputs in = lane_inputs(0xc0be5428u);
    in.s2.resize(kLanes);
    in.s3.resize(kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        in.s0[lane] = bits(1.0f + (static_cast<float>(x(lane)) + 0.5f) / kW);
        in.s1[lane] = bits(1.0f + (static_cast<float>(y(lane)) + 0.5f) / kH);
        in.s2[lane] = bits(kFace[lane % kFace.size()]);
        // Bias and explicit LOD are deliberately nonzero: the established CUBE route still
        // samples the single base level. The DREF arm uses this as a comparison reference.
        in.s3[lane] = bits((lane & 1u) ? 2000.0f : 0.5f);
        in.active[lane] = mask == 0u || (mask == 1u && lane % 3u != 0u);
    }
    return in;
}
inline std::vector<uint32_t> buffer_words(const LaneInputs& in) {
    std::vector<uint32_t> out;
    for (const auto* array : {&in.s0, &in.s1, &in.old_d})
        out.insert(out.end(), array->begin(), array->end());
    for (bool active : in.active) out.push_back(active ? 1u : 0u);
    for (const auto* array : {&in.s2, &in.s3}) out.insert(out.end(), array->begin(), array->end());
    return out;
}
inline std::vector<uint32_t> texture_words() {
    std::vector<uint32_t> out;
    for (uint32_t face = 0; face < 12u; ++face)
        for (uint32_t row = 0; row < kH; ++row)
            for (uint32_t column = 0; column < kW; ++column)
                out.push_back(bits(static_cast<float>(texel(column, row, face))));
    return out;
}
inline uint32_t expected(Arm arm, const LaneInputs& in, uint32_t lane) {
    if (!in.active[lane]) return in.old_d[lane];
    const uint32_t value = texel(x(lane), y(lane), kLayer[lane % kLayer.size()]);
    if (arm.opcode == 0x2f)
        return bits(std::bit_cast<float>(in.s3[lane]) < static_cast<float>(value) ? 1.0f : 0.0f);
    return bits(static_cast<float>(value));
}
inline LaneProgram program(Arm arm, uint32_t wave, uint64_t buffer = 0, uint64_t image = 0,
                           bool normalize = false) {
    using namespace prosper::gpu;
    std::vector<uint32_t> words;
    for (const auto [dst, src] :
         std::array<std::array<uint32_t, 2>, 5>{{{7, 6}, {8, 2}, {9, 3}, {10, 5}, {11, 6}}})
        words.push_back(vop1(1u, dst, 256u + src));
    const auto access = image_words(arm);
    words.insert(words.end(), access.begin(), access.end());
    LaneProgram p = lane_program(words, kLanes, wave, true, true);
    p.table.resources[0].gpu_addr = buffer;
    ShaderResource t{};
    t.cls = ResourceClass::Texture;
    t.binding = 5;
    t.sgpr_base = 16;
    t.img_dim = 3;
    t.format = DataFormat::Float32;
    t.num_components = 1;
    t.width = kW;
    t.height = kH;
    // The current compute backend materializes six stacked faces of a cube array. Single-cube
    // materialization is a separate known renderer issue; this test preserves its current domain.
    t.depth = 12;
    t.tile_mode = 0;
    t.declared_mip_levels = 1;
    t.gpu_addr = image;
    t.size = static_cast<uint32_t>(texture_words().size() * sizeof(uint32_t));
    t.unnormalized = normalize;
    t.depth_compare = arm.opcode == 0x2f;
    t.depth_compare_func = 1;   // LESS: reference < fetched depth
    for (uint32_t c = 0; c < 4; ++c) t.swizzle[c] = 4u + c;
    p.table.resources.push_back(t);
    p.config.user_sgprs.resize(32);
    return p;
}

}   // namespace prosper::test::ir::image_cube
