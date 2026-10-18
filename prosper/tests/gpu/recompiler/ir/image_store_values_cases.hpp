// MIMG storage data: compact VDATA by DMASK before inverse DST_SEL, and route load results
// forward. Distinct per-lane words and guard texels expose wrong ranks, halves and selectors.
#pragma once
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace prosper::test::ir::image_store_values {
inline constexpr uint32_t kLanes = 64, kW = 64, kH = 3;
enum class Kind { WordUint, WordFloat, HalfUint, HalfFloat };
struct Arm {
    Kind kind;
    uint32_t mask, selectors;
    bool load = false;
};
inline bool half(Kind k) {
    return k == Kind::HalfUint || k == Kind::HalfFloat;
}
inline bool integer(Kind k) {
    return k == Kind::HalfUint || k == Kind::WordUint;
}
inline constexpr std::array<std::array<uint32_t, 4>, 6> kSelectors{
    {{4, 5, 6, 7}, {6, 5, 4, 7}, {5, 6, 7, 4}, {0, 1, 4, 7}, {5, 5, 1, 0}, {1, 0, 1, 0}}};
inline std::vector<Arm> arms() {
    std::vector<Arm> out;
    out.reserve(159);
    for (Kind kind : {Kind::WordUint, Kind::WordFloat, Kind::HalfUint, Kind::HalfFloat})
        for (uint32_t mask = 1; mask < 16; ++mask) {
            if (kind == Kind::HalfUint && (mask & (mask + 1u))) continue;
            for (uint32_t s = 0; s < 3; ++s) out.push_back({kind, mask, s});
        }
    for (Kind kind : {Kind::WordUint, Kind::WordFloat})
        for (uint32_t s = 0; s < kSelectors.size(); ++s) out.push_back({kind, 15, s, true});
    return out;
}
struct Half {
    uint32_t half, word;
};
// AMD70648's half elements widened exactly: +/-0, normals and subnormals, no NaN payload
// or rounding ambiguity. These are binary format facts, independently worked from the fields.
inline constexpr std::array kHalf{Half{0, 0},
                                  Half{0x8000, 0x80000000},
                                  Half{0x3c00, 0x3f800000},
                                  Half{0xc000, 0xc0000000},
                                  Half{0x3800, 0x3f000000},
                                  Half{0x7bff, 0x477fe000},
                                  Half{0x0400, 0x38800000},
                                  Half{1, 0x33800000},
                                  Half{0x03ff, 0x387fc000},
                                  Half{0xbc01, 0xbf802000},
                                  Half{0x3555, 0x3eaaa000},
                                  Half{0x8001, 0xb3800000}};
inline Half half_value(uint32_t lane, uint32_t component) {
    return kHalf[(lane * 5u + component * 7u) % kHalf.size()];
}
inline uint32_t word_value(Kind kind, uint32_t lane, uint32_t component) {
    if (kind == Kind::HalfFloat)
        return half_value(lane, component * 2).half |
               (half_value(lane, component * 2 + 1).half << 16);
    if (kind == Kind::HalfUint)
        return ((0x8001u + lane * 173u + component * 97u) & 0xffffu) |
               (((0xfedcu - lane * 197u - component * 137u) & 0xffffu) << 16);
    if (kind == Kind::WordFloat)
        return std::bit_cast<uint32_t>(float(lane * 4u + component) - 31.5f);
    return 0x8192a3b4u ^ (lane * 1234567u + component * 173u);
}
inline uint32_t initial(uint32_t pixel, uint32_t component) {
    return 0x42000000u + pixel * 16u + component;
}
inline std::vector<uint32_t> image_words() {
    std::vector<uint32_t> out;
    out.reserve(std::size_t{kW} * kH * 4);
    for (uint32_t pixel = 0; pixel < kW * kH; ++pixel)
        for (uint32_t c = 0; c < 4; ++c) out.push_back(initial(pixel, c));
    return out;
}
inline bool active(uint32_t lane, uint32_t mask) {
    return mask == 0 || (mask == 1 && lane % 4 != 0);
}
inline std::vector<uint32_t> buffer_words(const Arm& a, uint32_t mask) {
    std::vector<uint32_t> out;
    out.reserve(std::size_t{5} * kLanes);
    for (uint32_t c = 0; c < 4; ++c)
        for (uint32_t lane = 0; lane < kLanes; ++lane) out.push_back(word_value(a.kind, lane, c));
    for (uint32_t lane = 0; lane < kLanes; ++lane) out.push_back(active(lane, mask) ? 1u : 0u);
    return out;
}
inline uint32_t stored_component(const Arm& a, uint32_t lane, uint32_t component) {
    if (!(a.mask & (1u << component))) return 0;
    const auto rank = uint32_t(std::popcount(a.mask & ((1u << component) - 1u)));
    if (!half(a.kind)) return word_value(a.kind, lane, rank);
    if (a.kind == Kind::HalfFloat) return half_value(lane, rank).word;
    return (word_value(a.kind, lane, rank / 2u) >> (16u * (rank % 2u))) & 0xffffu;
}
inline uint32_t expected_store(const Arm& a, uint32_t mask, uint32_t pixel, uint32_t c) {
    if (pixel < kW || pixel >= 2 * kW || !active(pixel % kW, mask)) return initial(pixel, c);
    for (uint32_t source = 0; source < 4; ++source)
        if (kSelectors[a.selectors][source] == c + 4u)
            return stored_component(a, pixel % kW, source);
    return 0; // admitted store selectors are permutations, so this is never reached
}
inline uint32_t expected_load(const Arm& a, uint32_t mask, uint32_t lane, uint32_t c) {
    if (!active(lane, mask)) return word_value(a.kind, lane, c);
    const uint32_t sel = kSelectors[a.selectors][c];
    if (sel == 0) return 0;
    if (sel == 1) return integer(a.kind) ? 1u : 0x3f800000u;
    return initial(kW + lane, sel - 4);
}
inline LaneProgram program(const Arm& a, uint32_t wave, uint64_t buffer = 0, uint64_t image = 0) {
    using namespace prosper::gpu;
    auto p = lane_program({}, kLanes, wave);
    p.code.clear();
    for (uint32_t array = 0; array < 5; ++array)
        p.code.insert(p.code.end(),
                      {0xe0302000u | (array * kLanes * 4u), 0x80020000u | ((array + 1u) << 8)});
    p.code.insert(p.code.end(), {vop1(1, 10, 256), vop1(1, 11, 129), 0xbf8c3f70u,
                                 0x7daa0a80u}); // v10=lane,v11=1; wait; CMPX
    p.code.insert(p.code.end(), {0xf0000008u | ((a.load ? 0u : 8u) << 18) | (a.mask << 8),
                                 0x0004010au | (half(a.kind) ? 1u << 31 : 0u)});
    if (a.load) {
        p.code.push_back(0xbefe04c1u);
        for (uint32_t c = 0; c < 4; ++c)
            p.code.insert(p.code.end(),
                          {0xe0702000u | (c * kLanes * 4u), 0x80020000u | ((c + 1u) << 8)});
    }
    p.code.push_back(0xbf810000u);
    p.table.resources[0].gpu_addr = buffer;
    p.table.resources[0].size = 5u * kLanes * 4u;
    ShaderResource t{};
    t.cls = ResourceClass::StorageImage;
    t.binding = 5;
    t.sgpr_base = 16;
    t.img_dim = 1;
    t.format = integer(a.kind) ? DataFormat::Uint32 : DataFormat::Float32;
    t.num_components = 4;
    t.width = kW;
    t.height = kH;
    t.declared_mip_levels = 1;
    t.gpu_addr = image;
    t.size = kW * kH * 16u;
    for (uint32_t c = 0; c < 4; ++c) t.swizzle[c] = kSelectors[a.selectors][c];
    p.table.resources.push_back(t);
    p.config.user_sgprs.resize(32);
    return p;
}
} // namespace prosper::test::ir::image_store_values
