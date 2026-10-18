// Operand-first MIMG witnesses. Address order follows AMD 70648 8.2.4: modifier or gradients,
// then spatial coordinates; an array layer is an independent float operand.
#pragma once
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <vector>

namespace prosper::test::ir::image_operands {
inline constexpr uint32_t kLanes = 64, kW = 64, kH = 8, kLayers = 3;
enum class Kind {
    Bias,
    Offset,
    LzOffset,
    Gradient,
    CoarseGradient,
    Array,
    ArrayAlt,
    ArrayLod,
    ArrayLz
};
struct Arm {
    Kind kind;
    bool nsa, normalize;
};
inline bool arrayed(Kind k) {
    return k >= Kind::Array;
}
inline bool gradient(Kind k) {
    return k == Kind::Gradient || k == Kind::CoarseGradient;
}
inline bool offset(Kind k) {
    return k == Kind::Offset || k == Kind::LzOffset;
}
inline uint32_t slots(Kind k) {
    return gradient(k) ? 6u : k == Kind::ArrayLod ? 4u : 3u;
}
inline std::vector<Arm> arms() {
    std::vector<Arm> out;
    for (Kind k : {Kind::Bias, Kind::Offset, Kind::LzOffset, Kind::Gradient, Kind::CoarseGradient,
                   Kind::Array, Kind::ArrayAlt, Kind::ArrayLod, Kind::ArrayLz})
        for (bool nsa : {false, true})
            for (bool normalize : {false, true}) out.push_back({k, nsa, normalize});
    return out;
}
inline uint32_t f(float x) {
    return std::bit_cast<uint32_t>(x);
}
inline uint32_t address_reg(const Arm& a, uint32_t slot) {
    constexpr std::array nsa{10u, 20u, 12u, 22u, 14u, 24u};
    return a.nsa ? nsa[slot] : 10u + slot;
}
// Golden packet encodings, rather than a production operation inventory.
inline std::vector<uint32_t> image_words(const Arm& a, bool bad_a16 = false) {
    uint32_t op = 0;
    switch (a.kind) {
        case Kind::Bias: op = 0x25; break;
        case Kind::Offset: op = 0x30; break;
        case Kind::LzOffset: op = 0x37; break;
        case Kind::Gradient: op = 0x22; break;
        case Kind::CoarseGradient: op = 0x68; break;
        case Kind::Array: op = 0x20; break;
        case Kind::ArrayAlt: op = 0xa0; break;
        case Kind::ArrayLod: op = 0x24; break;
        case Kind::ArrayLz: op = 0x27; break;
    }
    const uint32_t extra = a.nsa ? (slots(a.kind) + 2u) / 4u : 0u;
    std::vector<uint32_t> words{0xf0000100u | ((op & 0x7fu) << 18) | (op >> 7) |
                                    ((arrayed(a.kind) ? 5u : 1u) << 3) | (extra << 1),
                                0x00c4010au | (bad_a16 ? 1u << 30 : 0u)};
    words.resize(2 + extra);
    for (uint32_t slot = 1; a.nsa && slot < slots(a.kind); ++slot)
        words[2 + (slot - 1u) / 4u] |= address_reg(a, slot) << (8u * ((slot - 1u) % 4u));
    return words;
}
struct Point {
    uint32_t x, y, layer;
};
inline Point point(uint32_t lane) {
    return {lane % 62u + 1u, (lane * 3u + lane / 62u) % 6u + 1u, (lane + lane / 8u) % kLayers};
}
inline int32_t dx(uint32_t lane) {
    return lane % 2 ? 1 : -1;
}
inline int32_t dy(uint32_t lane) {
    return (lane / 2u) % 2 ? 1 : -1;
}
inline uint32_t texel(uint32_t x, uint32_t y, uint32_t layer) {
    return f(float(0x1000u + (layer * kH + y) * kW + x));
}
struct Data {
    std::array<std::vector<uint32_t>, 6> source;
    std::vector<uint32_t> old;
    std::vector<bool> active;
};
inline Data inputs(const Arm& a, uint32_t mask = 1) {
    Data in;
    for (auto& s : in.source) s.resize(kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        const Point p = point(lane);
        const float u = (float(p.x) + 0.5f) / (a.normalize ? 1.0f : float(kW));
        const float v = (float(p.y) + 0.5f) / (a.normalize ? 1.0f : float(kH));
        if (gradient(a.kind)) {
            // Quad-uniform derivatives; no assertion about coarse derivative reduction or LOD.
            in.source[0][lane] = f(0.5f / (a.normalize ? 1.0f : float(kW)));
            in.source[3][lane] = f(0.25f / (a.normalize ? 1.0f : float(kH)));
            in.source[4][lane] = f(u);
            in.source[5][lane] = f(v);
        } else if (arrayed(a.kind)) {
            in.source[0][lane] = f(u);
            in.source[1][lane] = f(v);
            in.source[2][lane] = f(float(p.layer));
        } else {
            if (offset(a.kind))
                in.source[0][lane] =
                    (uint32_t(dx(lane)) & 0x3fu) | ((uint32_t(dy(lane)) & 0x3fu) << 8);
            else
                in.source[0][lane] = f(float(int(lane % 3u) - 1) * 0.5f);
            in.source[1][lane] = f(u);
            in.source[2][lane] = f(v);
        }
        in.old.push_back(0xd0000000u | lane);
        in.active.push_back(mask == 0 || (mask == 1 && lane % 4 != 0));
    }
    return in;
}
inline uint32_t expected(const Arm& a, const Data& in, uint32_t lane) {
    if (!in.active[lane]) return in.old[lane];
    const Point p = point(lane);
    const uint32_t x = uint32_t(int32_t(p.x) + (offset(a.kind) ? dx(lane) : 0));
    const uint32_t y = uint32_t(int32_t(p.y) + (offset(a.kind) ? dy(lane) : 0));
    return texel(x, y, arrayed(a.kind) ? p.layer : 0u);
}
inline std::vector<uint32_t> buffer_words(const Data& in) {
    std::vector<uint32_t> out;
    for (const auto& s : in.source) out.insert(out.end(), s.begin(), s.end());
    out.insert(out.end(), in.old.begin(), in.old.end());
    for (bool active : in.active) out.push_back(active ? 1u : 0u);
    return out;
}
inline std::vector<uint32_t> texture_words(const Arm& a) {
    std::vector<uint32_t> out;
    for (uint32_t layer = 0; layer < (arrayed(a.kind) ? kLayers : 1u); ++layer)
        for (uint32_t y = 0; y < kH; ++y)
            for (uint32_t x = 0; x < kW; ++x) out.push_back(texel(x, y, layer));
    return out;
}
inline LaneProgram program(const Arm& a, uint32_t wave, uint64_t buffer = 0, uint64_t image = 0,
                           bool bad_a16 = false) {
    using namespace prosper::gpu;
    LaneProgram p = lane_program({}, kLanes, wave);
    p.code.clear();
    // An incorrect consecutive read of an NSA operand sees a finite poison coordinate.
    for (uint32_t reg = 10; reg < 16; ++reg)
        p.code.insert(p.code.end(), {vop1(1, reg, 0xff), f(0.75f)});
    const auto load = [&](uint32_t reg, uint32_t array) {
        p.code.insert(p.code.end(),
                      {0xe0302000u | (array * kLanes * 4u), 0x80020000u | (reg << 8)});
    };
    for (uint32_t slot = 0; slot < slots(a.kind); ++slot) load(address_reg(a, slot), slot);
    load(1, 6);
    load(4, 7);
    p.code.insert(p.code.end(), {0xbf8c3f70u, 0x7daa0880u});
    const auto image_op = image_words(a, bad_a16);
    p.code.insert(p.code.end(), image_op.begin(), image_op.end());
    p.code.insert(p.code.end(), {0xbefe04c1u, 0xe0702000u, 0x80020100u, 0xbf810000u});
    p.table.resources[0].gpu_addr = buffer;
    p.table.resources[0].size = 8u * kLanes * 4u;
    ShaderResource t{};
    t.cls = ResourceClass::Texture;
    t.binding = 5;
    t.sgpr_base = 16;
    t.img_dim = arrayed(a.kind) ? 5u : 1u;
    t.format = DataFormat::Float32;
    t.num_components = 1;
    t.width = kW;
    t.height = kH;
    t.depth = arrayed(a.kind) ? kLayers : 1u;
    t.tile_mode = 0;
    t.declared_mip_levels = 1;
    t.gpu_addr = image;
    t.size = uint32_t(texture_words(a).size() * sizeof(uint32_t));
    for (uint32_t c = 0; c < 4; ++c) t.swizzle[c] = 4u + c;
    t.unnormalized = a.normalize;
    t.mag_filter = t.min_filter = 0;
    p.table.resources.push_back(t);
    p.config.user_sgprs.resize(32);
    return p;
}
}   // namespace prosper::test::ir::image_operands
