// Gather-coordinate witnesses: independent spatial/channel/layer values, consecutive and NSA
// address layouts. Single-mip textures isolate coordinates from implicit-LOD semantics.
#pragma once
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <array>
#include <bit>
#include <cstdint>
#include <vector>

namespace prosper::test::ir::image_gather {
inline constexpr uint32_t kLanes = 64, kW = 64, kH = 8, kLayers = 3;
struct Arm {
    uint32_t opcode, component;
    bool arrayed, nsa, normalize;
};
inline bool offset(const Arm& a) {
    return a.opcode == 0x57;
}
inline uint32_t slots(const Arm& a) {
    return 2u + uint32_t(offset(a)) + uint32_t(a.arrayed);
}
inline uint32_t f(float x) {
    return std::bit_cast<uint32_t>(x);
}
inline uint32_t address_reg(const Arm& a, uint32_t slot) {
    constexpr std::array regs{10u, 20u, 12u, 22u};
    return a.nsa ? regs[slot] : 10u + slot;
}
inline std::vector<Arm> arms() {
    std::vector<Arm> out;
    out.reserve(96);
    for (uint32_t opcode : {0x40u, 0x47u, 0x57u})
        for (uint32_t component : {0u, 1u, 2u, 3u})
            for (bool arrayed : {false, true})
                for (bool nsa : {false, true})
                    for (bool normalize : {false, true})
                        out.push_back({opcode, component, arrayed, nsa, normalize});
    return out;
}
inline std::vector<uint32_t> words(const Arm& a, bool bad_a16 = false, bool bad_dmask = false) {
    const uint32_t dmask = bad_dmask ? 3u : 1u << a.component;
    std::vector<uint32_t> out{0xf0000000u | (a.opcode << 18) | (dmask << 8) |
                                  ((a.arrayed ? 5u : 1u) << 3) | (a.nsa ? 2u : 0u),
                              0x00c4010au | (bad_a16 ? 1u << 30 : 0u)};
    if (a.nsa) {
        out.push_back(0);
        for (uint32_t slot = 1; slot < slots(a); ++slot)
            out[2] |= address_reg(a, slot) << (8u * (slot - 1u));
    }
    return out;
}
struct Point {
    uint32_t x, y, layer;
};
inline Point point(uint32_t lane) {
    return {lane % 60u + 2u, (lane * 3u + lane / 60u) % 4u + 2u, (lane + lane / 8u) % kLayers};
}
inline int32_t dx(uint32_t lane) {
    return lane % 2 ? 1 : -1;
}
inline int32_t dy(uint32_t lane) {
    return (lane / 2u) % 2 ? 1 : -1;
}
inline uint32_t texel(uint32_t x, uint32_t y, uint32_t layer, uint32_t component) {
    return f(float(0x1000u + ((component * kLayers + layer) * kH + y) * kW + x));
}
struct Data {
    std::array<std::vector<uint32_t>, 4> source, old;
    std::vector<bool> active;
};
inline Data inputs(const Arm& a, uint32_t mask) {
    Data in;
    for (auto& s : in.source) s.resize(kLanes);
    for (auto& s : in.old) s.resize(kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        const auto p = point(lane);
        const uint32_t spatial = offset(a) ? 1u : 0u;
        in.source[spatial][lane] = f(float(p.x) / (a.normalize ? 1.0f : float(kW)));
        in.source[spatial + 1u][lane] = f(float(p.y) / (a.normalize ? 1.0f : float(kH)));
        if (offset(a))
            in.source[0][lane] = (uint32_t(dx(lane)) & 63u) | ((uint32_t(dy(lane)) & 63u) << 8);
        if (a.arrayed) in.source[spatial + 2u][lane] = f(float(p.layer));
        for (uint32_t c = 0; c < 4; ++c) in.old[c][lane] = 0xd0000000u | (c << 16) | lane;
        in.active.push_back(mask == 0 || (mask == 1 && lane % 4 != 0));
    }
    return in;
}
inline uint32_t expected(const Arm& a, const Data& in, uint32_t lane, uint32_t result) {
    if (!in.active[lane]) return in.old[result][lane];
    // Gather's ordered footprint around integer (u,v): (i0,j1),(i1,j1),(i1,j0),(i0,j0),
    // with i0=floor(u-0.5), j0=floor(v-0.5). Offsets apply to spatial axes only.
    constexpr std::array xx{0, 1, 1, 0}, yy{1, 1, 0, 0};
    const auto p = point(lane);
    const auto x = uint32_t(int32_t(p.x) - 1 + xx[result] + (offset(a) ? dx(lane) : 0));
    const auto y = uint32_t(int32_t(p.y) - 1 + yy[result] + (offset(a) ? dy(lane) : 0));
    return texel(x, y, a.arrayed ? p.layer : 0u, a.component);
}
inline std::vector<uint32_t> buffer_words(const Data& in) {
    std::vector<uint32_t> out;
    out.reserve(9u * kLanes);
    for (const auto& s : in.source) out.insert(out.end(), s.begin(), s.end());
    for (const auto& s : in.old) out.insert(out.end(), s.begin(), s.end());
    for (bool active : in.active) out.push_back(active ? 1u : 0u);
    return out;
}
inline std::vector<uint32_t> texture_words(const Arm& a) {
    std::vector<uint32_t> out;
    for (uint32_t layer = 0; layer < (a.arrayed ? kLayers : 1u); ++layer)
        for (uint32_t y = 0; y < kH; ++y)
            for (uint32_t x = 0; x < kW; ++x)
                for (uint32_t c = 0; c < 4; ++c) out.push_back(texel(x, y, layer, c));
    return out;
}
inline LaneProgram program(const Arm& a, uint32_t wave, uint64_t buffer = 0, uint64_t image = 0,
                           bool bad_a16 = false, bool bad_dmask = false) {
    using namespace prosper::gpu;
    auto p = lane_program({}, kLanes, wave);
    p.code.clear();
    for (uint32_t reg = 10; reg < 14; ++reg)
        p.code.insert(p.code.end(), {vop1(1, reg, 0xff), f(0.75f)});
    const auto load = [&](uint32_t reg, uint32_t array) {
        p.code.insert(p.code.end(),
                      {0xe0302000u | (array * kLanes * 4u), 0x80020000u | (reg << 8)});
    };
    for (uint32_t slot = 0; slot < slots(a); ++slot) load(address_reg(a, slot), slot);
    for (uint32_t c = 0; c < 4; ++c) load(1u + c, 4u + c);
    load(5, 8);
    p.code.insert(p.code.end(), {0xbf8c3f70u, 0x7daa0a80u}); // wait; CMPX_NE 0,v5
    const auto image_op = words(a, bad_a16, bad_dmask);
    p.code.insert(p.code.end(), image_op.begin(), image_op.end());
    p.code.push_back(0xbefe04c1u); // full EXEC for stores of old inactive results
    for (uint32_t c = 0; c < 4; ++c)
        p.code.insert(p.code.end(),
                      {0xe0702000u | (c * kLanes * 4u), 0x80020000u | ((1u + c) << 8)});
    p.code.push_back(0xbf810000u);
    p.table.resources[0].gpu_addr = buffer;
    p.table.resources[0].size = 9u * kLanes * 4u;
    ShaderResource t{};
    t.cls = ResourceClass::Texture;
    t.binding = 5;
    t.sgpr_base = 16;
    t.img_dim = a.arrayed ? 5u : 1u;
    t.format = DataFormat::Float32;
    t.num_components = 4;
    t.width = kW;
    t.height = kH;
    t.depth = a.arrayed ? kLayers : 1u;
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
} // namespace prosper::test::ir::image_gather
