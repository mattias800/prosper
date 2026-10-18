// AMD 70648, MIMG D16: compact selected components into consecutive halves. Independent
// binary16 examples and distinct old VGPRs expose wrong component order and stray writes.
#pragma once
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <array>
#include <bit>
#include <cstdint>
#include <vector>
namespace prosper::test::ir::image_d16 {
inline constexpr uint32_t kLanes = 64, kW = 64, kH = 4;
struct Example {
    uint32_t f32, f16;
};
// Worked in binary, independently of eval: 1+2^-11 ties down, 1+3*2^-11 ties up.
inline constexpr std::array kExamples{
    Example{0x00000000, 0x0000}, Example{0x80000000, 0x8000}, Example{0x3f800000, 0x3c00},
    Example{0xc0000000, 0xc000}, Example{0x3f000000, 0x3800}, Example{0x477fe000, 0x7bff},
    Example{0x38800000, 0x0400}, Example{0x33800000, 0x0001}, Example{0x387fc000, 0x03ff},
    Example{0x3f801000, 0x3c00}, Example{0x3f803000, 0x3c02}, Example{0xbf803000, 0xbc02},
    Example{0x7f800000, 0x7c00}, Example{0xff800000, 0xfc00}, Example{0x33000000, 0x0000},
    Example{0xb3800000, 0x8001}};
struct Arm {
    uint32_t opcode, dmask;
    bool nsa;
};
inline bool gather(const Arm& a) {
    return a.opcode == 0x47;
}
inline std::vector<Arm> arms() {
    std::vector<Arm> out;
    out.reserve(68);
    for (bool nsa : {false, true}) {
        for (uint32_t op : {0u, 0x27u})
            for (uint32_t mask = 1; mask < 16; ++mask) out.push_back({op, mask, nsa});
        for (uint32_t mask : {1u, 2u, 4u, 8u}) out.push_back({0x47, mask, nsa});
    }
    return out;
}
inline uint32_t x(uint32_t lane) {
    return lane % 60 + 2;
}
inline uint32_t y(uint32_t lane) {
    return (lane / 16) % 2 + 1;
}
inline Example texel(uint32_t xx, uint32_t yy, uint32_t c) {
    return kExamples[(xx + yy * 7u + c * 3u) % kExamples.size()];
}
struct Data {
    std::array<std::vector<uint32_t>, 2> source;
    std::array<std::vector<uint32_t>, 4> old;
    std::vector<bool> active;
};
inline Data inputs(const Arm& a, uint32_t mask) {
    Data in;
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        for (uint32_t axis = 0; axis < 2; ++axis) {
            const uint32_t value = axis ? y(lane) : x(lane);
            const float coordinate =
                (float(value) + (gather(a) ? 0.f : 0.5f)) / float(axis ? kH : kW);
            in.source[axis].push_back(a.opcode == 0 ? value : std::bit_cast<uint32_t>(coordinate));
        }
        for (uint32_t c = 0; c < 4; ++c) in.old[c].push_back(0xd0000000u | (c << 16) | lane);
        in.active.push_back(mask == 0 || (mask == 1 && lane % 4 != 0));
    }
    return in;
}
inline std::vector<uint32_t> buffer_words(const Data& in) {
    std::vector<uint32_t> out;
    out.reserve(std::size_t{7} * kLanes);
    for (const auto& s : in.source) out.insert(out.end(), s.begin(), s.end());
    for (const auto& s : in.old) out.insert(out.end(), s.begin(), s.end());
    for (bool active : in.active) out.push_back(active ? 1u : 0u);
    return out;
}
inline std::vector<uint32_t> texture_words() {
    std::vector<uint32_t> out;
    out.reserve(kW * kH * 4);
    for (uint32_t yy = 0; yy < kH; ++yy)
        for (uint32_t xx = 0; xx < kW; ++xx)
            for (uint32_t c = 0; c < 4; ++c) out.push_back(texel(xx, yy, c).f32);
    return out;
}
inline uint32_t expected(const Arm& a, const Data& in, uint32_t lane, uint32_t word) {
    if (!in.active[lane]) return in.old[word][lane];
    std::vector<uint32_t> half;
    if (gather(a)) {
        const uint32_t channel = uint32_t(std::countr_zero(a.dmask));
        constexpr std::array xx{0u, 1u, 1u, 0u}, yy{1u, 1u, 0u, 0u};
        for (uint32_t c = 0; c < 4; ++c)
            half.push_back(texel(x(lane) - 1 + xx[c], y(lane) - 1 + yy[c], channel).f16);
    } else {
        for (uint32_t c = 0; c < 4; ++c)
            if (a.dmask & (1u << c)) half.push_back(texel(x(lane), y(lane), c).f16);
    }
    if (word * 2 >= half.size()) return in.old[word][lane];
    return half[word * 2] | (word * 2 + 1 < half.size() ? half[word * 2 + 1] << 16 : 0u);
}
inline LaneProgram program(const Arm& a, uint32_t wave, uint64_t buffer = 0, uint64_t image = 0,
                           bool d16 = true, bool integer = false) {
    using namespace prosper::gpu;
    auto p = lane_program({}, kLanes, wave);
    p.code.clear();
    const auto load = [&](uint32_t reg, uint32_t array) {
        p.code.insert(p.code.end(), {0xe0302000u | (array * kLanes * 4), 0x80020000u | (reg << 8)});
    };
    p.code.insert(p.code.end(), {vop1(1, 11, 0xff), 0xdeadbeefu});
    load(10, 0);
    load(a.nsa ? 20u : 11u, 1);
    for (uint32_t c = 0; c < 4; ++c) load(1 + c, 2 + c);
    load(5, 6);
    p.code.insert(p.code.end(), {0xbf8c3f70u, 0x7daa0a80u}); // wait; CMPX_NE 0,v5
    p.code.insert(p.code.end(),
                  {0xf0000008u | (a.opcode << 18) | (a.dmask << 8) | (a.nsa ? 2u : 0u),
                   0x00c4010au | (d16 ? 1u << 31 : 0u)});
    if (a.nsa) p.code.push_back(20u);
    p.code.push_back(0xbefe04c1u); // restore EXEC to observe old inactive words and guard VGPRs
    for (uint32_t c = 0; c < 4; ++c)
        p.code.insert(p.code.end(), {0xe0702000u | (c * kLanes * 4), 0x80020000u | ((1 + c) << 8)});
    p.code.push_back(0xbf810000u);
    p.table.resources[0].gpu_addr = buffer;
    p.table.resources[0].size = 7u * kLanes * 4u;
    ShaderResource t{};
    t.cls = ResourceClass::Texture;
    t.binding = 5;
    t.sgpr_base = 16;
    t.img_dim = 1;
    t.format = integer ? DataFormat::Uint32 : DataFormat::Float32;
    t.num_components = 4;
    t.width = kW;
    t.height = kH;
    t.declared_mip_levels = 1;
    t.gpu_addr = image;
    t.size = kW * kH * 4u * sizeof(uint32_t);
    for (uint32_t c = 0; c < 4; ++c) t.swizzle[c] = 4 + c;
    t.mag_filter = t.min_filter = 0;
    p.table.resources.push_back(t);
    p.config.user_sgprs.resize(32);
    return p;
}
} // namespace prosper::test::ir::image_d16
