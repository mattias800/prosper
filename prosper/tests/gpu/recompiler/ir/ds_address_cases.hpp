// AMD 70648 DS offsets: bytes for ordinary accesses, 4/8-byte units for READ2/WRITE2,
// and 256-byte units for ST64. Reference accesses use independent host-computed addresses.
#pragma once
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <array>
#include <cstdint>
#include <cstddef>
#include <vector>
namespace prosper::test::ir::ds_address {
struct Form {
    uint32_t opcode, width, unit;
    bool store;
    uint32_t scale; // zero for byte-offset forms, otherwise bytes per OFFSET0/1 unit
};
inline constexpr std::array<Form, 14> forms{{
    {0x0d, 1, 1, true, 0},
    {0x4d, 2, 2, true, 0},
    {0xde, 3, 3, true, 0},
    {0xdf, 4, 4, true, 0},
    {0x36, 1, 1, false, 0},
    {0x76, 2, 2, false, 0},
    {0xfe, 3, 3, false, 0},
    {0xff, 4, 4, false, 0},
    {0x0e, 2, 1, true, 4},
    {0x37, 2, 1, false, 4},
    {0x0f, 2, 1, true, 256},
    {0x38, 2, 1, false, 256},
    {0x4e, 4, 2, true, 8},
    {0x77, 4, 2, false, 8},
}};
struct Case {
    uint32_t form, wave, literal;
};
inline std::vector<Case> cases() {
    std::vector<Case> out;
    out.reserve(68);
    for (uint32_t wave : {32u, 64u})
        for (uint32_t f = 0; f < forms.size(); ++f) {
            if (forms[f].scale)
                for (uint32_t literal : {0x0100u, 0x0202u, 0x0103u})
                    out.push_back({f, wave, literal});
            else
                for (uint32_t literal : {0u, 12u}) out.push_back({f, wave, literal});
        }
    return out;
}
inline bool active(uint32_t lane, uint32_t wave) {
    return lane != 0 && lane != wave - 1 && lane % 5 != 2;
}
inline uint32_t base(const Case& c, uint32_t lane) {
    return lane * (forms[c.form].scale == 256 ? 1024u : 64u);
}
inline uint32_t address(const Case& c, uint32_t lane, uint32_t component) {
    const auto& f = forms[c.form];
    if (!f.scale) return base(c, lane) + c.literal + component * 4;
    const uint32_t offset = component < f.unit ? c.literal & 255 : (c.literal >> 8) & 255;
    return base(c, lane) + offset * f.scale + (component % f.unit) * 4;
}
inline uint32_t source(const Case& c, uint32_t component) {
    const auto& f = forms[c.form];
    return f.scale && (c.literal & 255) == ((c.literal >> 8) & 255) ? component % f.unit
                                                                    : component;
}
inline uint32_t word(uint32_t lane, uint32_t k, bool old) {
    return (old ? 0xd0192735u : 0x8192a3b4u) ^ lane * 0x1020305u ^ k * 0x31415927u;
}
inline LaneProgram program(const Case& c) {
    using namespace prosper::gpu;
    const auto& f = forms[c.form];
    auto p = lane_program({}, c.wave, c.wave);
    p.code.clear();
    p.code.reserve(80);
    const std::array<uint32_t, 14> regs{2, 8, 9, 10, 11, 4, 16, 17, 18, 19, 24, 25, 26, 27};
    for (uint32_t a = 0; a < regs.size(); ++a) {
        p.code.push_back(0xe0302000u | (a * c.wave * 4));
        p.code.push_back(0x80020000u | (regs[a] << 8));
    }
    p.code.push_back(0xbf8c0000u);
    for (uint32_t k = 0; k < f.width; ++k) {
        if (source(c, k) != k) continue; // equal-offset writes use DATA0 once
        p.code.push_back(0xd8340000u);
        p.code.push_back(((f.store ? 16u : 8u) + k) << 8 | (24 + k));
    }
    p.code.insert(p.code.end(), {0xbf8c0000u, 0x7daa0880u});
    p.code.push_back(0xd8000000u | (f.opcode << 18) | c.literal);
    p.code.push_back(f.store ? (8u << 8) | ((f.scale ? 8 + f.unit : 0) << 16) | 2
                             : (16u << 24) | 2);
    p.code.push_back(0xbf8c0000u);
    if (f.store) {
        p.code.push_back(0xbefe04c1u);
        for (uint32_t k = 0; k < f.width; ++k) {
            p.code.push_back(0xd8d80000u);
            p.code.push_back(((16 + k) << 24) | (24 + k));
        }
    }
    p.code.insert(p.code.end(), {0xbf8c0000u, 0xbefe04c1u, 0xe0782000u, 0x80031000u, 0xbf810000u});
    p.config.lds_bytes = 65536;
    p.table.resources[0].binding = 2;
    p.table.resources[0].size = c.wave * 14 * 4;
    ShaderResource out;
    out.cls = ResourceClass::VertexBuffer;
    out.format = DataFormat::Uint32;
    out.num_components = 1;
    out.binding = 3;
    out.sgpr_base = 12;
    out.stride = 16;
    out.size = c.wave * 16 + 16;
    p.table.resources.push_back(out);
    return p;
}
struct Data {
    std::vector<uint32_t> input, initial, expected;
};
inline Data data(const Case& c) {
    const auto& f = forms[c.form];
    Data d;
    d.input.reserve(size_t{c.wave} * 14);
    d.initial.reserve(size_t{c.wave} * 4 + 4);
    for (uint32_t array = 0; array < 14; ++array)
        for (uint32_t lane = 0; lane < c.wave; ++lane)
            d.input.push_back(array == 0   ? base(c, lane)
                              : array < 5  ? word(lane, array - 1, false)
                              : array == 5 ? active(lane, c.wave)
                              : array < 10 ? word(lane, array - 6, true)
                                           : address(c, lane, array - 10));
    for (uint32_t k = 0; k < c.wave * 4 + 4; ++k) d.initial.push_back(0xa5f03c69u ^ k * 0x10203u);
    d.expected = d.initial;
    for (uint32_t lane = 0; lane < c.wave; ++lane)
        for (uint32_t k = 0; k < 4; ++k) {
            const bool live = active(lane, c.wave) && k < f.width;
            const uint32_t component = (live || (f.store && k < f.width)) ? source(c, k) : k;
            d.expected[size_t{lane} * 4 + k] = word(lane, component, !live);
        }
    return d;
}
} // namespace prosper::test::ir::ds_address
