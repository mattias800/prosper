// Project-owned buffer-load programs. Input and output use binding3; independent packed data
// uses binding4. Every requested byte, including the adjacent join word, lies in that backing.
#pragma once

#include "gpu/recompiler/ir/buffer_format_cases.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <cstdlib>
#include <iterator>

#include <array>
#include <cstdint>
#include <vector>

namespace prosper::test::ir::buffer_format {
using prosper::gpu::DataFormat;
struct Format {
    DataFormat format;
    BufferFieldKind kind;
    uint32_t component_bytes, mtbuf_format;
    bool packed;
};
// AMD 70648 BUF_FMT values for four components (three for 10_11_11).
inline const std::array<Format, 14> kFormats{{
    {DataFormat::Uint8, BufferFieldKind::Unsigned, 1, 60, false},
    {DataFormat::Sint8, BufferFieldKind::Signed, 1, 61, false},
    {DataFormat::Uint16, BufferFieldKind::Unsigned, 2, 69, false},
    {DataFormat::Sint16, BufferFieldKind::Signed, 2, 70, false},
    {DataFormat::Float16, BufferFieldKind::Half, 2, 71, false},
    {DataFormat::Unorm8, BufferFieldKind::Unorm, 1, 56, false},
    {DataFormat::Snorm8, BufferFieldKind::Snorm, 1, 57, false},
    {DataFormat::Unorm16, BufferFieldKind::Unorm, 2, 65, false},
    {DataFormat::Snorm16, BufferFieldKind::Snorm, 2, 66, false},
    {DataFormat::Uint2_10_10_10, BufferFieldKind::Unsigned, 0, 54, true},
    {DataFormat::Sint2_10_10_10, BufferFieldKind::Signed, 0, 55, true},
    {DataFormat::Unorm2_10_10_10, BufferFieldKind::Unorm, 0, 50, true},
    {DataFormat::Snorm2_10_10_10, BufferFieldKind::Snorm, 0, 51, true},
    {DataFormat::Float10_11_11, BufferFieldKind::Ufloat, 0, 36, true},
}};

struct Case {
    uint32_t format = 0, component = 0, wave = 32, raw_opcode = 0;
    bool typed = false, dynamic = false;
    uint32_t offset = 0;
    bool admitted() const { return raw_opcode || !(dynamic && kFormats[format].packed); }
};

inline std::vector<Case> cases() {
    std::vector<Case> out;
    for (uint32_t wave : {32u, 64u}) {
        for (uint32_t f = 0; f < kFormats.size(); ++f)
            for (bool typed : {false, true})
                for (bool dynamic : {false, true})
                    for (uint32_t component = 0; component < 4; ++component)
                        out.push_back({f, component, wave, 0, typed, dynamic, dynamic ? 1u : 0u});
        for (uint32_t opcode : {8u, 9u, 10u, 11u})
            for (uint32_t offset : {0u, 3u})
                out.push_back({0, 0, wave, opcode, false, true, offset});
    }
    return out;
}

inline LaneProgram program(const Case& c) {
    const uint32_t mode = c.dynamic ? 0x1000u : 0x2000u;
    const uint32_t first = c.raw_opcode ? 0xe0000000u | (c.raw_opcode << 18) | mode | c.offset
                           : c.typed    ? 0xe8000000u | (kFormats[c.format].mtbuf_format << 19) |
                                              (3u << 16) | mode | c.offset
                                        : 0xe00c0000u | mode | c.offset;
    // After the load, copy the chosen component to v1 under the same partial EXEC.
    const uint32_t move = vop1(1, 1, 256 + 1 + c.component);
    auto p = lane_program({first, 0x80030102u, 0xbf8c3f70u, move}, c.wave, c.wave);
    prosper::gpu::ShaderResource r;
    r.cls = c.raw_opcode ? prosper::gpu::ResourceClass::ConstantBuffer
                         : prosper::gpu::ResourceClass::VertexBuffer;
    r.format = c.raw_opcode ? DataFormat::Uint32 : kFormats[c.format].format;
    r.num_components = c.raw_opcode                                         ? 1
                       : kFormats[c.format].kind == BufferFieldKind::Ufloat ? 3
                                                                            : 4;
    r.binding = 4;
    r.sgpr_base = 12;
    r.stride = 16;
    r.size = c.wave * 16 + 16;
    r.fetch_index_mode = prosper::gpu::VertexFetchIndexMode::Shader;
    if (c.typed) r.fetch_pc = 10; // exact instruction site; metadata fallback is not admitted
    p.table.resources.push_back(r);
    return p;
}

inline BufferFieldShape shape(const Case& c, uint32_t component) {
    if (c.raw_opcode)
        return {c.raw_opcode & 1 ? BufferFieldKind::Signed : BufferFieldKind::Unsigned, 0,
                c.raw_opcode >= 10 ? 16u : 8u};
    const auto& f = kFormats[c.format];
    const bool mini = f.kind == BufferFieldKind::Ufloat;
    const uint32_t width = f.packed ? (mini            ? component < 2 ? 11u : 10u
                                       : component < 3 ? 10u
                                                       : 2u)
                                    : f.component_bytes * 8;
    const uint32_t bit_offset = f.packed ? (mini ? component * 11 : component * 10) : 0;
    const uint32_t norm = (1u << (width - (f.kind == BufferFieldKind::Snorm))) - 1;
    return {f.kind, bit_offset, width, static_cast<float>(norm)};
}

inline uint32_t component_sample(const Case& c, uint32_t lane, uint32_t component) {
    const auto s = shape(c, component);
    if (s.kind == BufferFieldKind::Half) {
        constexpr uint32_t half[] = {0, 0x8000, 0x3c00, 0xbc00, 0x3800, 0x0400, 0x7bff, 1};
        return half[(lane + component * 3) % std::size(half)];
    }
    if (s.kind == BufferFieldKind::Ufloat) {
        const uint32_t mantissa = s.width - 5;
        return lane % 8 == 0 ? 0u
                             : (14u + component) << mantissa |
                                   ((lane * 13u + component) & ((1u << mantissa) - 1));
    }
    const uint32_t mask = (1u << s.width) - 1;
    const std::array<uint32_t, 5> edges{0, 1, mask >> 1, (mask >> 1) + 1, mask};
    return lane < 5 ? edges[(lane + component) % edges.size()]
                    : (lane * 73 + component * 29 + 0x51) & mask;
}

struct Data {
    LaneInputs lanes;
    std::vector<uint32_t> input, payload, expected;
};

inline Data data(const Case& c) {
    Data d;
    d.lanes = lane_inputs(0x5384u, c.wave);
    d.payload.assign(c.wave * 4 + 4, 0xa5a5a5a5u);
    const auto write_byte = [&](uint32_t address, uint32_t value) {
        uint32_t& word = d.payload[address / 4];
        const uint32_t shift = (address % 4) * 8;
        word = (word & ~(255u << shift)) | ((value & 255u) << shift);
    };
    for (uint32_t lane = 0; lane < c.wave; ++lane) {
        d.lanes.s0[lane] = c.dynamic ? lane * 16 + lane % 4 : lane;
        d.lanes.s1[lane] = 0x72000000u + lane;
        d.lanes.active[lane] = lane != 0 && lane != c.wave - 1 && lane % 5 != 2;
        const uint32_t base = lane * 16 + (c.dynamic ? lane % 4 + c.offset : 0);
        const bool packed = !c.raw_opcode && kFormats[c.format].packed;
        const uint32_t components = c.raw_opcode                                         ? 1
                                    : kFormats[c.format].kind == BufferFieldKind::Ufloat ? 3
                                                                                         : 4;
        uint32_t packed_word = 0;
        for (uint32_t component = 0; component < components; ++component) {
            const auto s = shape(c, component);
            const uint32_t raw = component_sample(c, lane, component);
            if (packed)
                packed_word |= raw << s.offset;
            else
                for (uint32_t byte = 0; byte < s.width / 8; ++byte)
                    write_byte(base + component * (s.width / 8) + byte, raw >> (byte * 8));
        }
        if (packed) {
            for (uint32_t byte = 0; byte < 4; ++byte)
                write_byte(base + byte, packed_word >> (byte * 8));
        }
        uint32_t answer = 0;
        if (c.component >= components)
            answer = c.typed ? 0u : 0x3f800000u;
        else {
            const auto s = shape(c, c.component);
            const uint32_t raw = packed ? packed_word : component_sample(c, lane, c.component);
            const auto expected = field_truth(raw, s);
            // This finite witness domain must never invent a value for an unknown result.
            if (!expected) std::abort();
            answer = *expected;
        }
        d.expected.push_back(d.lanes.active[lane] ? answer : d.lanes.old_d[lane]);
    }
    for (const auto* values : {&d.lanes.s0, &d.lanes.s1, &d.lanes.old_d})
        d.input.insert(d.input.end(), values->begin(), values->end());
    for (bool active : d.lanes.active) d.input.push_back(active ? 1u : 0u);
    return d;
}
}   // namespace prosper::test::ir::buffer_format
