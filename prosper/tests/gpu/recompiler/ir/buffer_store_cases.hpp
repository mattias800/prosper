// AMD 70648 buffer field layouts; finite eighths have hand-worked half encodings.
// Expected storage is independent byte replacement, never a reload through prosper's decoder.
#pragma once
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <array>
#include <bit>
#include <cstdint>
#include <vector>

namespace prosper::test::ir::buffer_store {
using prosper::gpu::DataFormat;
enum class Kind { Integer, Half, Unorm, Snorm, Ufloat };
struct Format {
    DataFormat format;
    Kind kind;
    uint32_t bytes, encoding, components;
};
inline const std::array<Format, 14> formats{{
    {DataFormat::Uint8, Kind::Integer, 1, 60, 4},
    {DataFormat::Sint8, Kind::Integer, 1, 61, 4},
    {DataFormat::Uint16, Kind::Integer, 2, 69, 4},
    {DataFormat::Sint16, Kind::Integer, 2, 70, 4},
    {DataFormat::Float16, Kind::Half, 2, 71, 4},
    {DataFormat::Unorm8, Kind::Unorm, 1, 56, 4},
    {DataFormat::Snorm8, Kind::Snorm, 1, 57, 4},
    {DataFormat::Unorm16, Kind::Unorm, 2, 65, 4},
    {DataFormat::Snorm16, Kind::Snorm, 2, 66, 4},
    {DataFormat::Uint2_10_10_10, Kind::Integer, 0, 54, 4},
    {DataFormat::Sint2_10_10_10, Kind::Integer, 0, 55, 4},
    {DataFormat::Unorm2_10_10_10, Kind::Unorm, 0, 50, 4},
    {DataFormat::Snorm2_10_10_10, Kind::Snorm, 0, 51, 4},
    {DataFormat::Float10_11_11, Kind::Ufloat, 0, 36, 3},
}};
struct Case {
    uint32_t format, wave, offset;
    bool typed;
    uint32_t raw_opcode = 0;
};
inline std::vector<Case> cases() {
    std::vector<Case> out;
    for (uint32_t wave : {32u, 64u})
        for (uint32_t f = 0; f < formats.size(); ++f)
            for (bool typed : {false, true}) {
                const auto& fmt = formats[f];
                const uint32_t offsets = fmt.kind == Kind::Integer && fmt.bytes ? 4 / fmt.bytes : 1;
                for (uint32_t o = 0; o < offsets; ++o)
                    out.push_back({f, wave, o * fmt.bytes, typed});
            }
    return out;
}
// n/8, including values beyond the normalized interval. Both representations are literal truth.
inline constexpr std::array<int32_t, 13> eighths{-12, -8, -4, -2, 0, 1, 2, 4, 6, 8, 10, 12, 3};
inline constexpr std::array<uint32_t, 13> halves{0xbe00, 0xbc00, 0xb800, 0xb400, 0,
                                                 0x3000, 0x3400, 0x3800, 0x3a00, 0x3c00,
                                                 0x3d00, 0x3e00, 0x3600};
inline uint32_t sample_index(uint32_t lane, uint32_t component) {
    return (lane * 5 + component * 3) % eighths.size();
}
inline uint32_t input_word(const Case& c, uint32_t lane, uint32_t component) {
    if (formats[c.format].kind == Kind::Integer)
        return 0x8192a3b4u ^ lane * 0x1020305u ^ component * 0x31415927u;
    return std::bit_cast<uint32_t>(static_cast<float>(eighths[sample_index(lane, component)]) /
                                   8.0f);
}
inline uint32_t field(const Case& c, uint32_t lane, uint32_t component) {
    const auto& f = formats[c.format];
    const uint32_t width = f.bytes                  ? f.bytes * 8
                           : f.kind == Kind::Ufloat ? (component < 2 ? 11 : 10)
                                                    : (component < 3 ? 10 : 2);
    const uint32_t mask = (1u << width) - 1;
    if (f.kind == Kind::Integer) return input_word(c, lane, component) & mask;
    const uint32_t i = sample_index(lane, component);
    if (f.kind == Kind::Half) return halves[i];
    if (f.kind == Kind::Ufloat) return eighths[i] < 0 ? 0 : halves[i] >> (width == 11 ? 4 : 5);
    int32_t n = eighths[i];
    if (n > 8) n = 8;
    if (n < (f.kind == Kind::Snorm ? -8 : 0)) n = f.kind == Kind::Snorm ? -8 : 0;
    const uint32_t maximum = (1u << (width - (f.kind == Kind::Snorm))) - 1;
    const uint32_t magnitude = static_cast<uint32_t>(n < 0 ? -n : n) * maximum;
    uint32_t rounded = magnitude / 8;
    const uint32_t tail = magnitude % 8;
    if (tail > 4 || (tail == 4 && (rounded & 1))) ++rounded;
    return (n < 0 ? 0u - rounded : rounded) & mask;
}
inline bool active(uint32_t lane, uint32_t wave) {
    return lane != 0 && lane != wave - 1 && lane % 5 != 2;
}
inline LaneProgram program(const Case& c, uint32_t opcode = 7) {
    using namespace prosper::gpu;
    auto p = lane_program({}, c.wave, c.wave);
    p.code.clear();
    const std::array<uint32_t, 6> regs{8, 9, 10, 11, 6, 4};
    for (uint32_t a = 0; a < regs.size(); ++a) {
        p.code.push_back(0xe0302000u | (a * c.wave * 4));
        p.code.push_back(0x80020000u | (regs[a] << 8));
    }
    p.code.insert(p.code.end(), {0xbf8c3f70u, 0x7daa0880u});
    if (c.raw_opcode) opcode = c.raw_opcode;
    p.code.push_back((c.typed ? 0xe8000000u | (formats[c.format].encoding << 19) | (opcode << 16)
                              : 0xe0000000u | (opcode << 18)) |
                     0x2000u | c.offset);
    p.code.insert(p.code.end(), {0x80030806u, 0xbf8c3f70u, 0xbf810000u});
    p.table.resources[0].binding = 2;
    p.table.resources[0].size = c.wave * 6 * 4;
    ShaderResource r;
    r.cls = ResourceClass::VertexBuffer;
    r.format = c.raw_opcode ? DataFormat::Uint32 : formats[c.format].format;
    r.num_components = c.raw_opcode ? 1 : formats[c.format].components;
    r.binding = 3;
    r.sgpr_base = 12;
    r.stride = 16;
    r.size = c.wave * 16 + 16;
    r.fetch_index_mode = VertexFetchIndexMode::Shader;
    if (c.typed) r.fetch_pc = 14;
    p.table.resources.push_back(r);
    return p;
}
struct Data {
    std::vector<uint32_t> input, initial, expected;
};
inline Data data(const Case& c) {
    Data d;
    for (uint32_t component = 0; component < 4; ++component)
        for (uint32_t lane = 0; lane < c.wave; ++lane)
            d.input.push_back(input_word(c, lane, component));
    for (uint32_t lane = 0; lane < c.wave; ++lane) d.input.push_back((lane * 7) % c.wave);
    for (uint32_t lane = 0; lane < c.wave; ++lane) d.input.push_back(active(lane, c.wave));
    for (uint32_t word = 0; word < c.wave * 4 + 4; ++word)
        d.initial.push_back(0xa5f03c69u ^ word * 0x10203u);
    d.expected = d.initial;
    const auto put_byte = [&](uint32_t address, uint32_t byte) {
        const uint32_t shift = (address % 4) * 8;
        auto& word = d.expected[address / 4];
        word = (word & ~(255u << shift)) | ((byte & 255) << shift);
    };
    const auto& f = formats[c.format];
    for (uint32_t lane = 0; lane < c.wave; ++lane) {
        if (!active(lane, c.wave)) continue;
        const uint32_t address = ((lane * 7) % c.wave) * 16 + c.offset;
        if (c.raw_opcode) {
            // RDNA2 BUFFER_STORE_DWORD, X2, X4, X3; full words, independent of FORMAT.
            const uint32_t width = c.raw_opcode == 0x1c   ? 1
                                   : c.raw_opcode == 0x1d ? 2
                                   : c.raw_opcode == 0x1e ? 4
                                                          : 3;
            for (uint32_t k = 0; k < width; ++k)
                for (uint32_t byte = 0; byte < 4; ++byte)
                    put_byte(address + k * 4 + byte, input_word(c, lane, k) >> (byte * 8));
        } else if (!f.bytes) {
            uint32_t packed = 0;
            for (uint32_t k = 0; k < f.components; ++k)
                packed |= field(c, lane, k) << (f.kind == Kind::Ufloat ? k * 11 : k * 10);
            for (uint32_t byte = 0; byte < 4; ++byte)
                put_byte(address + byte, packed >> (byte * 8));
        } else
            for (uint32_t k = 0; k < f.components; ++k)
                for (uint32_t byte = 0; byte < f.bytes; ++byte)
                    put_byte(address + k * f.bytes + byte, field(c, lane, k) >> (byte * 8));
    }
    return d;
}
} // namespace prosper::test::ir::buffer_store
