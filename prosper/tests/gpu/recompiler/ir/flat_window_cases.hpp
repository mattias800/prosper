// Synthetic packets and byte-level expected values for the already-admitted window-load forms.
// AMD 70648 FLAT/GLOBAL load table: 08/09 byte, 0a/0b short, 0c..0f dword/x2/x4/x3.
// This is independent test data, not an opcode classifier used by lowering or admission.
#pragma once
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace prosper::test::ir::flat_window {
enum class Invalid { None, MissingWindow, MissingBase };
struct Load {
    uint32_t opcode, bytes, components;
    bool sign;
};
struct Case {
    uint32_t wave, segment, component;
    Load load;
    Invalid invalid = Invalid::None;
    bool admitted() const { return invalid == Invalid::None; }
};
inline std::vector<Case> cases() {
    constexpr std::array<Load, 8> loads{{{0x08, 1, 1, false},
                                         {0x09, 1, 1, true},
                                         {0x0a, 2, 1, false},
                                         {0x0b, 2, 1, true},
                                         {0x0c, 4, 1, false},
                                         {0x0d, 4, 2, false},
                                         {0x0e, 4, 4, false},
                                         {0x0f, 4, 3, false}}};
    std::vector<Case> out;
    out.reserve(64);
    for (uint32_t wave : {32u, 64u})
        for (uint32_t segment : {0u, 2u})
            for (const auto& load : loads)
                for (uint32_t component = 0; component < load.components; ++component)
                    out.push_back({wave, segment, component, load});
    for (uint32_t wave : {32u, 64u})
        for (uint32_t segment : {0u, 2u})
            for (Invalid invalid : {Invalid::MissingWindow, Invalid::MissingBase})
                out.push_back({wave, segment, 0, loads[4], invalid});
    return out;
}
inline LaneProgram program(const Case& c) {
    using namespace prosper::gpu;
    const std::vector<uint32_t> op{0xd70f6a0au,
                                   0x0002040cu, // v_add_co_u32 v10,vcc,s12,v2
                                   0x7e16020du,             // v_mov_b32 v11,s13
                                   0xdc000000u | (c.load.opcode << 18) | (c.segment << 14),
                                   0x107d000au,             // load v[16..],v[10:11],off
                                   0xbf8c3f70u,             // wait for the load
                                   0x7e020310u + c.component}; // v_mov_b32 v1,v[16+component]
    auto p = lane_program(op, c.wave, c.wave);
    constexpr uint32_t base = 0x00100000u;
    p.config.user_sgprs.assign(14, 0u);
    p.config.user_sgprs[12] = base;
    p.config.user_sgprs[13] = 0;
    ShaderResource window{};
    window.cls = ResourceClass::ConstantBuffer;
    window.format = DataFormat::Uint32;
    window.num_components = 1;
    window.binding = 2;
    window.gpu_addr = base;
    window.size = 128 * 4;
    window.stride = 4;
    window.flat_base_sgpr = c.invalid == Invalid::MissingBase ? 0xffffffffu : 12u;
    window.fetch_pc = 0xffffffffu;
    for (size_t pc = 0; pc < p.code.size();) {
        const auto in = rdna2_decode_one(p.code.data() + pc, p.code.size() - pc);
        if (!in.len_dwords) break;
        if (in.fmt == Rdna2Format::FLAT) {
            window.fetch_pc = static_cast<uint32_t>(pc);
            break;
        }
        pc += in.len_dwords;
    }
    if (c.invalid != Invalid::MissingWindow) p.table.resources.push_back(window);
    return p;
}
inline std::vector<uint32_t> backing_words() {
    std::vector<uint32_t> out(128);
    for (uint32_t i = 0; i < out.size(); ++i) out[i] = 0x13579bdfu ^ (i * 0x0103070bu);
    return out;
}
struct Data {
    std::vector<uint32_t> cbuf, input, expected;
};
inline Data data(const Case& c, bool all_active) {
    Data out;
    out.cbuf = backing_words();
    auto lanes = lane_inputs(0x464c4154u + c.wave, c.wave);
    std::vector<uint8_t> bytes(out.cbuf.size() * 4);
    for (size_t i = 0; i < bytes.size(); ++i)
        bytes[i] = static_cast<uint8_t>(out.cbuf[i / 4] >> ((i % 4) * 8));
    for (uint32_t lane = 0; lane < c.wave; ++lane) {
        // Naturally aligned shorts/dwords; all active addresses and every wide component fit.
        lanes.s0[lane] = 32u + 4u * lane +
                         (c.load.bytes == 1   ? lane % 4u
                          : c.load.bytes == 2 ? 2u * (lane % 2u)
                                              : 0u);
        if (all_active) lanes.active[lane] = true;
    }
    out.input.reserve(size_t{c.wave} * 4);
    for (const auto* array : {&lanes.s0, &lanes.s1, &lanes.old_d})
        out.input.insert(out.input.end(), array->begin(), array->end());
    for (bool active : lanes.active) out.input.push_back(active ? 1u : 0u);
    out.expected = out.input;
    for (uint32_t lane = 0; lane < c.wave; ++lane) {
        const size_t start = lanes.s0[lane] + c.component * 4u;
        uint32_t word = 0;
        for (uint32_t k = 0; k < c.load.bytes; ++k)
            word |= static_cast<uint32_t>(bytes[start + k]) << (k * 8u);
        if (c.load.sign) {
            const uint32_t bits = c.load.bytes * 8;
            if (word & (1u << (bits - 1))) word |= 0xffffffffu << bits;
        }
        out.expected[lane] = lanes.active[lane] ? word : lanes.old_d[lane];
    }
    return out;
}
}   // namespace prosper::test::ir::flat_window
