// Full-compute SMEM witnesses. Descriptor facts are synthetic; reads/proofs stay in production.
#pragma once
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace prosper::test::ir::smem_index {
enum class Invalid { None, MissingOffset, MissingAddress, NegativeImmediateOnly };
struct Case {
    bool pcrel;
    uint32_t wave, width, component, soffset, immediate;
    Invalid invalid = Invalid::None;
    bool admitted() const { return invalid == Invalid::None; }
};
inline std::vector<Case> cases() {
    std::vector<Case> out;
    out.reserve(628);
    for (uint32_t wave : {32u, 64u})
        for (uint32_t width : {1u, 2u, 4u, 8u, 16u})
            for (uint32_t k = 0; k < width; ++k) {
                for (const auto& offset : std::array<std::array<uint32_t, 2>, 6>{
                         {{0, 0}, {4, 0xfffffffcu}, {3, 1}, {508, 0}, {512, 0}, {12, 4}}})
                    out.push_back({false, wave, width, k, offset[0], offset[1]});
                for (uint32_t soffset : {0u, 4u, 16u, 20u})
                    out.push_back({true, wave, width, k, soffset, 0});
            }
    for (uint32_t wave : {32u, 64u}) {
        out.push_back({false, wave, 2, 1, 4, 0, Invalid::MissingOffset});
        out.push_back({true, wave, 2, 1, 4, 0, Invalid::MissingOffset});
        out.push_back({false, wave, 2, 1, 4, 0, Invalid::MissingAddress});
        out.push_back({false, wave, 2, 1, 0, 0xfffffffcu, Invalid::NegativeImmediateOnly});
    }
    return out;
}
inline std::vector<uint32_t> constant_words(bool pcrel) {
    std::vector<uint32_t> out;
    const uint32_t count = pcrel ? 5 : 128;
    out.reserve(count);
    for (uint32_t i = 0; i < count; ++i)
        out.push_back(std::bit_cast<uint32_t>((pcrel ? 3001.0f : 2001.0f) + static_cast<float>(i)));
    return out;
}
inline LaneProgram program(const Case& c) {
    using namespace prosper::gpu;
    std::vector<uint32_t> op;
    const auto move = [&](uint32_t reg, uint32_t value) {
        op.insert(op.end(), {0xbe8003ffu | (reg << 16), value});
    };
    if (c.pcrel) {
        op.insert(op.end(), {0xbe981f00u, 0x801818ffu, 0u, 0x82191980u});
        move(26, 20);
        move(27, 0x10005004u);
    }
    if (c.invalid != Invalid::MissingOffset) move(32, c.soffset);
    const uint32_t opcode = 8u + static_cast<uint32_t>(std::countr_zero(c.width));
    const uint32_t source = c.invalid == Invalid::NegativeImmediateOnly ? 125 : 32;
    op.insert(op.end(), {0xf4000000u | (opcode << 18) | (16u << 6) | (c.pcrel ? 12u : 0u),
                         (source << 25) | (c.immediate & 0x1fffffu), 0xbf8cc07fu,
                         0x7e020200u | (16u + c.component)});
    auto p = lane_program(op, c.wave, c.wave);
    if (c.pcrel) {
        const auto getpc = std::find(p.code.begin(), p.code.end(), 0xbe981f00u);
        const auto pc = static_cast<size_t>(getpc - p.code.begin());
        p.code[pc + 2] = static_cast<uint32_t>(p.code.size() * 4 - (pc + 1) * 4);
        const auto words = constant_words(true);
        p.code.insert(p.code.end(), words.begin(), words.end());
    } else {
        ShaderResource data;
        data.cls = ResourceClass::ConstantBuffer;
        data.format = DataFormat::Uint32;
        data.num_components = 1;
        data.binding = 2;
        data.gpu_addr = c.invalid == Invalid::MissingAddress ? 0 : 0x1000;
        data.size = 128 * 4;
        data.stride = 4;
        data.sgpr_base = 0;
        data.scalar_buffer_dword_count = 128;
        p.table.resources.push_back(data);
        uint32_t smem_pc = 0;
        for (size_t pc = 0; pc < p.code.size();) {
            const auto in = rdna2_decode_one(p.code.data() + pc, p.code.size() - pc);
            if (!in.len_dwords) break;
            if (in.fmt == Rdna2Format::SMEM) {
                smem_pc = static_cast<uint32_t>(pc);
                break;
            }
            pc += in.len_dwords;
        }
        ShaderResource selector;
        selector.cls = ResourceClass::ConstantBuffer;
        selector.format = DataFormat::Uint32;
        selector.num_components = 1;
        selector.binding = 4;
        selector.sgpr_base = 64;
        selector.stride = 4;
        selector.table_index_count = 2;
        selector.table_entry_stride = 120;
        selector.table_selector_mode = BufferTableSelectorMode::DynamicSbufferByteOffset;
        selector.table_load_pc = smem_pc;
        selector.table_entries.resize(2);
        p.table.resources.push_back(selector);
    }
    return p;
}
struct Data {
    std::vector<uint32_t> cbuf, input, expected;
};
inline Data data(const Case& c) {
    Data out;
    out.cbuf = constant_words(false);
    const auto lanes = lane_inputs(0x5a17u, c.wave);
    out.input.reserve(size_t{c.wave} * 4);
    for (const auto* array : {&lanes.s0, &lanes.s1, &lanes.old_d})
        out.input.insert(out.input.end(), array->begin(), array->end());
    for (bool active : lanes.active) out.input.push_back(active ? 1u : 0u);
    out.expected = out.input;
    const auto words = constant_words(c.pcrel);
    const uint32_t sum = c.soffset + c.immediate;
    const uint32_t index = sum / 4u + c.component;
    const uint32_t word = index < words.size() ? words[index] : 0;
    for (uint32_t lane = 0; lane < c.wave; ++lane)
        out.expected[lane] = lanes.active[lane] ? word : lanes.old_d[lane];
    return out;
}
} // namespace prosper::test::ir::smem_index
