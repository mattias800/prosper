// Synthetic scalar-buffer loads with genuine instruction-built V# words, shared by CPU tests
// and strict vertex/fragment SPIR-V validation. No resource is inferred from reflection defaults.
#pragma once

#include "gpu/execute/empty_scalar_buffer.hpp"
#include "gpu/execute/gpu_execute.hpp"

#include <cstdint>
#include <vector>

namespace prosper::test::graphics_empty_sbuffer {
inline void scalar_literal(std::vector<uint32_t>& words, uint32_t destination, uint32_t value) {
    words.push_back(0xbe8003ffu | (destination << 16u));
    words.push_back(value);
}

inline std::vector<uint32_t> program(bool pixel, uint32_t opcode = 0x8u, uint32_t stride = 0u,
                                     uint32_t records = 0u, uint32_t offset = 4u,
                                     bool missing_word = false, uint32_t soffset = 125u,
                                     uint32_t initial_soffset = UINT32_MAX) {
    std::vector<uint32_t> words;
    words.reserve(16u);
    // Keep the descriptor outside both actual graphics USER_DATA seed windows (PS s0..31,
    // VS s8..39). An omitted fourth word is then genuinely absent, not a supplied zero.
    scalar_literal(words, 64u, 0u);
    scalar_literal(words, 65u, stride << 16u);
    scalar_literal(words, 66u, records);
    if (!missing_word) scalar_literal(words, 67u, 0u);
    if (initial_soffset != UINT32_MAX) scalar_literal(words, soffset, initial_soffset);
    words.insert(words.end(), {0xf4000020u | (opcode << 18u), (soffset << 25u) | offset,
                               0x7e000200u, pixel ? 0xf800080fu : 0xf80008cfu, 0u, 0xbf810000u});
    return words;
}

inline prosper::gpu::ShaderResourceTable folded_resources(const std::vector<uint32_t>& words) {
    std::vector<prosper::gpu::SrtUse> uses;
    prosper::gpu::resolve_dynamic_fetch(words.data(), words.size(), nullptr, 0u, 0u, &uses);
    prosper::gpu::ShaderResourceTable table;
    for (const auto& use : uses)
        if (auto marker =
                prosper::gpu::empty_scalar_buffer_resource(use, words.data(), words.size()))
            table.resources.push_back(*marker);
    return table;
}
} // namespace prosper::test::graphics_empty_sbuffer
