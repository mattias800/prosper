// split_t8_fold_harness.hpp -- run the scalar const-fold over a small guest program and ask whether
// an image use was published. Shared by the split-T# proof tests whose programs take their table
// pointer from entry user data s[2:3].
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "gpu/execute/gpu_execute.hpp"

namespace prosper::gpu::test {

// A table of recognisable words, so a published descriptor is visibly the table's.
inline uint32_t* split_t8_table() {
    alignas(16) static std::array<uint32_t, 16> table{};
    return table.data();
}

// Runs the fold over `dwords` words of `code` with `nsgpr` entry user-data registers, all zero except
// the table pointer in s[pointer:pointer+1].
inline std::vector<SrtUse> split_t8_uses_for(const uint32_t* code, size_t dwords, int pointer,
                                             uint32_t nsgpr) {
    uint32_t* table = split_t8_table();
    for (uint32_t i = 0; i < 16; ++i) table[i] = 0xD1000000u + i;
    const auto base = reinterpret_cast<uint64_t>(table);
    std::vector<uint32_t> seed(nsgpr, 0u);
    seed[static_cast<size_t>(pointer)] = static_cast<uint32_t>(base);
    seed[static_cast<size_t>(pointer) + 1u] = static_cast<uint32_t>(base >> 32u);
    std::vector<SrtUse> result;
    resolve_dynamic_fetch(code, dwords, seed.data(), nsgpr, 0, &result);
    return result;
}

// The entry user data is s0..s3; the table pointer is s[2:3].
inline std::vector<SrtUse> split_t8_uses_for(const std::vector<uint32_t>& code) {
    return split_t8_uses_for(code.data(), code.size(), 2, 4u);
}

// Black Flag fragment program 0x407eecf000's body, verbatim (pc in dwords): pc3 s_load_dwordx8
// s[0:7], s[10:11]; pc5 s_load_dwordx4 s[8:11], s[10:11], 0x20; pc10 image_load using the T# in
// s[4:11]; pc16 a conditional branch to the block after the first s_endpgm (pc 21). Its table
// pointer is entry user data s[10:11] of s0..s11.
inline const std::vector<uint32_t>& split_t8_tail_block_body() {
    static const std::vector<uint32_t> body = {
        0xBFA00001u, 0x7E000F02u, 0x7E020F03u, 0xF40C0005u, 0xFA000000u, 0xF4080205u, 0xFA000020u,
        0xBF8CC07Fu, 0xF4201A80u, 0xFA000000u, 0xF0000108u, 0x00010000u, 0xBF8C0070u, 0x3600006Au,
        0x7D840080u, 0x8AEA6A7Eu, 0xBF840004u, 0xBEFE046Au, 0xF8001890u, 0x00000000u, 0xBF810000u,
    };
    return body;
}

// Whether an image (kind 0) use was published at `pc`.
inline bool split_t8_has_image_use(const std::vector<SrtUse>& uses, uint32_t pc) {
    return std::any_of(uses.begin(), uses.end(),
                       [pc](const SrtUse& u) { return u.kind == 0 && u.use_pc == pc; });
}

}   // namespace prosper::gpu::test
