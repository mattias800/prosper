// split_t8_fold_harness.hpp -- run the scalar const-fold over a small guest program and ask whether
// an image use was published. Shared by the split-T# proof tests whose programs take their table
// pointer from entry user data s[2:3].
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "gpu/execute/gpu_execute.hpp"

namespace prosper::gpu::test {

// A table of recognisable words, so a published descriptor is visibly the table's.
inline uint32_t* split_t8_table() {
    alignas(16) static uint32_t table[16];
    return table;
}

// The entry user data is s0..s3; the table pointer is s[2:3].
inline std::vector<SrtUse> split_t8_uses_for(const std::vector<uint32_t>& code) {
    uint32_t* table = split_t8_table();
    for (uint32_t i = 0; i < 16; ++i) table[i] = 0xD1000000u + i;
    const auto base = reinterpret_cast<uint64_t>(table);
    const uint32_t seed[4] = {0u, 0u, static_cast<uint32_t>(base),
                              static_cast<uint32_t>(base >> 32u)};
    std::vector<SrtUse> result;
    resolve_dynamic_fetch(code.data(), code.size(), seed, 4, 0, &result);
    return result;
}

// Whether an image (kind 0) use was published at `pc`.
inline bool split_t8_has_image_use(const std::vector<SrtUse>& uses, uint32_t pc) {
    return std::any_of(uses.begin(), uses.end(),
                       [pc](const SrtUse& u) { return u.kind == 0 && u.use_pc == pc; });
}

}   // namespace prosper::gpu::test
