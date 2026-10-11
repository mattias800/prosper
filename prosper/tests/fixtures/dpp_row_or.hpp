#pragma once

#include "dpp_row_max.hpp"

namespace prosper::test {
// Project-owned VOP2 encodings share the exact full-mask, FI0/BC0 row contract.
// Substitute the combiner in synthetic MAX/ADD streams, retaining each branch and static site.
inline std::vector<uint32_t> dpp_row_or(std::vector<uint32_t> words) {
    for (auto& word : words) {
        const uint32_t packet = word & 0xfe0001ffu;
        if (packet == 0x280000fau || packet == 0x4a0000fau)
            word = (word & 0x81ffffffu) | (0x1cu << 25u);
    }
    return words;
}

inline std::vector<uint32_t> dpp_row_or_program(std::initializer_list<uint32_t> shifts,
                                                bool narrowed_exec = false) {
    return dpp_row_or(dpp_row_max_program(shifts, narrowed_exec));
}
}  // namespace prosper::test
