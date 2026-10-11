#pragma once

// Project-owned ISA: an interpolation destination in the first selection is read on the
// skipped edge of a second selection. Its first join must retain both genuine definitions.
#include <cstdint>
#include <vector>

namespace prosper::test::fragment_vintrp_merge {
constexpr uint32_t interpolation(uint32_t opcode, uint32_t selector = 2) {
    return 0xc8000000u | (3u << 18) | (opcode << 16) | (1u << 8) | selector;
}
inline std::vector<uint32_t> program(uint32_t opcode = 1, bool one_arm = false,
                                     uint32_t selector = 2) {
    std::vector<uint32_t> code{
        0x7e060287u,   // v_mov_b32 v3,7: genuine pre-selection value
        0xbe800380u,   // s_mov_b32 s0,0
        0xbf068000u,   // s_cmp_eq_u32 s0,0
    };
    if (one_arm) {
        code.insert(code.end(), {0xbf840001u, interpolation(opcode, selector)});
    } else {
        code.insert(code.end(),
                    {0xbf840002u, 0x7e080289u, 0xbf820001u, interpolation(opcode, selector)});
    }
    code.insert(code.end(), {
                                0xbf068000u,   // fresh SCC for the second selection
                                0xbf840001u,   // s_cbranch_scc0 skips the v3 overwrite
                                0x2606068fu,   // v_and_b32 v3,15,v3
                                0xf8000801u,
                                0x00000003u,   // exp mrt0,v3,off,off,off done
                                0xbf810000u,
                            });
    return code;
}
} // namespace prosper::test::fragment_vintrp_merge
