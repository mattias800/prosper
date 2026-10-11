#pragma once

// Hand-built Wave32 branch join: one arm retires s20's B32 compare mask; a fresh
// compare replaces it before the next mask read. These are synthetic ISA words.
#include <cstdint>
#include <vector>

namespace prosper::test::fragment_compare_kill {
constexpr uint32_t end = 0xbf810000u;
constexpr uint32_t mov(uint32_t destination, uint32_t source) {
    return 0xbe800300u | (destination << 16) | source;
}
constexpr uint32_t compare(uint32_t opcode = 0xc2u, uint32_t source1 = 1) {
    return 0x7c0000f9u | (opcode << 17) | (source1 << 9);
}
constexpr uint32_t compare_sd(uint32_t destination, uint32_t source0 = 0,
                              bool scalar_source = false) {
    return 0x06068000u | (destination << 8) | source0 | (scalar_source ? 1u << 23 : 0u);
}

inline std::vector<uint32_t> program(bool unrelated_vector = false) {
    std::vector<uint32_t> code{
        compare(),    compare_sd(20),          // pc0: v_cmp_eq_u32_sdwa s20,v0,v1
        0x7d820300u,                        // pc2: v_cmp_lt_u32 vcc,v0,v1
        0xbe983c6au,                        // pc3: s_and_saveexec_b32 s24,vcc_lo
        0xbf880001u,                        // pc4: s_cbranch_execz pc6
        mov(20, 135),                       // pc5: s_mov_b32 s20,7 (taken arm)
        mov(126, 24),                       // pc6: restore EXEC_LO from s24
    };
    if (unrelated_vector) code.push_back(0x7e040287u); // v_mov_b32 v2,7
    code.insert(code.end(),
                {compare(), compare_sd(20), mov(126, 20), 0xf800180fu, 0x01000100u, end});
    return code;
}
} // namespace prosper::test::fragment_compare_kill
