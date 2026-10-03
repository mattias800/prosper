#pragma once

#include <cstdint>
#include <initializer_list>
#include <vector>

namespace prosper::test {
// Project-owned encodings: VOP2 MAX_U32 v1,v1,v1 and its DPP16 control word.
inline std::vector<uint32_t> dpp_row_max_program(std::initializer_list<uint32_t> shifts,
                                                 bool narrowed_exec = false, bool add = false) {
    std::vector<uint32_t> code;
    // Input v0 == input v2 selects EXEC; v1 remains genuine input on all lanes.
    if (narrowed_exec) code.push_back(0x7da40500u); // v_cmpx_eq_u32 v0,v2
    for (uint32_t shift : shifts) {
        code.push_back(((add ? 0x25u : 0x14u) << 25) | (1u << 17) | (1u << 9) | 0xfau);
        code.push_back(0xff000001u | ((0x110u + shift) << 8));
    }
    if (narrowed_exec) code.push_back(0xbefe04c1u); // s_mov_b64 exec,-1
    code.push_back(0xbf810000u);
    return code;
}
}  // namespace prosper::test
