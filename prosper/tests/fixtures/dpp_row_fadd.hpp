#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace prosper::test {
enum class DppRowFaddCase { Linear, DivergentSites, LoopAndCompletedPeer, LaterBarrierPhase };

// Project-owned encodings: distinct VDST=v1, rotated SRC0=v2 and unpermuted SRC1=v3.
// v0/v7 select the branch; v4 supplies each wave's trip count; v5/v6 independently select EXEC.
inline std::vector<uint32_t> dpp_row_fadd_program(DppRowFaddCase shape) {
    std::vector<uint32_t> code;
    const auto add = [&] {
        code.insert(code.end(),
                    {(3u << 25) | (1u << 17) | (3u << 9) | 0xfau, 0xff080002u | (0x128u << 8)});
    };
    const auto move_result = [&] { code.push_back(0x7e040301u); }; // v_mov_b32 v2,v1
    const auto branch = [&](uint32_t opcode) {
        const size_t site = code.size();
        code.push_back(0xbf800000u | (opcode << 16));
        return site;
    };
    const auto target = [&](size_t site, size_t destination) {
        code[site] |= static_cast<uint16_t>(static_cast<int32_t>(destination) -
                                            static_cast<int32_t>(site + 1));
    };
    if (shape == DppRowFaddCase::Linear) {
        add();
    } else if (shape == DppRowFaddCase::LoopAndCompletedPeer) {
        code.push_back(0x7e000504u); // v_readfirstlane_b32 s0,v4
        code.push_back(0xbf068000u); // s_cmp_eq_u32 s0,0
        const size_t finished = branch(5);
        code.push_back(0x7da40d05u); // v_cmpx_eq_u32 v5,v6
        const size_t loop = code.size();
        add();
        move_result();
        code.push_back(0x80808100u); // s_sub_u32 s0,s0,1
        code.push_back(0xbf068000u);
        target(branch(4), loop);
        code.push_back(0xbefe04c1u); // restore EXEC before the genuine output observation
        target(finished, code.size());
    } else {
        if (shape == DppRowFaddCase::LaterBarrierPhase) {
            code.push_back(0x7e100291u); // earlier phase needs no row scratch
            code.push_back(0xbf8a0000u);
        }
        code.push_back(0x7d820f00u); // v_cmp_lt_u32 v0,v7
        const size_t alternate = branch(6);
        add();
        move_result();
        add();
        const size_t merge = branch(2);
        target(alternate, code.size());
        code.insert(code.end(), {0x7e0202fau, 0xff092802u}); // bounded MOV v1,v2 ROW_ROR8
        move_result();
        add();
        target(merge, code.size());
    }
    code.insert(code.end(), {0xf8000941u, 0x00000001u, 0xbf810000u});
    return code;
}
} // namespace prosper::test
