#pragma once

#include <cstdint>
#include <initializer_list>
#include <cstddef>
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

enum class DppRowCfgCase { Mixed, DivergentSites, LoopAndCompletedPeer, LaterBarrierPhase };

// All inputs are genuine per-lane words. v0/v2 select a wave branch; v3 gives its real scalar
// trip count via READFIRSTLANE; v4/v5 select EXEC independently of the unsigned values in v1.
inline std::vector<uint32_t> dpp_row_cfg_program(DppRowCfgCase shape, bool only_add = false) {
    std::vector<uint32_t> code;
    const auto row = [&](uint32_t shift, bool maximum) {
        code.push_back(((maximum && !only_add ? 0x14u : 0x25u) << 25) | (1u << 17) | (1u << 9) |
                       0xfau);
        code.push_back(0xff000001u | ((0x110u + shift) << 8));
    };
    const auto branch = [&](uint32_t opcode) {
        const size_t site = code.size();
        code.push_back(0xbf800000u | (opcode << 16));
        return site;
    };
    const auto target = [&](size_t site, size_t destination) {
        const auto delta = static_cast<int32_t>(destination) - static_cast<int32_t>(site + 1);
        code[site] |= static_cast<uint16_t>(delta);
    };
    if (shape == DppRowCfgCase::Mixed) {
        branch(2); // s_branch to its actual next instruction: select the CFG service
        row(1, false);
        row(2, true);
        row(4, false);
        row(8, true);
    } else if (shape == DppRowCfgCase::LoopAndCompletedPeer) {
        code.push_back(0x7e000503u);   // v_readfirstlane_b32 s0,v3: each wave's supplied trip count
        code.push_back(0xbf068000u);   // s_cmp_eq_u32 s0,0
        const size_t finished = branch(5);   // s_cbranch_scc1 end
        code.push_back(0x7da40b04u);   // v_cmpx_eq_u32 v4,v5
        const size_t loop = code.size();
        row(1, false);
        row(2, true);
        code.push_back(0x80808100u);   // s_sub_u32 s0,s0,1
        code.push_back(0xbf068000u);   // s_cmp_eq_u32 s0,0
        target(branch(4), loop);   // s_cbranch_scc0 loop
        code.push_back(0xbefe04c1u);   // s_mov_b64 exec,-1
        target(finished, code.size());
    } else {
        if (shape == DppRowCfgCase::LaterBarrierPhase) {
            code.push_back(
                0x7e100291u);   // v_mov_b32 v8,17: first phase needs no row scratch plane
            code.push_back(0xbf8a0000u);   // s_barrier
        }
        code.push_back(0x7d820500u);   // v_cmp_lt_u32 v0,v2
        const size_t alternate = branch(6);   // s_cbranch_vccz else
        row(1, true);
        row(2, true);
        const size_t merge = branch(2);   // s_branch end
        target(alternate, code.size());
        row(1, false);
        row(4, true);
        row(8, false);
        target(merge, code.size());
    }
    code.push_back(0xbf810000u);
    return code;
}

inline std::vector<uint32_t> dpp_row_cfg_export_program(DppRowCfgCase shape,
                                                        bool only_add = false) {
    auto code = dpp_row_cfg_program(shape, only_add);
    code.insert(code.end() - 1, {0xf8000941u, 0x00000001u});   // EXP PRIM,v1: raw unsigned sink
    return code;
}
}   // namespace prosper::test
