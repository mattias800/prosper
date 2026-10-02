#pragma once
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace prosper::test::fragment_packet {
// Original project-owned RDNA2 instructions. This is a supplied register packet, NOT captured
// raster packing, a PS5 pixel oracle or evidence that any live draw may enter the packet route.
inline constexpr uint32_t poison_sentinel = 0xdeadbeefu;
inline constexpr uint32_t branch_false = 0x130bad01u, branch_true = 0x63060001u;
struct Case {
    uint32_t selected_lane = 63, data_base = 0x51000000u;
    bool inactive_source = false, leave_source_inactive = false, second_export = false;
};
inline void smov(std::vector<uint32_t>& p, uint32_t reg, uint32_t value) {
    p.insert(p.end(), {0xbe8003ffu | (reg << 16), value});
}
inline void vmov(std::vector<uint32_t>& p, uint32_t dst, uint32_t scalar) {
    p.push_back(0x7e000200u | (dst << 17) | scalar);
}
inline void exp(std::vector<uint32_t>& p, uint32_t en, uint32_t sources) {
    p.insert(p.end(), {0xf8001800u | en, sources}); // MRT0, DONE/VM retained
}
inline std::vector<uint32_t> guest(const Case& c) {
    std::vector<uint32_t> p;
    p.insert(p.end(), {0x7d8400ffu, c.selected_lane}); // V_CMP_EQ_U32 VCC,literal,v0
    p.push_back(0xbe94046au); // S_MOV_B64 s[20:21],VCC
    p.push_back(0xbe8c1014u); // S_BCNT1_I32_B64 s12,s[20:21]
    smov(p, 14, branch_false);
    const size_t branch = p.size();
    p.push_back(0xbf860000u); // VCCZ skips the observable scalar update
    smov(p, 14, branch_true);
    p[branch] |= static_cast<uint32_t>(p.size() - branch - 1);
    if (c.inactive_source) {
        p.push_back(0xbe90047eu); // S_MOV_B64 s[16:17],EXEC
        p.insert(p.end(), {0x7daa00ffu, 40}); // V_CMPX_NE_U32 EXEC,literal40,v0
    }
    p.insert(p.end(), {0xd760000fu, 264u | (168u << 9)}); // READLANE s15,v8,40
    if (c.inactive_source && !c.leave_source_inactive)
        p.push_back(0xbefe0410u); // restore EXEC from supplied/saved full mask
    vmov(p, 1, 12);
    vmov(p, 2, 14);
    vmov(p, 3, 15);
    exp(p, 15, 0x08030201u); // count, branch, readlane40, own raw v8
    if (c.second_export) {
        // A real common event AFTER the first EXP catches OpKill/Return of EXEC-off workers.
        p.insert(p.end(), {0xd7600013u, 264u | (168u << 9)}); // READLANE s19,v8,40
        vmov(p, 5, 19);
        exp(p, 1, 5);
    }
    p.push_back(0xbf810000u);
    return p;
}
inline prosper::gpu::FragmentInvocationPacket packet(const Case& c) {
    prosper::gpu::FragmentInvocationPacket p;
    p.guest_code = guest(c);
    p.slots_available.fill(true);
    p.export_enabled.fill(1);
    p.mask_state_available = true;
    p.exec_mask = UINT64_MAX;
    p.vcc_mask = 0;
    for (uint32_t reg : {0u, 1u, 2u, 3u, 5u, 8u}) {
        prosper::gpu::FragmentPacketVgpr column;
        column.reg = reg;
        for (uint32_t lane = 0; lane < 64; ++lane)
            column.words[lane] = reg == 0 ? lane : reg == 8 ? c.data_base + lane * 17
                                                                   : poison_sentinel;
        p.vgprs.push_back(column);
    }
    for (uint32_t reg : {12u, 14u, 15u, 16u, 17u, 19u, 20u, 21u})
        p.sgprs.emplace_back(reg, poison_sentinel);
    return p;
}
inline std::vector<uint32_t> expected(const Case& c,
                                     const prosper::gpu::FragmentInvocationPacket& p) {
    const uint32_t stride = (c.second_export ? 2u : 1u) * 12u;
    std::vector<uint32_t> out(stride * 64, 0);
    // Enumerate the logical packet's guest mask independently of the compiler/word packing.
    uint32_t ones = 0;
    bool any = false;
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const bool bit = ((p.exec_mask >> lane) & 1u) && lane == c.selected_lane;
        ones += bit;
        any |= bit;
    }
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const bool exec = ((p.exec_mask >> lane) & 1u) &&
            !(c.inactive_source && c.leave_source_inactive && lane == 40);
        const uint32_t base = stride * lane;
        const uint32_t header[] = {1, uint32_t(exec), p.export_enabled[lane], 0, 15, 0, 1, 1};
        std::copy(std::begin(header), std::end(header), out.begin() + base);
        out[base + 8] = exec ? ones : poison_sentinel;
        out[base + 9] = exec ? (any ? branch_true : branch_false) : poison_sentinel;
        out[base + 10] = exec ? c.data_base + 40 * 17 : poison_sentinel;
        out[base + 11] = c.data_base + lane * 17;
        if (c.second_export) {
            const uint32_t second[] = {1, uint32_t(exec), p.export_enabled[lane], 0, 1, 0, 1, 1};
            std::copy(std::begin(second), std::end(second), out.begin() + base + 12);
            out[base + 20] = exec ? c.data_base + 40 * 17 : poison_sentinel;
        }
    }
    return out;
}
} // namespace prosper::test::fragment_packet
