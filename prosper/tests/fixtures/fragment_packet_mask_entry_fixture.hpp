#pragma once
#include "fragment_packet_raw_masks_fixture.hpp"
#include "gpu/recompiler/fragment_packet_mask_requirements.hpp"

namespace prosper::test::fragment_mask_entry {
using namespace prosper::gpu;
namespace fp = fragment_packet;
namespace fd = fragment_definedness;
struct Case {
    std::string name;
    FragmentInvocationPacket packet;
    std::vector<uint32_t> expected;
};
inline FragmentInvocationPacket base() {
    auto p = fd::base();
    p.mask_state_available = false;
    p.exec_available = true;
    p.vcc_mask = UINT64_MAX; // absent storage deliberately disagrees with ordinary VCC writers
    p.scc = true;
    fd::column(p, 1, fp::poison_sentinel);
    return p;
}
inline void finish(FragmentInvocationPacket& p, uint32_t scalar) {
    fp::vmov(p.guest_code, 1, scalar);
    fp::exp(p.guest_code, 1, 1);
    p.guest_code.push_back(0xbf810000u);
}
inline Case exec_writer(uint64_t mask, bool boundary) {
    auto p = base();
    p.exec_available = false;
    for (auto& [reg, value] : p.sgprs)
        if (reg == 20 || reg == 21) value = uint32_t(mask >> (reg == 21 ? 32 : 0));
    p.guest_code = {
        0xbefe0414u}; // complete genuine numeric s20:21 -> EXEC, independent of OLD EXEC
    if (boundary) p.guest_code.push_back(0xbf820000u);
    finish(p, 0);
    return {"exec_writer_" + std::to_string(mask) + (boundary ? "_boundary" : "_same"), p,
            fd::expected(mask, 0x42230011u)};
}
inline Case old_scc(bool value) {
    auto p = base();
    p.scc_available = true;
    p.scc = value;
    fp::smov(p.guest_code, 6, fp::branch_false); // MOV leaves OLD SCC untouched
    p.guest_code.push_back(0xbf840002u);
    fp::smov(p.guest_code, 6, fp::branch_true);
    finish(p, 6);
    return {value ? "old_scc_one" : "old_scc_zero", p,
            fd::expected(UINT64_MAX, value ? fp::branch_true : fp::branch_false)};
}
inline Case new_scc(bool carry = true) {
    auto p = base();
    p.sgprs.emplace_back(2, UINT32_MAX);
    p.sgprs.emplace_back(3, carry ? 2 : 0);
    fp::smov(p.guest_code, 6, fp::branch_false);
    p.guest_code.push_back(0x800a0302u); // S_ADD_U32 s10,s2,s3: genuine carry, not OLD SCC
    p.guest_code.push_back(0xbf840002u);
    fp::smov(p.guest_code, 6, fp::branch_true);
    finish(p, 6);
    return {carry ? "defined_scc_carry_before_branch" : "defined_scc_no_carry_before_branch", p,
            fd::expected(UINT64_MAX, carry ? fp::branch_true : fp::branch_false)};
}
inline Case cmpx_vcc(bool nonzero) {
    auto p = base();
    p.vcc_available = true;
    p.vcc_mask = nonzero ? UINT64_MAX : 0;
    FragmentPacketVgpr index;
    index.reg = 0;
    for (uint32_t lane = 0; lane < 64; ++lane) index.words[lane] = lane;
    p.vgprs.push_back(index);
    p.guest_code = {0x7daa00ffu, 40}; // CMPX changes EXEC, NEVER defines VCC
    fp::smov(p.guest_code, 6, fp::branch_false);
    p.guest_code.push_back(0xbf860002u);
    fp::smov(p.guest_code, 6, fp::branch_true);
    finish(p, 6);
    return {nonzero ? "cmpx_old_vcc_nonzero" : "cmpx_old_vcc_zero", p,
            fd::expected(UINT64_MAX ^ (uint64_t(1) << 40),
                         nonzero ? fp::branch_true : fp::branch_false)};
}
inline Case scalar_vcc(uint64_t mask) {
    auto p = base();
    for (auto& [reg, value] : p.sgprs)
        if (reg == 20 || reg == 21) value = uint32_t(mask >> (reg == 21 ? 32 : 0));
    fp::smov(p.guest_code, 6, fp::branch_false);
    p.guest_code.push_back(0xbeea0414u); // genuine complete numeric VCC writer ignores EXEC
    p.guest_code.push_back(0xbf860002u);
    fp::smov(p.guest_code, 6, fp::branch_true);
    finish(p, 6);
    return {"scalar_vcc_writer_" + std::to_string(mask), p,
            fd::expected(UINT64_MAX, mask ? fp::branch_true : fp::branch_false)};
}
inline Case peer_before_exec() {
    auto p = exec_writer(UINT64_MAX, true);
    p.name = "exec_absent_during_genuine_inactive_peer_read";
    FragmentPacketVgpr peer;
    peer.reg = 8;
    for (uint32_t lane = 0; lane < 64; ++lane) peer.words[lane] = 0x61000000u + lane;
    p.packet.vgprs.push_back(peer);
    p.packet.guest_code.insert(
        p.packet.guest_code.begin(),
        {0xd7600006u, 264u | (168u << 9)}); // READLANE s6,v8,40 ignores EXEC
    p.packet.guest_code[4] = 0x7e020206u; // genuine peer result -> actual v1/raw EXP
    p.expected = fd::expected(UINT64_MAX, 0x61000028u);
    return p;
}
inline Case wqm_numeric() {
    constexpr uint64_t widened = uint64_t(15) << 40;
    auto c = exec_writer(uint64_t(1) << 40, true);
    c.name = "numeric_wqm_defines_exec_and_scc_without_entry_masks";
    c.packet.quad_topology = FragmentPacketQuadTopology::ConsecutiveLogicalQuads;
    c.packet.guest_code[0] = 0xbefe0a14u; // WQM_B64 EXEC,s20:21, not WQM(OLD EXEC)
    c.expected = fd::expected(widened, 0x42230011u);
    return c;
}
inline Case joined_vcc(bool select_first) {
    auto p = base();
    p.scc_available = true;
    p.scc = select_first;
    p.sgprs.emplace_back(22, 0);
    p.sgprs.emplace_back(23, 0);
    fp::smov(p.guest_code, 6, fp::branch_false);
    p.guest_code.insert(p.guest_code.end(),
                        {0xbf840002u, 0xbeea0414u, 0xbf820001u, 0xbeea0416u, 0xbf860002u});
    fp::smov(p.guest_code, 6, fp::branch_true);
    finish(p, 6);
    return {select_first ? "join_first_full_vcc_writer" : "join_second_full_vcc_writer", p,
            fd::expected(UINT64_MAX, select_first ? fp::branch_true : fp::branch_false)};
}
inline Case bypass_vcc(bool nonzero) {
    auto c = joined_vcc(false);
    c.name = nonzero ? "bypass_requires_old_vcc_nonzero" : "bypass_requires_old_vcc_zero";
    c.packet.guest_code[2] = 0xbf840003u; // genuine SCC0 path skips BOTH full VCC writers
    c.packet.vcc_available = true;
    c.packet.vcc_mask = nonzero ? UINT64_MAX : 0;
    c.expected = fd::expected(UINT64_MAX, nonzero ? fp::branch_true : fp::branch_false);
    return c;
}
inline std::vector<Case> cases() {
    std::vector<Case> out;
    auto simple = base();
    finish(simple, 0);
    out.push_back({"only_exec_demanded", simple, fd::expected(UINT64_MAX, 0x42230011u)});
    for (uint64_t mask : {uint64_t(0), fragment_raw_masks::asymmetric, UINT64_MAX})
        for (bool boundary : {false, true}) out.push_back(exec_writer(mask, boundary));
    auto compare = fp::packet({});
    compare.mask_state_available = false;
    compare.exec_available = true;
    compare.vcc_mask = UINT64_MAX;
    compare.scc = true;
    out.push_back({"compare_defines_vcc_before_branch", compare, fp::expected({}, compare)});
    out.push_back(new_scc());
    out.push_back(new_scc(false));
    out.push_back(old_scc(false));
    out.push_back(old_scc(true));
    out.push_back(cmpx_vcc(false));
    out.push_back(cmpx_vcc(true));
    out.push_back(scalar_vcc(0));
    out.push_back(scalar_vcc(fragment_raw_masks::asymmetric));
    out.push_back(peer_before_exec());
    out.push_back(wqm_numeric());
    out.push_back(joined_vcc(false));
    out.push_back(joined_vcc(true));
    out.push_back(bypass_vcc(false));
    out.push_back(bypass_vcc(true));
    return out;
}
inline constexpr uint32_t kCases = 22;
inline FragmentResourcePacket wave_input(const Case& c) {
    return fragment_raw_masks::wave_input({c.name, c.packet, c.expected});
}
} // namespace prosper::test::fragment_mask_entry
