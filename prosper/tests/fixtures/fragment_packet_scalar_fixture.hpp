#pragma once
#include "fixtures/fragment_packet_fixture.hpp"

namespace prosper::test::fragment_packet::scalar {
// Missing scratch allocation is deliberately NOT represented by supplied zero SGPR values.
// Original decoded instructions must establish every consumed word on all structural paths.
enum class Kind { BothArms, Readlane, Pair, SavedVcc, SuppliedEntry, AddSelect, Movk, SkippedRead };
struct Case {
    Kind kind = Kind::BothArms;
    bool scc = false;
    uint64_t exec = UINT64_MAX;
    uint64_t vcc = (uint64_t(1) << 63) | (uint64_t(1) << 40) | 1;
};
inline std::vector<Case> cases() {
    return {{Kind::BothArms,false}, {Kind::BothArms,true},
        {Kind::BothArms,true,uint64_t(1) << 63}, {Kind::Readlane},
        {Kind::Readlane,false,UINT64_MAX ^ (uint64_t(1) << 40)},
        {Kind::Readlane,false,0}, {Kind::Pair}, {Kind::SavedVcc},
        {Kind::SavedVcc,false,UINT64_MAX,0}, {Kind::SuppliedEntry},
        {Kind::AddSelect}, {Kind::Movk}, {Kind::SkippedRead}};
}
inline prosper::gpu::FragmentInvocationPacket packet(const Case& c) {
    auto p = prosper::test::fragment_packet::packet({});
    p.sgprs.clear(); // only true entry reads below receive supplied values
    p.guest_code.clear(); p.exec_mask = c.exec; p.vcc_mask = c.vcc; p.scc = c.scc;
    auto& code = p.guest_code;
    uint32_t channels = 1;
    switch (c.kind) {
        case Kind::BothArms:
            code.push_back(0xbf840003u); // SCC0 -> false arm (pc4)
            smov(code,14,branch_true);
            code.push_back(0xbf820002u); // unconditional -> join (pc6)
            smov(code,14,branch_false);
            vmov(code,1,14);
            break;
        case Kind::Readlane:
            p.sgprs = {{22,40}};
            code.insert(code.end(),{0xd760000fu,264u | (22u << 9)}); // s15 <- v8[lane40], ignores EXEC
            vmov(code,1,15);
            break;
        case Kind::Pair:
            smov(code,20,0x80000001u); smov(code,21,0x80000010u);
            code.push_back(0xbe8c1014u); // BCNT s12,s[20:21], four set bits
            vmov(code,1,12); vmov(code,2,21); vmov(code,3,20);
            channels = 7;
            break;
        case Kind::SavedVcc:
            code.push_back(0xbe94046au); // save physical VCC pair into missing scratch s20:s21
            code.push_back(0xbe8c1014u); // BCNT s12,s[20:21]
            vmov(code,1,12);
            break;
        case Kind::SuppliedEntry:
            p.sgprs = {{14,0x76543210u}};
            vmov(code,1,14); // consume real entry before the later overwrite
            smov(code,14,0xaaaaaaaa);
            break;
        case Kind::AddSelect:
            p.sgprs = {{24,branch_true},{25,branch_false}};
            smov(code,20,0xffffffffu); smov(code,21,2);
            code.push_back(0x800c1514u); // ADD_U32 s12,s20,s21: result1 and carry SCC
            code.push_back(0x850e1918u); // CSELECT_B32 s14,s24,s25 (preserves carry SCC)
            vmov(code,1,12); vmov(code,2,14);
            channels = 3;
            break;
        case Kind::Movk:
            code.push_back(0xb00c8001u); // MOVK_I32 s12,sign-extended0x8001
            vmov(code,1,12);
            break;
        case Kind::SkippedRead:
            code.push_back(0xbf820001u); // unconditional -> pc2, not a supplied-SCC specialization
            vmov(code,1,14);             // unreachable read; still inventoried as a supported opcode
            smov(code,14,branch_true);   // genuine definition on the only structural path
            vmov(code,1,14);
            break;
    }
    exp(code,channels,0x00030201u);
    code.push_back(0xbf810000u);
    return p;
}
inline std::vector<uint32_t> expected(const Case& c) {
    uint32_t channels = 1;
    std::array<uint32_t,3> payload{};
    switch (c.kind) {
        case Kind::BothArms: payload[0] = c.scc ? branch_true : branch_false; break;
        case Kind::Readlane: payload[0] = 0x51000000u + 40 * 17; break;
        case Kind::Pair:
            channels = 7; payload = {4,0x80000010u,0x80000001u}; break;
        case Kind::SavedVcc:
            for (uint32_t lane = 0; lane < 64; ++lane) payload[0] += (c.vcc >> lane) & 1u;
            break;
        case Kind::SuppliedEntry: payload[0] = 0x76543210u; break;
        case Kind::AddSelect: channels = 3; payload = {1,branch_true,0}; break;
        case Kind::Movk: payload[0] = 0xffff8001u; break;
        case Kind::SkippedRead: payload[0] = branch_true; break;
    }
    std::vector<uint32_t> out(64 * 12,0);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const bool active = (c.exec >> lane) & 1u;
        const uint32_t fields[] = {1,uint32_t(active),1,0,channels,0,1,1};
        std::copy(std::begin(fields),std::end(fields),out.begin() + lane * 12);
        for (uint32_t channel = 0; channel < 3; ++channel)
            if (channels & (1u << channel))
                out[lane * 12 + 8 + channel] = active ? payload[channel] : poison_sentinel;
    }
    return out;
}
} // namespace prosper::test::fragment_packet::scalar
