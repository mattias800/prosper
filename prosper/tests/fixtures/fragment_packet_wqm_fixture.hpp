#pragma once
#include "fragment_packet_fixture.hpp"

namespace prosper::test::fragment_packet::wqm {
// Project-owned instruction packets, never reconstructed raster waves. Raw scalar inputs and
// output expectations are independent of the emitted scratch/event representation.
enum class Source { Exec, Vcc, SavedVcc, ScalarPair, Empty, Full };
struct Case {
    Source source = Source::Exec;
    uint32_t destination = 126;
    uint64_t exec = uint64_t(1) << 63, vcc = uint64_t(1) << 40;
    uint64_t scalar = (uint64_t(1) << 35) | 1;
    uint64_t second = uint64_t(1) << 40;
    bool initial_scc = false, repeat = true;
};
inline uint32_t sop1(uint32_t op, uint32_t dst, uint32_t src) {
    return 0xbe800000u | (dst << 16) | (op << 8) | src;
}
inline uint64_t expand(uint64_t mask) {
    uint64_t result = 0;
    for (uint32_t quad = 0; quad < 16; ++quad) {
        bool any = false;
        for (uint32_t pixel = 0; pixel < 4; ++pixel) any |= (mask >> (4 * quad + pixel)) & 1u;
        if (any) for (uint32_t pixel = 0; pixel < 4; ++pixel)
            result |= uint64_t(1) << (4 * quad + pixel);
    }
    return result;
}
inline uint32_t population(uint64_t mask) {
    uint32_t n = 0;
    for (uint32_t bit = 0; bit < 64; ++bit) n += (mask >> bit) & 1u;
    return n;
}
inline uint64_t source_mask(const Case& c) {
    switch (c.source) {
        case Source::Exec: return c.exec;
        case Source::Vcc: case Source::SavedVcc: return c.vcc;
        case Source::ScalarPair: return c.scalar;
        case Source::Empty: return 0;
        case Source::Full: return UINT64_MAX;
    }
    return 0;
}
inline prosper::gpu::FragmentInvocationPacket packet(const Case& c) {
    auto p = fragment_packet::packet({});
    p.quad_topology = prosper::gpu::FragmentPacketQuadTopology::ConsecutiveLogicalQuads;
    p.exec_mask = c.exec; p.vcc_mask = c.vcc; p.scc = c.initial_scc;
    p.export_enabled[60] = 0; // Export eligibility must NOT remove a supplied quad source/worker.
    p.sgprs.clear();
    for (uint32_t reg : {12u, 14u, 16u, 17u, 19u, 20u, 21u, 28u, 36u, 37u})
        p.sgprs.emplace_back(reg, poison_sentinel);
    p.sgprs.emplace_back(24, branch_true); p.sgprs.emplace_back(25, branch_false);
    p.sgprs.emplace_back(30, uint32_t(c.scalar)); p.sgprs.emplace_back(31, uint32_t(c.scalar >> 32));
    p.sgprs.emplace_back(32, uint32_t(c.second)); p.sgprs.emplace_back(33, uint32_t(c.second >> 32));
    const uint32_t source = c.source == Source::Exec ? 126u : c.source == Source::Vcc ? 106u :
        c.source == Source::SavedVcc ? 20u : c.source == Source::ScalarPair ? 30u :
        c.source == Source::Empty ? 128u : 193u;
    p.guest_code = {sop1(4, 36, 126), sop1(4, 20, 106), sop1(10, c.destination, source),
        0x850e1918u, // S_CSELECT_B32 s14,s24,s25: actual WQM SCC before BCNT changes it
        sop1(4, 16, c.destination), sop1(16, 12, 16), sop1(16, 28, 106), sop1(4, 126, 16)};
    vmov(p.guest_code, 1, 12); vmov(p.guest_code, 2, 14); vmov(p.guest_code, 3, 28);
    vmov(p.guest_code, 8, 256); // Newly widened EXEC-off neighbors perform a real vector write.
    exp(p.guest_code, 15, 0x08030201u);
    p.guest_code.push_back(sop1(4, 126, 36)); // Restore ORIGINAL EXEC, not the widened mask.
    p.guest_code.insert(p.guest_code.end(), {0xd7600013u, 264u | (168u << 9)});
    vmov(p.guest_code, 3, 19); // READLANE40 ignores source EXEC after the original mask is restored.
    vmov(p.guest_code, 5, 24);
    vmov(p.guest_code, 2, 14); // Observe WQM SCC even when its widened EXEC was empty.
    exp(p.guest_code, 15, 0x03020805u);
    if (c.repeat) {
        p.guest_code.insert(p.guest_code.end(), {sop1(10, 126, 32), sop1(16, 12, 126)});
        vmov(p.guest_code, 1, 12);
        exp(p.guest_code, 1, 1);
    }
    p.guest_code.push_back(0xbf810000u);
    return p;
}
inline std::vector<uint32_t> expected(const Case& c) {
    const uint64_t wide = expand(source_mask(c)), later = expand(c.second);
    const uint64_t current_vcc = c.destination == 106 ? wide : c.vcc;
    const uint32_t records = c.repeat ? 3u : 2u;
    std::vector<uint32_t> out(64 * records * 12, 0);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const bool active = (wide >> lane) & 1u, old = (c.exec >> lane) & 1u;
        const bool next = (later >> lane) & 1u;
        const uint32_t base = lane * records * 12, enabled = lane == 60 ? 0u : 1u;
        const uint32_t first[] = {1, uint32_t(active), enabled, 0, 15, 0, 1, 1,
            active ? population(wide) : poison_sentinel,
            active ? (wide ? branch_true : branch_false) : poison_sentinel,
            active ? population(current_vcc) : poison_sentinel,
            active ? lane : 0x51000000u + lane * 17};
        const uint32_t word40 = (wide & (uint64_t(1) << 40)) ? 40u : 0x51000000u + 40 * 17;
        const uint32_t restore[] = {1, uint32_t(old), enabled, 0, 15, 0, 1, 1,
            old ? branch_true : poison_sentinel, first[11],
            old ? (wide ? branch_true : branch_false) : first[9], old ? word40 : first[10]};
        std::copy(std::begin(first), std::end(first), out.begin() + base);
        std::copy(std::begin(restore), std::end(restore), out.begin() + base + 12);
        if (c.repeat) {
            const uint32_t third[] = {1, uint32_t(next), enabled, 0, 1, 0, 1, 1,
                next ? population(later) : first[8], 0, 0, 0};
            std::copy(std::begin(third), std::end(third), out.begin() + base + 24);
        }
    }
    return out;
}
// A declared but branch-skipped WQM still allocates its common service. The service must not
// overwrite prior SCC or EXEC when no guest invocation requests it in this dispatcher step.
inline prosper::gpu::FragmentInvocationPacket skipped_scc_packet(bool initial_scc) {
    auto p = fragment_packet::packet({});
    p.quad_topology = prosper::gpu::FragmentPacketQuadTopology::ConsecutiveLogicalQuads;
    p.exec_mask = uint64_t(1) << 63; p.scc = initial_scc;
    p.sgprs.emplace_back(24, branch_true); p.sgprs.emplace_back(25, branch_false);
    p.guest_code = {0xbf820001u, sop1(10, 126, 126), 0x850e1918u};
    vmov(p.guest_code, 2, 14);
    exp(p.guest_code, 1, 2);
    p.guest_code.push_back(0xbf810000u);
    return p;
}
inline std::vector<uint32_t> skipped_scc_expected(bool initial_scc) {
    std::vector<uint32_t> out(64 * 12, 0);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const bool active = lane == 63;
        const uint32_t record[] = {1, uint32_t(active), 1, 0, 1, 0, 1, 1,
            active ? (initial_scc ? branch_true : branch_false) : poison_sentinel, 0, 0, 0};
        std::copy(std::begin(record), std::end(record), out.begin() + lane * 12);
    }
    return out;
}
} // namespace prosper::test::fragment_packet::wqm
