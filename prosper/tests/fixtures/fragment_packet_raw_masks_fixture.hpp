#pragma once
#include "fragment_packet_definedness_fixture.hpp"
#include "fragment_packet_wave_fixture.hpp"

namespace prosper::test::fragment_raw_masks {
using namespace prosper::gpu;
namespace f = fragment_definedness;
namespace fp = fragment_packet;
inline constexpr uint64_t asymmetric = (uint64_t(1) << 0) | (uint64_t(1) << 31) |
                                       (uint64_t(1) << 32) | (uint64_t(1) << 40) |
                                       (uint64_t(1) << 63);
struct Case {
    std::string name;
    FragmentInvocationPacket packet;
    std::vector<uint32_t> expected;
};
// The oracle uses only independent physical words and mask bit positions, never compiler state.
inline std::vector<uint32_t> expected(const FragmentInvocationPacket& p, uint64_t saved,
                                      uint64_t replacement) {
    std::vector<uint32_t> out(64 * 24);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const bool on = ((replacement >> lane) & 1u) != 0;
        const uint32_t first[]{1,
                               uint32_t(on),
                               p.export_enabled[lane],
                               0,
                               15,
                               0,
                               1,
                               1,
                               uint32_t(saved),
                               uint32_t(saved >> 32),
                               uint32_t(replacement),
                               uint32_t(replacement >> 32)};
        const uint32_t second[]{1,
                                uint32_t(on),
                                p.export_enabled[lane],
                                0,
                                1,
                                0,
                                1,
                                1,
                                on ? 0x42230011u : fp::poison_sentinel,
                                0,
                                0,
                                0};
        std::copy(std::begin(first), std::end(first), out.begin() + lane * 24);
        std::copy(std::begin(second), std::end(second), out.begin() + lane * 24 + 12);
    }
    return out;
}
inline Case overwritten(uint64_t mask, bool high, bool boundary, bool inverted = false,
                        bool from_vcc = false) {
    auto p = f::base();
    p.exec_mask = from_vcc ? 0 : mask;
    p.vcc_mask = mask;
    p.export_enabled[40] = 0; // owned helper-supplied observation is not export eligibility
    p.sgprs.emplace_back(30, UINT32_MAX);
    p.sgprs.emplace_back(31, UINT32_MAX);
    for (uint32_t reg : {1u, 2u, 3u, 4u, 5u}) f::column(p, reg, fp::poison_sentinel);
    p.guest_code.push_back(from_vcc ? 0xbe94046au : 0xbe94047eu); // actual saved mask producer
    if (inverted) p.guest_code.push_back(0xbe940814u); // S_NOT_B64 s20:21,s20:21 current source
    const uint64_t saved = inverted ? ~mask : mask;
    p.guest_code.push_back(0xbefe041eu); // full EXEC from genuinely owned ordinary scalar pair
    fp::vmov(p.guest_code, 1, 20);
    fp::vmov(p.guest_code, 2, 21);
    const uint32_t word = high ? 0x81000101u : 0x80000001u;
    fp::smov(p.guest_code, high ? 21 : 20, word);
    if (boundary) p.guest_code.push_back(0xbf820000u);
    p.guest_code.push_back(
        0xbe960414u); // numeric S_MOV_B64 s22:23,s20:21 after partial overwrite
    fp::vmov(p.guest_code, 3, 22);
    fp::vmov(p.guest_code, 4, 23);
    p.guest_code.push_back(0xbefe0416u); // exact current replacement pair -> logical EXEC
    fp::vmov(p.guest_code, 5, 0);
    fp::exp(p.guest_code, 15, 0x04030201u);
    fp::exp(p.guest_code, 1, 5);
    p.guest_code.push_back(0xbf810000u);
    const uint64_t replacement =
        high ? (uint64_t(word) << 32) | uint32_t(saved) : (saved & 0xffffffff00000000ull) | word;
    return {std::string(from_vcc ? "vcc_" : "exec_") + (high ? "high_" : "low_") +
                (boundary ? "boundary_" : "same_") + (inverted ? "not_" : "saved_") +
                std::to_string(mask),
            p, expected(p, saved, replacement)};
}
inline std::vector<Case> cases() {
    std::vector<Case> out;
    for (bool high : {false, true})
        for (bool boundary : {false, true})
            for (bool inverted : {false, true})
                out.push_back(overwritten(asymmetric, high, boundary, inverted));
    for (uint32_t lane : {0u, 31u, 32u, 40u, 63u})
        out.push_back(overwritten(uint64_t(1) << lane, true, false));
    out.push_back(overwritten(0, false, false));
    out.push_back(overwritten(UINT64_MAX, true, true));
    out.push_back(overwritten(asymmetric, false, true, false, true));
    return out;
}
inline constexpr uint32_t kCases = 16;
inline std::vector<Case> joins() {
    std::vector<Case> out;
    for (bool high : {false, true})
        for (bool taken : {false, true}) {
            auto c = overwritten(asymmetric, high, true);
            c.packet.scc = !taken;
            c.packet.guest_code.insert(c.packet.guest_code.begin() + 4, 0xbf840002u);
            // Original SCC0 skips only the two-word half writer. The join's mask domain is
            // ambiguous, but each physical word has genuine reaching DATA on BOTH predecessors.
            if (taken) c.expected = expected(c.packet, asymmetric, asymmetric);
            c.name =
                std::string(high ? "join_high_" : "join_low_") + (taken ? "skipped" : "written");
            out.push_back(std::move(c));
        }
    return out;
}
inline Case reused_producer() {
    auto c = overwritten(asymmetric, true, true);
    // Before the original save, replace EXEC with the genuine entry numeric s20:21 (all ones).
    // After that first save, consume both current words before replacing the saved mask again
    // from VCC. This gives the same destination two distinct materialization events.
    c.packet.guest_code.insert(c.packet.guest_code.begin(), 0xbefe0414u);
    c.packet.guest_code.insert(c.packet.guest_code.begin() + 2,
                               {0x7e0c0214u, 0x7e0e0215u, 0xbe94046au});
    f::column(c.packet, 6, fp::poison_sentinel);
    f::column(c.packet, 7, fp::poison_sentinel);
    // The first save's separate words are transferred back to scalar s24/s25 through real peers
    // after the second save: the source values remain current numeric DATA, not a revived alias.
    const auto insert = c.packet.guest_code.end() - 1;
    c.packet.guest_code.insert(insert,
                               {0xd7600018u, 262u | (128u << 9), 0xd7600019u, 263u | (191u << 9)});
    f::column(c.packet, 8, fp::poison_sentinel);
    f::column(c.packet, 9, fp::poison_sentinel);
    std::vector<uint32_t> sink;
    fp::vmov(sink, 8, 24);
    fp::vmov(sink, 9, 25);
    fp::exp(sink, 3, 0x00000908u);
    c.packet.guest_code.insert(c.packet.guest_code.end() - 1, sink.begin(), sink.end());
    // Existing sinks discriminate the later save from stale first-event all-ones words; the
    // third sink observes real numeric peer transfers of BOTH words from the earlier event.
    auto old = c.expected;
    c.expected.assign(64 * 36, 0);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        std::copy_n(old.begin() + lane * 24, 24, c.expected.begin() + lane * 36);
        const bool on = old[lane * 24 + 1] != 0;
        const uint32_t row[]{1,
                             uint32_t(on),
                             c.packet.export_enabled[lane],
                             0,
                             3,
                             0,
                             1,
                             1,
                             on ? UINT32_MAX : fp::poison_sentinel,
                             on ? UINT32_MAX : fp::poison_sentinel,
                             0,
                             0};
        std::copy(std::begin(row), std::end(row), c.expected.begin() + lane * 36 + 24);
    }
    c.name = "reused_destination_after_source_change_and_peer_numeric_transfer";
    return c;
}
inline Case wqm_saved() {
    auto c = overwritten(asymmetric, false, true);
    c.packet.quad_topology = FragmentPacketQuadTopology::ConsecutiveLogicalQuads;
    c.packet.guest_code[0] = 0xbe940a7eu;   // S_WQM_B64 s20:21,EXEC, actual service-produced mask
    uint64_t widened = 0;
    for (uint32_t lane = 0; lane < 64; lane += 4)
        if ((asymmetric >> lane) & 15u) widened |= uint64_t(15) << lane;
    c.expected = expected(c.packet, widened, (widened & 0xffffffff00000000ull) | 0x80000001u);
    c.name = "wqm_producer_current_raw_words";
    return c;
}
inline Case saveexec_scc(bool nonzero) {
    auto c = overwritten(asymmetric, true, true);
    c.packet.vcc_mask = nonzero ? asymmetric : 0;
    c.packet.scc = !nonzero;   // stale entry SCC must not satisfy either original branch
    c.packet.sgprs.emplace_back(6, 0x51000000u);
    f::column(c.packet, 6, fp::poison_sentinel);
    c.packet.guest_code[0] = 0xbe94246au;   // S_AND_SAVEEXEC saves OLD EXEC, SCC tests NEW EXEC
    std::vector<uint32_t> branch{0xbf840002u};   // SCC0 skips only the actual scalar marker update
    fp::smov(branch, 6, 0x61000000u);
    fp::vmov(branch, 6, 6);
    c.packet.guest_code.insert(c.packet.guest_code.begin() + 2, branch.begin(), branch.end());
    std::vector<uint32_t> sink;
    fp::exp(sink, 1, 6);
    c.packet.guest_code.insert(c.packet.guest_code.end() - 1, sink.begin(), sink.end());
    auto old = c.expected;
    c.expected.assign(64 * 36, 0);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        std::copy_n(old.begin() + lane * 24, 24, c.expected.begin() + lane * 36);
        const uint32_t row[]{1,
                             old[lane * 24 + 1],
                             c.packet.export_enabled[lane],
                             0,
                             1,
                             0,
                             1,
                             1,
                             nonzero ? 0x61000000u : 0x51000000u,
                             0,
                             0,
                             0};
        std::copy(std::begin(row), std::end(row), c.expected.begin() + lane * 36 + 24);
    }
    c.name = nonzero ? "saveexec_scc_new_nonzero_old_saved" : "saveexec_scc_new_zero_old_saved";
    return c;
}
inline Case not_scc(bool nonzero) {
    auto c = overwritten(nonzero ? asymmetric : UINT64_MAX, true, true, true);
    c.packet.scc = !nonzero;
    c.packet.sgprs.emplace_back(6, 0x51000000u);
    f::column(c.packet, 6, fp::poison_sentinel);
    // Observe actual NOT SCC after SCC-preserving MOV EXEC, independently of the later half write.
    std::vector<uint32_t> branch{0xbf840002u};
    fp::smov(branch, 6, 0x61000000u);
    fp::vmov(branch, 6, 6);
    c.packet.guest_code.insert(c.packet.guest_code.begin() + 3, branch.begin(), branch.end());
    std::vector<uint32_t> sink;
    fp::exp(sink, 1, 6);
    c.packet.guest_code.insert(c.packet.guest_code.end() - 1, sink.begin(), sink.end());
    const auto old = c.expected;
    c.expected.assign(64 * 36, 0);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        std::copy_n(old.begin() + lane * 24, 24, c.expected.begin() + lane * 36);
        const uint32_t row[]{1,
                             old[lane * 24 + 1],
                             c.packet.export_enabled[lane],
                             0,
                             1,
                             0,
                             1,
                             1,
                             nonzero ? 0x61000000u : 0x51000000u,
                             0,
                             0,
                             0};
        std::copy(std::begin(row), std::end(row), c.expected.begin() + lane * 36 + 24);
    }
    c.name = nonzero ? "not_scc_current_nonzero" : "not_scc_current_zero";
    return c;
}
inline Case scalar_data_add() {
    auto c = overwritten(asymmetric, true, false);
    f::column(c.packet, 6, fp::poison_sentinel);
    c.packet.guest_code.insert(c.packet.guest_code.begin() + 9,
                               {0x80181514u, 0x7e0c0218u});   // S_ADD_U32 s24,s20,s21; MOV v6,s24
    std::vector<uint32_t> sink;
    fp::exp(sink, 1, 6);
    c.packet.guest_code.insert(c.packet.guest_code.end() - 1, sink.begin(), sink.end());
    const uint32_t sum = uint32_t(asymmetric) + uint32_t(0x81000101u);   // independent uint wrap
    auto old = c.expected;
    c.expected.assign(64 * 36, 0);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        std::copy_n(old.begin() + lane * 24, 24, c.expected.begin() + lane * 36);
        const uint32_t row[]{
            1, old[lane * 24 + 1], c.packet.export_enabled[lane], 0, 1, 0, 1, 1, sum, 0, 0, 0};
        std::copy(std::begin(row), std::end(row), c.expected.begin() + lane * 36 + 24);
    }
    c.name = "current_mask_words_are_scalar_data_wrapping_add";
    return c;
}
inline std::vector<Case> distinct_wave_cases() {
    return {overwritten(asymmetric, true, true), overwritten(uint64_t(1) << 63, true, true),
            overwritten(0, true, true)};
}
inline FragmentResourcePacket wave_input(const Case& c) {
    auto input = fragment_resource_packet::base();
    const auto mode = input.invocation.float_mode;
    const auto flags = input.invocation.float_flags;
    input.invocation = c.packet;
    input.invocation.float_mode = mode;
    input.invocation.float_flags = flags;
    return input;   // fully hand-owned profile, not hardware entry/packing authority
}
}   // namespace prosper::test::fragment_raw_masks
