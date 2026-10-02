#pragma once
#include "fragment_packet_fixture.hpp"

namespace prosper::test::fragment_packet::mbcnt {
// These original guest programs consume owned register packets, not inferred raster waves.
// The oracle enumerates source-word bit positions independently of emitted event/scratch state.
enum class Source { Exec, Vcc, SavedVcc, Scalar, Vgpr, Inline, InlineFloat, Literal, Full };
enum class Accumulator { Vgpr, Scc, Scalar, Inline, InlineFloat, Literal };
struct Case {
    Source source = Source::SavedVcc;
    Accumulator first_accumulator = Accumulator::Vgpr;
    uint64_t exec = UINT64_MAX, mask = 0x8000000180000021ull;
    uint64_t scalar = 0xa500000140000081ull;
    uint32_t next_lane = 40;
    bool first_is_high = false;
    bool skip_first = false, skip_all = false, numeric_replacement = false;
};
inline uint32_t raw_lo(uint32_t lane) { return 0x80000021u ^ (lane * 0x01020409u); }
inline uint32_t raw_hi(uint32_t lane) { return 0x40000081u ^ (lane * 0x04080103u); }
inline uint32_t accumulator(uint32_t lane) { return 0xfffffff0u + lane * 3u; }
inline uint32_t first_accumulator(const Case& c, uint32_t lane) {
    switch (c.first_accumulator) {
        case Accumulator::Vgpr: return accumulator(lane);
        case Accumulator::Scc: return 1;
        case Accumulator::Scalar: return 0xfffffff4u;
        case Accumulator::Inline: return UINT32_MAX;
        case Accumulator::InlineFloat: return 0xbf800000u;
        case Accumulator::Literal: return 0xfffffffdu;
    }
    return 0;
}
inline uint32_t encoded_accumulator(Accumulator source) {
    switch (source) {
        case Accumulator::Vgpr: return 263;
        case Accumulator::Scc: return 253;
        case Accumulator::Scalar: return 27;
        case Accumulator::Inline: return 193;
        case Accumulator::InlineFloat: return 243;
        case Accumulator::Literal: return 255;
    }
    return 0;
}
inline uint32_t prefix(uint32_t word, uint32_t lane, bool high) {
    uint32_t count = 0;
    for (uint32_t bit = 0; bit < 32; ++bit) {
        const uint32_t position = bit + (high ? 32u : 0u);
        if (position < lane) count += (word >> bit) & 1u;
    }
    return count;
}
inline uint32_t population(uint64_t mask) {
    uint32_t count = 0;
    for (uint32_t bit = 0; bit < 64; ++bit) count += (mask >> bit) & 1u;
    return count;
}
inline void instruction(std::vector<uint32_t>& p, bool high, uint32_t dst,
                        uint32_t source, uint32_t acc, uint32_t literal = 0) {
    p.insert(p.end(), {0xd4000000u | ((high ? 0x366u : 0x365u) << 16) | dst,
                      source | (acc << 9)});
    if (source == 255 || acc == 255) p.push_back(literal);
}
inline uint32_t encoded_source(Source source, bool high) {
    switch (source) {
        case Source::Exec: return high ? 127 : 126;
        case Source::Vcc: return high ? 107 : 106;
        case Source::SavedVcc: return high ? 21 : 20;
        case Source::Scalar: return high ? 25 : 24;
        case Source::Vgpr: return high ? 262 : 261;
        case Source::Inline: return 161; // inline integer 33, two separated source bits
        case Source::InlineFloat: return high ? 247 : 243; // raw -4.0 / -1.0 bits
        case Source::Literal: return 255;
        case Source::Full: return 193; // inline integer -1, not an active-lane population
    }
    return 0;
}
inline prosper::gpu::FragmentInvocationPacket packet(const Case& c) {
    prosper::gpu::FragmentInvocationPacket p;
    p.slots_available.fill(true); p.export_enabled.fill(1);
    p.export_enabled[31] = 0; p.export_enabled[40] = 0;
    p.mask_state_available = true; p.exec_mask = c.exec; p.vcc_mask = c.mask; p.scc = true;
    for (uint32_t reg : {0u, 1u, 2u, 3u, 5u, 6u, 7u}) {
        prosper::gpu::FragmentPacketVgpr column; column.reg = reg;
        for (uint32_t lane = 0; lane < 64; ++lane)
            column.words[lane] = reg == 0 ? lane : reg == 5 ? raw_lo(lane) :
                reg == 6 ? raw_hi(lane) : reg == 7 ? accumulator(lane) : poison_sentinel;
        p.vgprs.push_back(column);
    }
    for (uint32_t reg : {12u, 14u, 16u, 17u, 20u, 21u})
        p.sgprs.emplace_back(reg, poison_sentinel);
    p.sgprs.emplace_back(24, uint32_t(c.scalar));
    p.sgprs.emplace_back(25, uint32_t(c.scalar >> 32));
    p.sgprs.emplace_back(27, 0xfffffff4u);
    auto& code = p.guest_code;
    code = {0xbe90047eu, 0xbe94046au}; // save complete EXEC and VCC independently of EXEC
    if (c.numeric_replacement) {
        smov(code, 20, uint32_t(c.scalar)); smov(code, 21, uint32_t(c.scalar >> 32));
    }
    const size_t skip = code.size();
    if (c.skip_first) code.push_back(0xbf820000u);
    instruction(code, c.first_is_high, 1, encoded_source(c.source, false),
                encoded_accumulator(c.first_accumulator), c.source == Source::Literal
                    ? uint32_t(c.mask) : first_accumulator(c, 0));
    instruction(code, true, 2, encoded_source(c.source, true), 257, uint32_t(c.mask >> 32));
    if (c.skip_first) code[skip] |= uint32_t(code.size() - skip - 1);
    exp(code, 15, 0x05070201u);
    // Different shared services reuse the MBCNT scratch after an EXP. READLANE40 must still
    // read a supplied EXEC-off/export-ineligible source; no worker is killed by the raw export.
    code.push_back(0xbe8c106au); // BCNT s12,VCC, including supplied inactive source bits
    vmov(code, 3, 12);
    code.insert(code.end(), {0xd760000eu, 261u | (168u << 9)}); // READLANE s14,v5,40
    vmov(code, 3, 14);
    exp(code, 15, 0x03050201u);
    code.insert(code.end(), {0x7d8400ffu, c.next_lane}); // new VCC = EXEC & (v0 == next_lane)
    code.push_back(0xbe94046au); // new saved-mask lifetime, different from the first event
    instruction(code, false, 1, encoded_source(c.source, false), 263, uint32_t(c.mask));
    instruction(code, true, 2, encoded_source(c.source, true), 257, uint32_t(c.mask >> 32));
    vmov(code, 3, 12); // observe the earlier whole-mask reduction, not the new compare
    exp(code, 15, 0x03070201u);
    code.push_back(0xbf810000u);
    if (c.skip_all) code.insert(code.begin(), 0xbf820000u | uint32_t(code.size() - 1));
    return p;
}
inline uint32_t source_word(const Case& c, uint32_t lane, bool high, bool later) {
    uint64_t mask = c.mask;
    if (later && (c.source == Source::Vcc || c.source == Source::SavedVcc))
        mask = c.next_lane < 64 && ((c.exec >> c.next_lane) & 1u)
            ? uint64_t(1) << c.next_lane : 0;
    switch (c.source) {
        case Source::Exec: mask = c.exec; break;
        case Source::Vcc: break;
        case Source::SavedVcc:
            if (!later && c.numeric_replacement) mask = c.scalar;
            break;
        case Source::Scalar: mask = c.scalar; break;
        case Source::Vgpr: return high ? raw_hi(lane) : raw_lo(lane);
        case Source::Inline: return 33;
        case Source::InlineFloat: return high ? 0xc0800000u : 0xbf800000u;
        case Source::Literal: break;
        case Source::Full: return UINT32_MAX;
    }
    return high ? uint32_t(mask >> 32) : uint32_t(mask);
}
inline std::vector<uint32_t> expected(const Case& c) {
    std::vector<uint32_t> out(64 * 36, 0);
    if (c.skip_all) return out;
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const bool active = (c.exec >> lane) & 1u;
        const uint32_t enabled = lane == 31 || lane == 40 ? 0u : 1u;
        const uint32_t acc = first_accumulator(c, lane);
        const uint32_t lo = active && !c.skip_first
            ? acc + prefix(source_word(c, lane, false, false), lane, c.first_is_high)
            : poison_sentinel;
        const uint32_t hi = active && !c.skip_first
            ? lo + prefix(source_word(c, lane, true, false), lane, true) : poison_sentinel;
        const uint32_t later_lo = active
            ? accumulator(lane) + prefix(source_word(c, lane, false, true), lane, false)
            : lo;
        const uint32_t later_hi = active
            ? later_lo + prefix(source_word(c, lane, true, true), lane, true) : hi;
        const uint32_t first[] = {1, uint32_t(active), enabled, 0, 15, 0, 1, 1,
                                 lo, hi, accumulator(lane), raw_lo(lane)};
        const uint32_t second[] = {1, uint32_t(active), enabled, 0, 15, 0, 1, 1,
                                  lo, hi, raw_lo(lane), active ? raw_lo(40) : poison_sentinel};
        const uint32_t third[] = {1, uint32_t(active), enabled, 0, 15, 0, 1, 1,
                                 later_lo, later_hi, accumulator(lane),
                                 active ? population(c.mask) : poison_sentinel};
        std::copy(std::begin(first), std::end(first), out.begin() + lane * 36);
        std::copy(std::begin(second), std::end(second), out.begin() + lane * 36 + 12);
        std::copy(std::begin(third), std::end(third), out.begin() + lane * 36 + 24);
    }
    return out;
}
inline std::vector<Case> cases() {
    std::vector<Case> result;
    for (const auto source : {Source::Exec, Source::Vcc, Source::SavedVcc, Source::Scalar,
                             Source::Vgpr, Source::Inline, Source::InlineFloat,
                             Source::Literal, Source::Full})
        for (uint64_t exec : {UINT64_MAX, uint64_t(1) << 63, uint64_t(0xaaaaaaaa55555555ull)}) {
            Case c; c.source = source; c.exec = exec; result.push_back(c);
        }
    for (uint32_t bit = 0; bit < 64; ++bit) {
        Case c; c.mask = uint64_t(1) << bit; c.exec = uint64_t(1) << 63; result.push_back(c);
    }
    Case empty; empty.source = Source::Vcc; empty.exec = 0; result.push_back(empty);
    for (auto acc : {Accumulator::Scc, Accumulator::Scalar, Accumulator::Inline,
                     Accumulator::InlineFloat, Accumulator::Literal}) {
        Case c; c.first_accumulator = acc; result.push_back(c);
    }
    Case skip; skip.skip_first = true; result.push_back(skip);
    Case ended; ended.skip_all = true; result.push_back(ended);
    Case replaced; replaced.numeric_replacement = true; result.push_back(replaced);
    return result;
}
} // namespace prosper::test::fragment_packet::mbcnt
