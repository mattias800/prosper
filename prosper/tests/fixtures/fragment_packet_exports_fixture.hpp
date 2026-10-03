#pragma once
#include "fragment_resource_packet_fixture.hpp"
#include "gpu/recompiler/fragment_packet_wave_data.hpp"

namespace prosper::test::fragment_packet_exports {
using namespace prosper::gpu;
namespace fp = fragment_packet;
inline constexpr uint64_t sparse = 1ull | (1ull << 31) | (1ull << 32) | (1ull << 40) | (1ull << 63);
inline bool active(uint32_t lane) {
    return lane == 0 || lane == 31 || lane == 32 || lane == 40 || lane == 63;
}
inline uint32_t value(uint32_t wave) {
    return wave == 1 ? 0 : 0x62470000u + wave * 0x100;
}
inline void exp(std::vector<uint32_t>& code, uint32_t target, uint32_t en, uint32_t sources,
                bool done, bool vm, bool compr = false) {
    code.insert(code.end(), {0xf8000000u | (target << 4) | en | (uint32_t(compr) << 10) |
                                 (uint32_t(done) << 11) | (uint32_t(vm) << 12),
                             sources});
}
inline FragmentResourcePacket base(uint32_t wave = 0) {
    auto p = fragment_resource_packet::base();
    auto& g = p.invocation;
    g.export_observation = FragmentPacketExportObservation::Architectural;
    g.vgprs.clear();
    g.sgprs = {{0, value(wave)},
               {20, uint32_t(sparse)},
               {21, uint32_t(sparse >> 32)},
               {22, UINT32_MAX},
               {23, UINT32_MAX}};
    g.exec_mask = sparse;
    g.export_enabled[40] = 0; // active source still demanded although commit is ineligible
    return p;
}
inline void column(FragmentResourcePacket& p, uint32_t reg, uint32_t bits,
                   uint64_t available = UINT64_MAX) {
    FragmentPacketVgpr c;
    c.reg = reg;
    c.available_mask = available;
    for (uint32_t lane = 0; lane < 64; ++lane) c.words[lane] = bits + lane;
    p.invocation.vgprs.push_back(c);
}
inline FragmentResourcePacket scratch(uint32_t wave = 0, bool inactive = false) {
    auto p = base(wave);
    if (inactive) p.invocation.exec_mask = 0;
    fp::vmov(p.invocation.guest_code, 1, 0);
    exp(p.invocation.guest_code, 0, 1, 1, true, true);
    p.invocation.guest_code.push_back(0xbf810000u);
    return p;
}
inline FragmentResourcePacket missing_active(bool missing) {
    auto p = base();
    column(p, 1, 0, sparse & ~(missing ? 1ull << 40 : 0));
    exp(p.invocation.guest_code, 0, 1, 1, false, false);
    exp(p.invocation.guest_code, 9, 0, 0, true, true);
    p.invocation.guest_code.push_back(0xbf810000u);
    return p; // first EXP VM0, lane40 export_enabled0: neither excuses its active read
}
inline FragmentResourcePacket peer(bool missing = false) {
    auto p = base();
    p.invocation.exec_mask = 1ull << 63;
    column(p, 8, 0x87000000u, missing ? UINT64_MAX ^ (1ull << 40) : UINT64_MAX);
    p.invocation.guest_code = {0xd760003cu, 264u | (168u << 9)}; // s60=v8[40], inactive peer
    fp::vmov(p.invocation.guest_code, 1, 60);
    exp(p.invocation.guest_code, 0, 1, 1, true, true);
    p.invocation.guest_code.push_back(0xbf810000u);
    return p;
}
inline FragmentResourcePacket wqm(bool missing = false) {
    auto p = base();
    p.invocation.exec_mask = 1ull << 63;
    column(p, 1, 0x77000000u, missing ? 1ull << 63 : UINT64_MAX);
    p.invocation.guest_code = {0xbefe0a7eu}; // S_WQM_B64 EXEC,EXEC reactivates60..63
    exp(p.invocation.guest_code, 0, 1, 1, true, true);
    p.invocation.guest_code.push_back(0xbf810000u);
    return p;
}
inline FragmentResourcePacket multiple(bool drain = true) {
    auto p = base();
    p.invocation.exec_mask = UINT64_MAX;
    for (uint32_t reg : {1u, 2u, 3u}) column(p, reg, 0x80000000u + reg * 0x10000);
    auto& code = p.invocation.guest_code;
    exp(code, 2, 3, 0x00000201u, false, true); // PC0, full EXEC, R/G
    if (drain) code.push_back(0xbf8c0000u); // actual full WAIT before altering pending EXEC
    code.push_back(0xbefe0414u); // PC3 when drained, genuine supplied sparse s20:21
    exp(code, 2, 4, 0x00030000u, false, false); // PC4, sparse B, does not update VM
    exp(code, 9, 0, 0, false, true); // PC6, last VM sparse, NO payload
    if (drain) code.push_back(0xbf8c0000u);
    code.push_back(0xbefe0416u); // PC9, genuine supplied full s22:23
    exp(code, 9, 0, 0, true, false); // PC10, final EXEC full, last VM still sparse
    code.push_back(0xbf810000u);
    return p;
}
// Every original writer remains identical between the unsafe and WAIT-completed programs. No
// supplied branch/mask value certifies timing: even a same-valued EXEC replacement must drain.
inline FragmentResourcePacket pending_write(uint32_t family, bool drain, uint32_t wave = 0) {
    auto p = base(wave);
    column(p, 1, 0x87000000u);
    column(p, 0, 0);
    exp(p.invocation.guest_code, 0, 1, 1, false, false);
    if (drain) p.invocation.guest_code.push_back(0xbf8c0000u);
    auto& code = p.invocation.guest_code;
    switch (family) {
        case 0: fp::vmov(code, 1, 0); break; // pending physical source word
        case 1: code.push_back(0xbefe0414u); break; // explicit numeric EXEC pair
        case 2: code.push_back(0xbe9e2414u); break; // SAVEEXEC s30:31,s20:21 implicit EXEC
        case 3: code.insert(code.end(), {0x7daa00ffu, 40}); break; // CMPX implicit EXEC
        case 4: code.push_back(0xbefe0a7eu); break; // WQM explicit EXEC pair
        case 5: code.insert(code.end(), {0x7d8400ffu, 40}); break; // ordinary CMP only VCC
        case 6: fp::vmov(code, 2, 0); break; // disjoint VGPR never pending
    }
    exp(code, 0, 1, 1, true, true);
    code.push_back(0xbf810000u);
    return p;
}
inline FragmentResourcePacket pending_join(bool both_wait, bool scc, uint32_t wave = 0) {
    auto p = base(wave);
    p.invocation.scc = scc;
    column(p, 1, 0x87000000u);
    auto& code = p.invocation.guest_code;
    exp(code, 0, 1, 1, false, false); // PC0
    code.push_back(0xbf840002u); // PC2, SCC0 -> PC5
    code.push_back(0xbf8c0000u); // PC3, first arm genuinely drains
    code.push_back(0xbf820001u); // PC4 -> PC6
    code.push_back(both_wait ? 0xbf8c0000u : 0xbf800000u); // PC5, second arm
    fp::vmov(code, 1, 0); // PC6, join writer
    exp(code, 0, 1, 1, true, true); // PC7
    code.push_back(0xbf810000u);
    return p;
}
// Independent original-word oracle: full 14-word transport at BOTH sites and all64 lanes.
// No production decoder, source-mask helper or decoded PCs enter these expectations.
inline std::vector<uint32_t> pending_records(uint32_t family, uint32_t wave) {
    std::vector<uint32_t> words(64 * 28);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const bool before = active(lane);
        const bool after = family == 3   ? before && lane != 40
                           : family == 4 ? lane < 4 || (lane >= 28 && lane < 36) ||
                                               (lane >= 40 && lane < 44) || lane >= 60
                                         : before;
        const uint32_t row[]{1,
                             uint32_t(before),
                             uint32_t(lane != 40),
                             0,
                             1,
                             0,
                             0,
                             0,
                             before ? 0x87000000u + lane : 0,
                             0,
                             0,
                             0,
                             0,
                             uint32_t(before),
                             1,
                             uint32_t(after),
                             uint32_t(lane != 40),
                             0,
                             1,
                             0,
                             1,
                             1,
                             after ? family == 0 ? value(wave) : 0x87000000u + lane : 0,
                             0,
                             0,
                             0,
                             family == 3 || family == 5 ? 5u : 4u,
                             uint32_t(after)};
        std::copy(std::begin(row), std::end(row), words.begin() + lane * 28);
    }
    return words;
}
inline FragmentResourcePacket compressed(uint32_t en = 15) {
    auto p = base();
    column(p, 5, 0x12348000u, sparse);
    column(p, 6, 0x7fff0000u, sparse);
    // v7/v8 are absent: compressed data MUST read physical VSRC0/1, never EN-indexed2/3.
    exp(p.invocation.guest_code, 3, en, 0x08070605u, true, true, true);
    p.invocation.guest_code.push_back(0xbf810000u);
    return p;
}
inline FragmentResourcePacket pending_image(bool drain, uint32_t wave = 0) {
    auto p = base(wave);
    column(p, 10, 0);
    column(p, 11, 0);
    p.invocation.vgprs[0].words.fill(0x3e000000u); // genuine exact 4x4 base-mip XY centers
    p.invocation.vgprs[1].words.fill(0x3e000000u);
    column(p, 23, value(wave)); // highest word of the FOUR-word image destination
    auto image = fragment_resource_packet::chain().images[0];
    for (uint32_t word = 0; word < 8; ++word)
        p.invocation.sgprs.emplace_back(4 + word, image.descriptor[word]);
    for (uint32_t word = 0; word < 4; ++word)
        p.invocation.sgprs.emplace_back(12 + word, image.sampler[word]);
    auto& code = p.invocation.guest_code;
    exp(code, 0, 1, 23, false, false); // original value before the wide MIMG destination write
    if (drain) code.push_back(0xbf8c0000u);
    image.pc = static_cast<uint32_t>(code.size());
    code.insert(code.end(), {0xf0000000u | (0x27u << 18) | 0xf00u | 8u,
                             10u | (20u << 8) | (1u << 16) | (3u << 21)});
    code.push_back(0xbf8c0000u);   // actual asynchronous MIMG completion before reading its values
    exp(code, 0, 15, 0x17161514u, true, true);
    code.push_back(0xbf810000u);
    p.images.push_back(std::move(image));
    return p;
}
inline std::vector<FragmentPacketWavePlacement> placements(const FragmentPacketKernel& k) {
    const uint32_t in = k.layout.input_words + 2, out = k.layout.output_words + 128;
    return {{19 + 2 * (in + 17), 11 + out + 29}, {19, 11 + 2 * (out + 29)}, {19 + in + 17, 11}};
}
inline FragmentResourcePacket previous_destination(bool missing = false) {
    auto p = base();
    p.invocation.exec_mask = 1ull << 63;
    p.invocation.sgprs.emplace_back(16, 0);   // hand-owned original M0 writer, NOT hardware ABI
    column(p, 0, 0);
    column(p, 1, 0);
    for (auto& c : p.invocation.vgprs) c.words.fill(0);
    column(p, 10, 0x3f800000u, missing ? (1ull << 61) | (1ull << 62) : 0x7ull << 60);
    p.invocation.vgprs.back().words.fill(0x3f800000u);
    p.parameter_cache.available = true;
    p.parameter_cache.quad_primitive.fill(1);
    p.parameter_cache.parameters = {{1, 0, 0, 0x3f800000u, 0x3f800000u, 0x40000000u}};
    auto& code = p.invocation.guest_code;
    code.push_back(0xbefc0310u);
    fragment_resource_packet::interp(code, 0, 10, 0, 0);   // PC1, only63 defines scratch
    code.push_back(0xbefe0a7eu);   // PC2, reactivate60..63
    fragment_resource_packet::interp(code, 1, 10, 0, 1);   // PC3, implicit OLD VDST read
    exp(code, 0, 1, 10, true, true);
    code.push_back(0xbf810000u);
    return p;
}
// Independently hand-derived single scratch EXP transport. Do NOT use the production decoder or
// decoded site/mask helper to generate expectations. Present active zero has observed bit1.
inline std::vector<uint32_t> scratch_records(uint32_t wave, bool inactive = false) {
    std::vector<uint32_t> words(64 * 14);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const bool on = !inactive && active(lane);
        const uint32_t row[]{
            1, uint32_t(on), uint32_t(lane != 40), 0, 1, 0, 1, 1, on ? value(wave) : 0, 0, 0, 0,
            1, uint32_t(on)};
        std::copy(std::begin(row), std::end(row), words.begin() + lane * 14);
    }
    return words;
}
}   // namespace prosper::test::fragment_packet_exports
