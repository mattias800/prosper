#pragma once
#include "fragment_packet_fixture.hpp"
#include "fragment_resource_packet_fixture.hpp"
#include "fragment_packet_mbcnt_fixture.hpp"
#include <array>
#include <string>
#include <utility>

namespace prosper::test::fragment_definedness {
using namespace prosper::gpu;
namespace fp = fragment_packet;
struct Case {
    std::string name;
    FragmentInvocationPacket packet;
    std::vector<uint32_t> expected;
    uint32_t failure_lane = UINT32_MAX, failure_pc = UINT32_MAX, failure_reg = UINT32_MAX;
    uint32_t failure_kind = 0;
};
inline constexpr uint32_t kIntegerRailCount = 30;
inline FragmentInvocationPacket base() {
    FragmentInvocationPacket p;
    p.slots_available.fill(true);
    p.export_enabled.fill(1);
    p.mask_state_available = true;
    p.exec_mask = UINT64_MAX;
    p.sgprs = {{0, 0x42230011u}, {20, UINT32_MAX}, {21, UINT32_MAX}};
    return p;
}
inline void column(FragmentInvocationPacket& p, uint32_t reg, uint32_t bits,
                   uint64_t available = UINT64_MAX) {
    FragmentPacketVgpr c;
    c.reg = reg;
    c.words.fill(bits);
    c.available_mask = available;
    p.vgprs.push_back(c);
}
inline std::vector<uint32_t> expected(uint64_t exec, uint32_t active,
                                      uint32_t inactive = 0xdeadbeefu) {
    std::vector<uint32_t> out(64 * 12);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const bool on = ((exec >> lane) & 1u) != 0;
        const uint32_t row[]{1, uint32_t(on), 1, 0, 1, 0, 1, 1, on ? active : inactive, 0, 0, 0};
        std::copy(std::begin(row), std::end(row), out.begin() + lane * 12);
    }
    return out;
}
inline FragmentInvocationPacket numeric_exec_pair(uint64_t mask) {
    auto p = base();
    p.sgprs.emplace_back(2, 0x73330042u);
    for (auto& [reg, value] : p.sgprs) {
        if (reg == 20) value = static_cast<uint32_t>(mask);
        if (reg == 21) value = static_cast<uint32_t>(mask >> 32);
    }
    fp::vmov(p.guest_code, 2, 0);   // full-EXEC writer establishes absent scratch for all raw rows
    p.guest_code.push_back(0xbefe0414u);   // original S_MOV_B64 EXEC,s[20:21] form
    fp::vmov(p.guest_code, 2, 2);   // scalar transfer controls this actual masked vector write
    fp::exp(p.guest_code, 1, 2);
    p.guest_code.push_back(0xbf810000u);
    return p;
}
inline FragmentInvocationPacket saved_mask_high_overwrite(bool boundary) {
    auto p = base();
    column(p, 1, 0xdeadbeefu);
    p.guest_code.push_back(0xbe94047eu);   // S_MOV_B64 s[20:21],EXEC: Bool, not numeric halves
    fp::smov(p.guest_code, 21, 0);   // legal high-only overwrite destroys the complete old mask
    if (boundary) p.guest_code.push_back(0xbf820000u);   // next-instruction dispatcher boundary
    p.guest_code.push_back(0xbefe0414u);   // cannot reuse the stale complete Bool mask
    fp::vmov(p.guest_code, 1, 0);
    fp::exp(p.guest_code, 1, 1);
    p.guest_code.push_back(0xbf810000u);
    return p;
}
inline std::vector<Case> cases() {
    std::vector<Case> out;
    auto p = base();
    fp::vmov(p.guest_code, 1, 0);
    fp::exp(p.guest_code, 1, 1);
    p.guest_code.push_back(0xbf810000u);
    out.push_back({"writer_only", p, expected(UINT64_MAX, 0x42230011u)});
    auto chain = base();
    column(chain, 0, 0);
    fp::vmov(chain.guest_code, 1, 256);
    fp::vmov(chain.guest_code, 2, 257);
    fp::exp(chain.guest_code, 1, 2);
    chain.guest_code.push_back(0xbf810000u);
    out.push_back({"source_present_zero", chain, expected(UINT64_MAX, 0)});
    auto missing = chain;
    missing.vgprs[0].available_mask &= ~(uint64_t(1) << 63);
    out.push_back({"source_absent_zero_lane63", missing, {}, 63, 0, 0, 1});
    auto masked = base();
    masked.exec_mask &= ~(uint64_t(1) << 40);
    column(masked, 0, 0, masked.exec_mask);
    column(masked, 1, 0xdeadbeefu);
    fp::vmov(masked.guest_code, 1, 256);
    fp::exp(masked.guest_code, 1, 1);
    masked.guest_code.push_back(0xbf810000u);
    out.push_back(
        {"inactive_source_unconsumed_owned_destination", masked, expected(masked.exec_mask, 0)});
    auto raw = base();
    raw.exec_mask = 0;
    raw.export_enabled.fill(0);
    fp::exp(raw.guest_code, 1, 1);
    raw.guest_code.push_back(0xbf810000u);
    out.push_back({"inactive_raw_payload_still_observed", raw, {}, 0, 0, 1, 4});
    auto restore = p;
    restore.exec_mask &= ~(uint64_t(1) << 40);
    restore.guest_code.insert(restore.guest_code.begin() + 1,
                              0xbefe0414u);   // restore full EXEC from s20:21
    out.push_back({"masked_writer_not_full_definition", restore, {}, 40, 2, 1, 4});
    auto branch = p;
    branch.guest_code.insert(branch.guest_code.begin(), 0xbf840001u);   // SCC0 skips actual MOV
    branch.scc = true;
    out.push_back({"executed_forward_writer", branch, expected(UINT64_MAX, 0x42230011u)});
    branch.scc = false;
    out.push_back({"skipped_forward_writer", branch, {}, 0, 2, 1, 4});
    auto overlap = base();
    fp::vmov(overlap.guest_code, 1, 257);
    fp::vmov(overlap.guest_code, 1, 0);
    fp::exp(overlap.guest_code, 1, 1);
    overlap.guest_code.push_back(0xbf810000u);
    out.push_back({"later_writer_cannot_launder_old_read", overlap, {}, 0, 0, 1, 1});
    for (bool owned : {true, false}) {
        auto peer = base();
        peer.exec_mask &= ~(uint64_t(1) << 40);
        if (owned) column(peer, 1, 0x73330040u, uint64_t(1) << 40);
        fp::vmov(peer.guest_code, 1, 0);
        peer.guest_code.insert(peer.guest_code.end(), {0xd7600003u, 257u | (168u << 9)});
        peer.guest_code.push_back(0xbefe0414u);
        fp::vmov(peer.guest_code, 2, 3);
        fp::exp(peer.guest_code, 1, 2);
        peer.guest_code.push_back(0xbf810000u);
        out.push_back({owned ? "selected_inactive_peer_owned" : "selected_inactive_peer_absent",
                       peer, owned ? expected(UINT64_MAX, 0x73330040u) : std::vector<uint32_t>{},
                       owned ? UINT32_MAX : 0u, owned ? UINT32_MAX : 1u, owned ? UINT32_MAX : 1u,
                       owned ? 0u : 3u});
    }
    auto widened = base();
    widened.exec_mask = 1;
    widened.quad_topology = FragmentPacketQuadTopology::ConsecutiveLogicalQuads;
    column(widened, 0, 0, 1);
    column(widened, 2, 0xdeadbeefu);
    fp::vmov(widened.guest_code, 1, 256);
    widened.guest_code.push_back(0xbefe0a7eu);   // S_WQM_B64 EXEC,EXEC
    fp::vmov(widened.guest_code, 2, 257);
    fp::exp(widened.guest_code, 1, 2);
    widened.guest_code.push_back(0xbf810000u);
    out.push_back({"wqm_newly_active_read_absent", widened, {}, 1, 2, 1, 1});
    column(widened, 1, 0, ~uint64_t(1));   // old neighbors supplied, lane0 defined by actual MOV
    out.push_back({"wqm_newly_active_read_owned", widened, expected(15, 0)});
    for (const auto source : {fp::mbcnt::Source::Exec, fp::mbcnt::Source::Vgpr}) {
        fp::mbcnt::Case c;
        c.source = source;
        auto counter = fp::mbcnt::packet(c);
        std::erase_if(counter.vgprs,
                      [](const auto& column) { return column.reg == 1 || column.reg == 2; });
        out.push_back({source == fp::mbcnt::Source::Exec ? "common_mask_counter_writer_only"
                                                         : "numeric_counter_writer_only",
                       counter, fp::mbcnt::expected(c)});
    }
    fp::Case later;
    later.second_export = true;
    auto events = fp::packet(later);
    std::erase_if(events.vgprs, [](const auto& c) { return c.reg != 0 && c.reg != 8; });
    out.push_back(
        {"distinct_readlane_event_after_source_change", events, fp::expected(later, events)});
    auto two_reads = base();
    fp::vmov(two_reads.guest_code, 1, 256);   // actual absent v0 read at PC0
    fp::vmov(two_reads.guest_code, 2, 259);   // distinct absent v3 read at PC1
    fp::exp(two_reads.guest_code, 1, 2);
    two_reads.guest_code.push_back(0xbf810000u);
    out.push_back({"two_bad_reads_keep_first_failure", two_reads, {}, 0, 0, 0, 1});
    column(two_reads, 0, 0);
    column(two_reads, 3, 0x73330003u);
    out.push_back({"same_two_reads_genuine_inputs", two_reads, expected(UINT64_MAX, 0x73330003u)});
    constexpr uint64_t low_bits = 0x80000001u;   // independent positions0/31
    constexpr uint64_t high_bits = uint64_t(0x80000101u) << 32;   // positions32/40/63
    for (const auto& [name, mask] : std::array<std::pair<const char*, uint64_t>, 5>{
             {{"numeric_exec_present_zero", 0},
              {"numeric_exec_all_ones", UINT64_MAX},
              {"numeric_exec_low_positions", low_bits},
              {"numeric_exec_high_positions", high_bits},
              {"numeric_exec_asymmetric_both_halves", low_bits | high_bits}}}) {
        const auto input = numeric_exec_pair(mask);
        out.push_back({name, input, expected(mask, 0x73330042u, 0x42230011u)});
    }
    auto overwrite = numeric_exec_pair(1 | (uint64_t(1) << 40));
    std::vector<uint32_t> high_zero;
    fp::smov(high_zero, 21, 0);
    overwrite.guest_code.insert(overwrite.guest_code.begin() + 1, high_zero.begin(),
                                high_zero.end());
    out.push_back(
        {"numeric_exec_current_high_overwrite", overwrite, expected(1, 0x73330042u, 0x42230011u)});
    for (bool scc : {false, true}) {
        auto join = overwrite;
        join.guest_code.insert(join.guest_code.begin() + 1,
                               0xbf840002u);   // SCC0 skips high writer
        join.scc = scc;
        const uint64_t mask = scc ? 1 : 1 | (uint64_t(1) << 40);
        out.push_back({scc ? "numeric_exec_join_current_zero" : "numeric_exec_join_entry_high",
                       join, expected(mask, 0x73330042u, 0x42230011u)});
        auto preserved_scc = numeric_exec_pair(low_bits | high_bits);
        preserved_scc.guest_code.insert(preserved_scc.guest_code.begin() + 2,
                                        0xbf840001u);   // AFTER transfer: SCC0 skips masked MOV
        preserved_scc.scc = scc;
        out.push_back(
            {scc ? "numeric_exec_preserves_scc_true" : "numeric_exec_preserves_scc_false",
             preserved_scc,
             expected(low_bits | high_bits, scc ? 0x73330042u : 0x42230011u, 0x42230011u)});
    }
    auto copy = numeric_exec_pair(low_bits | high_bits);
    copy.guest_code.insert(copy.guest_code.begin() + 1, 0xbe960414u);   // copy numeric pair to22:23
    copy.guest_code.insert(copy.guest_code.begin() + 2, high_zero.begin(), high_zero.end());
    copy.guest_code[4] = 0xbefe0416u;   // consume owned copy after original high word changed
    out.push_back({"numeric_exec_copied_pair_survives_source_change", copy,
                   expected(low_bits | high_bits, 0x73330042u, 0x42230011u)});
    auto disabled_entry = numeric_exec_pair(uint64_t(1) << 40);
    disabled_entry.exec_mask = 0;
    column(disabled_entry, 2, 0x42230011u);   // genuine old inactive raw payload, not scratch zero
    std::vector<uint32_t> unused_writer;
    fp::vmov(unused_writer, 3, 0);
    disabled_entry.guest_code.insert(disabled_entry.guest_code.begin() + 1,
                                     unused_writer.front());   // absent, unobserved scratch
    out.push_back({"numeric_exec_move_ignores_old_exec", disabled_entry,
                   expected(uint64_t(1) << 40, 0x73330042u, 0x42230011u)});
    return out;
}
inline FragmentResourcePacket resource_chain(bool lod, bool inactive) {
    auto p = fragment_resource_packet::chain(lod, inactive);
    for (auto& column : p.invocation.vgprs) {
        if (column.reg <= 1) {
            if (inactive) column.available_mask &= ~(uint64_t(1) << 40);
        } else if (inactive)
            column.available_mask = uint64_t(1) << 40;
    }
    std::erase_if(p.invocation.vgprs, [&](const auto& column) {
        if (column.reg <= 1) return false;
        // Only raw inactive export/selected-peer observations need entry payload. Other writer-
        // only temporaries are absent, not supplied poison just to satisfy register allocation.
        const bool observed = column.reg == 10 || column.reg == 11 || column.reg == 13 ||
                              column.reg == 14 || (column.reg >= 20 && column.reg <= 24);
        if (inactive && observed) return false;
        return true;
    });
    return p;
}
inline FragmentResourcePacket resource_missing_lod() {
    auto p = resource_chain(true, false);
    // Remove BOTH definitions of v12. Preserve the earlier arithmetic sink using real loaded
    // s16 rather than consuming v12 first; only IMAGE_SAMPLE_L's third coordinate reads it.
    p.invocation.guest_code[8] = 0x7e260280u;   // MOV v19,inline0 instead of MOV v12,s16
    p.invocation.guest_code[9] = 0x7e1a0210u;   // MOV v13,s16 instead of MUL from absent v12
    p.invocation.guest_code[13] = 0x7e260280u;   // MOV v19,inline0 instead of MOV v12,s18
    return p;
}
}   // namespace prosper::test::fragment_definedness
