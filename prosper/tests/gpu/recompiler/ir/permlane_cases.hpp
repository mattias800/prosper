// Test-owned selector arrays and row arithmetic, independent of the production packed-nibble graph.
#pragma once
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include <cstddef>
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>
namespace prosper::test::ir::permlane {
struct Table {
    uint32_t lo, hi;
    std::array<uint8_t, 16> picks;
    uint32_t src_lo = 20, src_hi = 21;
};
inline constexpr std::array kTables{
    Table{0x76543210, 0xfedcba98, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}},
    Table{0x87654321, 0x0fedcba9, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0}},
    Table{0x89abcdef, 0x01234567, {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
    Table{0, 0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    Table{UINT32_MAX, UINT32_MAX, {15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15}},
    Table{0x76543210, 0x76543210, {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7}, 255, 255},
    Table{0, UINT32_MAX, {0, 0, 0, 0, 0, 0, 0, 0, 15, 15, 15, 15, 15, 15, 15, 15}, 128, 193},
    Table{0x87654321, 0x0fedcba9, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0}, 255, 21},
    Table{0x87654321, 0x0fedcba9, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0}, 20, 255},
    Table{0x3f000000, 0x3f000000, {0, 0, 0, 0, 0, 0, 15, 3, 0, 0, 0, 0, 0, 0, 15, 3}, 240, 240},
};
inline uint32_t source_lane(uint32_t lane, const Table& table, bool across) {
    const uint32_t row = lane / 16;
    return (across ? row ^ 1u : row) * 16 + table.picks[lane % 16];
}
inline uint32_t value(const LaneInputs& in, uint32_t lane, const Table& table, bool across,
                      bool fetch, bool bound, bool alias = false) {
    const uint32_t peer = source_lane(lane, table, across);
    if (!fetch && !in.active[peer])
        // Compatibility only for BC=0; the inherited interaction is not a new ISA claim.
        return bound ? 0 : in.old_d[lane];
    return alias ? in.old_d[peer] : in.s0[peer];
}
inline LaneInputs inputs(uint32_t wave, uint32_t mask = 0) {
    auto in = lane_inputs(0x5372 + wave, wave);
    for (uint32_t lane = 0; lane < wave; ++lane) {
        in.s0[lane] = 0x57200000u ^ (lane * 0x01010101u) ^ (lane << 20);
        in.old_d[lane] = 0xbad00000u ^ (lane * 1009u);
        if (mask == 1) in.active[lane] = true;
        if (mask == 2) in.active[lane] = false;
    }
    return in;
}
inline std::vector<uint32_t> words(bool across, const Table& table, bool fetch, bool bound,
                                   bool alias = false) {
    std::vector<uint32_t> out;
    if (table.src_lo == 20) out.insert(out.end(), {0xbe9403ff, table.lo});
    if (table.src_hi == 21) out.insert(out.end(), {0xbe9503ff, table.hi});
    out.insert(out.end(), {(across ? 0xd7780001u : 0xd7770001u) | (uint32_t(fetch) << 11) |
                               (uint32_t(bound) << 12),
                           (alias ? 257u : 258u) | (table.src_lo << 9) | (table.src_hi << 18)});
    if (table.src_lo == 255 || table.src_hi == 255) {
        if (table.src_lo == 255 && table.src_hi == 255 && table.lo != table.hi) return {};
        out.push_back(table.src_lo == 255 ? table.lo : table.hi);
    }
    return out;
}
inline LaneProgram program(bool across, const Table& table, bool fetch, bool bound, uint32_t wave,
                           bool alias = false, bool full_exec = false) {
    auto p = lane_program(words(across, table, fetch, bound, alias), wave, wave);
    p.config.native_subgroup_size = wave;
    if (full_exec) {
        const auto it = std::find(p.code.begin(), p.code.end(), 0x7daa0880u);
        if (it == p.code.end())
            p.code.clear();
        else
            p.code.erase(it);
    }
    return p;
}
inline bool has_shuffle(const std::vector<uint32_t>& module) {
    for (size_t i = 5; i < module.size();) {
        const uint32_t n = module[i] >> 16;
        if (!n || n > module.size() - i) return false;
        if ((module[i] & 0xffffu) == 345) return true;
        i += n;
    }
    return false;
}
inline LaneRun execute(const std::vector<uint32_t>& module, const LaneInputs& in) {
    std::vector<uint32_t> buffer;
    for (const auto* a : {&in.s0, &in.s1, &in.old_d})
        buffer.insert(buffer.end(), a->begin(), a->end());
    for (bool a : in.active) buffer.push_back(a ? 1u : 0u);
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = static_cast<uint32_t>(in.lanes());
    auto out = vm.run_buffers(static_cast<uint32_t>(in.lanes()), {{3, buffer}}, 3);
    if (!vm.error.empty()) return {{}, vm.error};
    if (out.size() < in.lanes()) return {{}, "missing stored lanes"};
    out.resize(in.lanes());
    return {out, {}};
}
}   // namespace prosper::test::ir::permlane
