// A bottom-tested scalar loop retires an entry VCC mask on both exit paths.
// Its numeric halves remain ordinary loop DATA; no replacement mask may be invented.
#pragma once
#include "fixtures/arbitrary_placeholder.hpp"
#include "fixtures/native_execution_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace prosper::test::loop_vcc_data_exit {
struct Options {
    uint32_t trips = 4;
    uint32_t condition = 0;
    bool shift = false;
    bool entry_mask = true;
    bool missing_high = false;
    bool spill_high = false;
    bool known_launch = true;
};
inline gpu::ComputeShaderConfig config(Options o) {
    auto c = native_execution_launch();
    c.user_sgprs.resize(16);
    c.user_sgprs[4] = o.condition;
    if (!o.known_launch) c.dispatch_launch.known = 0;
    return c;
}
inline std::vector<uint32_t> program(Options o) {
    std::vector<uint32_t> code{
        0x7e040300u, // v2 = lane, output index
        0x7e020280u, // v1 = 0, conditional iteration count
        0xbe800380u, // s0 = 0
        o.entry_mask ? 0x7d840100u : 0xbf800000u, // VCC is initially a mask
        o.shift ? 0x8f6a8400u : 0x936a9000u, // header: vcc_lo = s0 * 16 (shift or multiply)
        o.missing_high ? 0xbeeb0311u
        : o.spill_high ? 0xbeeb0331u
                       : 0xbeeb0387u,
        0xbf068004u, // SCC = (supplied s4 == 0)
        0xbf840001u, // nested scalar if skips the count increment
        0x4a020281u, // v1 += 1
        0x80008100u, // s0 += 1
        0xbf0a8000u | ((128u + o.trips) << 8u), // s0 < trips
        0xbf85fff8u, // SCC1 -> header pc4
        0x7e06026au, // v3 = current numeric VCC_LO
        0x4a020303u, // v1 += v3
        0x7e06026bu, // v3 = current numeric VCC_HI
        0x4a020303u, // v1 += v3
        0x7d840687u, // VCC = (v3 == 7), checking which physical half was read
        0x02020296u, // v1 = VCC ? v1 : 22
        0xe0702000u,
        0x80020102u, // output[lane] = v1
        0xbf810000u,
    };
    if (o.spill_high) code = arbitrary_placeholder::with_missing_spill_root(code, 49);
    return code;
}
inline std::vector<uint32_t> compile(Options o, uint64_t address = 0) {
    const auto code = program(o);
    const auto c = config(o);
    const auto table = arbitrary_placeholder::table();
    return gpu::recompile_compute(code.data(), code.size(), &table, c,
                                  {gpu::RecompileDiagnosticStage::Compute, address});
}
inline uint32_t expected(Options o) {
    return (o.trips - 1u) * 16u + 7u + (o.condition == 0 ? o.trips : 0u);
}
inline bool has_switch(const std::vector<uint32_t>& module) {
    for (std::size_t at = 5; at < module.size();) {
        const auto width = module[at] >> 16u;
        if (!width || width > module.size() - at) return false;
        if ((module[at] & 65535u) == 251u) return true;
        at += width;
    }
    return false;
}
} // namespace prosper::test::loop_vcc_data_exit
