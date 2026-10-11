#pragma once

// Project-owned B64 predicates with separate old/current/new VCC generations.
// Shared by the public NGG compiler tests and strict SPIR-V representatives.
#include "fragment_compare_kill.hpp"
#include "ngg_lane_slots.hpp"
#include <cstdint>
#include <vector>

namespace prosper::test::ngg_bool_copy {
constexpr uint32_t mov(uint32_t dst, uint32_t source, bool pair = false) {
    return 0xbe800000u | (dst << 16) | ((pair ? 4u : 3u) << 8) | source;
}
inline std::vector<uint32_t> program(bool barrier = true, bool numeric = false,
                                     uint32_t compare_source = 0) {
    namespace compare = fragment_compare_kill;
    std::vector<uint32_t> code{
        // The real NGG ABI requires the guest to establish EXEC before VALU.
        ngg_slots::kExecFull,
        mov(106, 129, true),
        mov(30, 106),
        compare::compare(),
        compare::compare_sd(20, compare_source),
        mov(106, 20, true),
        0xd5010009u,
        129u | (137u << 9) | (106u << 18),
        0xbf860001u,
        0xbf800000u,
    };
    if (numeric) code.push_back(mov(31, 107));
    code.insert(code.end(), {mov(106, 193, true), mov(31, 107)});
    if (barrier) code.push_back(0xbf8a0000u);
    // Reuse the observable project-owned NGG export fixture. v0..3 are genuine
    // shell launch inputs; s21 is explicitly initialized before the primitive EXP.
    ngg_slots::scalar(code, 21, 7);
    ngg_slots::finish(code);
    return code;
}
} // namespace prosper::test::ngg_bool_copy
