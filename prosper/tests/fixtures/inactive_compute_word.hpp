#pragma once
// LLVM gfx1030 Wave64: a full-word MOV reads s16, not its old destination. The loop's
// future buffer load leaves a real SSA backedge word, even before that load is observed.
// Only the low 32 lanes store; the other 32 output sentinels must survive both trips.
#include "fixtures/arbitrary_placeholder.hpp"
#include <array>
#include <cstdint>
#include <vector>

namespace prosper::test::inactive_compute_word {
inline constexpr std::array kProgram{
    0xf4200402u, 0xfa000000u, 0xbf8cc07fu, 0xbe9e0380u, 0x7e12023cu, 0x7d841309u, 0xbe94047eu,
    0xd4c1006au, 0x00014100u, 0xbe96246au, 0x7e0c0210u, 0xe0702000u, 0x80020600u, 0xbefe0414u,
    0xe0302000u, 0x80010600u, 0xbf8c3f70u, 0x801e811eu, 0xbf0a821eu, 0xbf85fff2u, 0xbf810000u,
};
inline std::vector<uint32_t> program() {
    return {kProgram.begin(), kProgram.end()};
}
inline auto launch() {
    return arbitrary_placeholder::config();
}
inline auto resources(uint64_t input_address = 0, uint64_t output_address = 0) {
    auto result = arbitrary_placeholder::table(output_address);
    auto input = result.resources.front();
    input.binding = 2;
    input.sgpr_base = 4;
    input.gpu_addr = input_address;
    result.resources.push_back(input);
    return result;
}
inline std::vector<uint32_t> input(uint32_t seed) {
    std::vector<uint32_t> values(64);
    for (uint32_t lane = 0; lane < values.size(); ++lane) values[lane] = seed + lane;
    return values;
}
inline std::vector<uint32_t> expected(uint32_t seed) {
    std::vector<uint32_t> values(64, 0xdeadbeefu);
    for (uint32_t lane = 0; lane < 32; ++lane) values[lane] = seed;
    return values;
}
} // namespace prosper::test::inactive_compute_word
