// #5441: replayable guest programs keep the scalar pair across a runtime structured merge.
#pragma once

#include "gpu/recompiler/ir/vector_lane_harness.hpp"

#include <cstdint>
#include <vector>

namespace prosper::test::sop1_phi {
inline constexpr uint32_t kLow = 0x600dcafe, kHigh = 0x7ee1f00d;

inline ir::LaneProgram witness(bool has_else) {
    // s0 is runtime WorkgroupId.x. Partial EXEC comes from the harness's buffer.
    std::vector<uint32_t> body{0xbe8603ff, kLow, 0xbe8703ff, kHigh, 0xbf068000};
    body.push_back(has_else ? 0xbf850002 : 0xbf850001);
    body.push_back(0xbe860a81); // numeric WQM of 1
    if (has_else) body.insert(body.end(), {0xbf820001, 0xbe860ac5}); // WQM of -5
    body.insert(body.end(), {0xbefe04c1, 0x7e020206, 0x7e0a0207});
    auto p = ir::lane_program(body, 64, 64);
    p.code.insert(p.code.end() - 1, {0xe0702100, 0x80020500}); // high word in array 1
    p.config.tgid_x_en = true;
    p.config.dispatch_launch.dimensions = {2, 1, 1};
    return p;
}

inline std::vector<uint32_t> buffer() {
    std::vector<uint32_t> words(256, 0xdeadbeef);
    for (uint32_t lane = 0; lane < 64; ++lane) words[192 + lane] = lane % 8 == 0;
    return words;
}

inline uint64_t expected(bool has_else, uint32_t selector) {
    if (!selector && !has_else) return (uint64_t(kHigh) << 32) | kLow;
    return selector ? uint64_t{15} : ~uint64_t{0};
}
} // namespace prosper::test::sop1_phi
