#pragma once
// Explicit complete native subgroups with independently owned per-lane FragCoord.x values.
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "fixtures/native_fragment_spirv_oracle.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace prosper::test {
inline std::vector<std::array<uint32_t, 4>>
native_fragment_lane_values(const std::vector<uint32_t>& module, const std::vector<float>& x) {
    if (x.size() != 32 && x.size() != 64)
        throw std::runtime_error("fixture requires a complete 32/64-lane subgroup");
    bpermute_oracle::Interpreter vm(module, 4);
    if (!vm.error.empty()) throw std::runtime_error(vm.error);
    std::vector<bpermute_oracle::Lane> lanes;
    uint32_t output = 0;
    for (uint32_t lane = 0; lane < x.size(); ++lane) {
        auto [state, sink] = native_fragment_lane(
            vm, {},
            {{15, {std::bit_cast<uint32_t>(x[lane]), 0x3f000000u, 0, 0x3f800000u}},
             {41, {lane, 0, 0, 0}},
             {36, {static_cast<uint32_t>(x.size()), 0, 0, 0}}});
        lanes.push_back(std::move(state));
        output = sink;
    }
    native_fragment_subgroup(vm, lanes);
    std::vector<std::array<uint32_t, 4>> results;
    for (auto& lane : lanes) {
        const auto& memory = lane.memory.at(output);
        if (memory.words.size() != 4 ||
            !std::all_of(memory.valid.begin(), memory.valid.end(), [](bool v) { return v; }))
            throw std::runtime_error("undefined native fragment output");
        results.push_back({memory.words[0], memory.words[1], memory.words[2], memory.words[3]});
    }
    return results;
}
} // namespace prosper::test
