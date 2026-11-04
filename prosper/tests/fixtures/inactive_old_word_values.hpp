#pragma once
// Complete native cohorts with explicit varying colours and independently owned FragCoord.x.
#include "fixtures/native_fragment_spirv_oracle.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace prosper::test::inactive_old_word {
inline std::vector<std::array<uint32_t, 4>> values(const std::vector<uint32_t>& module,
                                                   const std::vector<float>& x,
                                                   const std::vector<float>& red,
                                                   bool supply_attribute = true) {
    if ((x.size() != 32 && x.size() != 64) || red.size() != x.size())
        throw std::runtime_error("inactive-word fixture requires a complete native cohort");
    bpermute_oracle::Interpreter vm(module, 4);
    if (!vm.error.empty()) throw std::runtime_error(vm.error);
    std::vector<bpermute_oracle::Lane> lanes;
    uint32_t output = 0;
    for (uint32_t lane = 0; lane < x.size(); ++lane) {
        std::map<uint32_t, std::array<uint32_t, 4>> locations;
        if (supply_attribute)
            locations[0] = {std::bit_cast<uint32_t>(red[lane]), 0, 0, 0x3f800000u};
        auto invocation = native_fragment_lane(
            vm, locations,
            {{15, {std::bit_cast<uint32_t>(x[lane]), 0x3f000000u, 0, 0x3f800000u}},
             {41, {lane, 0, 0, 0}},
             {36, {static_cast<uint32_t>(x.size()), 0, 0, 0}}});
        lanes.push_back(std::move(invocation.first));
        output = invocation.second;
    }
    native_fragment_subgroup(vm, lanes);
    std::vector<std::array<uint32_t, 4>> result;
    for (auto& lane : lanes) {
        const auto& memory = lane.memory.at(output);
        if (memory.words.size() != 4 || !std::all_of(memory.valid.begin(), memory.valid.end(),
                                                     [](bool valid) { return valid; }))
            throw std::runtime_error("undefined inactive-word fixture output");
        result.push_back({memory.words[0], memory.words[1], memory.words[2], memory.words[3]});
    }
    return result;
}
} // namespace prosper::test::inactive_old_word
