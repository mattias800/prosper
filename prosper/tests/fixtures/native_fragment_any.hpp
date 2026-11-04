#pragma once
// Complete, independently supplied native subgroups only; no missing-lane authority.
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace prosper::test {
inline bool native_fragment_any(bpermute_oracle::Interpreter& vm,
    std::vector<bpermute_oracle::Lane>& lanes, size_t site) {
    const auto& collective = vm.ins.at(site);
    const auto& a = collective.a;
        if (collective.op == 5111 || collective.op == 335) {
            const bool quad = collective.op == 5111;
            if ((lanes.size() != 32 && lanes.size() != 64) ||
                a.size() != (quad ? 3u : 4u) || vm.ty(a[0]).op != 20)
                throw std::runtime_error("unsupported native fragment any");
            std::vector<bool> values;
            for (auto& lane : lanes) {
                if (lane.done || lane.pc != site || (!quad && vm.word(lane, a[2]) != 3))
                    throw std::runtime_error("partial/divergent native fragment any");
                values.push_back(vm.boolean(lane, a[quad ? 2 : 3]));
            }
            const size_t group = quad ? 4 : lanes.size();
            for (size_t base = 0; base < lanes.size(); base += group) {
                const bool any = std::any_of(values.begin() + base, values.begin() + base + group,
                                             [](bool value) { return value; });
                for (size_t index = base; index < base + group; ++index) {
                    vm.set(lanes[index], a[1], a[0], {static_cast<uint32_t>(any)});
                    ++lanes[index].pc;
                }
            }
            return true;
        }
    return false;
}
} // namespace prosper::test
