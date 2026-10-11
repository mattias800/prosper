#include "gpu/recompiler/rdna2_readlane_route.hpp"
#include <algorithm>
#include <set>

namespace prosper::gpu {
bool has_ordinary_cfg_readlane(const std::vector<Rdna2Inst>& ins) {
    std::set<int> cfg_writelane_spill_arrays;
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361)
            cfg_writelane_spill_arrays.insert(in.dst.value);
    }
    return std::any_of(ins.begin(), ins.end(), [&](const Rdna2Inst& in) {
        return !in.is_end && in.fmt == Rdna2Format::VOP3 && in.opcode == 0x360 &&
               !cfg_writelane_spill_arrays.contains(in.src[0].value);
    });
}

} // namespace prosper::gpu
