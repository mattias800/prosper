#include <cstdint>
#include <vector>
#include "gpu/recompiler/spirv/fragment_reconvergence.hpp"
#include <cstddef>

namespace prosper::gpu {
FragmentReconvergence fragment_spirv_reconvergence(const std::vector<uint32_t>& module) {
    if (module.size() < 5 || module[0] != 0x07230203u) return FragmentReconvergence::Malformed;
    bool required = false;
    for (size_t pc = 5; pc < module.size();) {
        const auto count = module[pc] >> 16;
        const auto op = module[pc] & 0xffffu;
        if (!count || count > module.size() - pc) return FragmentReconvergence::Malformed;
        if (op == 16) { // OpExecutionMode, not an arbitrary literal equal to the mode number
            if (count < 3) return FragmentReconvergence::Malformed;
            if (module[pc + 2] == kMaximallyReconvergesKHR) {
                if (count != 3) return FragmentReconvergence::Malformed;
                required = true;
            }
        }
        pc += count;
    }
    return required ? FragmentReconvergence::Required : FragmentReconvergence::None;
}
} // namespace prosper::gpu
