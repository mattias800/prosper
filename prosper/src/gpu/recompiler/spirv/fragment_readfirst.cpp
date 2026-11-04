// AMD 70648 12.8: choose the lowest EXEC lane, or lane zero when EXEC is empty.
#include "gpu/recompiler/spirv/fragment_reconvergence.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstddef>
#include <vector>

namespace prosper::gpu {
namespace {
bool require_reconvergence(SpirvCompute& b) {
    // Vulkan helper invocations must participate in these ordinary group operations. The
    // execution mode supplies that contract; merely enabling the device feature does not.
    for (size_t pc = 0; pc < b.exec.size();) {
        const auto count = b.exec[pc] >> 16;
        if (!count || count > b.exec.size() - pc) return false;
        if ((b.exec[pc] & 0xffffu) == Op_ExecutionMode && count == 3 &&
            b.exec[pc + 1] == b.f_main && b.exec[pc + 2] == kMaximallyReconvergesKHR)
            return true;
        pc += count;
    }
    std::vector<uint32_t> extension;
    b.pstr(extension, "SPV_KHR_maximal_reconvergence");
    b.putv(b.exts, Op_Extension, extension);
    b.put(b.exec, Op_ExecutionMode, {b.f_main, kMaximallyReconvergesKHR});
    return true;
}
} // namespace

uint32_t fragment_readfirst_value(SpirvCompute& b, uint32_t source, uint32_t exec_mask) {
    if (!b.is_fragment || !source || !exec_mask || (b.wave_size != 32 && b.wave_size != 64) ||
        !require_reconvergence(b))
        return 0;
    const auto zero = b.uconst(0);
    const auto low = b.fragment_wave_ballot_half(exec_mask, 0);
    const auto low_nonempty = b.ucmp(Op_INotEqual, low, zero);
    uint32_t fallback = zero;
    if (b.wave_size == 64) {
        const auto high = b.fragment_wave_ballot_half(exec_mask, 1);
        const auto high_lane = b.ibin(Op_IAdd, b.find_ilsb(high), b.uconst(32));
        fallback = b.sel(b.ucmp(Op_INotEqual, high, zero), high_lane, zero);
    }
    const auto lane = b.sel(low_nonempty, b.find_ilsb(low), fallback);
    // Like the existing native READLANE, access and the scalar write ignore guest EXEC.
    return b.subgroup_shuffle(source, lane);
}
} // namespace prosper::gpu
