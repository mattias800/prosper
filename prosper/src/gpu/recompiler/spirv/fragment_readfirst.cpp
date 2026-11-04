// AMD 70648 12.8: choose the lowest EXEC lane, or lane zero when EXEC is empty.
#include <cstdint>
#include "gpu/recompiler/rdna2_native_fragment_collectives.hpp"
#include "gpu/recompiler/spirv/fragment_reconvergence.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstddef>
#include <vector>

namespace prosper::gpu {
namespace {
bool require_reconvergence(SpirvCompute& b) {
    bool maximal = false, full_quads = false;
    for (size_t pc = 0; pc < b.exec.size();) {
        const auto count = b.exec[pc] >> 16;
        if (!count || count > b.exec.size() - pc) return false;
        if ((b.exec[pc] & 0xffffu) == Op_ExecutionMode && count == 3 && b.exec[pc + 1] == b.f_main) {
            maximal |= b.exec[pc + 2] == kMaximallyReconvergesKHR;
            full_quads |= b.exec[pc + 2] == kRequireFullQuadsKHR;
        }
        pc += count;
    }
    const auto extension = [&](const char* name) {
        std::vector<uint32_t> words;
        b.pstr(words, name);
        b.putv(b.exts, Op_Extension, words);
    };
    if (!maximal) {
        extension("SPV_KHR_maximal_reconvergence");
        b.put(b.exec, Op_ExecutionMode, {b.f_main, kMaximallyReconvergesKHR});
    }
    if (!full_quads) {
        extension("SPV_KHR_quad_control");
        b.put(b.caps, Op_Capability, {5087}); // QuadControlKHR
        b.put(b.exec, Op_ExecutionMode, {b.f_main, kRequireFullQuadsKHR});
    }
    return true;
}
bool native_fragment(const SpirvCompute& b) {
    return b.is_fragment && !b.is_compute && !b.is_vertex && !b.is_fragment_packet() &&
           !b.ngg_one_lane && !b.ngg_workgroup_shell &&
           b.guest_stage == GuestShaderStage::Fragment &&
           b.physical_domain == PhysicalExecutionDomain::Invocation &&
           (b.wave_size == 32 || b.wave_size == 64);
}
} // namespace

uint32_t fragment_collective_entry_exec(SpirvCompute& b) {
    if (!native_fragment_collective_retry_active() || !native_fragment(b) ||
        !require_reconvergence(b)) return 0;
    // Hardware starts in exact execution: helpers exist but do not own a valid pixel.
    return b.logical_not(b.helper_invocation());
}
uint32_t fragment_collective_wqm(SpirvCompute& b, uint32_t active) {
    if (!native_fragment_collective_retry_active() || !native_fragment(b) || !active ||
        !require_reconvergence(b)) return 0;
    const auto result = b.id();
    b.put(b.code, 5111, {b.t_bool, result, active}); // OpGroupNonUniformQuadAnyKHR
    return result;
}

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
