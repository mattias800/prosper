// Arithmetic statements relocated by the case-route extractor.
#include "gpu/recompiler/ir/lower/flat_window_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
std::array<uint32_t, 2> lower_flat_window_component(SpirvCompute& b, uint32_t byte0,
                                                    const FlatAccessInfo& access, uint32_t c) {
    const uint32_t addr = access.bits == 32 ? b.ibin(Op_IAdd, byte0, b.uconst(c * 4)) : byte0;
    const uint32_t idx = b.ibin(Op_ShiftRightLogical, addr, b.uconst(2));
    return {addr, idx};
}
uint32_t lower_flat_window_byte_offset(SpirvCompute& b, uint32_t addr_lo, uint32_t base_lo) {
    const uint32_t byte0 = b.ibin(Op_ISub, addr_lo, base_lo);
    return byte0;
}
}   // namespace prosper::gpu::ir
