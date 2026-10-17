// Bind resolved IDs only; source reads, window resolution, accesses and predicates stay in emit_alu.
#include "gpu/recompiler/ir/lower/flat_window_lowering.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/ops/flat_window.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <array>
#include <cstdint>
namespace prosper::gpu::ir {
std::array<uint32_t, 2> lower_flat_window_component(SpirvCompute& b, uint32_t byte0,
                                                    const FlatAccessInfo& access, uint32_t c) {
    const auto built = build_flat_window_component(access.bits, c);
    const auto values = emit_spirv(b, built.block, std::array{byte0});
    return {values[built.byte_address.index], values[built.dword_index.index]};
}
uint32_t lower_flat_window_byte_offset(SpirvCompute& b, uint32_t addr_lo, uint32_t base_lo) {
    const auto built = build_flat_window_byte_offset();
    return emit_spirv(b, built.block, std::array{addr_lo, base_lo})[built.byte_offset.index];
}
}   // namespace prosper::gpu::ir
