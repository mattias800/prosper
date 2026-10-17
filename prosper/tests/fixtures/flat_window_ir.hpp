// The same production value lowerings feed the validator and the per-lane integer test.
#pragma once
#include "gpu/recompiler/ir/lower/flat_window_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <array>
#include <cstdint>
namespace prosper::test::ir::flat_window {
inline void emit_fixture(prosper::gpu::SpirvCompute& b, uint32_t bits = 32,
                         uint32_t component = 3) {
    using namespace prosper::gpu;
    const uint32_t address = b.load_input(0), base = b.load_input(1);
    const uint32_t active = b.ucmp(Op_INotEqual, b.load_input(2), b.uconst(0));
    const uint32_t old = b.load_input(3);
    const uint32_t byte0 = prosper::gpu::ir::lower_flat_window_byte_offset(b, address, base);
    FlatAccessInfo access{};
    access.bits = bits;
    const auto [addr, idx] =
        prosper::gpu::ir::lower_flat_window_component(b, byte0, access, component);
    const std::array outputs{byte0, addr, idx};
    for (uint32_t k = 0; k < outputs.size(); ++k)
        b.store_output_word(b.sel(active, outputs[k], old), 3, k);
}
} // namespace prosper::test::ir::flat_window
