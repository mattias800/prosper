// A strict-validation fixture shared with per-lane execution: all four SMEM definitions.
#pragma once
#include "gpu/recompiler/ir/ops/smem_index.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <array>
#include <cstdint>
namespace prosper::test::ir::smem_index {
inline constexpr std::array<uint32_t, 5> words{0x10203040u, 0x99887766u, 0u, 0xabcdef01u,
                                               0x76543210u};
inline void emit_fixture(prosper::gpu::SpirvCompute& b) {
    using namespace prosper::gpu::ir;
    const uint32_t input = b.load_input(0);
    const std::array<uint32_t, 4> values{
        emit_smem_index_value(b, build_smem_dword_index(0xfffffffcu), input),
        emit_smem_index_value(b, build_smem_component_index(3), input),
        emit_smem_index_value(b, build_smem_constant_word(words), input),
        emit_smem_index_value(b, build_smem_table_selector(120), input)};
    for (uint32_t k = 0; k < values.size(); ++k) b.store_output_word(values[k], 4, k);
}
} // namespace prosper::test::ir::smem_index
