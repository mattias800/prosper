// Pure SMEM index values. The caller owns source proof, bounds, accesses and selector state.
#pragma once
#include "gpu/recompiler/ir/ir.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu {
struct SpirvCompute;
}
namespace prosper::gpu::ir {
struct SmemIndexBlock {
    Block block;
    Value value;
};
SmemIndexBlock build_smem_dword_index(uint32_t immediate);
SmemIndexBlock build_smem_component_index(uint32_t component);
SmemIndexBlock build_smem_constant_word(std::span<const uint32_t> words);
SmemIndexBlock build_smem_table_selector(uint32_t stride);
uint32_t emit_smem_index_value(SpirvCompute& b, const SmemIndexBlock& built, uint32_t input);
} // namespace prosper::gpu::ir
