// Pure SMEM index and embedded-word values; guards, accesses and proof stay with caller.
#pragma once
#include <cstdint>
#include <span>
namespace prosper::gpu {
struct SpirvCompute;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
uint32_t lower_smem_pcrel_index(SpirvCompute& b, uint32_t soff, const Rdna2Inst& in);
uint32_t lower_smem_pcrel_component(SpirvCompute& b, uint32_t idx, uint32_t k);
uint32_t lower_smem_constant_word(SpirvCompute& b, uint32_t kidx, std::span<const uint32_t> words);
uint32_t lower_smem_table_selector(SpirvCompute& b, uint32_t soff_bits, uint32_t stride);
uint32_t lower_smem_dynamic_index(SpirvCompute& b, uint32_t soff_bits, const Rdna2Inst& in);
uint32_t lower_smem_dynamic_component(SpirvCompute& b, uint32_t idx0, uint32_t k);
}   // namespace prosper::gpu::ir
