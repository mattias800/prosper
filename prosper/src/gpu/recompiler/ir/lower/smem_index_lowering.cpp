// Bind the already-resolved value. All guest source reads, accesses and proof remain at the call.
#include "gpu/recompiler/ir/lower/smem_index_lowering.hpp"
#include "gpu/recompiler/ir/ops/smem_index.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
uint32_t lower_smem_dynamic_component(SpirvCompute& b, uint32_t idx0, uint32_t k) {
    return emit_smem_index_value(b, build_smem_component_index(k), idx0);
}
uint32_t lower_smem_dynamic_index(SpirvCompute& b, uint32_t soff_bits, const Rdna2Inst& in) {
    return emit_smem_index_value(b, build_smem_dword_index(in.literal), soff_bits);
}
uint32_t lower_smem_table_selector(SpirvCompute& b, uint32_t soff_bits, uint32_t stride) {
    return emit_smem_index_value(b, build_smem_table_selector(stride), soff_bits);
}
uint32_t lower_smem_constant_word(SpirvCompute& b, uint32_t kidx, std::span<const uint32_t> words) {
    return emit_smem_index_value(b, build_smem_constant_word(words), kidx);
}
uint32_t lower_smem_pcrel_component(SpirvCompute& b, uint32_t idx, uint32_t k) {
    return emit_smem_index_value(b, build_smem_component_index(k), idx);
}
uint32_t lower_smem_pcrel_index(SpirvCompute& b, uint32_t soff, const Rdna2Inst& in) {
    return emit_smem_index_value(b, build_smem_dword_index(in.literal), soff);
}
}   // namespace prosper::gpu::ir
