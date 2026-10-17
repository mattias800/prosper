// Extractor-relocated value expressions only; no memory accesses or proof state.
#include "gpu/recompiler/ir/lower/smem_index_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
uint32_t lower_smem_dynamic_component(SpirvCompute& b, uint32_t idx0, uint32_t k) {
    uint32_t kidx = k ? b.ibin(Op_IAdd, idx0, b.uconst(k)) : idx0;
    return kidx;
}
uint32_t lower_smem_dynamic_index(SpirvCompute& b, uint32_t soff_bits, const Rdna2Inst& in) {
    uint32_t idx0 =
        b.ibin(Op_ShiftRightLogical, b.ibin(Op_IAdd, soff_bits, b.uconst(in.literal)), b.uconst(2));
    return idx0;
}
uint32_t lower_smem_table_selector(SpirvCompute& b, uint32_t soff_bits, uint32_t stride) {
    const struct {
        uint32_t table_entry_stride;
    } resource{stride};
    const uint32_t value = b.ibin(Op_UDiv, soff_bits, b.uconst(resource.table_entry_stride));
    return value;
}
uint32_t lower_smem_constant_word(SpirvCompute& b, uint32_t kidx, std::span<const uint32_t> words) {
    const struct {
        std::span<const uint32_t> second;
    } table{words};
    const auto* pcrel = &table;
    uint32_t value = b.uconst(0); // bounded V# OOB contract
    for (uint32_t t = 0; t < pcrel->second.size(); ++t)
        value = b.sel(b.ucmp(Op_IEqual, kidx, b.uconst(t)), b.uconst(pcrel->second[t]), value);
    return value;
}
uint32_t lower_smem_pcrel_component(SpirvCompute& b, uint32_t idx, uint32_t k) {
    uint32_t kidx = k ? b.ibin(Op_IAdd, idx, b.uconst(k)) : idx;
    return kidx;
}
uint32_t lower_smem_pcrel_index(SpirvCompute& b, uint32_t soff, const Rdna2Inst& in) {
    uint32_t idx =
        b.ibin(Op_ShiftRightLogical, b.ibin(Op_IAdd, soff, b.uconst(in.literal)), b.uconst(2));
    return idx;
}
}   // namespace prosper::gpu::ir
