// Match the relocated SMEM expression nesting and ascending constant-select append order.
#include "gpu/recompiler/ir/ops/smem_index.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include <array>
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
SmemIndexBlock build_smem_dword_index(uint32_t immediate) {
    SmemIndexBlock out;
    Builder ir(out.block);
    const Value soffset = ir.input(Type::U32);
    out.value = ir.lshr(ir.iadd(soffset, ir.c32(immediate)), ir.c32(2));
    return out;
}
SmemIndexBlock build_smem_component_index(uint32_t component) {
    SmemIndexBlock out;
    Builder ir(out.block);
    const Value index = ir.input(Type::U32);
    out.value = component ? ir.iadd(index, ir.c32(component)) : index;
    return out;
}
SmemIndexBlock build_smem_constant_word(std::span<const uint32_t> words) {
    SmemIndexBlock out;
    Builder ir(out.block);
    const Value index = ir.input(Type::U32);
    Value value = ir.c32(0); // The caller's bounded PC-relative V# word, including OOB zero.
    for (uint32_t t = 0; t < words.size(); ++t)
        value = ir.select(ir.ieq(index, ir.c32(t)), ir.c32(words[t]), value);
    out.value = value;
    return out;
}
SmemIndexBlock build_smem_table_selector(uint32_t stride) {
    SmemIndexBlock out;
    Builder ir(out.block);
    out.value = ir.udiv(ir.input(Type::U32), ir.c32(stride));
    return out;
}
uint32_t emit_smem_index_value(SpirvCompute& b, const SmemIndexBlock& built, uint32_t input) {
    return emit_spirv(b, built.block, std::array{input})[built.value.index];
}
} // namespace prosper::gpu::ir
