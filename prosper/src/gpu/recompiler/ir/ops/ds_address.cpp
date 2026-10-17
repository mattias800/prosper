// Preserve the old append/read order, including the paired read's high-word indices.
#include "gpu/recompiler/ir/ops/ds_address.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include <cstdint>
#include <vector>
namespace prosper::gpu::ir {
DsAddressBlock build_ds_byte_address(uint32_t literal, bool gds) {
    DsAddressBlock out;
    Builder ir(out.block);
    Value addr = ir.iadd(ir.input(Type::U32), ir.c32(literal));
    if (gds) addr = ir.and_(addr, ir.c32(0xffffu));
    out.outputs = {ir.lshr(addr, ir.c32(2))};
    return out;
}
DsAddressBlock build_ds_pair_address(uint32_t offset0, uint32_t offset1) {
    DsAddressBlock out;
    Builder ir(out.block);
    const Value base = ir.lshr(ir.input(Type::U32), ir.c32(2));
    const Value idx0 = ir.iadd(base, ir.c32(offset0));
    const Value idx1 = ir.iadd(base, ir.c32(offset1));
    out.outputs = {idx0, idx1};
    return out;
}
DsAddressBlock build_ds_pair_words() {
    DsAddressBlock out;
    Builder ir(out.block);
    const Value idx0 = ir.input(Type::U32);
    const Value hi0 = ir.iadd(idx0, ir.c32(1));
    const Value idx1 = ir.input(Type::U32);
    const Value hi1 = ir.iadd(idx1, ir.c32(1));
    out.outputs = {idx0, hi0, idx1, hi1};
    return out;
}
DsAddressBlock build_ds_word_index(uint32_t component) {
    DsAddressBlock out;
    Builder ir(out.block);
    const Value index = ir.input(Type::U32);
    out.outputs = {component ? ir.iadd(index, ir.c32(component)) : index};
    return out;
}
std::vector<uint32_t> emit_ds_address_values(SpirvCompute& b, const DsAddressBlock& built,
                                             const DsAddressValueReader& read) {
    const auto ids = emit_spirv_resolving_inputs(b, built.block, read);
    std::vector<uint32_t> outputs;
    outputs.reserve(built.outputs.size());
    for (Value output : built.outputs) outputs.push_back(ids[output.index]);
    return outputs;
}
} // namespace prosper::gpu::ir
