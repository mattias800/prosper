// Field truncation/encoding, field placement, and store indices shared by eval and SPIR-V.
#include "gpu/recompiler/ir/ops/buffer_store.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include <cstdint>
#include <functional>
#include <vector>
namespace prosper::gpu::ir {
namespace {
Value input(BufferStoreBlock& out, Builder& ir, uint32_t what) {
    out.inputs.push_back(what);
    return ir.input(Type::U32);
}
Value field(Builder& ir, BufferStoreKind kind, uint32_t width, const std::function<Value()>& read) {
    // Keep integer's nested read/constant evaluation order; normalized and half helpers read first.
    if (kind == BufferStoreKind::Integer) return ir.and_(read(), ir.c32((1u << width) - 1));
    if (kind == BufferStoreKind::Half) return ir.pack_half_lo(read());
    if (kind == BufferStoreKind::Ufloat) return ir.pack_unsigned_float(read(), width - 5);
    return ir.pack_normalized_field(read(), width, kind == BufferStoreKind::Snorm);
}
}   // namespace
BufferStoreBlock build_buffer_store_fields(uint32_t component, uint32_t bytes,
                                           uint32_t field_mask) {
    BufferStoreBlock out;
    Builder ir(out.block);
    const Value addr = input(out, ir, 4);
    const Value caddr = component ? ir.iadd(addr, ir.c32(component * bytes)) : addr;
    const Value cidx = ir.lshr(caddr, ir.c32(2));
    const Value bitpos = ir.shl(ir.and_(caddr, ir.c32(3)), ir.c32(3));
    const Value mask = ir.shl(ir.c32(field_mask), bitpos);
    const Value value = ir.shl(ir.and_(input(out, ir, component), ir.c32(field_mask)), bitpos);
    out.outputs = {cidx, mask, value};
    return out;
}
BufferStoreBlock build_buffer_store_index(uint32_t component) {
    BufferStoreBlock out;
    Builder ir(out.block);
    const Value index = input(out, ir, 4);
    out.outputs = {component ? ir.iadd(index, ir.c32(component)) : index};
    return out;
}
BufferStoreBlock build_buffer_store_word(const BufferStoreShape& shape) {
    BufferStoreBlock out;
    Builder ir(out.block);
    Value acc = ir.c32(0);
    const bool mini = shape.kind == BufferStoreKind::Ufloat;
    for (uint32_t k = 0; k < shape.count && k < shape.physical_count; ++k) {
        const uint32_t offset = mini ? (k == 0   ? 0
                                        : k == 1 ? 11
                                                 : 22)
                                     : (k == 0   ? 0
                                        : k == 1 ? 10
                                        : k == 2 ? 20
                                                 : 30);
        const uint32_t width = mini ? (k < 2 ? 11 : 10) : (k < 3 ? 10 : 2);
        Value value = field(ir, shape.kind, width, [&] { return input(out, ir, k); });
        if (offset) value = ir.shl(value, ir.c32(offset));
        acc = ir.or_(acc, value);
    }
    out.outputs = {acc};
    return out;
}
BufferStoreBlock build_buffer_store_dword(const BufferStoreShape& shape, uint32_t dword) {
    BufferStoreBlock out;
    Builder ir(out.block);
    Value acc = ir.c32(0);
    for (uint32_t k = 0; k < shape.count; ++k) {
        const uint32_t byte = k * shape.bytes;
        if (byte / 4 != dword) continue;
        Value value = field(ir, shape.kind, shape.bytes * 8, [&] { return input(out, ir, k); });
        const uint32_t shift = (byte % 4) * 8;
        if (shift) value = ir.shl(value, ir.c32(shift));
        acc = ir.or_(acc, value);
    }
    const Value index = dword ? ir.iadd(input(out, ir, 4), ir.c32(dword)) : input(out, ir, 4);
    out.outputs = {index, acc};
    return out;
}
std::vector<uint32_t> emit_buffer_store_values(SpirvCompute& b, const BufferStoreBlock& built,
                                               const BufferStoreValueReader& read) {
    const auto ids = emit_spirv_resolving_inputs(
        b, built.block, [&](uint32_t slot) { return read(built.inputs[slot]); });
    std::vector<uint32_t> outputs;
    outputs.reserve(built.outputs.size());
    for (Value value : built.outputs) outputs.push_back(ids[value.index]);
    return outputs;
}
}   // namespace prosper::gpu::ir
