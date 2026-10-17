// buffer_address.cpp — the MUBUF/MTBUF address definition (buffer_address.hpp).
#include "gpu/recompiler/ir/ops/buffer_address.hpp"

#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"

#include <cstdio>
#include <cstdlib>

namespace prosper::gpu::ir {

namespace {

Value element_address(Builder& ir, const BufferAddressShape& shape,
                      const std::function<Value(BufferAddressInput)>& read) {
    if (shape.form == BufferAddressForm::FoldedZero) return ir.c32(0);
    if (shape.form == BufferAddressForm::FoldedElement) {
        // The pre-IR emitter read the element in its own statement, before STRIDE's constant.
        const Value element = read(BufferAddressInput::Element);
        return ir.imul(element, ir.c32(shape.stride));
    }
    Value addr = ir.c32(shape.offset);
    if (shape.index)
        addr = ir.iadd(addr, ir.imul(read(BufferAddressInput::Index), ir.c32(shape.stride)));
    if (shape.voffset) addr = ir.iadd(addr, read(BufferAddressInput::Voffset));
    return ir.iadd(addr, read(BufferAddressInput::Soffset));
}

}   // namespace

BufferAddressBlock build_buffer_address(const BufferAddressShape& shape) {
    BufferAddressBlock out;
    Builder ir(out.block);
    const auto read = [&](BufferAddressInput what) {
        out.inputs.push_back(what);
        return ir.input(Type::U32);
    };
    out.addr = element_address(ir, shape, read);
    out.idx = ir.lshr(out.addr, ir.c32(2));
    return out;
}

std::array<Known, 2> eval_buffer_address(const BufferAddressShape& shape,
                                         const std::array<Known, 4>& words) {
    const BufferAddressBlock built = build_buffer_address(shape);
    std::vector<Known> inputs;
    inputs.reserve(built.inputs.size());
    for (BufferAddressInput what : built.inputs)
        inputs.push_back(words[static_cast<size_t>(what)]);
    const std::vector<Known> values = evaluate(built.block, inputs);
    return {values[built.addr.index], values[built.idx.index]};
}

std::array<uint32_t, 2> emit_buffer_address(SpirvCompute& b, const BufferAddressShape& shape,
                                            const BufferAddressReader& read) {
    if (!read) {
        std::fprintf(stderr, "[ir] buffer address reader missing\n");
        std::abort();
    }
    const BufferAddressBlock built = build_buffer_address(shape);
    const std::vector<uint32_t> ids = emit_spirv_resolving_inputs(
        b, built.block, [&](uint32_t slot) { return read(built.inputs[slot]); });
    return {ids[built.addr.index], ids[built.idx.index]};
}

}   // namespace prosper::gpu::ir
