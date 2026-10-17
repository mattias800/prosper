// Preserve the relocated subtraction, component-add and truncating shift, in that order.
#include "gpu/recompiler/ir/ops/flat_window.hpp"
#include "gpu/recompiler/ir/builder.hpp"
namespace prosper::gpu::ir {
FlatWindowOffsetBlock build_flat_window_byte_offset() {
    FlatWindowOffsetBlock out;
    Builder ir(out.block);
    const Value address = ir.input(Type::U32);
    const Value base = ir.input(Type::U32);
    out.byte_offset = ir.isub(address, base);
    return out;
}
FlatWindowComponentBlock build_flat_window_component(uint32_t bits, uint32_t component) {
    FlatWindowComponentBlock out;
    Builder ir(out.block);
    const Value byte0 = ir.input(Type::U32);
    // Even component zero emits the old IAdd for a dword access.
    out.byte_address = bits == 32 ? ir.iadd(byte0, ir.c32(component * 4)) : byte0;
    out.dword_index = ir.lshr(out.byte_address, ir.c32(2));
    return out;
}
} // namespace prosper::gpu::ir
