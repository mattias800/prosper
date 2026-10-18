// AMD70648 MIMG data elements: raw words retain their bits; D16 consumes packed halves.
// The resolved forward/inverse DST_SEL map is a value operand, never recomputed here.
#include "gpu/recompiler/ir/ops/image_store_values.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include <cstdlib>
namespace prosper::gpu::ir {
ImageStorageValue build_image_storage_word(ImageStorageWordKind kind, uint32_t half) {
    ImageStorageValue out;
    Builder ir(out.block);
    out.sources.push_back(0);
    const Value word = ir.input(Type::U32);
    if (kind == ImageStorageWordKind::Raw) {
        out.value = word;
    } else {
        if (half > 1) std::abort();
        if (kind == ImageStorageWordKind::FloatHalf) {
            out.value = ir.unpack_half(word, half);
        } else {
            // Keep the old shift then mask, including its constant allocation order.
            const Value shifted = ir.lshr(word, ir.c32(half * 16u));
            out.value = ir.and_(shifted, ir.c32(0xffffu));
        }
    }
    return out;
}
ImageStorageValue build_image_storage_select(int source, uint32_t constant) {
    ImageStorageValue out;
    Builder ir(out.block);
    if (source >= 0) {
        out.sources.push_back(static_cast<uint32_t>(source));
        out.value = ir.input(Type::U32);
    } else {
        out.value = ir.c32(constant);
    }
    return out;
}
uint32_t emit_image_storage_value(SpirvCompute& b, const ImageStorageValue& value,
                                  std::span<const uint32_t> words) {
    const auto ids = emit_spirv_resolving_inputs(
        b, value.block, [&](uint32_t slot) { return words[value.sources[slot]]; });
    return ids[value.value.index];
}
} // namespace prosper::gpu::ir
