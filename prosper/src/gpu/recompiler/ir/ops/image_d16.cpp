// AMD 70648 MIMG D16 packing, preserving the two pre-IR packing paths and their call order.
#include "gpu/recompiler/ir/ops/image_d16.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include <array>
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
ImageD16Block build_image_d16_word(ImageD16Form form) {
    ImageD16Block out;
    Builder ir(out.block);
    const Value low = ir.input(Type::U32);
    if (form == ImageD16Form::SelectedLow) {
        out.word = ir.image_pack_half_lo(low);
    } else {
        const Value high = ir.input(Type::U32);
        if (form == ImageD16Form::GatherPair) {
            out.word = ir.image_pack_half2(low, high);
        } else {
            const Value lo_half = ir.image_pack_half_lo(low);
            const Value hi_half = ir.image_pack_half_lo(high);
            out.word = ir.or_(lo_half, ir.shl(hi_half, ir.c32(16)));
        }
    }
    return out;
}
Known eval_image_d16_word(ImageD16Form form, Known low, Known high) {
    const auto built = build_image_d16_word(form);
    const std::array<Known, 2> inputs{low, high};
    return evaluate(built.block, inputs)[built.word.index];
}
uint32_t emit_image_d16_word(SpirvCompute& b, ImageD16Form form, uint32_t low, uint32_t high) {
    const auto built = build_image_d16_word(form);
    const std::array<uint32_t, 2> inputs{low, high};
    return emit_spirv(b, built.block, inputs)[built.word.index];
}
} // namespace prosper::gpu::ir
