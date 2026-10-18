#include "gpu/recompiler/ir/ops/image_operand_coordinates.hpp"
#include "gpu/recompiler/ir/ops/image_coordinates.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include <cstdint>

namespace prosper::gpu::ir {
ImageOperandCoordinateBlock build_image_operand_coordinate(bool normalize, uint32_t extent) {
    ImageOperandCoordinateBlock out;
    Builder ir(out.block);
    const Value word = ir.input(Type::U32);
    out.coordinate = normalize ? build_normalized_image_coordinate(ir, word, extent) : word;
    return out;
}
Known eval_image_operand_coordinate(bool normalize, uint32_t extent, Known coordinate) {
    const auto built = build_image_operand_coordinate(normalize, extent);
    const Known inputs[] = {coordinate};
    return evaluate(built.block, inputs)[built.coordinate.index];
}
uint32_t emit_image_operand_coordinate(SpirvCompute& b, bool normalize, uint32_t extent,
                                       uint32_t coordinate) {
    const auto built = build_image_operand_coordinate(normalize, extent);
    const uint32_t inputs[] = {coordinate};
    return emit_spirv(b, built.block, inputs)[built.coordinate.index];
}
} // namespace prosper::gpu::ir
