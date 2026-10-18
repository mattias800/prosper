// A single spatial operand of a MIMG access. The caller keeps the read in its original
// argument position; the definition only passes through or scales that already-read word.
#pragma once
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include <cstdint>
namespace prosper::gpu {
struct SpirvCompute;
}
namespace prosper::gpu::ir {
struct ImageOperandCoordinateBlock {
    Block block;
    Value coordinate;
};
ImageOperandCoordinateBlock build_image_operand_coordinate(bool normalize, uint32_t extent);
Known eval_image_operand_coordinate(bool normalize, uint32_t extent, Known coordinate);
uint32_t emit_image_operand_coordinate(SpirvCompute& b, bool normalize, uint32_t extent,
                                       uint32_t coordinate);
} // namespace prosper::gpu::ir
