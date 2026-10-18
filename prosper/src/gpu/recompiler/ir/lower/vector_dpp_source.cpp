// Lower only the compute source value; reserved fragment/scratch paths keep their existing helpers.
#include "gpu/recompiler/ir/lower/vector_dpp_source_lowering.hpp"
#include "gpu/recompiler/ir/ops/vector_dpp_source.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <array>

namespace prosper::gpu::ir {
DppRowSourceIds emit_dpp_row_shr_source(SpirvCompute& b, uint32_t source, uint32_t active,
                                        uint32_t shift, bool bound_control) {
    Block block;
    Builder ir(block);
    const Value word = ir.input(Type::U32), exec = ir.input(Type::Bool);
    const auto result = dpp_row_shr_source(ir, word, exec, shift, bound_control);
    const std::array<uint32_t, 2> inputs{source, active};
    const auto ids = emit_spirv(b, block, inputs);
    return {ids[result.word.index], ids[result.in_bounds.index]};
}
uint32_t emit_dpp_quad_source(SpirvCompute& b, uint32_t source, uint32_t active, uint32_t control) {
    Block block;
    Builder ir(block);
    const Value word = ir.input(Type::U32), exec = ir.input(Type::Bool);
    const Value result = dpp_quad_source(ir, word, exec, control);
    const std::array<uint32_t, 2> inputs{source, active};
    return emit_spirv(b, block, inputs)[result.index];
}
}   // namespace prosper::gpu::ir
namespace prosper::gpu {
uint32_t SpirvCompute::subgroup_quad_permute(uint32_t value, uint32_t ctrl, uint32_t active) {
    return ir::emit_dpp_quad_source(*this, value, active, ctrl);
}
}   // namespace prosper::gpu
