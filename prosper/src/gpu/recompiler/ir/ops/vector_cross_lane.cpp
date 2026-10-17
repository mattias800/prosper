// vector_cross_lane.cpp — cross-lane VALU values (vector_cross_lane.hpp).
#include "gpu/recompiler/ir/ops/vector_cross_lane.hpp"

#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"

#include <array>

namespace prosper::gpu::ir {

Value readlane_b32(Builder& ir, Value source, Value selector, uint32_t wave_size) {
    // Appended in the pre-IR emitter's order: the lane id, the wave mask, then the wave base.
    const Value native_lane = ir.lane_id();
    const Value wave_mask = ir.c32(wave_size - 1u);
    const Value wave_base = ir.and_(native_lane, ir.c32(~(wave_size - 1u)));
    const Value source_lane = ir.or_(wave_base, ir.and_(selector, wave_mask));
    return ir.shuffle(source, source_lane);
}

uint32_t emit_readlane_b32(SpirvCompute& b, uint32_t source, uint32_t selector,
                           uint32_t wave_size) {
    Block block;
    Builder ir(block);
    const Value s0 = ir.input(Type::U32), s1 = ir.input(Type::U32);
    const Value d = readlane_b32(ir, s0, s1, wave_size);
    const std::array<uint32_t, 2> inputs{source, selector};
    return emit_spirv(b, block, inputs)[d.index];
}

} // namespace prosper::gpu::ir
