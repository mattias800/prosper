// Append in the native helpers' order: data shuffle precedes the source-EXEC shuffle.
#include "gpu/recompiler/ir/ops/vector_dpp_source.hpp"

namespace prosper::gpu::ir {
DppRowSource dpp_row_shr_source(Builder& ir, Value source, Value active, uint32_t shift,
                                bool bound_control) {
    const Value lane = ir.lane_id();
    const Value row_lane = ir.and_(lane, ir.c32(15));
    const Value in_bounds = ir.uge(row_lane, ir.c32(shift));
    const Value peer = ir.select(in_bounds, ir.isub(lane, ir.c32(shift)), lane);
    const Value shuffled = ir.shuffle(source, peer);
    const Value source_active = ir.shuffle(ir.select(active, ir.c32(1), ir.c32(0)), peer);
    const Value word = ir.select(ir.ine(source_active, ir.c32(0)), shuffled, ir.c32(0));
    return {bound_control ? ir.select(in_bounds, word, ir.c32(0)) : word, in_bounds};
}

Value dpp_quad_source(Builder& ir, Value source, Value active, uint32_t control) {
    const Value lane = ir.lane_id();
    const Value quad_lane = ir.and_(lane, ir.c32(3));
    Value selected = ir.c32(control & 3u);
    for (uint32_t output_lane = 1; output_lane < 4; ++output_lane)
        selected = ir.select(ir.ieq(quad_lane, ir.c32(output_lane)),
                             ir.c32((control >> (2u * output_lane)) & 3u), selected);
    const Value peer = ir.or_(ir.and_(lane, ir.c32(~3u)), selected);
    const Value shuffled = ir.shuffle(source, peer);
    const Value source_active = ir.shuffle(ir.select(active, ir.c32(1), ir.c32(0)), peer);
    return ir.select(ir.ine(source_active, ir.c32(0)), shuffled, ir.c32(0));
}
} // namespace prosper::gpu::ir
