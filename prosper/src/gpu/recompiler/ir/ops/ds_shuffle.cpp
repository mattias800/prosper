// Pure shuffle values, appended in the native ALU helpers' original emission order.
#include "gpu/recompiler/ir/ops/ds_shuffle.hpp"

#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <array>
#include <cstdint>
#include <functional>

namespace prosper::gpu::ir {
namespace {
Value gather_swizzle(Builder& ir, Value peer, const DsSwizzleInputs& inputs) {
    const Value shuffled = ir.shuffle(inputs.source(), peer);
    const Value active_word = ir.select(inputs.active(), ir.c32(1), ir.c32(0));
    const Value fetched_active = ir.shuffle(active_word, peer);
    return ir.select(ir.ine(fetched_active, ir.c32(0)), shuffled, ir.c32(0));
}
} // namespace

Value ds_bpermute_b32(Builder& ir, Value address, Value source, Value active, Value byte_offset) {
    const Value lane = ir.lane_id();
    // Addition wraps at 32 bits before byte alignment is removed; only index bits [6:2] select.
    const Value selected = ir.and_(ir.lshr(ir.iadd(address, byte_offset), ir.c32(2)), ir.c32(31));
    const Value peer = ir.or_(ir.and_(lane, ir.c32(~31u)), selected);
    const Value shuffled = ir.shuffle(source, peer);
    const Value source_active = ir.shuffle(ir.select(active, ir.c32(1), ir.c32(0)), peer);
    return ir.select(ir.ine(source_active, ir.c32(0)), shuffled, ir.c32(0));
}

Value ds_swizzle_quad_b32(Builder& ir, uint32_t control, const DsSwizzleInputs& inputs) {
    const Value lane = ir.lane_id();
    const Value quad_lane = ir.and_(lane, ir.c32(3));
    Value selected = ir.c32(control & 3u);
    for (uint32_t output_lane = 1; output_lane < 4; ++output_lane)
        selected = ir.select(ir.ieq(quad_lane, ir.c32(output_lane)),
                             ir.c32((control >> (2u * output_lane)) & 3u), selected);
    const Value peer = ir.or_(ir.and_(lane, ir.c32(~3u)), selected);
    return gather_swizzle(ir, peer, inputs);
}

Value ds_swizzle_mask_b32(Builder& ir, uint32_t control, const DsSwizzleInputs& inputs) {
    const Value lane = ir.lane_id();
    const Value and_mask = ir.c32(control & 0x1fu);
    const Value or_mask = ir.c32((control >> 5) & 0x1fu);
    const Value xor_mask = ir.c32((control >> 10) & 0x1fu);
    const Value lane_in_group = ir.and_(lane, ir.c32(0x1fu));
    const Value selected = ir.xor_(ir.or_(ir.and_(lane_in_group, and_mask), or_mask), xor_mask);
    const Value peer = ir.or_(ir.and_(lane, ir.c32(~0x1fu)), selected);
    return gather_swizzle(ir, peer, inputs);
}

uint32_t emit_ds_bpermute_b32(SpirvCompute& b, uint32_t address, uint32_t source, uint32_t active,
                              uint32_t byte_offset) {
    b.mark_subgroup_min32();
    Block block;
    Builder ir(block);
    const Value a = ir.input(Type::U32), s = ir.input(Type::U32);
    const Value exec = ir.input(Type::Bool), offset = ir.input(Type::U32);
    const Value d = ds_bpermute_b32(ir, a, s, exec, offset);
    const std::array<uint32_t, 4> inputs{address, source, active, byte_offset};
    return emit_spirv(b, block, inputs)[d.index];
}

uint32_t emit_ds_swizzle_b32(SpirvCompute& b, uint32_t control,
                             const std::function<uint32_t()>& source, uint32_t active) {
    // The quad only needs four contiguous host lanes; the bitmask form needs an aligned group32.
    if (!(control & 0x8000u)) b.mark_subgroup_min32();
    Block block;
    Builder ir(block);
    const DsSwizzleInputs inputs{[&] { return ir.input(Type::U32); },
                                 [&] { return ir.input(Type::Bool); }};
    const Value d = control & 0x8000u ? ds_swizzle_quad_b32(ir, control, inputs)
                                      : ds_swizzle_mask_b32(ir, control, inputs);
    return emit_spirv_resolving_inputs(
        b, block, [&](uint32_t slot) { return slot == 0 ? source() : active; })[d.index];
}
}   // namespace prosper::gpu::ir
