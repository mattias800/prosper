// The existing CUBE face clamp/round and vertical-stack transform, defined once for evaluation
// and emission. FMin/FMax intentionally retain the old GLSL instructions (not NMin/NMax).
#include "gpu/recompiler/ir/ops/image_cube_coordinates.hpp"

#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"

#include <bit>
#include <vector>

namespace prosper::gpu::ir {

ImageCubeCoordinateBlock build_image_cube_coordinates(uint32_t first_slot) {
    ImageCubeCoordinateBlock out;
    out.slots = {first_slot, first_slot + 1u, first_slot + 2u};
    Builder ir(out.block);
    const Value x = ir.input(Type::U32), y = ir.input(Type::U32), fid = ir.input(Type::U32);
    const Value one = ir.c32(std::bit_cast<uint32_t>(1.0f)),
                zero = ir.c32(std::bit_cast<uint32_t>(0.0f));
    const Value uf = ir.f32sub(x, one);
    const Value vf = ir.f32max(ir.f32min(ir.f32sub(y, one), one), zero);
    const Value layer =
        ir.f32roundeven(ir.f32max(ir.f32min(fid, ir.c32(std::bit_cast<uint32_t>(5.0f))), zero));
    const Value v6 = ir.f32mul(ir.f32add(layer, vf), ir.c32(std::bit_cast<uint32_t>(1.0f / 6.0f)));
    out.values = {uf, vf, layer, v6};
    return out;
}

std::array<Known, 4> eval_image_cube_coordinates(const std::array<Known, 3>& coordinates) {
    const auto built = build_image_cube_coordinates();
    const auto values = evaluate(built.block, coordinates);
    std::array<Known, 4> out{};
    for (size_t i = 0; i < out.size(); ++i) out[i] = values[built.values[i].index];
    return out;
}

std::array<uint32_t, 4> emit_image_cube_coordinates(SpirvCompute& b, uint32_t first_slot,
                                                    const ImageAddressSlotReader& read) {
    const auto built = build_image_cube_coordinates(first_slot);
    const auto ids = emit_spirv_resolving_inputs(
        b, built.block, [&](uint32_t slot) { return read(built.slots[slot]); });
    std::array<uint32_t, 4> out{};
    for (size_t i = 0; i < out.size(); ++i) out[i] = ids[built.values[i].index];
    return out;
}

} // namespace prosper::gpu::ir
