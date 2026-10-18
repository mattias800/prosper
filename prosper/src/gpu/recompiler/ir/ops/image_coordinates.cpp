// image_coordinates.cpp — the MIMG spatial-coordinate definition (image_coordinates.hpp).
#include "gpu/recompiler/ir/ops/image_coordinates.hpp"

#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"

#include <bit>
#include <cstdio>
#include <cstdlib>

namespace prosper::gpu::ir {

namespace {

[[noreturn]] void malformed(const char* what) {
    std::fprintf(stderr, "[ir] image coordinates: %s\n", what);
    std::abort();
}

// FORCE_UNNORMALIZED: coordinate * (1 / extent), the reciprocal rounded to f32 once, as the
// pre-IR emitter allocated it (the constant before the multiply).
Value normalized(Builder& ir, Value coordinate, uint32_t extent) {
    if (!extent) return ir.c32(0);
    return ir.f32mul(coordinate, ir.c32(std::bit_cast<uint32_t>(1.0f / static_cast<float>(extent))));
}

}   // namespace

ImageCoordinateBlock build_image_coordinates(const ImageCoordinateShape& shape) {
    if (shape.axes < 2 || shape.axes > 3) malformed("axes must be 2 or 3");
    if (shape.form == ImageCoordinateForm::Packed16 && shape.axes != 2)
        malformed("A16 packed coordinates are 2D only");
    ImageCoordinateBlock out;
    Builder ir(out.block);
    const auto read = [&](uint32_t slot) {
        out.inputs.push_back(slot);
        return ir.input(Type::U32);
    };
    if (shape.form == ImageCoordinateForm::Packed16) {
        const Value packed = read(0);
        out.coord[0] = ir.bfe_u(packed, ir.c32(0), ir.c32(16));
        out.coord[1] = ir.bfe_u(packed, ir.c32(16), ir.c32(16));
    } else {
        for (uint32_t axis = 0; axis < shape.axes; ++axis) out.coord[axis] = read(axis);
    }
    if (shape.normalize)
        for (uint32_t axis = 0; axis < shape.axes; ++axis)
            out.coord[axis] = normalized(ir, out.coord[axis], shape.extent[axis]);
    return out;
}

std::array<Known, 3> eval_image_coordinates(const ImageCoordinateShape& shape,
                                            const std::array<Known, 3>& address) {
    const ImageCoordinateBlock built = build_image_coordinates(shape);
    std::vector<Known> inputs;
    inputs.reserve(built.inputs.size());
    for (uint32_t slot : built.inputs) inputs.push_back(address[slot]);
    const std::vector<Known> values = evaluate(built.block, inputs);
    std::array<Known, 3> out{};
    for (uint32_t axis = 0; axis < shape.axes; ++axis) out[axis] = values[built.coord[axis].index];
    return out;
}

std::array<uint32_t, 3> emit_image_coordinates(SpirvCompute& b, const ImageCoordinateShape& shape,
                                               const ImageAddressSlotReader& read) {
    if (!read) malformed("address reader missing");
    const ImageCoordinateBlock built = build_image_coordinates(shape);
    const std::vector<uint32_t> ids = emit_spirv_resolving_inputs(
        b, built.block, [&](uint32_t slot) { return read(built.inputs[slot]); });
    std::array<uint32_t, 3> out{};
    for (uint32_t axis = 0; axis < shape.axes; ++axis) out[axis] = ids[built.coord[axis].index];
    return out;
}

}   // namespace prosper::gpu::ir
