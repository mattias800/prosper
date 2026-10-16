// AMD70648 op324 and327: Z wins ties, then Y, then X; MA retains the axis sign.
#include "gpu/recompiler/ir/ops/vector_cube_axis.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
namespace {
struct Prepared {
    Value x, y, z, zmaj, ymaj, zneg, yneg, xneg;
};
Prepared prepare(Builder& ir, const TernaryInputs& in) {
    const Value x = in.s0(), y = in.s1(), z = in.s2();
    const Value ax = ir.f32abs(x), ay = ir.f32abs(y), az = ir.f32abs(z);
    const Value zmaj = ir.booland_(ir.f32ge(az, ax), ir.f32ge(az, ay));
    const Value ymaj = ir.f32ge(ay, ax);
    const Value zneg = ir.f32lt(z, ir.c32(0));
    const Value yneg = ir.f32lt(y, ir.c32(0));
    const Value xneg = ir.f32lt(x, ir.c32(0));
    // Keep even unused prelude instructions: they are part of the existing module byte contract.
    return {x, y, z, zmaj, ymaj, zneg, yneg, xneg};
}
Value cube_id(Builder& ir, const TernaryInputs& in) {
    const auto p = prepare(ir, in);
    const Value idz = ir.select(p.zneg, ir.c32(0x40a00000), ir.c32(0x40800000));
    const Value idy = ir.select(p.yneg, ir.c32(0x40400000), ir.c32(0x40000000));
    const Value idx = ir.select(p.xneg, ir.c32(0x3f800000), ir.c32(0));
    return ir.select(p.zmaj, idz, ir.select(p.ymaj, idy, idx));
}
Value cube_ma(Builder& ir, const TernaryInputs& in) {
    const auto p = prepare(ir, in);
    const Value maj = ir.select(p.zmaj, p.z, ir.select(p.ymaj, p.y, p.x));
    return ir.f32mul(ir.c32(0x40000000), maj);
}
constexpr TernaryIntegerDef kDefs[] = {
    {Rdna2Format::VOP3, 0x144, cube_id},
    {Rdna2Format::VOP3, 0x147, cube_ma},
};
}   // namespace
std::span<const TernaryIntegerDef> vector_cube_axis_defs() {
    return kDefs;
}
const TernaryIntegerDef* find_vector_cube_axis_op(Rdna2Format fmt, uint32_t op) {
    for (const auto& d : kDefs)
        if (d.fmt == fmt && d.opcode == op) return &d;
    return nullptr;
}
}   // namespace prosper::gpu::ir
