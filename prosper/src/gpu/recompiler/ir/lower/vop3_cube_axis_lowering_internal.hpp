// Borrow root callbacks synchronously; constructing a route does not insert a destination.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/ir/lower/vop3_cube_axis_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
class Vop3CubeAxisLowering : LoweringContext {
public:
    Vop3CubeAxisLowering(const LoweringContext& c, const CubeAxisSource& read,
                         const CubeAxisResult& modify)
        : LoweringContext(c), vreg(rs.vreg), fv(read), fresult(modify) {}
    bool lower();

private:
    decltype(RegState::vreg)& vreg;
    const CubeAxisSource& fv;
    const CubeAxisResult& fresult;
};
}   // namespace prosper::gpu::ir
