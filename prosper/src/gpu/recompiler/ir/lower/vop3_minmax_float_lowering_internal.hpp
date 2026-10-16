// Callback references live only for the synchronous dispatch; construction inserts no destination.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/ir/lower/vop3_minmax_float_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
class Vop3MinmaxFloatLowering : LoweringContext {
public:
    Vop3MinmaxFloatLowering(const LoweringContext& c, const Vop3MinmaxFloatSource& read,
                            const Vop3MinmaxFloatResult& modify)
        : LoweringContext(c), vreg(rs.vreg), fv(read), fresult(modify) {}
    bool lower();

private:
    decltype(RegState::vreg)& vreg;
    const Vop3MinmaxFloatSource& fv;
    const Vop3MinmaxFloatResult& fresult;
};
}   // namespace prosper::gpu::ir
