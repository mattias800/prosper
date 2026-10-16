// The callback references are consumed within one dispatch call; construction inserts no VDST.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/ir/lower/vop3_mad_float_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
class Vop3MadFloatLowering : LoweringContext {
public:
    Vop3MadFloatLowering(const LoweringContext& context, const Vop3FloatSource& read,
                         const Vop3FloatResult& modify)
        : LoweringContext(context), vreg(rs.vreg), fv(read), fresult(modify) {}
    bool lower();

private:
    decltype(RegState::vreg)& vreg;
    const Vop3FloatSource& fv;
    const Vop3FloatResult& fresult;
};
}   // namespace prosper::gpu::ir
