// These references live only through one synchronous dispatch, with no destination insertion.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/ir/lower/vop3_div_fixup_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
class Vop3DivFixupLowering : LoweringContext {
public:
    Vop3DivFixupLowering(const LoweringContext& c, const DivFixupSource& read,
                         const DivFixupResult& modify)
        : LoweringContext(c), vreg(rs.vreg), fv(read), fresult(modify) {}
    bool lower();

private:
    decltype(RegState::vreg)& vreg;
    const DivFixupSource& fv;
    const DivFixupResult& fresult;
};
}   // namespace prosper::gpu::ir
