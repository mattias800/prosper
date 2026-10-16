// Construction captures the previous destination without inserting a new register.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/ir/lower/vop3_minmax_half_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
class Vop3MinmaxHalfLowering : LoweringContext {
public:
    Vop3MinmaxHalfLowering(const LoweringContext& c, uint32_t old, const MinmaxHalfResult& modify)
        : LoweringContext(c), vreg(rs.vreg), old_d(old), fresult(modify) {}
    bool lower();

private:
    decltype(RegState::vreg)& vreg;
    const uint32_t old_d;
    const MinmaxHalfResult& fresult;
};
}   // namespace prosper::gpu::ir
