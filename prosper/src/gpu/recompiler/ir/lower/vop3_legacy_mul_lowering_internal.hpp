// Keep source names/order and destination map ownership without an early insertion.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/ir/lower/vop3_legacy_mul_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
class Vop3LegacyMulLowering : LoweringContext {
public:
    Vop3LegacyMulLowering(const LoweringContext& c, const LegacyMulSource& read,
                          const LegacyMulResult& modify)
        : LoweringContext(c), vreg(rs.vreg), fv(read), fresult(modify) {}
    bool lower();

private:
    decltype(RegState::vreg)& vreg;
    const LegacyMulSource& fv;
    const LegacyMulResult& fresult;
};
}   // namespace prosper::gpu::ir
