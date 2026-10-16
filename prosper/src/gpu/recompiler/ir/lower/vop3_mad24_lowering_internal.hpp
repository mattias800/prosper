// Preserve the old scope names and avoid inserting VDST before the original assignment.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
class Vop3Mad24Lowering : LoweringContext {
public:
    explicit Vop3Mad24Lowering(const LoweringContext& context)
        : LoweringContext(context), vreg(rs.vreg) {}
    bool lower();

private:
    decltype(RegState::vreg)& vreg;
};
}   // namespace prosper::gpu::ir
