// Keep the old scope names, including a reference bound without a destination lookup.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
class Vop3MinmaxLowering : LoweringContext {
public:
    explicit Vop3MinmaxLowering(const LoweringContext& context)
        : LoweringContext(context), vreg(rs.vreg) {}
    bool lower();

private:
    decltype(RegState::vreg)& vreg;
};
}   // namespace prosper::gpu::ir
