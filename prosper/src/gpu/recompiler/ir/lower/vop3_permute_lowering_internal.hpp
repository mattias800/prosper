// Retain original scope names without inserting VDST during construction.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
class Vop3PermuteLowering : LoweringContext {
public:
    explicit Vop3PermuteLowering(const LoweringContext& context)
        : LoweringContext(context), vreg(rs.vreg) {}
    bool lower();

private:
    decltype(RegState::vreg)& vreg;
};
}   // namespace prosper::gpu::ir
