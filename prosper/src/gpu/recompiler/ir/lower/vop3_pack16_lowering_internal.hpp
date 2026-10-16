// Keep the extracted scope names without inserting any destination at construction.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
class Vop3Pack16Lowering : LoweringContext {
public:
    explicit Vop3Pack16Lowering(const LoweringContext& context)
        : LoweringContext(context), vreg(rs.vreg) {}
    bool lower();

private:
    decltype(RegState::vreg)& vreg;
};
}   // namespace prosper::gpu::ir
