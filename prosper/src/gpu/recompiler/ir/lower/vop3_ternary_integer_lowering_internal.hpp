// The original b/rs/in/ok/val/vreg names, with no register lookup in the constructor.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
class Vop3TernaryIntegerLowering : LoweringContext {
public:
    explicit Vop3TernaryIntegerLowering(const LoweringContext& context)
        : LoweringContext(context), vreg(rs.vreg) {}
    bool lower();

private:
    decltype(RegState::vreg)& vreg;
};
}   // namespace prosper::gpu::ir
