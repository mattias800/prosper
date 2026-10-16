// Borrow existing float source reads without inserting a destination during construction.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/ir/lower/vop3_pack16_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
class Vop3RtzPackLowering : LoweringContext {
public:
    Vop3RtzPackLowering(const LoweringContext& c, const RtzPackSource& read)
        : LoweringContext(c), vreg(rs.vreg), fv(read) {}
    bool lower();

private:
    decltype(RegState::vreg)& vreg;
    const RtzPackSource& fv;
};
}   // namespace prosper::gpu::ir
