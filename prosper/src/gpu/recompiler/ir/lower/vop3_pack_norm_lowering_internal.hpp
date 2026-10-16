// Callback lifetime is the synchronous dispatch; reference binding inserts no destination.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/ir/lower/vop3_pack_norm_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
class Vop3PackNormLowering : LoweringContext {
public:
    Vop3PackNormLowering(const LoweringContext& c, const Vop3PackNormSource& read)
        : LoweringContext(c), vreg(rs.vreg), fv(read) {}
    bool lower();

private:
    decltype(RegState::vreg)& vreg;
    const Vop3PackNormSource& fv;
};
}   // namespace prosper::gpu::ir
