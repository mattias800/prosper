// Callback references are synchronous; the destination is inserted only by the old value stage.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/ir/lower/vop3_ldexp_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
class Vop3LdexpLowering : LoweringContext {
public:
    Vop3LdexpLowering(const LoweringContext& c, const Vop3LdexpSource& read,
                      const Vop3LdexpResult& modify)
        : LoweringContext(c), vreg(rs.vreg), fv(read), fresult(modify) {}
    bool lower();

private:
    decltype(RegState::vreg)& vreg;
    const Vop3LdexpSource& fv;
    const Vop3LdexpResult& fresult;
};
}   // namespace prosper::gpu::ir
