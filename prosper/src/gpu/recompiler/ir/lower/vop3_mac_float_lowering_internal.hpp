// Borrow the original float callbacks and implicit accumulator; create no destination early.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/ir/lower/vop3_mad_float_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
class Vop3MacFloatLowering : LoweringContext {
public:
    Vop3MacFloatLowering(const LoweringContext& c, const Vop3FloatSource& read,
                         const Vop3FloatResult& modify, uint32_t accumulator)
        : LoweringContext(c), vreg(rs.vreg), fv(read), fresult(modify), old_d(accumulator) {}
    bool lower();

private:
    decltype(RegState::vreg)& vreg;
    const Vop3FloatSource& fv;
    const Vop3FloatResult& fresult;
    const uint32_t old_d;
};
}   // namespace prosper::gpu::ir
