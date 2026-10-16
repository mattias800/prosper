// The old scope names and borrowed callbacks; construction inserts no destination.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/ir/lower/vop3_binary_float_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
class Vop3BinaryFloatLowering : LoweringContext {
public:
    Vop3BinaryFloatLowering(const LoweringContext& c, const BinaryFloatSource& read,
                            const BinaryFloatResult& modify)
        : LoweringContext(c), vreg(rs.vreg), fv(read), fresult(modify) {}
    bool lower();

private:
    decltype(RegState::vreg)& vreg;
    const BinaryFloatSource& fv;
    const BinaryFloatResult& fresult;
};
}   // namespace prosper::gpu::ir
