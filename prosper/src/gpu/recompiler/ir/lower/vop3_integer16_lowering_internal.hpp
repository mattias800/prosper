// Preserve scope names and the captured old D without inserting VDST during construction.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
class Vop3Integer16Lowering : LoweringContext {
public:
    Vop3Integer16Lowering(const LoweringContext& context, uint32_t previous)
        : LoweringContext(context), vreg(rs.vreg), old_d(previous) {}
    bool lower();

private:
    decltype(RegState::vreg)& vreg;
    const uint32_t old_d;
};
}   // namespace prosper::gpu::ir
