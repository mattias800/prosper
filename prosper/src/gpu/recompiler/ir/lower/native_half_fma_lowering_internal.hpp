// Borrow the original source/result callbacks without inserting a destination.
#pragma once
#include <cstdint>
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/ir/lower/native_half_fma_lowering.hpp"
namespace prosper::gpu::ir {
class NativeHalfFmaLowering : LoweringContext {
public:
    NativeHalfFmaLowering(const LoweringContext& context, uint32_t old,
                          const NativeHalfFmaSource& read, const NativeHalfFmaResult& modify)
        : LoweringContext(context), vreg(rs.vreg), old_d(old), val(read), fresult(modify) {}
    void lower();

private:
    decltype(RegState::vreg)& vreg;
    uint32_t old_d;
    const NativeHalfFmaSource& val;
    const NativeHalfFmaResult& fresult;
};
}   // namespace prosper::gpu::ir
