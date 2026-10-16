// Borrow the existing per-half reader at the old value-switch position; own no register state.
#pragma once
#include "gpu/recompiler/ir/lower/vop3_binary_float_lowering.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
class Vop3pPackedFloatLowering {
public:
    Vop3pPackedFloatLowering(SpirvCompute& b, const Rdna2Inst& in,
                             const PackedFloatHalfSource& half, bool high, uint32_t& r)
        : b(b), in(in), half(half), high(high), r(r) {}
    bool lower();

private:
    SpirvCompute& b;
    const Rdna2Inst& in;
    const PackedFloatHalfSource& half;
    const bool high;
    uint32_t& r;
};
}   // namespace prosper::gpu::ir
