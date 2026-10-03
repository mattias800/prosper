// Structured SPIR-V control flow shared by the fragment-draw transaction emitters (count/assembly
// in fragment_draw_gpu.cpp, validation/replay in fragment_draw_commit_gpu.cpp).
#pragma once
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu::fragment_draw_flow {

inline uint32_t begin_if(SpirvCompute& b, uint32_t condition) {
    const auto body = b.id(), end = b.id();
    b.emit_selmerge(end);
    b.emit_condbranch(condition, body, end);
    b.emit_label(body);
    return end;
}
inline void end_if(SpirvCompute& b, uint32_t end) {
    b.emit_branch(end);
    b.emit_label(end);
}

// for (index = 0; index < count; ++index) body(index). `count` is evaluated once, before entry.
template <typename Body>
void bounded_loop(SpirvCompute& b, uint32_t count, Body body) {
    uint32_t pointer_type = 0;
    const auto index = b.function_var(b.t_u32, pointer_type);
    b.store_function(index, b.uconst(0));
    const auto header = b.id(), active = b.id(), next = b.id(), end = b.id();
    b.emit_branch(header);
    b.emit_label(header);
    const auto value = b.load_function(b.t_u32, index);
    const auto condition = b.ucmp(Op_ULessThan, value, count);
    b.emit_loopmerge(end, next);
    b.emit_condbranch(condition, active, end);
    b.emit_label(active);
    body(value);
    b.emit_branch(next);
    b.emit_label(next);
    b.store_function(index, b.ibin(Op_IAdd, value, b.uconst(1)));
    b.emit_branch(header);
    b.emit_label(end);
}

struct PixelCoordinate {
    uint32_t valid, pixel;
};
inline PixelCoordinate pixel_center(SpirvCompute& b, uint32_t raw, uint32_t extent) {
    // IEEE binary32 encoding of positive i+0.5, for 0<=i<8192. This is a transport/index
    // check on a host raster observation, not guest floating arithmetic or interpolation.
    // Integer decoding avoids float-to-integer undefined behavior on malformed NaNs/infinities.
    const auto exponent =
        b.ibin(Op_BitwiseAnd, b.ibin(Op_ShiftRightLogical, raw, b.uconst(23)), b.uconst(255));
    auto valid = b.land(b.ucmp(Op_UGreaterThanEqual, exponent, b.uconst(126)),
                        b.ucmp(Op_ULessThanEqual, exponent, b.uconst(139)));
    valid = b.land(
        valid, b.ucmp(Op_IEqual, b.ibin(Op_BitwiseAnd, raw, b.uconst(0x80000000u)), b.uconst(0)));
    const auto safe_exponent = b.sel(valid, exponent, b.uconst(126));
    const auto shift = b.ibin(Op_ISub, b.uconst(150), safe_exponent);   // checked 11..24
    const auto significand =
        b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, raw, b.uconst(0x7fffffu)), b.uconst(0x800000u));
    const auto fraction_mask =
        b.ibin(Op_ISub, b.ibin(Op_ShiftLeftLogical, b.uconst(1), shift), b.uconst(1));
    const auto half = b.ibin(Op_ShiftLeftLogical, b.uconst(1), b.ibin(Op_ISub, shift, b.uconst(1)));
    const auto pixel = b.ibin(Op_ShiftRightLogical, significand, shift);
    valid =
        b.land(valid, b.ucmp(Op_IEqual, b.ibin(Op_BitwiseAnd, significand, fraction_mask), half));
    valid = b.land(valid, b.ucmp(Op_ULessThan, pixel, extent));
    return {valid, pixel};
}

} // namespace prosper::gpu::fragment_draw_flow
