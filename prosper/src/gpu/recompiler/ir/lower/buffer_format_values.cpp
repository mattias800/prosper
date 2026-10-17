// Verbatim pure statement runs extracted from the MUBUF/MTBUF loaded-word paths.
#include "gpu/recompiler/ir/lower/buffer_format_values.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu::ir {

uint32_t lower_buffer_static_field(SpirvCompute& b, uint32_t dw, uint32_t boff, uint32_t comp_bytes,
                                   bool is_half, bool is_uint, bool is_sint, bool is_snorm,
                                   float norm) {
    uint32_t value;
    value = is_half   ? b.unpack_half(dw, boff ? 1u : 0u)
            : is_uint ? b.bfe_u(dw, b.uconst(boff), b.uconst(comp_bytes * 8))
            : is_sint ? b.bfe_s(dw, b.uconst(boff), b.uconst(comp_bytes * 8))
                      : b.unpack_norm(dw, boff, comp_bytes * 8, is_snorm, norm);
    return value;
}

BufferStaticLocation lower_buffer_static_location(SpirvCompute& b, uint32_t idx, uint32_t sk,
                                                  uint32_t comp_bytes) {
    uint32_t byte_off = sk * comp_bytes;
    uint32_t drel = byte_off / 4, boff = (byte_off % 4) * 8;
    uint32_t did = drel ? b.ibin(Op_IAdd, idx, b.uconst(drel)) : idx;
    return {did, boff};
}

uint32_t lower_buffer_dynamic_field(SpirvCompute& b, uint32_t joined, uint32_t comp_bytes,
                                    bool dyn_norm, bool is_snorm, float norm, bool is_sint) {
    uint32_t value;
    value = dyn_norm  ? b.unpack_norm(joined, 0, comp_bytes * 8, is_snorm, norm)
            : is_sint ? b.bfe_s(joined, b.uconst(0), b.uconst(comp_bytes * 8))
                      : b.bfe_u(joined, b.uconst(0), b.uconst(comp_bytes * 8));
    return value;
}

uint32_t lower_buffer_dynamic_join(SpirvCompute& b, uint32_t joined, uint32_t dw1, uint32_t shift) {
    const uint32_t inv_shift =
        b.ibin(Op_BitwiseAnd, b.ibin(Op_ISub, b.uconst(32), shift), b.uconst(31));
    const uint32_t upper = b.ibin(Op_ShiftLeftLogical, dw1, inv_shift);
    joined = b.ibin(Op_BitwiseOr, joined,
                    b.sel(b.ucmp(Op_IEqual, shift, b.uconst(0)), b.uconst(0), upper));
    return joined;
}

BufferComponentLocation lower_buffer_dynamic_location(SpirvCompute& b, uint32_t addr, uint32_t sk,
                                                      uint32_t comp_bytes) {
    const uint32_t caddr = sk ? b.ibin(Op_IAdd, addr, b.uconst(sk * comp_bytes)) : addr;
    const uint32_t cidx = b.ibin(Op_ShiftRightLogical, caddr, b.uconst(2));
    const uint32_t shift =
        b.ibin(Op_ShiftLeftLogical, b.ibin(Op_BitwiseAnd, caddr, b.uconst(3)), b.uconst(3));
    return {cidx, shift};
}

uint32_t lower_buffer_half_field(SpirvCompute& b, uint32_t joined) {
    uint32_t value;
    value = b.unpack_half(joined, 0);
    return value;
}

uint32_t lower_buffer_half_join(SpirvCompute& b, uint32_t joined, uint32_t dw1, uint32_t shift) {
    const uint32_t inv_shift =
        b.ibin(Op_BitwiseAnd, b.ibin(Op_ISub, b.uconst(32), shift), b.uconst(31));
    const uint32_t upper = b.ibin(Op_ShiftLeftLogical, dw1, inv_shift);
    joined = b.ibin(Op_BitwiseOr, joined,
                    b.sel(b.ucmp(Op_IEqual, shift, b.uconst(0)), b.uconst(0), upper));
    return joined;
}

BufferComponentLocation lower_buffer_half_location(SpirvCompute& b, uint32_t addr, uint32_t sk) {
    const uint32_t caddr = sk ? b.ibin(Op_IAdd, addr, b.uconst(sk * 2)) : addr;
    const uint32_t cidx = b.ibin(Op_ShiftRightLogical, caddr, b.uconst(2));
    const uint32_t shift =
        b.ibin(Op_ShiftLeftLogical, b.ibin(Op_BitwiseAnd, caddr, b.uconst(3)), b.uconst(3));
    return {cidx, shift};
}

uint32_t lower_buffer_packed_field(SpirvCompute& b, uint32_t dw, uint32_t sk, bool packed_10_11_11,
                                   bool is_uint, bool is_sint, bool is_snorm) {
    uint32_t value;
    uint32_t boff = packed_10_11_11 ? (sk == 0   ? 0u
                                       : sk == 1 ? 11u
                                                 : 22u)
                                    : (sk == 0   ? 0u
                                       : sk == 1 ? 10u
                                       : sk == 2 ? 20u
                                                 : 30u);
    uint32_t bits = packed_10_11_11 ? (sk < 2 ? 11u : 10u) : (sk < 3 ? 10u : 2u);
    if (packed_10_11_11) {
        value = b.unpack_ufloat(dw, boff, bits);
    } else if (is_uint) {
        value = b.bfe_u(dw, b.uconst(boff), b.uconst(bits));
    } else if (is_sint) {
        value = b.bfe_s(dw, b.uconst(boff), b.uconst(bits));
    } else {
        float field_norm = is_snorm ? (bits == 2 ? 1.0f : 511.0f) : (bits == 2 ? 3.0f : 1023.0f);
        value = b.unpack_norm(dw, boff, bits, is_snorm, field_norm);
    }
    return value;
}

uint32_t lower_buffer_raw_field(SpirvCompute& b, uint32_t joined, uint32_t raw_bits,
                                bool raw_signed) {
    uint32_t value;
    value = raw_signed ? b.bfe_s(joined, b.uconst(0), b.uconst(raw_bits))
                       : b.bfe_u(joined, b.uconst(0), b.uconst(raw_bits));
    return value;
}

uint32_t lower_buffer_raw_join(SpirvCompute& b, uint32_t joined, uint32_t dw1, uint32_t shift) {
    const uint32_t inv_shift =
        b.ibin(Op_BitwiseAnd, b.ibin(Op_ISub, b.uconst(32), shift), b.uconst(31));
    const uint32_t upper = b.ibin(Op_ShiftLeftLogical, dw1, inv_shift);
    joined = b.ibin(Op_BitwiseOr, joined,
                    b.sel(b.ucmp(Op_IEqual, shift, b.uconst(0)), b.uconst(0), upper));
    return joined;
}

BufferShiftedWord lower_buffer_raw_shift(SpirvCompute& b, uint32_t addr, uint32_t dw0) {
    const uint32_t byte_in_dw = b.ibin(Op_BitwiseAnd, addr, b.uconst(3));
    const uint32_t shift = b.ibin(Op_ShiftLeftLogical, byte_in_dw, b.uconst(3));
    uint32_t joined = b.ibin(Op_ShiftRightLogical, dw0, shift);
    return {shift, joined};
}

} // namespace prosper::gpu::ir
