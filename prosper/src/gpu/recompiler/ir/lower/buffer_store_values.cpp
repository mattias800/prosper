// Verbatim pure RAW segments from the buffer store paths.
#include "gpu/recompiler/ir/lower/buffer_store_values.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
BufferStoreWord lower_buffer_store_packed_dword(SpirvCompute& b, const Rdna2Inst& in, uint32_t idx,
                                                uint32_t d, uint32_t store_n, uint32_t comp_bytes,
                                                bool is_half, bool is_snorm, float norm,
                                                const BufferStoreRead& vread) {
    uint32_t acc = b.uconst(0);
    for (uint32_t k = 0; k < store_n; k++) {
        uint32_t byte_off = k * comp_bytes;
        if (byte_off / 4 != d) continue;
        uint32_t field =
            is_half ? b.pack_half_lo(vread(in.dst.value + (int)k))
                    : b.pack_norm(vread(in.dst.value + (int)k), comp_bytes * 8, is_snorm, norm);
        uint32_t sh = (byte_off % 4) * 8;
        if (sh) field = b.ibin(Op_ShiftLeftLogical, field, b.uconst(sh));
        acc = b.ibin(Op_BitwiseOr, acc, field);
    }
    uint32_t did = d ? b.ibin(Op_IAdd, idx, b.uconst(d)) : idx;
    return {did, acc};
}
uint32_t lower_buffer_store_packed_word(SpirvCompute& b, const Rdna2Inst& in, uint32_t store_n,
                                        uint32_t packed_word_ncomp, bool packed_10_11_11,
                                        bool is_uint, bool is_sint, bool is_snorm,
                                        const BufferStoreRead& vread) {
    uint32_t acc = b.uconst(0);
    for (uint32_t k = 0; k < store_n && k < packed_word_ncomp; k++) {
        const uint32_t boff = packed_10_11_11 ? (k == 0   ? 0u
                                                 : k == 1 ? 11u
                                                          : 22u)
                                              : (k == 0   ? 0u
                                                 : k == 1 ? 10u
                                                 : k == 2 ? 20u
                                                          : 30u);
        const uint32_t bits = packed_10_11_11 ? (k < 2 ? 11u : 10u) : (k < 3 ? 10u : 2u);
        uint32_t field;
        if (packed_10_11_11) {
            // pack_ufloat's argument is MANTISSA bits, not the field width: an 11-bit
            // field is 5 exponent + 6 mantissa, a 10-bit field 5 + 5. The same
            // routine already packs an R11G11B10 texel on the storage-image write
            // path (image_write), so this is the established conversion reached
            // through the buffer descriptor instead of the image one.
            field = b.pack_ufloat(vread(in.dst.value + (int)k), bits - 5u);
        } else if (is_uint || is_sint) {
            // The raw integer field, truncated to its width. Signed values keep
            // two's complement in `bits` bits, which is what bfe_s reads back.
            field =
                b.ibin(Op_BitwiseAnd, vread(in.dst.value + (int)k), b.uconst((1u << bits) - 1u));
        } else {
            const float field_norm =
                is_snorm ? (bits == 2 ? 1.0f : 511.0f) : (bits == 2 ? 3.0f : 1023.0f);
            field = b.pack_norm(vread(in.dst.value + (int)k), bits, is_snorm, field_norm);
        }
        if (boff) field = b.ibin(Op_ShiftLeftLogical, field, b.uconst(boff));
        acc = b.ibin(Op_BitwiseOr, acc, field);
    }
    return acc;
}
uint32_t lower_buffer_store_index(SpirvCompute& b, uint32_t idx, uint32_t k) {
    uint32_t kidx = k ? b.ibin(Op_IAdd, idx, b.uconst(k)) : idx;
    return kidx;
}
BufferStoreFields lower_buffer_store_fields(SpirvCompute& b, const Rdna2Inst& in, uint32_t addr,
                                            uint32_t k, uint32_t comp_bytes, uint32_t field_mask,
                                            const BufferStoreRead& vread) {
    const uint32_t caddr = k ? b.ibin(Op_IAdd, addr, b.uconst(k * comp_bytes)) : addr;
    const uint32_t cidx = b.ibin(Op_ShiftRightLogical, caddr, b.uconst(2));
    const uint32_t bitpos =
        b.ibin(Op_ShiftLeftLogical, b.ibin(Op_BitwiseAnd, caddr, b.uconst(3)), b.uconst(3));
    const uint32_t mask = b.ibin(Op_ShiftLeftLogical, b.uconst(field_mask), bitpos);
    const uint32_t v =
        b.ibin(Op_ShiftLeftLogical,
               b.ibin(Op_BitwiseAnd, vread(in.dst.value + (int)k), b.uconst(field_mask)), bitpos);
    return {cidx, mask, v};
}
}   // namespace prosper::gpu::ir
