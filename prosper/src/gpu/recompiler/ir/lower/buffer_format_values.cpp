// Pure values extracted from the MUBUF/MTBUF loaded-word paths. Decisions and accesses stay
// at their original caller sites; these wrappers bind already-read words to the shared IR.
#include "gpu/recompiler/ir/lower/buffer_format_values.hpp"

#include "gpu/recompiler/ir/ops/buffer_format.hpp"
#include <cstdint>

namespace prosper::gpu::ir {

uint32_t lower_buffer_static_field(SpirvCompute& b, uint32_t dw, uint32_t boff, uint32_t comp_bytes,
                                   bool is_half, bool is_uint, bool is_sint, bool is_snorm,
                                   float norm) {
    const auto kind = is_half    ? BufferFieldKind::Half
                      : is_uint  ? BufferFieldKind::Unsigned
                      : is_sint  ? BufferFieldKind::Signed
                      : is_snorm ? BufferFieldKind::Snorm
                                 : BufferFieldKind::Unorm;
    return emit_buffer_field(b, dw, {kind, boff, comp_bytes * 8, norm});
}

BufferStaticLocation lower_buffer_static_location(SpirvCompute& b, uint32_t idx, uint32_t sk,
                                                  uint32_t comp_bytes) {
    const uint32_t byte_off = sk * comp_bytes;
    return {emit_buffer_index_offset(b, idx, byte_off / 4), (byte_off % 4) * 8};
}

uint32_t lower_buffer_dynamic_field(SpirvCompute& b, uint32_t joined, uint32_t comp_bytes,
                                    bool dyn_norm, bool is_snorm, float norm, bool is_sint) {
    const auto kind = dyn_norm  ? (is_snorm ? BufferFieldKind::Snorm : BufferFieldKind::Unorm)
                      : is_sint ? BufferFieldKind::Signed
                                : BufferFieldKind::Unsigned;
    return emit_buffer_field(b, joined, {kind, 0, comp_bytes * 8, norm});
}

uint32_t lower_buffer_dynamic_join(SpirvCompute& b, uint32_t joined, uint32_t dw1, uint32_t shift) {
    return emit_buffer_high_join(b, joined, dw1, shift);
}

BufferComponentLocation lower_buffer_dynamic_location(SpirvCompute& b, uint32_t addr, uint32_t sk,
                                                      uint32_t comp_bytes) {
    const auto ids = emit_buffer_component_location(b, addr, sk, comp_bytes);
    return {ids[0], ids[1]};
}

uint32_t lower_buffer_half_field(SpirvCompute& b, uint32_t joined) {
    return emit_buffer_field(b, joined, {BufferFieldKind::Half, 0, 16});
}

uint32_t lower_buffer_half_join(SpirvCompute& b, uint32_t joined, uint32_t dw1, uint32_t shift) {
    return emit_buffer_high_join(b, joined, dw1, shift);
}

BufferComponentLocation lower_buffer_half_location(SpirvCompute& b, uint32_t addr, uint32_t sk) {
    const auto ids = emit_buffer_component_location(b, addr, sk, 2);
    return {ids[0], ids[1]};
}

uint32_t lower_buffer_packed_field(SpirvCompute& b, uint32_t dw, uint32_t sk, bool packed_10_11_11,
                                   bool is_uint, bool is_sint, bool is_snorm) {
    const uint32_t offset = packed_10_11_11 ? (sk == 0   ? 0u
                                               : sk == 1 ? 11u
                                                         : 22u)
                                            : (sk == 0   ? 0u
                                               : sk == 1 ? 10u
                                               : sk == 2 ? 20u
                                                         : 30u);
    const uint32_t bits = packed_10_11_11 ? (sk < 2 ? 11u : 10u) : (sk < 3 ? 10u : 2u);
    const auto kind = packed_10_11_11 ? BufferFieldKind::Ufloat
                      : is_uint       ? BufferFieldKind::Unsigned
                      : is_sint       ? BufferFieldKind::Signed
                      : is_snorm      ? BufferFieldKind::Snorm
                                      : BufferFieldKind::Unorm;
    const float norm = is_snorm ? (bits == 2 ? 1.0f : 511.0f) : (bits == 2 ? 3.0f : 1023.0f);
    return emit_buffer_field(b, dw, {kind, offset, bits, norm});
}

uint32_t lower_buffer_raw_field(SpirvCompute& b, uint32_t joined, uint32_t raw_bits,
                                bool raw_signed) {
    return emit_buffer_field(
        b, joined, {raw_signed ? BufferFieldKind::Signed : BufferFieldKind::Unsigned, 0, raw_bits});
}

uint32_t lower_buffer_raw_join(SpirvCompute& b, uint32_t joined, uint32_t dw1, uint32_t shift) {
    return emit_buffer_high_join(b, joined, dw1, shift);
}

BufferShiftedWord lower_buffer_raw_shift(SpirvCompute& b, uint32_t addr, uint32_t dw0) {
    const auto ids = emit_buffer_raw_shift(b, addr, dw0);
    return {ids[0], ids[1]};
}

uint32_t lower_buffer_low_shift(SpirvCompute& b, uint32_t word, uint32_t shift) {
    return emit_buffer_low_shift(b, word, shift);
}

uint32_t lower_buffer_next_dword(SpirvCompute& b, uint32_t index) {
    return emit_buffer_index_offset(b, index, 1);
}

} // namespace prosper::gpu::ir
