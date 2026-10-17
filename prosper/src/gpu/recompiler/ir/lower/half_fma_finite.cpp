#include <array>
#include <cstdint>
#include "gpu/recompiler/ir/lower/half_fma_finite.hpp"
#include "gpu/recompiler/ir/ops/half_fma_finite.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
bool use_half_fma_finite(const SpirvCompute& b, const Rdna2Inst& in) {
    if (in.omod || in.clamp) return false;
    bool available = false;
    uint8_t mode = 0;
    if (b.is_fragment || b.is_fragment_packet()) {
        available = b.fragment_float_mode.available;
        mode = b.fragment_float_mode.value;
    } else if (b.is_compute && !b.ngg_workgroup_shell && !b.ngg_one_lane) {
        available = b.compute_pgm_rsrc1 != UINT32_MAX;
        mode = uint8_t(b.compute_pgm_rsrc1 >> 12);
    } else {
        available = b.vertex_float_mode.available;
        mode = b.vertex_float_mode.value;
    }
    // AMD70648 MODE: half RTE (bits 3:2=0), preserve half inputs/outputs (bits 7:6=3).
    // Bounded finite nonzero results do not depend on FP16_OVFL. F32 mode is independent.
    return available && (mode & 0xcc) == 0xc0;
}
namespace {
uint32_t half_bits(SpirvCompute& b, uint32_t raw, uint32_t component) {
    if (component) raw = b.ibin(Op_ShiftRightLogical, raw, b.uconst(16));
    return b.ibin(Op_BitwiseAnd, raw, b.uconst(0xffff));
}
uint32_t source_bits(SpirvCompute& b, const Operand& operand, uint32_t raw, uint32_t component,
                     bool absolute, bool negate) {
    uint32_t inline_dword;
    if (inline_16bit_operand_dword(operand, inline_dword)) raw = b.uconst(inline_dword);
    auto bits = half_bits(b, raw, component);
    if (absolute) bits = b.ibin(Op_BitwiseAnd, bits, b.uconst(0x7fff));
    if (negate) bits = b.ibin(Op_BitwiseXor, bits, b.uconst(0x8000));
    return bits;
}
uint32_t select_exact(SpirvCompute& b, const std::array<uint32_t, 3>& raw, uint32_t fallback) {
    const auto out = emit_half_fma_finite(b, raw[0], raw[1], raw[2]);
    return b.sel(out[1], out[0], fallback);
}
uint32_t join(SpirvCompute& b, uint32_t low, uint32_t high) {
    return b.ibin(Op_BitwiseOr, low, b.ibin(Op_ShiftLeftLogical, high, b.uconst(16)));
}
}   // namespace
uint32_t emit_native_half_fma_finite(SpirvCompute& b, const Rdna2Inst& in,
                                     const std::array<uint32_t, 3>& raw, uint32_t fallback) {
    std::array<uint32_t, 3> halves;
    for (uint32_t k = 0; k < 3; ++k)
        halves[k] = source_bits(b, in.src[k], raw[k], (in.vop3p_opsel >> k) & 1, in.src_abs[k],
                                in.src_neg[k]);
    return select_exact(b, halves, fallback);
}
uint32_t emit_packed_half_fma_finite(SpirvCompute& b, const Rdna2Inst& in,
                                     const std::array<std::array<uint32_t, 3>, 2>& raw,
                                     uint32_t fallback) {
    std::array<uint32_t, 2> halves;
    for (uint32_t high = 0; high < 2; ++high) {
        std::array<uint32_t, 3> sources;
        const auto selectors = high ? in.vop3p_opsel_hi : in.vop3p_opsel;
        for (uint32_t k = 0; k < 3; ++k)
            sources[k] = source_bits(b, in.src[k], raw[high][k], (selectors >> k) & 1, false,
                                     high ? ((in.vop3p_neg_hi >> k) & 1) != 0 : in.src_neg[k]);
        halves[high] = select_exact(b, sources, half_bits(b, fallback, high));
    }
    return join(b, halves[0], halves[1]);
}
uint32_t emit_packed_fmac_half_finite(SpirvCompute& b, uint32_t a, uint32_t c, uint32_t old_d,
                                      uint32_t fallback) {
    std::array<uint32_t, 2> halves;
    for (uint32_t high = 0; high < 2; ++high)
        halves[high] = select_exact(
            b, {half_bits(b, a, high), half_bits(b, c, high), half_bits(b, old_d, high)},
            half_bits(b, fallback, high));
    return join(b, halves[0], halves[1]);
}
}   // namespace prosper::gpu::ir
