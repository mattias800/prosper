// Packed float route moved with extract_case_routes.py; arithmetic is unchanged here.
#include "gpu/recompiler/ir/lower/vop3p_packed_values.hpp"
#include "gpu/recompiler/ir/lower/half_fma_finite.hpp"
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/ir/lower/vop3_binary_float_lowering.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <optional>

namespace prosper::gpu::ir {
namespace {
struct Vop3pPackedValuesLowering : LoweringContext {
    std::optional<bool> lower();
};
std::optional<bool> Vop3pPackedValuesLowering::lower() {
    if (in.opcode >= 0x0Eu && in.opcode <= 0x12u) {  // v_pk_fma/add/mul/min/max_f16
        const uint32_t old_d = vreg_old(b, rs, in.dst.value);
        std::array<std::array<uint32_t, 3>, 2> raw_sources{};
        auto half = [&](int source, bool high_result) -> uint32_t {
            uint32_t v = val(in.src[source]);
            raw_sources[high_result][source] = v;
            const uint8_t selectors = high_result ? in.vop3p_opsel_hi : in.vop3p_opsel;
            const uint32_t sel = (selectors >> source) & 1u;
            // An inline constant is the raw dword its width produces (see
            // inline_16bit_operand_dword): a float contributes its f16 encoding in the LOW
            // half and ZERO in the high one — exact even for 1/(2*pi) — and an int its
            // sign-extended two's-complement bits, so inline 1 reads as the f16 denormal
            // 0x0001, NOT 1.0. Either way the half select applies; it is not a no-op. This
            // path used to replicate the float into both halves (#2119, kernel 32r14a).
            uint32_t inline_dword;
            if (inline_16bit_operand_dword(in.src[source], inline_dword)) {
                v = b.unpack_half(b.uconst(inline_dword), sel);
            } else {
                v = b.unpack_half(v, sel);
            }
            const bool negate =
                high_result ? ((in.vop3p_neg_hi >> source) & 1u) != 0 : in.src_neg[source];
            return negate ? b.fneg(v) : v;
        };
        auto operation = [&](bool high) {
            uint32_t r;
            if (!ir::lower_vop3p_packed_float(b, in, std::cref(half), high, r)) switch (in.opcode) {
                    case 0x0E:
                        r = b.fbin(Op_FAdd, b.fbin(Op_FMul, half(0, high), half(1, high)),
                                   half(2, high));
                        break;
                    default: ok = false; return b.uconst(0);
                }
            if (in.clamp) r = b.clamp01(r);
            return r;
        };
        const uint32_t lo = b.pack_half_lo(operation(false));
        const uint32_t hi =
            b.ibin(Op_ShiftLeftLogical, b.pack_half_lo(operation(true)), b.uconst(16));
        uint32_t packed = b.ibin(Op_BitwiseOr, lo, hi);
        if (ok && in.opcode == 0x0e && ir::use_half_fma_finite(b, in))
            packed = ir::emit_packed_half_fma_finite(b, in, raw_sources, packed);
        rs.vreg[in.dst.value] = packed;
        predicate_write(b, rs, in.dst.value, old_d);
        return true;
    }
    return std::nullopt;
}

}   // namespace

std::optional<bool> lower_vop3p_packed_values(SpirvCompute& b, RegState& rs, const Rdna2Inst& in,
                                              bool& ok, bool allow_wave) {
    return Vop3pPackedValuesLowering{{b, rs, in, ok, allow_wave}}.lower();
}
}   // namespace prosper::gpu::ir
