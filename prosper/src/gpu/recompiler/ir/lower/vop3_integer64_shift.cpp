// Relocated from emit_alu with extract_case_routes.py. Numeric semantics are unchanged here.
#include "gpu/recompiler/ir/lower/vop3_integer64_shift_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_integer64_shift_internal.hpp"
#include "gpu/recompiler/ir/ops/integer64_shifts.hpp"
namespace prosper::gpu::ir {
bool lower_vop3_integer64_shift(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok) {
    Vop3Integer64ShiftLowering lower{b, rs, in, ok, rs.vreg};
    if (in.opcode == kVop3OpcodeLshlrevB64) {
        lower.lshl();
        return true;
    }
    if (in.opcode == kVop3OpcodeLshrrevB64) {
        lower.lshr();
        return true;
    }
    if (in.opcode == 0x301) {
        // New arithmetic-right admission: unused SRC2 is canonical zero; the pair/source,
        // destination and integer modifier guards below remain the existing LSHR boundary.
        if (in.has_modifier || in.has_dpp || in.has_sdwa || (in.words[1] & (0x1ffu << 18))) {
            ok = false;
            return true;
        }
        lower.lshr();
        return true;
    }
    return false;
}
void Vop3Integer64ShiftLowering::lshl() {
    // GTA V constructs a per-lane bit as `1ull << lane` immediately after MBCNT. This
    // bounded admission keeps SRC1 at the exact inline integer 1; general register-pair
    // and floating-inline sources need wider source-span/provenance handling.
    const uint32_t opsel = (in.words[0] >> 11) & 0xFu;
    const bool exact_one = in.src[1].kind == OperandKind::InlineInt && in.src[1].value == 1;
    if (in.dst.value < 0 || in.dst.value >= 255 || !exact_one || in.src_abs[0] || in.src_abs[1] ||
        in.src_abs[2] || in.src_neg[0] || in.src_neg[1] || in.src_neg[2] || in.clamp || in.omod ||
        opsel) {
        ok = false;
    } else {
        const uint32_t amount = emit_integer64_shift_count(b, [&] { return val(in.src[0]); });
        const auto& def = *find_integer64_shift(in.fmt, in.opcode);
        const uint32_t result = emit_integer64_shift_result(b, def, [&](uint32_t source) {
            return source == 2 ? amount : b.uconst(source == 0 ? 1 : 0);
        });
        const int hi_dst = in.dst.value + 1;
        const uint32_t old_hi = vreg_old(b, rs, hi_dst);
        const auto words = emit_integer64_shift_words(b, result);
        vreg[in.dst.value] = words[0];
        vreg[hi_dst] = words[1];
        predicate_write(b, rs, hi_dst, old_hi);
    }
}
void Vop3Integer64ShiftLowering::lshr() {
    // GTA V's compact-BVH kernels shift a VGPR pair by either a scalar scratch word or
    // a VGPR lane value. RDNA masks the count to six bits. Keep the pair and modifier
    // admission exact: SRC1 must be an addressable consecutive register pair, and this
    // integer operation has no float modifiers or output modifier spelling.
    const uint32_t opsel = (in.words[0] >> 11) & 0xFu;
    const bool source_pair =
        in.src[1].kind == OperandKind::VGPR && in.src[1].value >= 0 && in.src[1].value < 255;
    if (in.dst.value < 0 || in.dst.value >= 255 || !source_pair || in.src_abs[0] || in.src_abs[1] ||
        in.src_abs[2] || in.src_neg[0] || in.src_neg[1] || in.src_neg[2] || in.clamp || in.omod ||
        opsel) {
        ok = false;
    } else {
        const uint32_t amount = emit_integer64_shift_count(b, [&] { return val(in.src[0]); });
        Operand high_source = in.src[1];
        ++high_source.value;
        const auto& def = *find_integer64_shift(in.fmt, in.opcode);
        const uint32_t result = emit_integer64_shift_result(b, def, [&](uint32_t source) {
            return source == 2 ? amount : val(source == 0 ? in.src[1] : high_source);
        });
        const int hi_dst = in.dst.value + 1;
        const uint32_t old_hi = vreg_old(b, rs, hi_dst);
        const auto words = emit_integer64_shift_words(b, result);
        vreg[in.dst.value] = words[0];
        vreg[hi_dst] = words[1];
        predicate_write(b, rs, hi_dst, old_hi);
    }
}
} // namespace prosper::gpu::ir
