// Existing late VOP3 integer leaves. Admission and destination stages remain in emit_alu.
#include "gpu/recompiler/ir/lower/vop3_ternary_integer_lowering_internal.hpp"
#include "gpu/recompiler/ir/lower/vop3_ternary_integer_lowering.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include <cstdint>
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
bool Vop3TernaryIntegerLowering::lower() {
    const auto* def = find_vector_ternary_integer_op(in.fmt, in.opcode);
    if (!def) return false;
    if (in.src_abs[0] || in.src_abs[1] || in.src_abs[2] || in.src_neg[0] || in.src_neg[1] ||
        in.src_neg[2] || in.omod || ((in.words[0] >> 11) & 0xFu) != 0) {
        // This integer three-operand family has no defined ABS/NEG/OMOD/OP_SEL
        // forms (ISA 70648 12.12; llvm-mc prints none), and these lowerings never read
        // them. Refuse an encoding that sets one rather than silently computing the
        // unmodified result. The caller retains the final CLAMP boundary, which is
        // gated after the chain. No program in the local corpus sets one (#5143).
        ok = false;
        return true;
    }
    vreg[in.dst.value] =
        emit_ternary_integer_op(b, *def, [this](uint32_t source) { return val(in.src[source]); });
    return true;
}

bool lower_vop3_ternary_integer(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                                bool allow_wave) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3TernaryIntegerLowering({b, rs, in, ok, allow_wave}).lower();
}
}   // namespace prosper::gpu::ir
