// Existing late VOP3 integer leaves. Admission and destination stages remain in emit_alu.
#include "gpu/recompiler/ir/lower/vop3_ternary_integer_lowering_internal.hpp"
#include "gpu/recompiler/ir/lower/vop3_ternary_integer_lowering.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include <cstdint>
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
bool Vop3TernaryIntegerLowering::lower() {
    if ((in.opcode == kVop3OpcodeAdd3U32 || in.opcode == 0x346 ||
         in.opcode == kVop3OpcodeAndOrB32 || in.opcode == 0x372 || in.opcode == 0x178 ||
         in.opcode == 0x36F || in.opcode == 0x345 || in.opcode == 0x347) &&
        (in.src_abs[0] || in.src_abs[1] || in.src_abs[2] || in.src_neg[0] || in.src_neg[1] ||
         in.src_neg[2] || in.omod || ((in.words[0] >> 11) & 0xFu) != 0)) {
        // The integer three-operand family below has no defined ABS/NEG/OMOD/OP_SEL
        // forms (ISA 70648 12.12; llvm-mc prints none), and these lowerings never read
        // them. Refuse an encoding that sets one rather than silently computing the
        // unmodified result, as v_bfm_b32 and v_lshrrev_b64 above already do. CLAMP is
        // gated after the chain. No program in the local corpus sets one (#5143).
        ok = false;
        return true;
    }
    const auto* def = find_vector_ternary_integer_op(in.fmt, in.opcode);
    if (!def) return false;
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
