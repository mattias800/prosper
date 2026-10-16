// Preserve both original read stages; SAD arithmetic lives only in the operation definitions.
#include "gpu/recompiler/ir/lower/vop3_sad_lowering_internal.hpp"
#include "gpu/recompiler/ir/lower/vop3_sad_lowering.hpp"
#include "gpu/recompiler/ir/ops/vector_sad.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
namespace {
uint32_t emit_v_sad_subword(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok) {
    const uint32_t s0 = operand_bits(b, rs, in, in.src[0], &ok);
    const uint32_t s1 = operand_bits(b, rs, in, in.src[1], &ok);
    const uint32_t s2 = operand_bits(b, rs, in, in.src[2], &ok);
    if (!ok) return b.uconst(0);

    const auto* def = find_vector_sad_op(in.fmt, in.opcode);
    if (!def) {
        ok = false;
        return b.uconst(0);
    }
    const uint32_t sources[] = {s0, s1, s2};
    return emit_ternary_integer_op(b, *def, [&](uint32_t source) { return sources[source]; });
}
} // namespace
bool Vop3SadLowering::lower() {
    if (in.opcode == 0x15A || in.opcode == 0x15B || in.opcode == 0x15C ||
        in.opcode == 0x171) {   // v_sad_u8 / v_sad_hi_u8 / v_sad_u16 / v_msad_u8
        vreg[in.dst.value] = emit_v_sad_subword(b, rs, in, ok);
    } else if (in.opcode == 0x15D) {   // v_sad_u32 = |s0-s1| (unsigned) + s2
        const auto* def = find_vector_sad_op(in.fmt, in.opcode);
        if (!def) {
            ok = false;
            return true;
        }
        vreg[in.dst.value] = emit_ternary_integer_op(
            b, *def, [this](uint32_t source) { return val(in.src[source]); });
    } else {
        return false;
    }
    return true;
}
bool lower_vop3_sad(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok, bool allow_wave) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3SadLowering({b, rs, in, ok, allow_wave}).lower();
}
}   // namespace prosper::gpu::ir
