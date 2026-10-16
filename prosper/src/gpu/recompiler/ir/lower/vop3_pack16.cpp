// Preserve selected-half source handling; shared IR owns only the whole-dword value.
#include "gpu/recompiler/ir/lower/vop3_pack16_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_pack16_lowering_internal.hpp"
#include "gpu/recompiler/ir/ops/vector_pack16.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
uint32_t Vop3Pack16Lowering::packed_half(int k) {
    const Operand& operand = in.src[k];
    // OPSEL[k] selects a half of an inline constant exactly as it does of a
    // register — see inline_16bit_operand_dword. Both inline branches used to
    // ignore it, so `1.0` read 0x3c00 and inline `1` read 1 from the HIGH half
    // where hardware reads 0 (#2119, kernel 32r14d).
    const uint32_t sel = (in.vop3p_opsel >> k) & 1u;
    uint32_t inline_dword;
    if (inline_16bit_operand_dword(operand, inline_dword))
        return b.uconst((inline_dword >> (16u * sel)) & 0xFFFFu);
    return b.bfe_u(val(operand), b.uconst(16u * sel), b.uconst(16));
}
bool Vop3Pack16Lowering::lower() {
    const auto* def = find_vector_pack16_op(in.fmt, in.opcode);
    if (!def) return false;
    if (in.src_abs[0] || in.src_abs[1] || in.src_neg[0] || in.src_neg[1] || in.clamp || in.omod) {
        ok = false;
        return true;
    }
    vreg[in.dst.value] = emit_ternary_integer_op(
        b, *def, [&](uint32_t slot) { return packed_half(static_cast<int>(slot)); });
    return true;
}
bool lower_vop3_pack16(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                       bool allow_wave) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3Pack16Lowering({b, rs, in, ok, allow_wave}).lower();
}
}   // namespace prosper::gpu::ir
