#include "gpu/recompiler/compiler_choices.hpp"
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include "gpu/recompiler/ir/lower/vop3_integer_float_rte_lowering.hpp"
#include "gpu/recompiler/float32_rte_support.hpp"
#include "gpu/recompiler/ir/lower/launch_float_mode.hpp"
#include "gpu/recompiler/ir/ops/vector_integer_float_rte_e64.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu::ir {
bool lower_vop3_integer_float_rte(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok) {
    const auto* def = find_vector_integer_float_rte_e64_op(in.fmt, in.opcode);
    if (!def) return false;
    const auto mode = launch_float_mode(b);
    // Preserve the previous unmatched boundary until BOTH guest and executing-device
    // rounding authority exist. A CPU evaluator's nominal RTE is not device authority.
    if (!mode.mode_available || (mode.mode & 3u) != 0) return false;
    const auto rte32 = compiler_choice(CompilerChoice::Float32Rte, [] {
        return float32_rte_admitted() ? int64_t{1} : int64_t{0};
    });
    if (rte32 != 1) return false;
    const uint32_t selector = in.words[1] & 0x1ffu;
    if (in.src[0] != decode_src_field(selector)) return false;
    const bool ordinary_sgpr = in.src[0].kind == OperandKind::SGPR;
    if (!ordinary_sgpr && in.src[0].kind != OperandKind::VGPR) return false;
    // operand_bits has a legacy zero fallback for absent register storage. This new
    // admission requires a supplied value instead. Presence is only a necessary condition:
    // the existing resolver below still owns mask/spill/token refusals.
    const int reg = in.src[0].value;
    if (ordinary_sgpr ? !rs.sreg.contains(reg) && !rs.sreg_input.contains(reg)
                      : !rs.vreg.contains(reg))
        return false;
    if (in.has_modifier || in.has_sdwa || in.has_dpp || in.has_literal || in.len_dwords != 2 ||
        (in.words[0] & 0xff00u) || (in.words[1] & ~0x1ffu) || in.clamp || in.omod ||
        in.src_abs[0] || in.src_abs[1] || in.src_abs[2] || in.src_neg[0] || in.src_neg[1] ||
        in.src_neg[2] || in.dst.kind != OperandKind::VGPR || in.n_src != 1) {
        ok = false;
        return true;
    }
    if (!ok) return true;
    const uint32_t source = operand_bits(b, rs, in, in.src[0], &ok);
    if (!ok) return true;
    b.declare_float32_rte();
    rs.vreg[in.dst.value] = emit_conversion_op(b, *def, source);
    return true;
}
} // namespace prosper::gpu::ir
