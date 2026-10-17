// vop3_cndmask.cpp — v_cndmask_b32_e64: SRC2's lane mask selects SRC1 or SRC0. The select is the
// IR definition (ir/ops/vector_cndmask.cpp); the mask read and the source modifiers are emit_alu's,
// unchanged.
#include "gpu/recompiler/ir/lower/vop3_mask_lowering_internal.hpp"
#include "gpu/recompiler/ir/ops/vector_cndmask.hpp"

namespace prosper::gpu::ir {

std::optional<bool> Vop3MaskLowering::route_cndmask() {
    const VectorCndmaskDef* def = find_vector_cndmask(in.fmt, in.opcode);
    if (!def) return std::nullopt;
    const Operand& s2 = in.src[2];
    uint32_t m = 0;   // src2 is an SGPR-pair (or VCC) wave mask
    if (s2.value == 106 || s2.value == 107) {
        // Wave32 e64/SDWA forms may explicitly select either physical VCC word as an
        // independent one-dword mask. Prefer that typed lifetime when present. Only
        // VCC_LO may fall back to the architectural implicit predicate; VCC_HI is a
        // distinct word in Wave32 and must have its own proven mask lifetime.
        auto it = rs.sreg_bool.find(s2.value);
        if (rs.sreg_bool_b32.contains(s2.value) && it != rs.sreg_bool.end())
            m = it->second;
        else if (s2.value == 106)
            m = rs.vcc;
    } else if (s2.kind == OperandKind::SGPR) {
        auto it = rs.sreg_bool.find(s2.value);
        if (it != rs.sreg_bool.end()) {
            m = it->second;
        } else if (b.ngg_one_lane) {
            // The exact Astro NGG projection represents guest lane zero. Some wrappers
            // construct a literal/dynamic B64 lane mask in ordinary scalar DATA registers
            // rather than through a mask-domain instruction (7f5f: s4:s5=0xaaaaaaaa...
            // before pc3870). For lane zero, consuming that pair as a wave mask is exactly
            // its low dword's bit zero. Keep arbitrary vertex shaders fail-closed.
            auto data = rs.sreg.find(s2.value);
            if (data != rs.sreg.end())
                m = b.ucmp(Op_INotEqual, b.ibin(Op_BitwiseAnd, data->second, b.uconst(1)),
                           b.uconst(0));
        }
    }
    // fv (not val): cndmask is float-modifier-capable and compilers emit it with -v/|v|
    // sources (sign-select idioms). Raw val() silently dropped neg/abs — the shader
    // recompiled "successfully" and computed the un-negated value. fv == val when no
    // modifier bits are set.
    if (!m)
        ok = false;
    else
        vreg[in.dst.value] = emit_vector_cndmask(b, *def, m, fv(1), fv(0));
    return true;
}

bool lower_vop3_cndmask(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                        bool allow_wave, const Vop3MaskSource& fv) {
    return Vop3MaskLowering({b, rs, in, ok, allow_wave}, &fv).route_cndmask().has_value();
}

}   // namespace prosper::gpu::ir
