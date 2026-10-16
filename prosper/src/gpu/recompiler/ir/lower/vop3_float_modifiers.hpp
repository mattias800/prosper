// Numeric VOP3 modifiers, mechanically moved from the root. No register or proof state is owned.
#pragma once
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/ir/lower/float_output_modifiers.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
template <class Read>
struct Vop3FloatModifiers {
    SpirvCompute& b;
    const Rdna2Inst& in;
    Read& val;
    bool& clamp_routed;
    bool& ok;
    // Float source with its VOP3 modifiers applied: abs then a sign-bit-exact negate
    // (hardware order neg(abs(x))). Returns raw bits (fbin/fext re-bitcast), so it's a
    // drop-in for val() in the FLOAT ops only — integer VOP3 ops keep val() (modifiers are
    // float-domain; assemblers don't set them on int ops).
    uint32_t fv(int k) const {
        uint32_t bits = val(in.src[k]);
        if (in.src_abs[k]) bits = b.fext1(Glsl_FAbs, bits);
        if (in.src_neg[k]) bits = b.fneg(bits);
        return bits;
    }
    // Output modifiers on a FLOAT result: OMOD scale (×2/×4/×0.5) then CLAMP saturate to [0,1]
    // (hardware order: clamp(omod(x))). Wrap each float op's result through this. Opcodes
    // that route through it mark the CLAMP bit consumed; a set CLAMP on any opcode that
    // does NOT (integer ops, cndmask, the pack family) means unmodeled INTEGER saturation
    // (ISA 6.5) or an unhandled combination — the dispatch tail rejects it fail-visibly
    // instead of silently dropping the saturation.
    uint32_t fresult(uint32_t bits) const {
        clamp_routed = true;
        return modify_float_result(b, in, bits, ok);
    }
};
} // namespace prosper::gpu::ir
