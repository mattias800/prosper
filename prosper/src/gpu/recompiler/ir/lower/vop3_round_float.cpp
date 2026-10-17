// New e64 rounding forms share the existing definitions. No CFG, mask, proof,
// definedness or folding policy is changed; the root keeps its final predicate_write.
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/ir/lower/vop3_binary_float_lowering.hpp"
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/ir/lower/vop3_round_float_lowering.hpp"
#include "gpu/recompiler/ir/lower/f32_round_input.hpp"
#include "gpu/recompiler/ir/ops/vector_round_float.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdio>
#include <cstdlib>
namespace prosper::gpu::ir {
bool lower_vop3_round_float(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                            const BinaryFloatSource& fv, const BinaryFloatResult& fresult) {
    const auto* def = find_vector_round_float_op(in.fmt, in.opcode);
    if (!def) return false;
    // Ordinary F32 VOP3 has no OP_SEL or modifiers for absent S1/S2.
    // CONFIDENCE: MED -- admit canonical zero unused fields only. Whether an unused
    // literal selector consumes another dword is a separate encoding contract.
    if (in.has_dpp || in.has_sdwa || ((in.words[0] >> 11) & 0xfu) ||
        ((in.words[1] >> 9) & 0x3ffffu) || in.src_abs[1] || in.src_abs[2] || in.src_neg[1] ||
        in.src_neg[2]) {
        ok = false;
        return true;
    }
    if (!fv || !fresult) {
        std::fprintf(stderr, "[ir] rounding float callbacks missing\n");
        std::abort();
    }
    const auto source = fv(0);
    if (!ok) return true;
    const auto result = emit_f32_rounding_op(b, in, *def, source, ok);
    if (!ok) return true;
    rs.vreg[in.dst.value] = fresult(result);
    return true;
}
bool lower_vop3_float_values_erased(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                                    bool allow_wave, const BinaryFloatSource& fv,
                                    const BinaryFloatResult& fresult) {
    if (lower_vop3_round_float(b, rs, in, ok, fv, fresult)) return true;
    return lower_vop3_binary_float_erased(b, rs, in, ok, allow_wave, fv, fresult);
}
}   // namespace prosper::gpu::ir
