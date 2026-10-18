// CLASS_F64 reads a binary64 source and a single raw U32 class mask. Neither FP modes nor
// shaderFloat64 are involved: the definition classifies the two integer words directly.
#include "gpu/recompiler/ir/lower/vopc_lowering_internal.hpp"
#include "gpu/recompiler/ir/ops/vector_compare.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstdint>

namespace prosper::gpu::ir {
uint32_t VopcLowering::compare_class_f64() {
    if (in.has_sdwa || in.has_dpp) {
        refuse("class-f64-sdwa-dpp");
        return 0;
    }
    const Operand& source = in.src[0];
    const Operand& mask = in.src[1];
    // The existing uniform-compare proof observes the first word. Admit a scalar pair only
    // when BOTH words satisfy its existing scalar-data predicate; do not invent a new proof.
    if ((source.kind == OperandKind::SGPR || source.kind == OperandKind::Special) &&
        !scalar_integer64_operand(source)) {
        refuse("class-f64-unproved-scalar-pair");
        return 0;
    }
    if (mask.kind != OperandKind::VGPR && !scalar_data_operand(mask)) {
        refuse("class-f64-unproved-class-mask");
        return 0;
    }
    // A class mask has integer semantics. No contract has established inline FP or special
    // mask operands other than SGPR_NULL; keep those forms outside this admission.
    if (mask.kind == OperandKind::InlineFloat ||
        (mask.kind == OperandKind::Special && mask.value != 125)) {
        refuse("class-f64-class-mask-unsupported");
        return 0;
    }
    const Integer64Halves source_bits = float64_source(0, ra);
    if (!ok) return 0;
    if (const VectorCompareDef* def = find_vector_compare(eff))
        return emit_vector_compare_pairs(b, *def, source_bits.lo, source_bits.hi, rc, b.uconst(0));
    ok = false;
    return 0;
}
} // namespace prosper::gpu::ir
