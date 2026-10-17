// vector_compare.hpp — the ONE definition of each VOPC comparison's per-lane value (ADR 0012).
//
// A VOPC compare is two halves. This header is the first: a VectorCompareDef's `build` appends the
// lane's test result as a Bool, from its two source words after the lowering has applied the
// source modifiers. The second half, writing EXEC & result to VCC, an SGPR pair or (V_CMPX) EXEC,
// is the same for every compare and lives in ir/lower/vopc_mask_write.cpp; no definition here
// knows about masks, destinations or EXEC. The lowering emits a definition (emit_vector_compare)
// and tests evaluate it (eval_vector_compare) through the same function.
//
// Definitions are keyed by the V_CMP_* opcode. A V_CMPX_* (opcode + 0x10, vopc_is_cmpx) and the
// VOP3 (e64) encoding of either decode as VOPC with the same number and share the definition.
#pragma once

#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"

#include <cstdint>
#include <span>

namespace prosper::gpu {
struct SpirvCompute;
}

namespace prosper::gpu::ir {

struct VectorCompareIn {
    Value s0, s1;
};

struct VectorCompareDef {
    uint32_t opcode;   // the V_CMP_* opcode
    Value (*build)(Builder& ir, const VectorCompareIn& in);   // returns a Bool
};

// One table per type group, each in its own ops/vector_compare_<group>.cpp.
std::span<const VectorCompareDef> vector_compare_f32_defs();

// Every group's definitions, and the one for a V_CMP_* opcode (nullptr when it has none yet).
std::span<const std::span<const VectorCompareDef>> vector_compare_groups();
const VectorCompareDef* find_vector_compare(uint32_t cmp_opcode);

// One lane, on known sources: 1 when the test passes, 0 when it fails.
Known eval_vector_compare(const VectorCompareDef& def, Known s0, Known s1);
// The SPIR-V Bool id of the test result, given the ids of the (already modified) sources.
uint32_t emit_vector_compare(SpirvCompute& b, const VectorCompareDef& def, uint32_t s0,
                             uint32_t s1);

} // namespace prosper::gpu::ir
