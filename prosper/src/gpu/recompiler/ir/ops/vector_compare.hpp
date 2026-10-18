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

// A 32-bit group reads s0 and s1. A 64-bit group (`pair_sources`) reads each source as two U32
// words, s0:s0_hi and s1:s1_hi, the halves of the guest register pair (or the extended constant)
// the lowering prepared; it packs them itself (Builder::pack_u64), so the pack is part of its bytes.
struct VectorCompareIn {
    Value s0, s1;
    Value s0_hi = {}, s1_hi = {};   // pair_sources only
};

struct VectorCompareDef {
    uint32_t opcode;   // the V_CMP_* opcode
    Value (*build)(Builder& ir, const VectorCompareIn& in);   // returns a Bool
    bool pair_sources = false;   // true for the 64-bit groups
};

// One table per type group, each in its own ops/vector_compare_<group>.cpp.
std::span<const VectorCompareDef> vector_compare_f32_defs();
std::span<const VectorCompareDef> vector_compare_i32_u32_defs();
std::span<const VectorCompareDef> vector_compare_i16_u16_defs();
// F16 definitions receive unpacked F32 bits from the existing lowering preparation.
std::span<const VectorCompareDef> vector_compare_f16_defs();
// CLASS_F32 receives raw SRC0 after integer ABS/NEG and raw SRC1.
std::span<const VectorCompareDef> vector_compare_class_f32_defs();
// CLASS_F64 reads SRC0's raw pair and only the low word of SRC1 (the class mask).
std::span<const VectorCompareDef> vector_compare_class_f64_defs();
// I64/U64 definitions read both halves of each source (pair_sources).
std::span<const VectorCompareDef> vector_compare_i64_u64_defs();
// F64 definitions read both halves of each source as one binary64 word (pair_sources).
std::span<const VectorCompareDef> vector_compare_f64_defs();

// Every group's definitions, and the one for a V_CMP_* opcode (nullptr when it has none yet).
std::span<const std::span<const VectorCompareDef>> vector_compare_groups();
const VectorCompareDef* find_vector_compare(uint32_t cmp_opcode);

// One lane, on known sources: 1 when the test passes, 0 when it fails.
Known eval_vector_compare(const VectorCompareDef& def, Known s0, Known s1);
// The SPIR-V Bool id of the test result, given the ids of the (already modified) sources.
uint32_t emit_vector_compare(SpirvCompute& b, const VectorCompareDef& def, uint32_t s0,
                             uint32_t s1);
// The same for a pair_sources definition, given each source as its low and high words. Calling
// the 32-bit form with a pair_sources definition, or this one without, is a lowering bug and aborts.
Known eval_vector_compare_pairs(const VectorCompareDef& def, Known s0_lo, Known s0_hi, Known s1_lo,
                                Known s1_hi);
uint32_t emit_vector_compare_pairs(SpirvCompute& b, const VectorCompareDef& def, uint32_t s0_lo,
                                   uint32_t s0_hi, uint32_t s1_lo, uint32_t s1_hi);

} // namespace prosper::gpu::ir
