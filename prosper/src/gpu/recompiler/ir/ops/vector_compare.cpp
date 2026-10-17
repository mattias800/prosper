// vector_compare.cpp — lookup, evaluation and emission shared by every VOPC comparison definition.
#include "gpu/recompiler/ir/ops/vector_compare.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/ir.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>

namespace prosper::gpu::ir {

namespace {

struct BuiltCompare {
    Block block;
    Value result;
};

// Inputs: s0, s1, then (pair_sources) s0_hi, s1_hi. Inputs emit nothing, so their order is free.
BuiltCompare build(const VectorCompareDef& def, bool pairs) {
    if (def.pair_sources != pairs) {
        std::fprintf(stderr, "[ir] vector compare 0x%x called with the wrong source width\n",
                     def.opcode);
        std::abort();
    }
    BuiltCompare built;
    Builder ir(built.block);
    VectorCompareIn in;
    in.s0 = ir.input(Type::U32);
    in.s1 = ir.input(Type::U32);
    if (pairs) {
        in.s0_hi = ir.input(Type::U32);
        in.s1_hi = ir.input(Type::U32);
    }
    built.result = def.build(ir, in);
    return built;
}

}   // namespace

std::span<const std::span<const VectorCompareDef>> vector_compare_groups() {
    // A new group adds its table here.
    static const std::array<std::span<const VectorCompareDef>, 7> groups = {
        vector_compare_f32_defs(), vector_compare_i32_u32_defs(),   vector_compare_i16_u16_defs(),
        vector_compare_f16_defs(), vector_compare_class_f32_defs(), vector_compare_i64_u64_defs(),
        vector_compare_f64_defs(),
    };
    return groups;
}

const VectorCompareDef* find_vector_compare(uint32_t cmp_opcode) {
    for (const auto group : vector_compare_groups())
        for (const VectorCompareDef& def : group)
            if (def.opcode == cmp_opcode) return &def;
    return nullptr;
}

Known eval_vector_compare(const VectorCompareDef& def, Known s0, Known s1) {
    const BuiltCompare built = build(def, false);
    const Known inputs[] = {s0, s1};
    return evaluate(built.block, inputs)[built.result.index];
}

uint32_t emit_vector_compare(SpirvCompute& b, const VectorCompareDef& def, uint32_t s0,
                             uint32_t s1) {
    const BuiltCompare built = build(def, false);
    const uint32_t inputs[] = {s0, s1};
    return emit_spirv(b, built.block, inputs)[built.result.index];
}

Known eval_vector_compare_pairs(const VectorCompareDef& def, Known s0_lo, Known s0_hi, Known s1_lo,
                                Known s1_hi) {
    const BuiltCompare built = build(def, true);
    const Known inputs[] = {s0_lo, s1_lo, s0_hi, s1_hi};
    return evaluate(built.block, inputs)[built.result.index];
}

uint32_t emit_vector_compare_pairs(SpirvCompute& b, const VectorCompareDef& def, uint32_t s0_lo,
                                   uint32_t s0_hi, uint32_t s1_lo, uint32_t s1_hi) {
    const BuiltCompare built = build(def, true);
    const uint32_t inputs[] = {s0_lo, s1_lo, s0_hi, s1_hi};
    return emit_spirv(b, built.block, inputs)[built.result.index];
}

} // namespace prosper::gpu::ir
