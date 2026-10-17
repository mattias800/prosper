// AMD70648 12.12: sixteen selector nibbles gather within a row, or from its paired row.
// Expressions retain the old helper/lowering order, including late old-destination reads.
#include "gpu/recompiler/ir/ops/vector_permlane.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
namespace prosper::gpu::ir {
namespace {
Value gather(Builder& ir, const PermlaneInputs& x, bool across_rows, bool fetch_inactive,
             bool bound_ctrl) {
    const Value lane = ir.lane_id();
    const Value row_lane = ir.and_(lane, ir.c32(15));
    const Value selector_word =
        ir.select(ir.ult(row_lane, ir.c32(8)), x.selectors_lo, x.selectors_hi);
    const Value selector_index = ir.and_(row_lane, ir.c32(7));
    const Value selector_shift = ir.shl(selector_index, ir.c32(2));
    const Value selected = ir.and_(ir.lshr(selector_word, selector_shift), ir.c32(15));
    Value row_base = ir.and_(lane, ir.c32(~15u));
    if (across_rows) row_base = ir.xor_(row_base, ir.c32(16));
    const Value source_lane = ir.or_(row_base, selected);
    const Value shuffled = ir.shuffle(x.source, source_lane);
    Value result = shuffled;
    if (!fetch_inactive) {
        const Value active_word = ir.select(x.active(), ir.c32(1), ir.c32(0));
        const Value fetched_active = ir.shuffle(active_word, source_lane);
        const Value source_active = ir.ine(fetched_active, ir.c32(0));
        // CONFIDENCE: MED: preserve the inherited FI=0/BC=0 inactive-source choice exactly.
        // Table90 does not give combined FI/BC pseudocode; this migration does not correct it.
        result = ir.select(source_active, shuffled, bound_ctrl ? ir.c32(0) : x.old_d());
    }
    return result;
}
enum class Input { Source, SelectorsLo, SelectorsHi, Active, OldD };
struct Built {
    Block block;
    std::vector<Input> inputs;
    Value d;
};
Built build(bool across_rows, bool fetch_inactive, bool bound_ctrl) {
    Built out;
    Builder ir(out.block);
    const auto input = [&](Input logical, Type type) {
        out.inputs.push_back(logical);
        return ir.input(type);
    };
    const Value source = input(Input::Source, Type::U32);
    const Value lo = input(Input::SelectorsLo, Type::U32);
    const Value hi = input(Input::SelectorsHi, Type::U32);
    const PermlaneInputs x{source, lo, hi, [&] { return input(Input::Active, Type::Bool); },
                           [&] { return input(Input::OldD, Type::U32); }};
    out.d = across_rows ? permlanex16_b32(ir, x, fetch_inactive, bound_ctrl)
                        : permlane16_b32(ir, x, fetch_inactive, bound_ctrl);
    return out;
}
}   // namespace
Value permlane16_b32(Builder& ir, const PermlaneInputs& x, bool fetch_inactive, bool bound_ctrl) {
    return gather(ir, x, false, fetch_inactive, bound_ctrl);
}
Value permlanex16_b32(Builder& ir, const PermlaneInputs& x, bool fetch_inactive, bool bound_ctrl) {
    return gather(ir, x, true, fetch_inactive, bound_ctrl);
}
uint32_t emit_permlane_b32(SpirvCompute& b, uint32_t source, uint32_t selectors_lo,
                           uint32_t selectors_hi, bool across_rows, bool fetch_inactive,
                           bool bound_ctrl, const PermlaneInputReader& active,
                           const PermlaneInputReader& old_d) {
    if ((!fetch_inactive && !active) || (!fetch_inactive && !bound_ctrl && !old_d)) {
        std::fprintf(stderr, "[ir] permlane input callback missing\n");
        std::abort();
    }
    const Built out = build(across_rows, fetch_inactive, bound_ctrl);
    return emit_spirv_resolving_inputs(b, out.block, [&](uint32_t slot) {
        switch (out.inputs[slot]) {
            case Input::Source: return source;
            case Input::SelectorsLo: return selectors_lo;
            case Input::SelectorsHi: return selectors_hi;
            case Input::Active: return active();
            case Input::OldD: return old_d();
        }
        std::abort();
    })[out.d.index];
}
}   // namespace prosper::gpu::ir
