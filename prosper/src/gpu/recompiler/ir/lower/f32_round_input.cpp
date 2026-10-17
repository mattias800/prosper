// AMD70648 section 12's MODE rule applies to CEIL/FLOOR (section 12.8 has no override).
// Keep pure IR FLOOR uses, conversion overrides, source ownership and EXEC outside this policy.
#include "gpu/recompiler/ir/lower/f32_round_input.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/ops/float_input_modifier.hpp"
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <optional>
namespace prosper::gpu::ir {
namespace {
std::optional<uint8_t> producing_mode(const SpirvCompute& b) {
    if (b.is_fragment || b.is_fragment_packet()) {
        if (b.fragment_float_mode.available) return b.fragment_float_mode.value;
    } else if (b.is_compute && !b.ngg_workgroup_shell) {
        if (b.compute_pgm_rsrc1 != UINT32_MAX) return uint8_t(b.compute_pgm_rsrc1 >> 12);
    } else if (b.vertex_float_mode.available) {
        return b.vertex_float_mode.value;
    }
    return std::nullopt;
}
bool constant_is_mode_independent(const SpirvCompute& b, uint32_t source) {
    for (const auto& [bits, id] : b.uconst_cache) {
        if (id != source) continue;
        const auto magnitude = bits & 0x7fffffffu;
        return magnitude == 0 || magnitude >= 0x00800000u;
    }
    return false;
}
} // namespace
uint32_t emit_f32_rounding_op(SpirvCompute& b, const Rdna2Inst& in, const VectorOpDef& def,
                              uint32_t source, bool& ok) {
    if (!ok) return 0;
    if (!rdna2_instruction_effects(in).f32_rounding_input_sensitive)
        return emit_vector_op(b, def, source, 0);
    const auto mode = producing_mode(b);
    if (!mode) {
        if (constant_is_mode_independent(b, source)) return emit_vector_op(b, def, source, 0);
        b.stage_reject_pc = in.pc;
        b.stage_reject_reason = "f32-round-input-mode-unavailable";
        ok = false;
        return 0;
    }
    if (*mode & 0x10) return emit_vector_op(b, def, source, 0);
    Block block;
    Builder ir(block);
    const auto input = build_flush_f32_input(ir, ir.input(Type::U32));
    const auto result = def.build(ir, {input, ir.input(Type::U32)});
    const auto preserved = build_preserve_f32_zero(ir, input, result);
    const uint32_t inputs[] = {source, 0};
    return emit_spirv(b, block, inputs)[preserved.index];
}
} // namespace prosper::gpu::ir
