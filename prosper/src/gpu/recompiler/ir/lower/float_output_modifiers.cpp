// Launch state selects OMOD policy; no register/mark/CFG authority is changed here.
#include "gpu/recompiler/ir/lower/float_output_modifiers.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
FloatOutputModifier float_output_modifier(SpirvCompute& b, const Rdna2Inst& in,
                                          FloatResultWidth width, bool& ok) {
    FloatOutputModifier result{0, 0, width};
    if (!in.omod) return result;
    bool mode_available = false, flags_available = false, ieee = false;
    uint8_t mode = 0;
    if (b.is_fragment || b.is_fragment_packet()) {
        mode_available = b.fragment_float_mode.available;
        mode = b.fragment_float_mode.value;
        flags_available = b.fragment_float_flags.available;
        ieee = b.fragment_float_flags.ieee_mode;
    } else if (b.is_compute && !b.ngg_workgroup_shell) {
        mode_available = flags_available = b.compute_pgm_rsrc1 != UINT32_MAX;
        mode = (b.compute_pgm_rsrc1 >> 12) & 0xff;
        ieee = (b.compute_pgm_rsrc1 & (1u << 23)) != 0;
    } else {
        mode_available = flags_available = b.vertex_float_mode.available;
        mode = b.vertex_float_mode.value;
        ieee = b.vertex_float_mode.ieee_mode;
    }
    const uint8_t output_bit = width == FloatResultWidth::F32 ? 0x20 : 0x80;
    if ((mode_available && (mode & output_bit)) || (flags_available && ieee)) return result;
    if (!mode_available || !flags_available || width == FloatResultWidth::F64) {
        b.stage_reject_pc = in.pc;
        b.stage_reject_reason = width == FloatResultWidth::F64
                                    ? "float-omod-f64-representation-unavailable"
                                    : "float-omod-launch-mode-unavailable";
        ok = false;
        return result;
    }
    result.scale = in.omod;
    result.rounding = (mode >> (width == FloatResultWidth::F32 ? 0 : 2)) & 3;
    return result;
}
uint32_t modify_float_result(SpirvCompute& b, const Rdna2Inst& in, uint32_t bits, bool& ok) {
    const auto& effects = rdna2_instruction_effects(in);
    const auto width = effects.vdst_half ? FloatResultWidth::F16 : FloatResultWidth::F32;
    bits = emit_float_output_modifier(b, bits, float_output_modifier(b, in, width, ok));
    if (in.clamp) bits = b.clamp01(bits);
    return bits;
}
} // namespace prosper::gpu::ir
