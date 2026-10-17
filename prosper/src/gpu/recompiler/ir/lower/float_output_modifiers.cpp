// Launch state selects OMOD policy; no register/mark/CFG authority is changed here.
#include "gpu/recompiler/ir/lower/float_output_modifiers.hpp"
#include "gpu/recompiler/ir/lower/launch_float_mode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
FloatOutputModifier float_output_modifier(SpirvCompute& b, const Rdna2Inst& in,
                                          FloatResultWidth width, bool& ok) {
    FloatOutputModifier result{0, 0, width};
    if (!in.omod) return result;
    const LaunchFloatMode launch = launch_float_mode(b);
    const bool mode_available = launch.mode_available, flags_available = launch.flags_available;
    const bool ieee = launch.ieee;
    const uint8_t mode = launch.mode;
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
