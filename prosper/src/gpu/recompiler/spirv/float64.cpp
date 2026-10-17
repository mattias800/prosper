// float64.cpp — SpirvCompute's binary64 types and relations (float64_support.hpp, #5325).
//
// Only reached after the lowering checked float64_admitted(): a module that declares Float64 is
// invalid on a device without shaderFloat64, so the declaration is lazy and gated, never ambient.
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {

namespace {
// SPIR-V 1.6 unified: Capability Float64 (10) and DenormPreserve (4464, SPV_KHR_float_controls);
// ExecutionMode DenormPreserve (4459), whose literal operand is the bit width.
constexpr uint32_t kCapFloat64 = 10, kCapDenormPreserve = 4464, kModeDenormPreserve = 4459;
} // namespace

void SpirvCompute::declare_float_controls_extension() {
    // One OpExtension per module: the SignedZeroInfNanPreserve declaration and the F64
    // DenormPreserve declaration both need SPV_KHR_float_controls.
    if (float_controls_extension_declared) return;
    float_controls_extension_declared = true;
    std::vector<uint32_t> o;
    pstr(o, "SPV_KHR_float_controls");
    putv(exts, Op_Extension, o);
}

uint32_t SpirvCompute::t_f64() {
    if (t_f64_cache) return t_f64_cache;
    put(caps, Op_Capability, {kCapFloat64});
    // AMD 70648 6.4: FP_DENORM[7:6] = 1 or 3 lets binary64 subnormal inputs reach the operation.
    // The lowering refuses (or flushes itself) for the other modes, so every subnormal an F64
    // relation sees here is one the guest meant to keep; DenormPreserve 64 stops the host from
    // flushing it. CONFIDENCE: HIGH (Vulkan float controls; RDNA2 ISA Table 24).
    put(caps, Op_Capability, {kCapDenormPreserve});
    declare_float_controls_extension();
    put(exec, Op_ExecutionMode, {f_main, kModeDenormPreserve, 64});
    t_f64_cache = id();
    put(types, Op_TypeFloat, {t_f64_cache, 64});
    return t_f64_cache;
}

uint32_t SpirvCompute::fcmp64(uint32_t cmpop, uint32_t a64, uint32_t b64) {
    const uint32_t type = t_f64();
    const uint32_t x = id();
    put(code, Op_Bitcast, {type, x, a64});
    const uint32_t y = id();
    put(code, Op_Bitcast, {type, y, b64});
    const uint32_t r = id();
    put(code, cmpop, {t_bool, r, x, y});
    return r;
}

}   // namespace prosper::gpu
