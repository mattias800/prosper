// float64_support.hpp — the DEVICE GATE on binary64 (F64) guest instructions (#5325).
//
// An F64 instruction is emitted through SPIR-V `OpTypeFloat 64`, which needs the Float64
// capability (VkPhysicalDeviceFeatures::shaderFloat64 ENABLED on the executing device), and its
// subnormal inputs must reach the relation unflushed when the guest MODE allows them. Vulkan's
// default float contract lets a driver flush them, so an F64 module also declares
// `OpExecutionMode DenormPreserve 64` (SPV_KHR_float_controls), which needs
// VkPhysicalDeviceFloatControlsProperties::shaderDenormPreserveFloat64 and API 1.2 or the enabled
// VK_KHR_shader_float_controls extension.
//
// The contract mirrors publish_float_controls_support (rdna2_to_spirv.hpp), and for the same
// reasons:
//   * Nothing is admitted until a device owner publishes. Offline callers with no device refuse
//     every F64 instruction by name ("f64-device-unavailable"), which is the fail-visible state:
//     a module declaring Float64 that a device cannot execute is invalid, not slow.
//   * Every publisher is ANDed in. The renderer's device and the compute backend's private device
//     execute the same modules, so the weaker device decides, and the verdict only ever goes
//     true -> false. Publish at device creation, before the first guest submit.
// `shader_inspect` publishes a capable device so the corpus gate measures what RADV executes.
#pragma once

namespace prosper::gpu {

// `shader_float64_enabled`: shaderFloat64 was ENABLED at vkCreateDevice (never merely advertised).
// `denorm_preserve_float64`: the device reports shaderDenormPreserveFloat64.
// `float_controls_permitted`: effective API >= 1.2, or VK_KHR_shader_float_controls enabled.
void publish_float64_support(bool shader_float64_enabled, bool denorm_preserve_float64,
                             bool float_controls_permitted);

// True when a guest F64 instruction may be emitted: some device owner published, and all could.
bool float64_admitted();

// Restore the unpublished state. Test hook only.
void reset_float64_support_for_test();

} // namespace prosper::gpu
