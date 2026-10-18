// Device authority for modules declaring RoundingModeRTE 32. Guest MODE is a separate check.
#pragma once

namespace prosper::gpu {

// Publish after device creation: the RTE32 property, independent 32-bit rounding controls,
// and effective Vulkan 1.2 or an enabled VK_KHR_shader_float_controls extension.
// Publishers are ANDed; absence is unavailable, and a weaker device cannot be overridden.
void publish_float32_rte_support(bool rte32, bool independent32, bool permitted);
bool float32_rte_admitted();
void reset_float32_rte_support_for_test();

} // namespace prosper::gpu
