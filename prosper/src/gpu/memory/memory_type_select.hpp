// memory_type_select.hpp — which Vulkan memory type a renderer allocation should use (#3888).
//
// Vulkan hands every buffer and image a `memoryTypeBits` mask of the types it may live in and leaves
// the choice to the application. The spec orders memory types by their property FLAGS -- a type
// whose flags are a strict subset of another's comes first (VkPhysicalDeviceMemoryProperties,
// "memoryTypes" ordering rules) -- and says nothing about heaps. A type with NO flags is a subset of
// every other type's flags, so a driver that exposes one must list it first, and nothing requires it
// to be on a device-local heap (NVIDIA's is reported to sit on the system heap -- unverified here).
// So "the first compatible type" is not "the fastest compatible type". On a discrete GPU whose
// driver lists a flag-less system-memory type first, an image allocated that way lives in system
// RAM and every sample or attachment access crosses the bus.
// Rendering stays correct, and no correctness test can see the loss.
//
// THE POLICY:
//
//     find_memory_type(bits, required)
//         the lowest-indexed type allowed by `bits` whose flags contain all of `required`.
//     select_memory_type(bits, required, preferred)
//         among those, the lowest-indexed one whose flags also contain all of `preferred`;
//         when none does, find_memory_type(bits, required).
//     select_gpu_only_memory_type(bits)
//         select_memory_type(bits, 0, DEVICE_LOCAL): prefer VRAM, and fall back to any compatible
//         type rather than failing an allocation the old rule would have made.
//
// Use select_gpu_only_memory_type for anything only the GPU reads or writes: colour and depth
// targets, sampled and storage images, GPU-only buffers. Keep find_memory_type with explicit flags
// for memory the CPU maps (staging, readback, host-written uniform/storage data). An explicit
// HOST_VISIBLE request is never widened or re-routed by this header, so staging stays host-visible.
//
// Why "the FIRST device-local type" and not a finer ranking: the same ordering rule puts a plain
// DEVICE_LOCAL type ahead of DEVICE_LOCAL|HOST_VISIBLE, so on a discrete card without resizable BAR
// the first device-local type is the big VRAM heap, not the 256 MiB BAR window. On RADV -- this
// project's Strix Halo APU (type 0: DEVICE_LOCAL on the 43 GiB device-local heap, ahead of the
// host-heap types) and AMD discrete cards alike -- the first compatible type for an image is already
// device-local, so this policy picks the same type the pre-#3888 code did there; the
// `[mem-placement]` lines (gpu/diagnostics/memory_placement_log.hpp) print both and say `same`.
// A software rasterizer with a single all-flags type is likewise unaffected.
//
// Pure: no Vulkan calls, no globals, no environment. Unit test: tests/gpu/memory/
// test_memory_type_select.cpp. CONFIDENCE: HIGH for the selection rule (it is the spec's own
// recommended "required, then preferred" search); MED that any shipping driver actually lists a
// host type first among an optimal image's allowed types -- the fix is cheap insurance either way.
#pragma once
#include <vulkan/vulkan.h>

#include <cstdint>

namespace prosper::gpu {

inline uint32_t memory_type_count(const VkPhysicalDeviceMemoryProperties& props) {
    return props.memoryTypeCount < VK_MAX_MEMORY_TYPES ? props.memoryTypeCount
                                                       : static_cast<uint32_t>(VK_MAX_MEMORY_TYPES);
}

// The lowest-indexed type allowed by `bits` whose flags contain all of `required`, or UINT32_MAX.
inline uint32_t find_memory_type(const VkPhysicalDeviceMemoryProperties& props, uint32_t bits,
                                 VkMemoryPropertyFlags required) {
    const uint32_t count = memory_type_count(props);
    for (uint32_t i = 0; i < count; ++i)
        if ((bits & (1u << i)) && (props.memoryTypes[i].propertyFlags & required) == required)
            return i;
    return UINT32_MAX;
}

// `required` must hold; `preferred` is honoured when some allowed type also has it.
inline uint32_t select_memory_type(const VkPhysicalDeviceMemoryProperties& props, uint32_t bits,
                                   VkMemoryPropertyFlags required,
                                   VkMemoryPropertyFlags preferred) {
    const uint32_t best = find_memory_type(props, bits, required | preferred);
    return best != UINT32_MAX ? best : find_memory_type(props, bits, required);
}

// A resource only the GPU touches: device-local when any allowed type is, else any allowed type.
inline uint32_t select_gpu_only_memory_type(const VkPhysicalDeviceMemoryProperties& props,
                                            uint32_t bits) {
    return select_memory_type(props, bits, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

}  // namespace prosper::gpu
