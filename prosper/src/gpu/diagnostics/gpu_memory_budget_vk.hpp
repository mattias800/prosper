// gpu_memory_budget_vk.hpp — the Vulkan-typed face of gpu_memory_budget.
//
// It is a separate header from `gpu_memory_budget.hpp` so that one can stay free of Vulkan types
// and be included anywhere. Everything here is a pure pass-through plus a counter: the arguments
// reaching the driver are byte-for-byte what the call site passed, and removing this header would
// change no rendered pixel.
//
// USE THESE INSTEAD OF CALLING vkAllocateMemory/vkFreeMemory DIRECTLY. A site that calls the driver
// directly is invisible to the budget, and an invisible ALLOCATION under-reports while an invisible
// FREE looks like a leak that is not there -- the second is worse, because a number that drifts
// upward is exactly the evidence somebody will act on. `prosper/tools/vkmem_coverage.py` finds any
// site that has slipped past this, and a ctest case runs it.
#pragma once
#include "gpu/diagnostics/gpu_memory_budget.hpp"

#include <vulkan/vulkan.h>

namespace prosper::gpu {

// Record a device's heap layout from its properties. Call once per device, before allocating.
inline void set_device_heaps(const VkPhysicalDeviceMemoryProperties& props) {
    uint32_t type_heap[VK_MAX_MEMORY_TYPES] = {};
    uint64_t heap_size[VK_MAX_MEMORY_HEAPS] = {};
    const uint32_t types = props.memoryTypeCount < VK_MAX_MEMORY_TYPES
        ? props.memoryTypeCount : VK_MAX_MEMORY_TYPES;
    const uint32_t heaps = props.memoryHeapCount < VK_MAX_MEMORY_HEAPS
        ? props.memoryHeapCount : VK_MAX_MEMORY_HEAPS;
    for (uint32_t i = 0; i < types; i++) type_heap[i] = props.memoryTypes[i].heapIndex;
    for (uint32_t i = 0; i < heaps; i++) heap_size[i] = (uint64_t)props.memoryHeaps[i].size;
    set_device_heaps(type_heap, types, heap_size, heaps);
}

// vkAllocateMemory, counted. Only a VK_SUCCESS is counted, and `out` is whatever the driver wrote.
//
// A FAILURE reports the budget from here, which is the only place that sees every failure. It was
// once wired at a single call site instead, and that made a property of the instrument out of a
// property of one of its many sites: a renderer allocation could fail and still print no
// number, which is precisely the #3533 scenario the whole thing exists for.
inline VkResult allocate_device_memory(VkDevice device, const VkMemoryAllocateInfo* info,
                                       VkDeviceMemory* out) {
    const VkResult status = vkAllocateMemory(device, info, nullptr, out);
    if (status == VK_SUCCESS) {
        if (info)
            note_device_alloc((uint64_t)*out, info->memoryTypeIndex,
                              (uint64_t)info->allocationSize);
        report_device_memory_periodically();
        return status;
    }
    report_allocation_failure(static_cast<int>(status),
                              info ? (uint64_t)info->allocationSize : 0,
                              info ? info->memoryTypeIndex : 0);
    return status;
}

// vkFreeMemory, counted. VK_NULL_HANDLE is a legal no-op for both the driver and the counter.
inline void free_device_memory(VkDevice device, VkDeviceMemory memory) {
    note_device_free((uint64_t)memory);
    report_device_memory_periodically();
    vkFreeMemory(device, memory, nullptr);
}

}  // namespace prosper::gpu
