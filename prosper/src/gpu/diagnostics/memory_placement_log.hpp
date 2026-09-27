// memory_placement_log.hpp — choose a GPU-only resource's memory type and say where it went (#3888).
//
// `choose_gpu_only_memory_type(cls, props, bits)` is gpu/memory/memory_type_select.hpp's
// select_gpu_only_memory_type plus ONE stderr line per allocation class per process:
//
//   [mem-placement] sampled-texture: type 1 (DEVICE_LOCAL) heap 0 (device-local, 16384 MiB);
//                   first-compatible type 0 heap 1 (host, 32768 MiB) -- MOVED to device-local
//
// The line exists because placement is invisible everywhere else: a texture in system memory renders
// exactly the same pixels, and on this project's development APU the first compatible type is
// already device-local, so nobody here can see the discrete-GPU case directly. The `first-compatible` half is
// what the pre-#3888 rule (the lowest-indexed allowed type, whatever its heap) would have chosen,
// computed from the same inputs, so one run shows both policies side by side: `-- same` means this
// change did not move that class on this device. `NOT DEVICE_LOCAL` means no device-local type
// accepted the resource and the allocation fell back to host memory -- a performance problem on a
// discrete card worth reporting.
//
// On by default, like gpu_memory_budget: at most one line per class for the life of the process, and
// the selection it logs is the one the allocation uses, so it changes no pixel. The choice itself is
// NOT observation (it decides the memory type); only the line is.
#pragma once
#include "gpu/memory/memory_type_select.hpp"

#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>

namespace prosper::gpu {

enum class GpuOnlyMemoryClass : uint8_t {
    ColorTarget,             // render_runner: the pass's primary colour attachment
    ColorTarget1,            // render_runner: the legacy second colour attachment
    ExtraColorTarget,        // render_runner: MRT slots 2..N
    DepthTarget,             // render_runner: the pass's depth/stencil attachment
    PersistentColorTarget,   // render_runner: retained render-to-texture colour targets
    RestoredDepthTarget,     // render_runner: a retained depth target recreated from a seed
    SampledTexture,          // render_runner: sampled/storage textures, incl. the persistent cache
    RetainedDepthArray,      // retained_depth_array_gpu: GPU-resident depth-array snapshots
    ComputeImage,            // live_compute: compute-bound sampled/storage images
    DetileOutput,            // gpu_detile_upload: the compute detiler's output buffer
    PresentSlot,             // present_blit: the GPU-present slot images
    Count
};

inline const char* gpu_only_memory_class_name(GpuOnlyMemoryClass c) {
    switch (c) {
    case GpuOnlyMemoryClass::ColorTarget: return "color-target";
    case GpuOnlyMemoryClass::ColorTarget1: return "color-target-1";
    case GpuOnlyMemoryClass::ExtraColorTarget: return "extra-color-target";
    case GpuOnlyMemoryClass::DepthTarget: return "depth-target";
    case GpuOnlyMemoryClass::PersistentColorTarget: return "persistent-color-target";
    case GpuOnlyMemoryClass::RestoredDepthTarget: return "restored-depth-target";
    case GpuOnlyMemoryClass::SampledTexture: return "sampled-texture";
    case GpuOnlyMemoryClass::RetainedDepthArray: return "retained-depth-array";
    case GpuOnlyMemoryClass::ComputeImage: return "compute-image";
    case GpuOnlyMemoryClass::DetileOutput: return "detile-output";
    case GpuOnlyMemoryClass::PresentSlot: return "present-slot";
    case GpuOnlyMemoryClass::Count: break;
    }
    return "?";
}

// "type 1 (DEVICE_LOCAL|HOST_VISIBLE) heap 0 (device-local, 16384 MiB)", or "none".
inline void describe_memory_type(const VkPhysicalDeviceMemoryProperties& props, uint32_t type,
                                 char* out, size_t out_size) {
    if (type >= memory_type_count(props)) { std::snprintf(out, out_size, "none"); return; }
    const VkMemoryPropertyFlags f = props.memoryTypes[type].propertyFlags;
    char flags[96] = {};
    size_t n = 0;
    auto add = [&](VkMemoryPropertyFlags bit, const char* name) {
        if (!(f & bit) || n >= sizeof(flags)) return;
        n += static_cast<size_t>(
            std::snprintf(flags + n, sizeof(flags) - n, "%s%s", n ? "|" : "", name));
    };
    add(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "DEVICE_LOCAL");
    add(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, "HOST_VISIBLE");
    add(VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, "HOST_COHERENT");
    add(VK_MEMORY_PROPERTY_HOST_CACHED_BIT, "HOST_CACHED");
    add(VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT, "LAZY");
    if (n == 0) std::snprintf(flags, sizeof(flags), "no flags");
    const uint32_t heap = props.memoryTypes[type].heapIndex;
    const bool heap_ok = heap < props.memoryHeapCount && heap < VK_MAX_MEMORY_HEAPS;
    const bool heap_local =
        heap_ok && (props.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
    const uint64_t heap_mib = heap_ok ? static_cast<uint64_t>(props.memoryHeaps[heap].size >> 20) : 0;
    std::snprintf(out, out_size, "type %u (%s) heap %u (%s, %" PRIu64 " MiB)", type, flags, heap,
                  heap_local ? "device-local" : "host", heap_mib);
}

inline uint32_t choose_gpu_only_memory_type(GpuOnlyMemoryClass cls,
                                            const VkPhysicalDeviceMemoryProperties& props,
                                            uint32_t bits) {
    const uint32_t chosen = select_gpu_only_memory_type(props, bits);
    static std::atomic<bool> logged[static_cast<size_t>(GpuOnlyMemoryClass::Count)] = {};
    const size_t slot = static_cast<size_t>(cls);
    if (slot < static_cast<size_t>(GpuOnlyMemoryClass::Count) &&
        !logged[slot].exchange(true, std::memory_order_relaxed)) {
        const uint32_t first = find_memory_type(props, bits, 0);   // the pre-#3888 rule
        char now[160], before[160];
        describe_memory_type(props, chosen, now, sizeof(now));
        describe_memory_type(props, first, before, sizeof(before));
        const bool device_local = chosen < memory_type_count(props) &&
            (props.memoryTypes[chosen].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        std::fprintf(stderr, "[mem-placement] %s: %s; first-compatible %s -- %s\n",
                     gpu_only_memory_class_name(cls), now, before,
                     chosen == UINT32_MAX ? "NO COMPATIBLE TYPE"
                     : !device_local      ? "NOT DEVICE_LOCAL (no device-local type accepted it)"
                     : chosen == first    ? "same"
                                          : "MOVED to device-local");
    }
    return chosen;
}

}  // namespace prosper::gpu
