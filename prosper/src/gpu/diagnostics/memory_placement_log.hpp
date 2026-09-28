// memory_placement_log.hpp — allocate a GPU-only resource's memory, and say where it went
// (#3888, #3897).
//
// `allocate_gpu_only_memory(device, cls, props, bits, info, &memory)` is the call every GPU-only
// renderer allocation makes. It tries gpu/memory/memory_type_select.hpp's candidates in order --
// device-local types first, then the others the resource allows -- and moves to the next ONLY when
// vkAllocateMemory returns VK_ERROR_OUT_OF_DEVICE_MEMORY. On an 8 GB discrete card a title that
// runs out of VRAM therefore keeps running, slowly, from system memory, instead of failing the
// allocation (#3897). When every candidate fails, the caller sees the error it saw before this
// helper existed and keeps its own failure path (drop the target, skip the draw, ...).
// `allocate_gpu_only(cls, props, bits, bytes, try_type)` is the same policy over a caller-supplied
// allocator, for sites that allocate through a pool; it is also what the unit test drives.
//
// Placement is invisible everywhere else -- a texture in system memory renders exactly the same
// pixels, and on this project's development APU the first compatible type is already device-local
// -- so it is reported three ways:
//
//   [mem-placement] sampled-texture: type 1 (DEVICE_LOCAL) heap 0 (device-local, 16384 MiB);
//                   first-compatible type 0 heap 1 (host, 32768 MiB) -- MOVED to device-local
//
// once per allocation class and memory type. The FIRST line of a class is exactly the #3896 line;
// a later allocation of the same class landing on a DIFFERENT type prints its own line, marked
// `[another type for this class]`, so a class that starts in VRAM and later spills is visible
// (the #3896 review's once-per-class blind spot). `first-compatible` is what the pre-#3888 rule
// (the lowest-indexed allowed type, whatever its heap) would have chosen; `-- same` means #3888
// did not move that class on this device.
//
//   [mem-placement] depth-target: FELL BACK after 2 attempt(s): type 0 (...) ran out of device
//                   memory; landed on type 2 (...) -- NOT DEVICE_LOCAL (system memory: ...)
//
// once per class and landed type, whenever an allocation succeeded only on a later candidate; and
// an `ALLOCATION FAILED` line once per class when every candidate failed. Separately, every GPU-only
// placement that lands off device-local memory while the device HAS device-local memory feeds the
// `gpu-memory-off-device` perf alarm (diagnostics/perf, #3891), which fires on any.
//
// PROSPER_GPU_MEM_FORCE_OOM=<class>[,<class>...]|all is a gated diagnostic that makes the device-
// local attempts of ONE allocation per named class fail with VK_ERROR_OUT_OF_DEVICE_MEMORY without
// calling the driver, so the fallback, its log line and the alarm can be exercised on a machine
// that never runs out of VRAM. PROSPER_GPU_MEM_FORCE_OOM_AT=<n> picks that class's n-th allocation
// (default 1; the alarm engine ignores everything before the first flip, so pick one after it).
// A class whose resource allows no non-device-local type then fails outright, which is the
// all-candidates-failed path. Unset, it costs one cached load per allocation.
//
// On by default, like gpu_memory_budget: the lines are bounded (classes x memory types) and change
// no pixel. The retry itself is NOT observation -- it decides where a resource lives -- and only
// runs after the driver has already refused.
#pragma once
#include "diagnostics/perf/perf_ledger.hpp"
#include "gpu/diagnostics/gpu_memory_budget_vk.hpp"
#include "gpu/memory/memory_type_select.hpp"

#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace prosper::gpu {

enum class GpuOnlyMemoryClass : uint8_t {
    ColorTarget,             // render_runner: the pass's primary colour attachment
    ColorTarget1,            // render_runner: the legacy second colour attachment
    ExtraColorTarget,        // render_runner: MRT slots 2..N
    DepthTarget,             // render_runner: the pass's depth/stencil attachment
    PersistentColorTarget,   // render_runner: retained render-to-texture colour targets
    RestoredDepthTarget,     // render_runner: a retained depth target recreated from a seed
    SampledTexture,          // render_runner: sampled/storage textures, incl. the persistent cache
    RetainedDepthArray,      // retained_depth_{array,cube}_gpu: GPU-resident depth snapshots
    ComputeImage,            // live_compute: compute-bound sampled/storage images
    DetileOutput,            // gpu_detile_upload: the compute detiler's output buffer
    PresentSlot,             // present_blit: the GPU-present slot images
    Count
};
static_assert(static_cast<size_t>(GpuOnlyMemoryClass::Count) <=
                  prosper::diagnostics::perf::kGpuMemoryClassSlots,
              "every GPU-only memory class needs a perf-ledger breakdown slot");
static_assert(static_cast<size_t>(GpuOnlyMemoryClass::Count) <= 32,
              "force_oom_class_mask is a 32-bit mask");

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

inline bool memory_type_is_device_local(const VkPhysicalDeviceMemoryProperties& props,
                                        uint32_t type) {
    return type < memory_type_count(props) &&
           (props.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
}

// Does the device have ANY usable device-local type? On a device without one (a software
// rasterizer reporting none), a host placement is the only placement and not worth an alarm.
inline bool device_has_device_local_memory(const VkPhysicalDeviceMemoryProperties& props) {
    return find_memory_type(props, ~0u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != UINT32_MAX;
}

// ---- PROSPER_GPU_MEM_FORCE_OOM (gated diagnostic) ---------------------------------------------

// Parse a class list: "all", or comma-separated class names. Unknown names are ignored (and
// reported by the caller). Returns a bit per GpuOnlyMemoryClass.
inline uint32_t parse_force_oom_classes(const char* spec) {
    if (!spec || !*spec) return 0;
    const uint32_t all = (1u << static_cast<uint32_t>(GpuOnlyMemoryClass::Count)) - 1u;
    uint32_t mask = 0;
    const char* p = spec;
    while (*p) {
        const char* e = std::strchr(p, ',');
        const size_t len = e ? static_cast<size_t>(e - p) : std::strlen(p);
        if (len == 3 && std::strncmp(p, "all", 3) == 0) mask |= all;
        for (uint32_t c = 0; c < static_cast<uint32_t>(GpuOnlyMemoryClass::Count); ++c) {
            const char* name = gpu_only_memory_class_name(static_cast<GpuOnlyMemoryClass>(c));
            if (std::strlen(name) == len && std::strncmp(p, name, len) == 0) mask |= 1u << c;
        }
        if (!e) break;
        p = e + 1;
    }
    return mask;
}

struct ForceOomConfig {
    uint32_t classes = 0;
    uint64_t at = 1;   // 1-based ordinal of the class's allocation to fail
};

inline const ForceOomConfig& force_oom_config() {
    static const ForceOomConfig config = [] {
        ForceOomConfig c;
        const char* spec = std::getenv("PROSPER_GPU_MEM_FORCE_OOM");
        c.classes = parse_force_oom_classes(spec);
        if (const char* at = std::getenv("PROSPER_GPU_MEM_FORCE_OOM_AT")) {
            const unsigned long long v = std::strtoull(at, nullptr, 10);
            if (v) c.at = v;
        }
        if (spec && *spec)
            std::fprintf(stderr,
                         "[mem-placement] PROSPER_GPU_MEM_FORCE_OOM=\"%s\" (class mask 0x%x, "
                         "allocation #%llu of each): DIAGNOSTIC -- device-local attempts of that "
                         "allocation fail without calling the driver\n",
                         spec, c.classes, (unsigned long long)c.at);
        return c;
    }();
    return config;
}

// Should THIS allocation of `cls` have its device-local attempts fail? Counts the class's
// allocations only while the diagnostic is armed for it.
inline bool force_gpu_only_oom(GpuOnlyMemoryClass cls) {
    const ForceOomConfig& c = force_oom_config();
    const uint32_t slot = static_cast<uint32_t>(cls);
    if (slot >= static_cast<uint32_t>(GpuOnlyMemoryClass::Count) || !(c.classes & (1u << slot)))
        return false;
    static std::atomic<uint64_t> ordinal[static_cast<size_t>(GpuOnlyMemoryClass::Count)] = {};
    return ordinal[slot].fetch_add(1, std::memory_order_relaxed) + 1 == c.at;
}

// ---- reporting ---------------------------------------------------------------------------------

// Log and count one completed GPU-only allocation (see the header comment for the three lines).
inline void note_gpu_only_placement(GpuOnlyMemoryClass cls,
                                    const VkPhysicalDeviceMemoryProperties& props, uint32_t bits,
                                    uint64_t bytes, const MemoryAllocationOutcome& o,
                                    bool forced = false) {
    const size_t slot = static_cast<size_t>(cls);
    if (slot >= static_cast<size_t>(GpuOnlyMemoryClass::Count)) return;
    const char* name = gpu_only_memory_class_name(cls);
    constexpr size_t kClasses = static_cast<size_t>(GpuOnlyMemoryClass::Count);
    static std::atomic<uint32_t> logged_types[kClasses] = {};
    static std::atomic<uint32_t> logged_fallbacks[kClasses] = {};
    static std::atomic<bool> logged_failure[kClasses] = {};
    const char* forced_note = forced ? " (forced by PROSPER_GPU_MEM_FORCE_OOM)" : "";

    if (!o.ok()) {
        if (logged_failure[slot].exchange(true, std::memory_order_relaxed)) return;
        char preferred[160];
        describe_memory_type(props, o.preferred, preferred, sizeof(preferred));
        if (o.candidates == 0)
            std::fprintf(stderr,
                         "[mem-placement] %s: NO COMPATIBLE TYPE (memoryTypeBits 0x%x) -- "
                         "allocation refused\n", name, bits);
        else
            std::fprintf(stderr,
                         "[mem-placement] %s: ALLOCATION FAILED: result %d after %u of %u "
                         "candidate type(s)%s; preferred %s\n",
                         name, static_cast<int>(o.result), o.attempts, o.candidates, forced_note,
                         preferred);
        return;
    }

    const bool device_local = memory_type_is_device_local(props, o.type);
    if (o.fell_back()) prosper::diagnostics::perf::add(prosper::diagnostics::perf::Counter::GpuMemoryFallbacks);
    if (!device_local && device_has_device_local_memory(props))
        prosper::diagnostics::perf::note_gpu_memory_off_device(slot, name, bytes);

    const uint32_t bit = o.type < 32 ? (1u << o.type) : 0u;
    if (o.fell_back()) {
        if (logged_fallbacks[slot].fetch_or(bit, std::memory_order_relaxed) & bit) return;
        char preferred[160], landed[160];
        describe_memory_type(props, o.preferred, preferred, sizeof(preferred));
        describe_memory_type(props, o.type, landed, sizeof(landed));
        std::fprintf(stderr,
                     "[mem-placement] %s: FELL BACK after %u attempt(s): %s ran out of device "
                     "memory%s; landed on %s -- %s\n",
                     name, o.attempts, preferred, forced_note, landed,
                     device_local ? "still DEVICE_LOCAL"
                                  : "NOT DEVICE_LOCAL (system memory: correct but slower on a "
                                    "discrete GPU; gpu-memory-off-device alarm)");
        return;
    }
    const uint32_t before_types = logged_types[slot].fetch_or(bit, std::memory_order_relaxed);
    if (before_types & bit) return;
    const uint32_t first = find_memory_type(props, bits, 0);   // the pre-#3888 rule
    char now[160], before[160];
    describe_memory_type(props, o.type, now, sizeof(now));
    describe_memory_type(props, first, before, sizeof(before));
    std::fprintf(stderr, "[mem-placement] %s: %s; first-compatible %s -- %s%s\n", name, now,
                 before,
                 !device_local     ? "NOT DEVICE_LOCAL (no device-local type accepted it)"
                 : o.type == first ? "same"
                                   : "MOVED to device-local",
                 before_types ? " [another type for this class]" : "");
}

// ---- allocation --------------------------------------------------------------------------------

// The GPU-only policy over a caller-supplied allocator: `try_type(type) -> VkResult` allocates from
// that memory type (through a pool, if the caller has one) and returns the driver's result. With
// `force_device_local_oom`, device-local attempts fail with VK_ERROR_OUT_OF_DEVICE_MEMORY without
// calling try_type (PROSPER_GPU_MEM_FORCE_OOM; the unit test passes it directly).
template <class TryType>
MemoryAllocationOutcome allocate_gpu_only(GpuOnlyMemoryClass cls,
                                          const VkPhysicalDeviceMemoryProperties& props,
                                          uint32_t bits, uint64_t bytes, TryType&& try_type,
                                          bool force_device_local_oom = false) {
    const MemoryAllocationOutcome o = allocate_with_memory_type_fallback(
        props, bits, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, [&](uint32_t type) -> VkResult {
            if (force_device_local_oom && memory_type_is_device_local(props, type))
                return VK_ERROR_OUT_OF_DEVICE_MEMORY;
            return try_type(type);
        });
    note_gpu_only_placement(cls, props, bits, bytes, o, force_device_local_oom);
    return o;
}

// vkAllocateMemory (through the counted gpu_memory_budget wrapper) for a GPU-only resource.
// `info.allocationSize` and `info.pNext` are the caller's; `info.memoryTypeIndex` is overwritten
// with the type that succeeded (on failure: the preferred type, for the caller's error message).
// On failure `*out` is VK_NULL_HANDLE and the result is the driver's error (or
// kNoCompatibleMemoryType when `bits` allows nothing prosper can use).
inline VkResult allocate_gpu_only_memory(VkDevice device, GpuOnlyMemoryClass cls,
                                         const VkPhysicalDeviceMemoryProperties& props,
                                         uint32_t bits, VkMemoryAllocateInfo& info,
                                         VkDeviceMemory* out) {
    *out = VK_NULL_HANDLE;
    const MemoryAllocationOutcome o = allocate_gpu_only(
        cls, props, bits, info.allocationSize,
        [&](uint32_t type) {
            info.memoryTypeIndex = type;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            const VkResult r = allocate_device_memory(device, &info, &memory);
            if (r == VK_SUCCESS) *out = memory;
            return r;
        },
        force_gpu_only_oom(cls));
    info.memoryTypeIndex = o.type;
    return o.result;
}

}  // namespace prosper::gpu
