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
//     memory_type_candidates(bits, required, preferred)
//         EVERY type select_memory_type could have returned, in the order an allocation should try
//         them: the preferred ones first, then the rest, each group by ascending index. Its first
//         entry is select_memory_type's answer.
//     allocate_with_memory_type_fallback(bits, required, preferred, try_type)
//         try the candidates in that order, moving to the next ONLY on
//         VK_ERROR_OUT_OF_DEVICE_MEMORY (#3897). Any other error, and "every candidate ran out",
//         surface as the result, so a caller that failed before still fails, with the same code.
//
// AMD DEVICE_COHERENT / DEVICE_UNCACHED types are never candidates, for any request: allocating
// from one is only valid with VkPhysicalDeviceCoherentMemoryFeaturesAMD::deviceCoherentMemory
// enabled, and prosper never enables it. RADV lists them after their ordinary equivalents, so the
// first-match rule did not reach them in practice -- but a mask that excluded the ordinary
// device-local type could, and the fallback order above would otherwise walk straight into them.
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
// Pure: no Vulkan calls, no globals, no environment (allocate_with_memory_type_fallback calls only
// the function it is given). Unit test: tests/gpu/memory/
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

// Flags of memory types prosper may never allocate from (see the header comment).
inline constexpr VkMemoryPropertyFlags kUnusableMemoryTypeFlags =
    VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD;

// Allowed by `bits`, usable by prosper at all, and carrying every flag in `required`.
inline bool memory_type_matches(const VkPhysicalDeviceMemoryProperties& props, uint32_t bits,
                                uint32_t type, VkMemoryPropertyFlags required) {
    if (type >= memory_type_count(props) || !(bits & (1u << type))) return false;
    const VkMemoryPropertyFlags f = props.memoryTypes[type].propertyFlags;
    return (f & kUnusableMemoryTypeFlags) == 0 && (f & required) == required;
}

// The lowest-indexed type allowed by `bits` whose flags contain all of `required`, or UINT32_MAX.
inline uint32_t find_memory_type(const VkPhysicalDeviceMemoryProperties& props, uint32_t bits,
                                 VkMemoryPropertyFlags required) {
    const uint32_t count = memory_type_count(props);
    for (uint32_t i = 0; i < count; ++i)
        if (memory_type_matches(props, bits, i, required)) return i;
    return UINT32_MAX;
}

// Every type allowed by `bits` with all of `required`: those that also have all of `preferred`
// first, then the others, each group by ascending index. Returns how many were written to `out`.
inline uint32_t memory_type_candidates(const VkPhysicalDeviceMemoryProperties& props,
                                       uint32_t bits, VkMemoryPropertyFlags required,
                                       VkMemoryPropertyFlags preferred,
                                       uint32_t (&out)[VK_MAX_MEMORY_TYPES]) {
    const uint32_t count = memory_type_count(props);
    uint32_t n = 0;
    for (uint32_t i = 0; i < count; ++i)
        if (memory_type_matches(props, bits, i, required | preferred)) out[n++] = i;
    for (uint32_t i = 0; i < count; ++i)
        if (memory_type_matches(props, bits, i, required) &&
            !memory_type_matches(props, bits, i, required | preferred))
            out[n++] = i;
    return n;
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

// What an allocation through allocate_with_memory_type_fallback did.
struct MemoryAllocationOutcome {
    // VK_SUCCESS; or the first error that was not VK_ERROR_OUT_OF_DEVICE_MEMORY; or
    // VK_ERROR_OUT_OF_DEVICE_MEMORY when every candidate ran out; or
    // kNoCompatibleMemoryType when there was no candidate to try.
    VkResult result = VK_ERROR_FEATURE_NOT_PRESENT;
    uint32_t type = UINT32_MAX;        // the type that succeeded; on failure, `preferred`
    uint32_t preferred = UINT32_MAX;   // the first candidate: what select_memory_type returns
    uint32_t attempts = 0;             // try_type calls made
    uint32_t candidates = 0;           // how many types were eligible
    bool ok() const { return result == VK_SUCCESS; }
    // Succeeded, but not on the type it preferred: an earlier candidate ran out of memory.
    bool fell_back() const { return ok() && type != preferred; }
};

// The result when no allowed type satisfies the request. Matches what the live compute pool
// already returned for a UINT32_MAX memory type, so a caller sees one code for "nothing to try".
inline constexpr VkResult kNoCompatibleMemoryType = VK_ERROR_FEATURE_NOT_PRESENT;

// Try memory_type_candidates(bits, required, preferred) in order with `try_type(type) -> VkResult`,
// retrying the next ONLY after VK_ERROR_OUT_OF_DEVICE_MEMORY. `required` is honoured by every
// attempt: a HOST_VISIBLE request can land on another host-visible type, never on a type the CPU
// cannot map. Pure: all side effects are try_type's.
template <class TryType>
MemoryAllocationOutcome allocate_with_memory_type_fallback(
        const VkPhysicalDeviceMemoryProperties& props, uint32_t bits,
        VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred, TryType&& try_type) {
    MemoryAllocationOutcome out;
    uint32_t candidates[VK_MAX_MEMORY_TYPES];
    out.candidates = memory_type_candidates(props, bits, required, preferred, candidates);
    if (out.candidates == 0) { out.result = kNoCompatibleMemoryType; return out; }
    out.preferred = out.type = candidates[0];
    for (uint32_t i = 0; i < out.candidates; ++i) {
        const VkResult r = try_type(candidates[i]);
        ++out.attempts;
        if (r == VK_SUCCESS) { out.result = VK_SUCCESS; out.type = candidates[i]; return out; }
        if (r != VK_ERROR_OUT_OF_DEVICE_MEMORY) { out.result = r; return out; }
    }
    out.result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    return out;
}

}  // namespace prosper::gpu
