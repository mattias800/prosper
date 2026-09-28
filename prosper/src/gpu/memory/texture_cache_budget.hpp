// texture_cache_budget.hpp — how many bytes may the persistent texture-image cache hold?
//
// The renderer keeps uploaded guest textures resident in a byte-bounded LRU cache
// (tests/fixtures/render_runner.h). Until #3873 that bound was `clamp(heap / 8, 1 GiB, 4 GiB)`
// of the largest device-local heap. The divisor was chosen only so an 8 GiB machine kept the
// historical 1 GiB default; nothing derived it. On the 8-16 GB discrete cards most users run, it
// hands textures 1-2 GiB while most of the card sits idle, and a title whose working set is larger
// re-uploads it every frame (Outer Wilds, CPU-decoded to RGBA8 at 1 GiB: ~2,000-2,600 misses and
// ~12 GB re-uploaded per 5 s, 6.7-7.6 fps against ~10 fps when the set fits).
//
// THE POLICY, when VK_EXT_memory_budget is available (the pure function below; every constant is
// named in this header so the unit test and a reader see the same numbers):
//
//     heuristic   = clamp(heap_size / 8, 1 GiB, 4 GiB)                 (the old rule)
//     non_texture = max(heap_usage, prosper_held) - texture_bytes     (saturating)
//     target      = heap_budget * 80 / 100
//     live        = target - non_texture                              (saturating)
//     discrete:  budget = max(1 GiB, min(live, heap_size * 3 / 4))
//     unified:   budget = max(1 GiB, min(live, heap_size * 3 / 4, heuristic))
//
// `heap_budget`/`heap_usage` are the driver's figures for the heap the textures allocate from:
// the budget is what the OS/driver lets THIS process use (it moves as other applications allocate),
// and the usage is this process's own total, which includes allocations prosper's own counter
// (gpu_memory_budget) cannot see, such as swapchain images and driver-internal objects. The larger of
// the two usage figures is taken so neither blind spot can inflate the texture share. The texture
// cache's own bytes are subtracted back out, which is what keeps the rule stable: a growing cache
// raises usage and texture_bytes equally and does not shrink its own budget.
//
// UNIFIED MEMORY. On an integrated GPU the device-local heap is system RAM (43 GiB on this project's
// development APU), so "80% of the budget" would hand textures tens of GiB and starve the host.
// There the old heuristic stays the ceiling -- which also makes the default behaviour on such a
// machine identical to before -- and the live headroom can only LOWER it. "Unified" is decided by
// the caller from VkPhysicalDeviceType (anything that is not a discrete GPU), NOT from a
// DEVICE_LOCAL|HOST_VISIBLE memory type on the big heap: resizable BAR exposes exactly that type on
// discrete cards too, and would misclassify every modern discrete GPU as an APU.
//
// FLOOR. 1 GiB -- the old rule's minimum -- always wins, including over the 75% ceiling on a very
// small heap, so no device is given less than the historical default. Under real pressure the driver
// pages rather than failing, and the eviction path (not this number) is what bounds the cache.
//
// Without the extension (have_budget == false) the answer is the old heuristic, unchanged.
// CONFIDENCE: HIGH for the arithmetic; MED for the 80% target, which is a headroom choice (it leaves
// room for render-target growth between re-evaluations and for driver padding), not a measurement.
#pragma once
#include <cstdint>

namespace prosper::gpu {

inline constexpr uint64_t kTextureCacheMiB = 1024ull * 1024ull;
inline constexpr uint64_t kTextureCacheGiB = 1024ull * kTextureCacheMiB;
inline constexpr uint64_t kTextureCacheFloorBytes = 1ull * kTextureCacheGiB;
inline constexpr uint64_t kTextureCacheHeuristicCeilingBytes = 4ull * kTextureCacheGiB;
inline constexpr uint64_t kTextureCacheHeuristicDivisor = 8;
inline constexpr uint64_t kTextureCacheTargetPercentOfBudget = 80;
inline constexpr uint64_t kTextureCacheCeilingPercentOfHeap = 75;

struct TextureCacheBudgetInputs {
    uint64_t heap_size = 0;      // VkMemoryHeap::size of the heap persistent textures allocate from
    bool have_budget = false;    // VK_EXT_memory_budget enabled and its figures valid
    uint64_t heap_budget = 0;    // VkPhysicalDeviceMemoryBudgetPropertiesEXT::heapBudget[heap]
    uint64_t heap_usage = 0;     // ...::heapUsage[heap] (this process, as the driver counts it)
    uint64_t prosper_held = 0;   // gpu_memory_budget's device_bytes_held(heap)
    uint64_t texture_bytes = 0;  // bytes the persistent texture cache holds now (subset of both)
    bool unified = false;        // the heap is system RAM (non-discrete device)
};

enum class TextureCacheBudgetSource : uint8_t {
    heuristic,  // no live budget: clamp(heap / 8, 1 GiB, 4 GiB)
    live,       // the live headroom figure was used as-is
    ceiling,    // discrete: clipped to 75% of the heap
    unified,    // unified memory: clipped to the old heuristic
    floor,      // raised to the 1 GiB floor
};

struct TextureCacheBudget {
    uint64_t bytes = 0;
    TextureCacheBudgetSource source = TextureCacheBudgetSource::heuristic;
    uint64_t non_texture = 0;  // the non-texture usage the answer subtracted (0 for heuristic)
    uint64_t target = 0;       // 80% of heap_budget (0 for heuristic)
};

// The old rule, kept verbatim: the fallback without the extension, and the unified-memory ceiling.
uint64_t texture_cache_heuristic_budget(uint64_t heap_size);

// The policy above. Pure: no Vulkan, no environment, no clock.
TextureCacheBudget texture_cache_budget(const TextureCacheBudgetInputs& in);

// Whether a re-evaluated budget differs enough from the applied one to replace it (and log it):
// by at least max(128 MiB, current / 16). Small drift in the driver's figures would otherwise
// re-log every period and nudge the eviction bound back and forth for nothing.
bool texture_cache_budget_changed_materially(uint64_t current, uint64_t proposed);

const char* texture_cache_budget_source_name(TextureCacheBudgetSource source);

// Which heap the budget is sized from. `recorded_heap` is the heap of the memory type the first
// retained texture image actually got (UINT32_MAX before any). It is used only when it is a
// DEVICE-LOCAL heap; otherwise -- not recorded yet, out of range, or a host heap because no
// device-local type accepted the image (memory_type_select.hpp's fallback; before #3888 the
// renderer took the first compatible type, which a driver may put on a system heap) -- the answer
// is the largest device-local heap, which is what the pre-#3873 rule always used. Sizing
// from a host heap as if it were discrete VRAM would hand the cache most of system RAM, exactly the
// starvation the unified cap exists to prevent. UINT32_MAX when no heap is device-local.
uint32_t texture_cache_budget_heap(const uint64_t* heap_sizes, const bool* heap_device_local,
                                   uint32_t heap_count, uint32_t recorded_heap);

// The limit the cache actually enforces: PROSPER_BACKEND_TEXTURE_CACHE_MB, when set, is an absolute
// override of everything above; otherwise the policy's figure, or the historical 1 GiB before the
// device has been sized (policy_bytes == 0).
uint64_t resolve_texture_cache_limit(bool have_override, uint64_t override_bytes,
                                     uint64_t policy_bytes);

}  // namespace prosper::gpu
