// memory_type_select — which memory type a GPU-only resource gets (#3888), over synthetic
// VkPhysicalDeviceMemoryProperties. No device is created; nothing here calls Vulkan.
//
// Every expected index below is read off the layout by hand, not computed by the function under
// test. The layouts: a driver listing a flag-less host type first (the #3888 failure), an
// NVIDIA-like discrete layout with and without resizable BAR (the flag-less type 0 on the system heap
// is REPORTED for NVIDIA, not verified on hardware here), an AMD discrete layout with and without
// resizable BAR, this project's Strix Halo APU exactly as `vulkaninfo` reported it on 2026-09-28,
// and a software rasterizer with one all-flags type. The mutation arms this file was checked against
// are listed in the PR.
#include "gpu/diagnostics/memory_placement_log.hpp"
#include "gpu/memory/memory_type_select.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <utility>

using namespace prosper::gpu;

static int fails = 0;
static void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "[ok]  " : "[FAIL]", what);
    if (!ok) ++fails;
}

static constexpr VkMemoryPropertyFlags DL = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
static constexpr VkMemoryPropertyFlags HV = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
static constexpr VkMemoryPropertyFlags HC = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
static constexpr VkMemoryPropertyFlags HCACHED = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
static constexpr VkMemoryPropertyFlags AMD_DC = VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD;
static constexpr VkMemoryPropertyFlags AMD_DU = VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD;
static constexpr uint64_t MiB = 1024ull * 1024ull;
static constexpr uint64_t GiB = 1024ull * MiB;

struct Heap { uint64_t size; bool device_local; };
struct Type { VkMemoryPropertyFlags flags; uint32_t heap; };

static VkPhysicalDeviceMemoryProperties layout(std::initializer_list<Heap> heaps,
                                               std::initializer_list<Type> types) {
    VkPhysicalDeviceMemoryProperties p{};
    for (const Heap& h : heaps) {
        p.memoryHeaps[p.memoryHeapCount].size = h.size;
        p.memoryHeaps[p.memoryHeapCount].flags = h.device_local ? VK_MEMORY_HEAP_DEVICE_LOCAL_BIT : 0;
        ++p.memoryHeapCount;
    }
    for (const Type& t : types) {
        p.memoryTypes[p.memoryTypeCount].propertyFlags = t.flags;
        p.memoryTypes[p.memoryTypeCount].heapIndex = t.heap;
        ++p.memoryTypeCount;
    }
    return p;
}

static uint32_t bits(std::initializer_list<uint32_t> types) {
    uint32_t b = 0;
    for (uint32_t t : types) b |= 1u << t;
    return b;
}

int main() {
    // ---- 1. The #3888 failure: a flag-less host type listed first ---------------------------------
    {
        const auto p = layout({{8 * GiB, true}, {32 * GiB, false}},
                              {{0, 1},     // 0: no flags, system heap
                               {DL, 0}});  // 1: VRAM
        check(select_gpu_only_memory_type(p, bits({0, 1})) == 1,
              "host type first: a GPU-only image takes the device-local type 1, not type 0");
        check(find_memory_type(p, bits({0, 1}), 0) == 0,
              "host type first: the pre-#3888 rule (first compatible) would have taken type 0");
        check(select_gpu_only_memory_type(p, bits({0})) == 0,
              "host type first: no device-local type allowed -> falls back to the host type");
        check(select_gpu_only_memory_type(p, bits({1})) == 1,
              "host type first: only the device-local type allowed -> it");
        check(select_gpu_only_memory_type(p, 0) == UINT32_MAX,
              "no allowed type at all -> UINT32_MAX (the allocation is refused, as before)");
    }

    // ---- 2. NVIDIA-like discrete layout, no resizable BAR ----------------------------------------
    // heaps: 0 = VRAM 8 GiB, 1 = system 32 GiB, 2 = 256 MiB BAR window.
    // types: 0 none/sys, 1 DL/VRAM, 2 HV|HC/sys, 3 HV|HC|CACHED/sys, 4 DL|HV|HC/BAR.
    {
        const auto p = layout({{8 * GiB, true}, {32 * GiB, false}, {256 * MiB, true}},
                              {{0, 1}, {DL, 0}, {HV | HC, 1}, {HV | HC | HCACHED, 1},
                               {DL | HV | HC, 2}});
        const uint32_t image = bits({0, 1, 4});
        check(select_gpu_only_memory_type(p, image) == 1,
              "NVIDIA-like: an optimal image goes to type 1 (VRAM heap)...");
        check(select_gpu_only_memory_type(p, image) != 4,
              "NVIDIA-like: ...not to the 256 MiB BAR type 4, although that is device-local too");
        check(find_memory_type(p, image, 0) == 0,
              "NVIDIA-like: the pre-#3888 rule would have put it in the system-heap type 0");
        const uint32_t buffer = bits({0, 1, 2, 3, 4});
        check(find_memory_type(p, buffer, HV | HC) == 2,
              "NVIDIA-like: explicit HOST_VISIBLE|COHERENT staging stays on host type 2");
        check(select_memory_type(p, buffer, HV | HC, HCACHED) == 3,
              "NVIDIA-like: readback preferring CACHED takes type 3");
        check(select_memory_type(p, bits({0, 1, 2, 4}), HV | HC, HCACHED) == 2,
              "NVIDIA-like: CACHED not allowed -> still a HOST_VISIBLE|COHERENT type (2), never DL-only");
    }

    // ---- 3. NVIDIA-like with resizable BAR: the DL|HV type now sits on the VRAM heap -------------
    {
        const auto p = layout({{16 * GiB, true}, {32 * GiB, false}},
                              {{0, 1}, {DL, 0}, {HV | HC, 1}, {HV | HC | HCACHED, 1},
                               {DL | HV | HC, 0}});
        check(select_gpu_only_memory_type(p, bits({0, 1, 4})) == 1,
              "NVIDIA-like + ReBAR: an optimal image takes plain DL type 1");
        check(select_gpu_only_memory_type(p, bits({0, 4})) == 4,
              "NVIDIA-like + ReBAR: only host type 0 and DL|HV type 4 allowed -> 4 (device-local)");
        check(find_memory_type(p, bits({0, 1, 2, 3, 4}), HV | HC) == 2,
              "NVIDIA-like + ReBAR: explicit host-visible staging still takes system type 2, not "
              "the device-local BAR type");
    }

    // ---- 4. AMD discrete (RADV-style), no resizable BAR -----------------------------------------
    // types: 0 DL/VRAM, 1 HV|HC/sys, 2 DL|HV|HC/BAR 256 MiB, 3 HV|HC|CACHED/sys.
    {
        const auto p = layout({{16 * GiB, true}, {32 * GiB, false}, {256 * MiB, true}},
                              {{DL, 0}, {HV | HC, 1}, {DL | HV | HC, 2}, {HV | HC | HCACHED, 1}});
        const uint32_t image = bits({0, 1, 2, 3});
        check(select_gpu_only_memory_type(p, image) == 0,
              "AMD dGPU: an image takes type 0 (VRAM)");
        check(select_gpu_only_memory_type(p, image) == find_memory_type(p, image, 0),
              "AMD dGPU: identical to the pre-#3888 first-compatible choice (no behaviour change)");
        check(select_gpu_only_memory_type(p, bits({1, 2, 3})) == 2,
              "AMD dGPU: type 0 not allowed -> the device-local BAR type 2 before host type 1");
        check(find_memory_type(p, image, HV | HC) == 1,
              "AMD dGPU: explicit host-visible staging takes host type 1");
    }

    // ---- 5. AMD discrete with resizable BAR -------------------------------------------------------
    {
        const auto p = layout({{16 * GiB, true}, {32 * GiB, false}},
                              {{DL, 0}, {HV | HC, 1}, {DL | HV | HC, 0}, {HV | HC | HCACHED, 1}});
        check(select_gpu_only_memory_type(p, bits({0, 1, 2, 3})) == 0,
              "AMD dGPU + ReBAR: an image takes plain DL type 0");
        check(select_gpu_only_memory_type(p, bits({1, 2})) == 2,
              "AMD dGPU + ReBAR: plain DL not allowed -> DL|HV type 2, not host type 1");
    }

    // ---- 6. This project's APU (RADV Strix Halo), from vulkaninfo on 2026-09-28 ------------------
    // heap 0: 21.58 GiB, NOT device-local (GTT). heap 1: 43.16 GiB, device-local.
    // Types 1, 4, 6 are the 32-bit-address variants and accept no image. 7-10 are AMD
    // device-coherent/uncached variants.
    {
        const auto p = layout({{23172218880ull, false}, {46344445952ull, true}},
                              {{DL, 1}, {DL, 1}, {HV | HC, 0}, {DL | HV | HC, 1}, {DL | HV | HC, 1},
                               {HV | HC | HCACHED, 0}, {HV | HC | HCACHED, 0}, {DL | AMD_DC | AMD_DU, 1},
                               {HV | HC | AMD_DC | AMD_DU, 0}, {DL | HV | HC | AMD_DC | AMD_DU, 1},
                               {HV | HC | HCACHED | AMD_DC | AMD_DU, 0}});
        const uint32_t image = bits({0, 2, 3, 5, 7, 8, 9, 10});   // "usable for: color images"
        check(select_gpu_only_memory_type(p, image) == 0,
              "Strix Halo APU: an image takes type 0 (DL, heap 1)");
        check(select_gpu_only_memory_type(p, image) == find_memory_type(p, image, 0),
              "Strix Halo APU: identical to the pre-#3888 choice (this box must not change)");
        check(select_gpu_only_memory_type(p, bits({2, 3, 5})) == 3,
              "Strix Halo APU: type 0 not allowed -> DL|HV type 3 on heap 1, not GTT type 2");
        check(find_memory_type(p, image, HV | HC) == 2,
              "Strix Halo APU: explicit host-visible staging still takes GTT type 2");
        check(select_memory_type(p, image, HV | HC, HCACHED) == 5,
              "Strix Halo APU: readback preferring CACHED takes type 5");
    }

    // ---- 7. Software rasterizer: one type with every host flag ----------------------------------
    {
        const auto p = layout({{8 * GiB, true}}, {{DL | HV | HC | HCACHED, 0}});
        check(select_gpu_only_memory_type(p, bits({0})) == 0, "single all-flags type: chosen");
        check(find_memory_type(p, bits({0}), HV | HC) == 0, "single all-flags type: staging too");
    }

    // ---- 8. Required flags are ALL required, and a type the bits exclude is never chosen --------
    {
        // A non-coherent host type (HV only) is listed before the coherent one, as the ordering
        // rule requires; a coherent request must skip it.
        const auto p = layout({{8 * GiB, true}, {32 * GiB, false}},
                              {{DL, 0}, {HV, 1}, {HV | HC, 1}});
        check(find_memory_type(p, bits({0, 1, 2}), HV | HC) == 2,
              "HOST_VISIBLE|COHERENT skips a HOST_VISIBLE-only type");
        check(find_memory_type(p, bits({0, 1}), HV | HC) == UINT32_MAX,
              "coherent type excluded by the bits -> UINT32_MAX, not a non-coherent substitute");
        check(select_memory_type(p, bits({0, 1, 2}), HV | HC, DL) == 2,
              "a preference that no required-satisfying type has is dropped, never the requirement");
        check(select_gpu_only_memory_type(p, bits({1, 2})) == 1,
              "fallback with no DL allowed takes the FIRST allowed type");
    }

    // ---- 9. An out-of-range memoryTypeCount is clamped, never read past the array ----------------
    {
        auto p = layout({{8 * GiB, true}}, {{0, 0}});
        p.memoryTypeCount = 40;   // > VK_MAX_MEMORY_TYPES
        check(memory_type_count(p) == VK_MAX_MEMORY_TYPES, "memoryTypeCount clamps to 32");
        check(find_memory_type(p, 0x80000000u, 0) == 31, "bit 31 is still reachable after clamping");
    }

    // ---- 10. The placement line's description, and that logging chooses what selection chooses --
    {
        const auto p = layout({{8 * GiB, true}, {32 * GiB, false}}, {{0, 1}, {DL, 0}, {DL | HV | HC, 0}});
        char d[160];
        describe_memory_type(p, 1, d, sizeof(d));
        check(std::strcmp(d, "type 1 (DEVICE_LOCAL) heap 0 (device-local, 8192 MiB)") == 0,
              "describe: device-local type");
        describe_memory_type(p, 0, d, sizeof(d));
        check(std::strcmp(d, "type 0 (no flags) heap 1 (host, 32768 MiB)") == 0,
              "describe: flag-less host type");
        describe_memory_type(p, 2, d, sizeof(d));
        check(std::strcmp(d, "type 2 (DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT) heap 0 "
                             "(device-local, 8192 MiB)") == 0,
              "describe: flag list");
        describe_memory_type(p, UINT32_MAX, d, sizeof(d));
        check(std::strcmp(d, "none") == 0, "describe: no type");
        check(choose_gpu_only_memory_type(GpuOnlyMemoryClass::SampledTexture, p, bits({0, 1})) == 1,
              "choose_gpu_only_memory_type (logged) returns the device-local type");
        check(choose_gpu_only_memory_type(GpuOnlyMemoryClass::SampledTexture, p, bits({0})) == 0,
              "choose_gpu_only_memory_type falls back too (second call, already logged)");
    }

    std::printf("%s (%d failure%s)\n", fails ? "FAIL" : "PASS", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
