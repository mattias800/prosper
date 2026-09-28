// memory_type_select — which memory type a GPU-only resource gets (#3888), over synthetic
// VkPhysicalDeviceMemoryProperties. No device is created; nothing here calls Vulkan.
//
// Every expected index below is read off the layout by hand, not computed by the function under
// test. The layouts: a driver listing a flag-less host type first (the #3888 failure), an
// NVIDIA-like discrete layout with and without resizable BAR (the flag-less type 0 on the system heap
// is REPORTED for NVIDIA, not verified on hardware here), an AMD discrete layout with and without
// resizable BAR, this project's Strix Halo APU exactly as `vulkaninfo` reported it on 2026-09-28,
// and a software rasterizer with one all-flags type. Sections 11-14 cover #3897: the fallback order,
// a fake allocator that injects VK_ERROR_OUT_OF_DEVICE_MEMORY on its first N attempts, the
// gpu-memory-off-device ledger counters, and AMD device-coherent exclusion. The mutation arms this
// file was checked against are listed in the PRs.
#include "diagnostics/perf/perf_ledger.hpp"
#include "gpu/diagnostics/memory_placement_log.hpp"
#include "gpu/memory/memory_type_select.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
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

    // ---- 10. The placement line's description ---------------------------------------------------
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
    }

    // ---- 11. Candidate order: preferred first, then the rest, each by index; coherent never ------
    {
        // NVIDIA-like, no ReBAR: 0 none/sys, 1 DL/VRAM, 2 HV|HC/sys, 3 HV|HC|CACHED/sys, 4 DL|HV|HC/BAR.
        const auto p = layout({{8 * GiB, true}, {32 * GiB, false}, {256 * MiB, true}},
                              {{0, 1}, {DL, 0}, {HV | HC, 1}, {HV | HC | HCACHED, 1},
                               {DL | HV | HC, 2}});
        uint32_t c[VK_MAX_MEMORY_TYPES] = {};
        const uint32_t n = memory_type_candidates(p, bits({0, 1, 2, 3, 4}), 0, DL, c);
        check(n == 5 && c[0] == 1 && c[1] == 4 && c[2] == 0 && c[3] == 2 && c[4] == 3,
              "candidates (GPU-only): DL types 1, 4 first, then host types 0, 2, 3");
        const uint32_t h = memory_type_candidates(p, bits({0, 1, 2, 3, 4}), HV | HC, HCACHED, c);
        check(h == 3 && c[0] == 3 && c[1] == 2 && c[2] == 4,
              "candidates (host-visible, prefer CACHED): 3, then 2, 4 -- never 0 or 1");
        check(memory_type_candidates(p, 0, 0, DL, c) == 0, "no allowed type -> no candidate");
    }
    {
        // Strix Halo APU (section 6's layout): types 7-10 are AMD device-coherent/uncached.
        const auto p = layout({{23172218880ull, false}, {46344445952ull, true}},
                              {{DL, 1}, {DL, 1}, {HV | HC, 0}, {DL | HV | HC, 1}, {DL | HV | HC, 1},
                               {HV | HC | HCACHED, 0}, {HV | HC | HCACHED, 0}, {DL | AMD_DC | AMD_DU, 1},
                               {HV | HC | AMD_DC | AMD_DU, 0}, {DL | HV | HC | AMD_DC | AMD_DU, 1},
                               {HV | HC | HCACHED | AMD_DC | AMD_DU, 0}});
        uint32_t c[VK_MAX_MEMORY_TYPES] = {};
        const uint32_t n = memory_type_candidates(p, bits({0, 2, 3, 5, 7, 8, 9, 10}), 0, DL, c);
        check(n == 4 && c[0] == 0 && c[1] == 3 && c[2] == 2 && c[3] == 5,
              "APU candidates: 0, 3 (device-local), 2, 5 (host); coherent 7-10 never listed");
        check(select_gpu_only_memory_type(p, bits({2, 7})) == 2,
              "APU: a mask whose only device-local type is coherent (7) takes host type 2, not 7");
        check(select_gpu_only_memory_type(p, bits({7, 9})) == UINT32_MAX,
              "APU: a mask of coherent types only -> no type (allocating one needs a feature "
              "prosper does not enable)");
        check(find_memory_type(p, bits({8, 2}), HV | HC) == 2 &&
                  find_memory_type(p, bits({8}), HV | HC) == UINT32_MAX,
              "APU: explicit host-visible requests skip coherent type 8 too");
    }

    // ---- 12. allocate_with_memory_type_fallback over a fake allocator -------------------------
    // The fake fails the first `fail_first` attempts with `failure` and records each type tried.
    struct Fake {
        uint32_t fail_first = 0;
        VkResult failure = VK_ERROR_OUT_OF_DEVICE_MEMORY;
        uint32_t tried[VK_MAX_MEMORY_TYPES] = {};
        uint32_t calls = 0;
        VkResult operator()(uint32_t type) {
            tried[calls] = type;
            return calls++ < fail_first ? failure : VK_SUCCESS;
        }
    };
    {
        // NVIDIA-like, no ReBAR (as above). Image mask {0, 1, 4}: order 1, 4, 0.
        const auto p = layout({{8 * GiB, true}, {32 * GiB, false}, {256 * MiB, true}},
                              {{0, 1}, {DL, 0}, {HV | HC, 1}, {HV | HC | HCACHED, 1},
                               {DL | HV | HC, 2}});
        const uint32_t image = bits({0, 1, 4});
        Fake ok;
        auto o = allocate_with_memory_type_fallback(p, image, 0, DL, std::ref(ok));
        check(o.ok() && o.type == 1 && o.attempts == 1 && !o.fell_back() && ok.calls == 1,
              "fallback: no failure -> the preferred type, one attempt");
        Fake one{1};
        o = allocate_with_memory_type_fallback(p, image, 0, DL, std::ref(one));
        check(o.ok() && o.type == 4 && o.preferred == 1 && o.fell_back() && one.tried[0] == 1 &&
                  one.tried[1] == 4,
              "fallback: VRAM type 1 out of memory -> the next device-local type 4");
        Fake two{2};
        o = allocate_with_memory_type_fallback(p, image, 0, DL, std::ref(two));
        check(o.ok() && o.type == 0 && o.attempts == 3 && two.tried[2] == 0,
              "fallback: both device-local types out of memory -> host type 0");
        Fake all{99};
        o = allocate_with_memory_type_fallback(p, image, 0, DL, std::ref(all));
        check(!o.ok() && o.result == VK_ERROR_OUT_OF_DEVICE_MEMORY && o.attempts == 3 &&
                  o.type == 1 && all.calls == 3,
              "fallback: every candidate out of memory -> VK_ERROR_OUT_OF_DEVICE_MEMORY surfaced, "
              "type reports the preferred one, nothing tried twice");
        Fake host_oom{1, VK_ERROR_OUT_OF_HOST_MEMORY};
        o = allocate_with_memory_type_fallback(p, image, 0, DL, std::ref(host_oom));
        check(!o.ok() && o.result == VK_ERROR_OUT_OF_HOST_MEMORY && host_oom.calls == 1,
              "fallback: a non-device-OOM error is surfaced at once, never retried elsewhere");
        Fake lost{1, VK_ERROR_DEVICE_LOST};
        o = allocate_with_memory_type_fallback(p, image, 0, DL, std::ref(lost));
        check(!o.ok() && o.result == VK_ERROR_DEVICE_LOST && lost.calls == 1,
              "fallback: VK_ERROR_DEVICE_LOST is surfaced at once");
        Fake none;
        o = allocate_with_memory_type_fallback(p, 0, 0, DL, std::ref(none));
        check(!o.ok() && o.result == kNoCompatibleMemoryType && none.calls == 0 &&
                  o.type == UINT32_MAX,
              "fallback: no candidate -> kNoCompatibleMemoryType without calling the allocator");

        // Required flags hold on every attempt: host-visible staging (mask allows all five).
        Fake staging{1};
        o = allocate_with_memory_type_fallback(p, bits({0, 1, 2, 3, 4}), HV | HC, 0,
                                               std::ref(staging));
        check(o.ok() && o.type == 3 && staging.tried[0] == 2,
              "fallback: HOST_VISIBLE|COHERENT type 2 out of memory -> host-visible type 3");
        Fake staging_all{99};
        o = allocate_with_memory_type_fallback(p, bits({0, 1, 2, 3, 4}), HV | HC, 0,
                                               std::ref(staging_all));
        bool host_visible_only = staging_all.calls == 3;
        for (uint32_t i = 0; i < staging_all.calls; ++i)
            host_visible_only &= (p.memoryTypes[staging_all.tried[i]].propertyFlags & (HV | HC)) ==
                                 (HV | HC);
        check(!o.ok() && host_visible_only,
              "fallback: a host-visible request never falls to a non-host-visible type (0 or 1)");
    }

    // ---- 13. allocate_gpu_only: log, count, alarm, forced OOM ----------------------------------
    {
        using namespace prosper::diagnostics::perf;
        auto counter = [](Counter c) { return ledger().counters[static_cast<size_t>(c)].load(); };
        auto per_class = [](GpuOnlyMemoryClass c) {
            return ledger().gpu_memory_off_device[static_cast<size_t>(c)].load();
        };
        const auto p = layout({{8 * GiB, true}, {32 * GiB, false}}, {{0, 1}, {DL, 0}, {DL | HV | HC, 0}});
        const uint32_t image = bits({0, 1});   // DL type 1, then host type 0
        const uint64_t off0 = counter(Counter::GpuMemoryOffDevice);
        const uint64_t fb0 = counter(Counter::GpuMemoryFallbacks);
        const uint64_t bytes0 = counter(Counter::GpuMemoryOffDeviceBytes);

        Fake fine;
        auto o = allocate_gpu_only(GpuOnlyMemoryClass::DepthTarget, p, image, 4096, std::ref(fine));
        check(o.ok() && o.type == 1 && counter(Counter::GpuMemoryOffDevice) == off0 &&
                  counter(Counter::GpuMemoryFallbacks) == fb0,
              "gpu-only: a device-local placement counts nothing");

        Fake oom{1};
        o = allocate_gpu_only(GpuOnlyMemoryClass::DepthTarget, p, image, 1 << 20, std::ref(oom));
        check(o.ok() && o.type == 0 && counter(Counter::GpuMemoryFallbacks) == fb0 + 1 &&
                  counter(Counter::GpuMemoryOffDevice) == off0 + 1 &&
                  counter(Counter::GpuMemoryOffDeviceBytes) == bytes0 + (1 << 20) &&
                  per_class(GpuOnlyMemoryClass::DepthTarget) >= 1 &&
                  std::strcmp(ledger().gpu_memory_class_names[static_cast<size_t>(
                                  GpuOnlyMemoryClass::DepthTarget)].load(),
                              "depth-target") == 0,
              "gpu-only: VRAM out of memory -> host type 0, one fallback, one off-device placement "
              "of 1 MiB, named depth-target");

        Fake forced;
        o = allocate_gpu_only(GpuOnlyMemoryClass::PresentSlot, p, image, 64, std::ref(forced), true);
        check(o.ok() && o.type == 0 && forced.calls == 1 && forced.tried[0] == 0 &&
                  counter(Counter::GpuMemoryOffDevice) == off0 + 2,
              "gpu-only: forced OOM skips every device-local type without calling the allocator");

        Fake forced_dl_only;
        o = allocate_gpu_only(GpuOnlyMemoryClass::PresentSlot, p, bits({1, 2}), 64,
                              std::ref(forced_dl_only), true);
        check(!o.ok() && o.result == VK_ERROR_OUT_OF_DEVICE_MEMORY && forced_dl_only.calls == 0,
              "gpu-only: forced OOM on a device-local-only mask is the all-failed path");

        // A resource that allows no device-local type on a device that HAS one: counted (no fallback).
        Fake host_only;
        const uint64_t fb_before = counter(Counter::GpuMemoryFallbacks);
        o = allocate_gpu_only(GpuOnlyMemoryClass::SampledTexture, p, bits({0}), 64, std::ref(host_only));
        check(o.ok() && o.type == 0 && counter(Counter::GpuMemoryOffDevice) == off0 + 3 &&
                  counter(Counter::GpuMemoryFallbacks) == fb_before && !o.fell_back(),
              "gpu-only: no device-local type allowed -> off-device counted, not a fallback");

        // A device with no device-local memory at all: host is the only placement, not an alarm.
        const auto soft = layout({{8 * GiB, false}}, {{HV | HC, 0}});
        Fake sw;
        o = allocate_gpu_only(GpuOnlyMemoryClass::SampledTexture, soft, bits({0}), 64, std::ref(sw));
        check(o.ok() && counter(Counter::GpuMemoryOffDevice) == off0 + 3,
              "gpu-only: a device without device-local memory counts nothing");

        // Coherent exclusion reaches the GPU-only path too.
        const auto coh = layout({{8 * GiB, true}, {32 * GiB, false}},
                                {{DL, 0}, {HV | HC, 1}, {DL | AMD_DC | AMD_DU, 0}});
        Fake c;
        o = allocate_gpu_only(GpuOnlyMemoryClass::ComputeImage, coh, bits({1, 2}), 64, std::ref(c));
        check(o.ok() && o.type == 1 && c.calls == 1 && c.tried[0] == 1,
              "gpu-only: a device-coherent type is never tried (host type 1 is)");
    }

    // ---- 14. PROSPER_GPU_MEM_FORCE_OOM class parsing ------------------------------------------
    {
        const uint32_t all = (1u << static_cast<uint32_t>(GpuOnlyMemoryClass::Count)) - 1u;
        check(parse_force_oom_classes(nullptr) == 0 && parse_force_oom_classes("") == 0,
              "force-oom: unset -> nothing armed");
        check(parse_force_oom_classes("all") == all, "force-oom: all");
        check(parse_force_oom_classes("sampled-texture") ==
                  1u << static_cast<uint32_t>(GpuOnlyMemoryClass::SampledTexture),
              "force-oom: one class by its [mem-placement] name");
        check(parse_force_oom_classes("color-target,depth-target") ==
                  ((1u << static_cast<uint32_t>(GpuOnlyMemoryClass::ColorTarget)) |
                   (1u << static_cast<uint32_t>(GpuOnlyMemoryClass::DepthTarget))),
              "force-oom: a list, and color-target does not also arm color-target-1");
        check(parse_force_oom_classes("bogus") == 0, "force-oom: an unknown name arms nothing");
    }

    // ---- 15. #3902: a pool releases its idle cache and retries the SAME type once on OOM -------
    {
        int allocs = 0, releases = 0;
        // Driver OOM first, success after the pool gave memory back.
        VkResult r = allocate_releasing_pool_on_oom(
            [&] { return ++allocs == 1 ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_SUCCESS; },
            [&] { ++releases; return size_t{3}; });
        check(r == VK_SUCCESS && allocs == 2 && releases == 1,
              "pool-oom: OOM -> release cache -> same type retried once -> success");
        // Nothing cached: no pointless retry, the OOM goes on to the type fallback.
        allocs = releases = 0;
        r = allocate_releasing_pool_on_oom(
            [&] { ++allocs; return VK_ERROR_OUT_OF_DEVICE_MEMORY; },
            [&] { ++releases; return size_t{0}; });
        check(r == VK_ERROR_OUT_OF_DEVICE_MEMORY && allocs == 1 && releases == 1,
              "pool-oom: empty cache -> no retry, OOM returned");
        // Still OOM after the release: retried exactly once, OOM returned (fallback continues).
        allocs = releases = 0;
        r = allocate_releasing_pool_on_oom(
            [&] { ++allocs; return VK_ERROR_OUT_OF_DEVICE_MEMORY; },
            [&] { ++releases; return size_t{5}; });
        check(r == VK_ERROR_OUT_OF_DEVICE_MEMORY && allocs == 2 && releases == 1,
              "pool-oom: still out after release -> exactly one retry");
        // Other errors and success: the cache is left alone.
        allocs = releases = 0;
        r = allocate_releasing_pool_on_oom([&] { ++allocs; return VK_ERROR_OUT_OF_HOST_MEMORY; },
                                           [&] { ++releases; return size_t{5}; });
        check(r == VK_ERROR_OUT_OF_HOST_MEMORY && allocs == 1 && releases == 0,
              "pool-oom: OUT_OF_HOST_MEMORY is returned at once, cache kept");
        allocs = releases = 0;
        r = allocate_releasing_pool_on_oom([&] { ++allocs; return VK_SUCCESS; },
                                           [&] { ++releases; return size_t{5}; });
        check(r == VK_SUCCESS && allocs == 1 && releases == 0, "pool-oom: success keeps the cache");
        // Through the type fallback: device-local type 0 OOMs, the pool frees, type 0 is retried
        // and succeeds -- the host type is never reached.
        const auto nv = layout({{8 * GiB, true}, {32 * GiB, false}}, {{DL, 0}, {HV | HC, 1}});
        int tried_host = 0, calls0 = 0, released = 0;
        const MemoryAllocationOutcome o = allocate_with_memory_type_fallback(
            nv, bits({0, 1}), 0, DL, [&](uint32_t type) {
                if (type == 1) { ++tried_host; return VK_SUCCESS; }
                return allocate_releasing_pool_on_oom(
                    [&] { return ++calls0 == 1 ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_SUCCESS; },
                    [&] { ++released; return size_t{1}; });
            });
        check(o.ok() && o.type == 0 && !o.fell_back() && tried_host == 0 && released == 1,
              "pool-oom: released VRAM is used before the fallback reaches system memory");
    }

    std::printf("%s (%d failure%s)\n", fails ? "FAIL" : "PASS", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
