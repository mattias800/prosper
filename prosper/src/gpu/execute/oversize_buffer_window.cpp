#include "gpu/execute/oversize_buffer_window.hpp"

#include <map>
#include <mutex>
#include <utility>

// Implemented by the HLE memory tracker (hle/memory/renderer_tracked_mapping.hpp). Declared here rather
// than included so the gpu layer does not gain an include edge into hle/.
extern "C" uint64_t prosper_renderer_guest_mapped_readable_prefix(uint64_t addr, uint64_t bytes);
extern "C" int prosper_renderer_guest_address_tracked(uint64_t addr);
extern "C" uint64_t prosper_renderer_tracked_mapping_generation();

namespace prosper::gpu {

bool clamp_oversized_buffer_window(uint64_t base, uint64_t window_bytes, uint64_t stride,
                                   uint64_t cap, const MappedMemoryProbe& probe,
                                   uint64_t& clamped_bytes) {
    if (!stride) stride = 1;
    if (window_bytes <= cap || !base || base > UINT64_MAX - window_bytes) return false;
    if (!probe.readable_prefix || !probe.any_mapped) return false;

    const uint64_t prefix = probe.readable_prefix(base, window_bytes);
    if (!prefix || prefix > cap || prefix > window_bytes) return false;

    // Nothing else in the window may be mapped: an island past the leading run is memory the shader
    // could legitimately read, and clamping would turn it into silent zeros.
    const uint64_t end = base + window_bytes;
    uint64_t addr = (base + prefix + kMappingGranule - 1) / kMappingGranule * kMappingGranule;
    for (; addr < end; addr += kMappingGranule)
        if (probe.any_mapped(addr, kMappingGranule)) return false;

    const uint64_t clamped = prefix - prefix % stride;
    if (clamped < stride) return false;
    clamped_bytes = clamped;
    return true;
}

namespace {

struct CachedClamp {
    uint64_t generation = 0;
    bool ok = false;
    uint64_t clamped = 0;
};

MappedMemoryProbe live_probe() {
    MappedMemoryProbe probe;
    probe.readable_prefix = [](uint64_t addr, uint64_t bytes) {
        return prosper_renderer_guest_mapped_readable_prefix(addr, bytes);
    };
    // Probing both ends of the granule covers a mapping that starts inside it.
    probe.any_mapped = [](uint64_t addr, uint64_t bytes) {
        return prosper_renderer_guest_address_tracked(addr) != 0 ||
               prosper_renderer_guest_address_tracked(addr + bytes - 1) != 0;
    };
    return probe;
}

} // namespace

bool clamp_oversized_buffer_window_live(uint64_t base, uint64_t window_bytes, uint64_t stride,
                                        uint64_t cap, uint64_t& clamped_bytes) {
    if (window_bytes <= cap) return false;
    static std::mutex mutex;
    static std::map<std::pair<uint64_t, uint64_t>, CachedClamp> cache;
    const uint64_t generation = prosper_renderer_tracked_mapping_generation();
    const auto key = std::make_pair(base, window_bytes);
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto found = cache.find(key);
        if (found != cache.end() && found->second.generation == generation) {
            clamped_bytes = found->second.clamped;
            return found->second.ok;
        }
    }
    uint64_t clamped = 0;
    const bool ok = clamp_oversized_buffer_window(base, window_bytes, stride, cap, live_probe(), clamped);
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (cache.size() > 256) cache.clear();   // bounded: a handful of distinct windows is typical
        cache[key] = {generation, ok, clamped};
    }
    if (ok) clamped_bytes = clamped;
    return ok;
}

} // namespace prosper::gpu
