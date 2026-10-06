#include "gpu/execute/oversize_buffer_window.hpp"

#include "gpu/agc/agc_shader_layout.hpp"
#include "gpu/execute/gpu_execute.hpp"

#include <map>
#include <mutex>
#include <utility>

// Implemented by the HLE memory tracker (hle/memory/renderer_tracked_mapping.hpp). Declared here rather
// than included so the gpu layer does not gain an include edge into hle/.
extern "C" uint64_t prosper_renderer_guest_mapped_readable_prefix(uint64_t addr, uint64_t bytes);
extern "C" int prosper_renderer_guest_address_tracked(uint64_t addr);
extern "C" uint64_t prosper_renderer_tracked_mapping_generation();

namespace prosper::gpu {

namespace {

// The resource layer's per-buffer copy cap, the same 256 MiB the fold's raw-use admission applies.
constexpr uint64_t kRawBufferCap = 0x10000000ull;

uint64_t round_to_records(uint64_t run, uint64_t stride) {
    if (!stride) stride = 1;
    const uint64_t clamped = run - run % stride;
    return clamped < stride ? 0 : clamped;
}

}   // namespace

uint64_t buffer_descriptor_window_bytes(const uint32_t v[4]) {
    const uint64_t stride = (v[1] >> 16) & 0x3FFFu;
    const uint64_t records = v[2];
    return stride ? records * stride : records;
}

bool oversized_window_mapped_run(uint64_t base, uint64_t window_bytes, uint64_t cap,
                                 const MappedMemoryProbe& probe, uint64_t& run_bytes) {
    if (window_bytes <= cap || window_bytes > kMaxScannedWindow) return false;
    if (!base || base > UINT64_MAX - window_bytes) return false;
    if (!probe.readable_prefix || !probe.any_mapped) return false;

    const uint64_t prefix = probe.readable_prefix(base, window_bytes);
    if (!prefix || prefix > cap || prefix > window_bytes) return false;

    // Nothing else in the window may be mapped: an island past the leading run is memory the shader
    // could legitimately read, and clamping would turn it into silent zeros.
    const uint64_t end = base + window_bytes;
    uint64_t addr = (base + prefix + kMappingGranule - 1) / kMappingGranule * kMappingGranule;
    for (; addr < end; addr += kMappingGranule)
        if (probe.any_mapped(addr, kMappingGranule)) return false;

    run_bytes = prefix;
    return true;
}

bool clamp_oversized_buffer_window(uint64_t base, uint64_t window_bytes, uint64_t stride,
                                   uint64_t cap, const MappedMemoryProbe& probe,
                                   uint64_t& clamped_bytes) {
    uint64_t run = 0;
    if (!oversized_window_mapped_run(base, window_bytes, cap, probe, run)) return false;
    const uint64_t clamped = round_to_records(run, stride);
    if (!clamped) return false;
    clamped_bytes = clamped;
    return true;
}

namespace {

using RunFinder = std::function<bool(uint64_t base, uint64_t window_bytes, uint64_t& run_bytes)>;

// One place turns a raw V# into the window the proof scans and the NUM_RECORDS it publishes, for both
// the injected-probe and the live path.
bool clamp_descriptor_with(const uint32_t v[4], const RunFinder& find_run,
                           uint32_t& clamped_records) {
    const DecodedBufferDescriptor d = decode_buffer_descriptor(v);
    uint64_t run = 0;
    if (!find_run(d.base, buffer_descriptor_window_bytes(v), run)) return false;
    const uint64_t clamped = round_to_records(run, d.stride);
    if (!clamped) return false;
    clamped_records = static_cast<uint32_t>(clamped / (d.stride ? d.stride : 1u));
    return true;
}

}   // namespace

bool clamp_oversized_buffer_descriptor(const uint32_t v[4], uint64_t cap,
                                       const MappedMemoryProbe& probe, uint32_t& clamped_records) {
    return clamp_descriptor_with(
        v,
        [&](uint64_t base, uint64_t window, uint64_t& run) {
            return oversized_window_mapped_run(base, window, cap, probe, run);
        },
        clamped_records);
}

namespace {

struct CachedRun {
    uint64_t generation = 0;
    bool ok = false;
    uint64_t run = 0;
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

bool live_mapped_run(uint64_t base, uint64_t window_bytes, uint64_t cap, uint64_t& run_bytes) {
    if (window_bytes <= cap) return false;
    static std::mutex mutex;
    static std::map<std::pair<uint64_t, uint64_t>, CachedRun> cache;
    const uint64_t generation = prosper_renderer_tracked_mapping_generation();
    const auto key = std::make_pair(base, window_bytes);
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto found = cache.find(key);
        if (found != cache.end() && found->second.generation == generation) {
            run_bytes = found->second.run;
            return found->second.ok;
        }
    }
    uint64_t run = 0;
    const bool ok = oversized_window_mapped_run(base, window_bytes, cap, live_probe(), run);
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (cache.size() > 256)
            cache.clear();   // bounded: a handful of distinct windows is typical
        cache[key] = {generation, ok, run};
    }
    if (ok) run_bytes = run;
    return ok;
}

size_t resolve_marked(std::vector<SrtUse>& uses, size_t first,
                      const std::function<bool(const uint32_t*, uint32_t&)>& clamp) {
    size_t clamped_count = 0;
    size_t write = first;
    for (size_t read = first; read < uses.size(); ++read) {
        SrtUse& use = uses[read];
        if (use.oversize_window) {
            uint32_t records = 0;
            if (!clamp(use.v4.data(), records)) continue;   // refused: dropped, as before the mark
            use.v4[2] = records;
            use.oversize_window = false;
            ++clamped_count;
        }
        if (write != read) uses[write] = use;
        ++write;
    }
    uses.resize(write);
    return clamped_count;
}

}   // namespace

bool clamp_oversized_buffer_window_live(uint64_t base, uint64_t window_bytes, uint64_t stride,
                                        uint64_t cap, uint64_t& clamped_bytes) {
    uint64_t run = 0;
    if (!live_mapped_run(base, window_bytes, cap, run)) return false;
    const uint64_t clamped = round_to_records(run, stride);
    if (!clamped) return false;
    clamped_bytes = clamped;
    return true;
}

size_t resolve_oversized_buffer_windows(std::vector<SrtUse>& uses, size_t first) {
    return resolve_marked(uses, first, [](const uint32_t* v, uint32_t& records) {
        return clamp_descriptor_with(
            v,
            [](uint64_t base, uint64_t window, uint64_t& run) {
                return live_mapped_run(base, window, kRawBufferCap, run);
            },
            records);
    });
}

size_t resolve_oversized_buffer_windows(std::vector<SrtUse>& uses, size_t first,
                                        const MappedMemoryProbe& probe) {
    return resolve_marked(uses, first, [&probe](const uint32_t* v, uint32_t& records) {
        return clamp_oversized_buffer_descriptor(v, kRawBufferCap, probe, records);
    });
}

}   // namespace prosper::gpu
