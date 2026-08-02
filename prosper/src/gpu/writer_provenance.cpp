#include "writer_provenance.hpp"

#include <atomic>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace prosper::gpu {
namespace {

constexpr size_t kMaxEvents = 65536;
std::mutex g_mutex;
struct EventKey {
    GuestWriterKind kind;
    uint64_t addr;
    bool operator==(const EventKey& other) const { return kind == other.kind && addr == other.addr; }
};
struct EventKeyHash {
    size_t operator()(const EventKey& key) const {
        return (static_cast<size_t>(key.kind) << 61) ^
               static_cast<size_t>(key.addr ^ (key.addr >> 32));
    }
};
std::unordered_map<EventKey, GuestWriteEvent, EventKeyHash> g_events;
std::atomic<uint64_t> g_sequence{0};

uint64_t span_end(uint64_t addr, uint64_t size) {
    if (size > std::numeric_limits<uint64_t>::max() - addr)
        return std::numeric_limits<uint64_t>::max();
    return addr + size;
}

} // namespace

// PROSPER_PROVENANCE_ADDR=0xADDR[:SIZE] — report every recorded write overlapping one guest range,
// naming the writer kind and its identity (for a compute buffer, the program's code address). This
// answers "who writes this address", which nothing else could: the history was queryable only from
// C++ via last_guest_write_overlap(), and the dimension probes select by resource shape, so a small
// counter or an indirect-argument word had no diagnostic surface at all (#1742).
struct ProvenanceAddrWatch {
    bool active = false;
    uint64_t addr = 0;
    uint64_t size = 4;
};

const ProvenanceAddrWatch& provenance_addr_watch() {
    static const ProvenanceAddrWatch watch = [] {
        ProvenanceAddrWatch w;
        const char* value = std::getenv("PROSPER_PROVENANCE_ADDR");
        if (!value || !*value) return w;
        char* end = nullptr;
        const unsigned long long addr = std::strtoull(value, &end, 0);
        if (!addr || end == value) return w;
        w.addr = addr;
        if (end && *end == ':') {
            const unsigned long long size = std::strtoull(end + 1, nullptr, 0);
            if (size) w.size = size;
        }
        w.active = true;
        std::fprintf(stderr, "[provenance] watching guest range 0x%llx..0x%llx\n",
                     (unsigned long long)w.addr, (unsigned long long)(w.addr + w.size));
        return w;
    }();
    return watch;
}

bool writer_provenance_enabled() {
    const bool explicitly_on = writer_provenance_full_enabled() || provenance_addr_watch().active;
    const char* dimension_mode = std::getenv("PROSPER_PROVENANCE_DIM");
    const char* resource_hash_mode = std::getenv("PROSPER_RESOURCE_HASH_DIM");
    const char* timeline_depth_hash_mode = std::getenv("PROSPER_GPU_TIMELINE_DEPTH_HASH_DIM");
    return explicitly_on || (dimension_mode && *dimension_mode) ||
           (resource_hash_mode && *resource_hash_mode) ||
           (timeline_depth_hash_mode && *timeline_depth_hash_mode);
}

bool writer_provenance_full_enabled() {
    const char* value = std::getenv("PROSPER_WRITER_PROVENANCE");
    // The address watch needs the same unfiltered retention: the ranges it exists to explain are
    // single words, which the dimension probes' large-resource filter would discard.
    return provenance_addr_watch().active ||
           (value && *value && std::strcmp(value, "0") && std::strcmp(value, "off"));
}

const char* guest_writer_kind_name(GuestWriterKind kind) {
    switch (kind) {
        case GuestWriterKind::ColorTarget: return "color";
        case GuestWriterKind::ComputeBuffer: return "compute-buffer";
        case GuestWriterKind::DmaData: return "dma-data";
        case GuestWriterKind::WriteData: return "write-data";
    }
    return "unknown";
}

uint64_t record_guest_write(GuestWriterKind kind, uint64_t addr, uint64_t size,
                            uint64_t submit, uint64_t item, uint64_t order, uint64_t identity,
                            uint32_t width, uint32_t height) {
    if (!addr || !size) return 0;
    GuestWriteEvent event;
    event.sequence = g_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    event.kind = kind;
    event.addr = addr;
    event.size = size;
    event.submit = submit;
    event.item = item;
    event.order = order;
    event.identity = identity;
    event.width = width;
    event.height = height;
    if (const ProvenanceAddrWatch& watch = provenance_addr_watch();
        watch.active && addr < span_end(watch.addr, watch.size) &&
        watch.addr < span_end(addr, size)) {
        // Every overlap, uncapped and unaggregated: the question this answers is "which writer", and
        // a rate limit would hide a writer that only appears after a scene change. The switch is
        // off by default and selects a single range, so the volume is bounded by the guest's own
        // writes to that one address.
        std::fprintf(stderr,
                     "[provenance] %-14s addr=0x%llx size=%llu submit=%llu item=%llu order=%llu "
                     "identity=0x%llx seq=%llu\n",
                     guest_writer_kind_name(kind), (unsigned long long)addr,
                     (unsigned long long)size, (unsigned long long)submit,
                     (unsigned long long)item, (unsigned long long)order,
                     (unsigned long long)identity, (unsigned long long)event.sequence);
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    EventKey key{kind, addr};
    if (g_events.size() == kMaxEvents && !g_events.count(key)) {
        auto oldest = g_events.begin();
        for (auto it = g_events.begin(); it != g_events.end(); ++it)
            if (it->second.sequence < oldest->second.sequence) oldest = it;
        g_events.erase(oldest);
    }
    g_events[key] = event;
    return event.sequence;
}

std::optional<GuestWriteEvent> last_guest_write_overlap(uint64_t addr, uint64_t size,
                                                        uint64_t before_sequence) {
    if (!addr || !size) return std::nullopt;
    const uint64_t end = span_end(addr, size);
    std::lock_guard<std::mutex> lock(g_mutex);
    std::optional<GuestWriteEvent> latest;
    for (const auto& pair : g_events) {
        const GuestWriteEvent& event = pair.second;
        if (event.sequence >= before_sequence) continue;
        if (addr < span_end(event.addr, event.size) && event.addr < end &&
            (!latest || event.sequence > latest->sequence))
            latest = event;
    }
    return latest;
}

void reset_guest_write_history_for_test() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_events.clear();
    g_sequence.store(0, std::memory_order_relaxed);
}

} // namespace prosper::gpu
