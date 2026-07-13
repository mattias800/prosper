#include "writer_provenance.hpp"

#include <atomic>
#include <cstring>
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

bool writer_provenance_enabled() {
    const char* explicit_mode = std::getenv("PROSPER_WRITER_PROVENANCE");
    const char* dimension_mode = std::getenv("PROSPER_PROVENANCE_DIM");
    const char* resource_hash_mode = std::getenv("PROSPER_RESOURCE_HASH_DIM");
    const char* timeline_depth_hash_mode = std::getenv("PROSPER_GPU_TIMELINE_DEPTH_HASH_DIM");
    const bool explicitly_on = explicit_mode && *explicit_mode &&
                               std::strcmp(explicit_mode, "0") &&
                               std::strcmp(explicit_mode, "off");
    return explicitly_on || (dimension_mode && *dimension_mode) ||
           (resource_hash_mode && *resource_hash_mode) ||
           (timeline_depth_hash_mode && *timeline_depth_hash_mode);
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
