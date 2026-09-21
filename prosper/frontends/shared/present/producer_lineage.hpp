// Completed renderer work carried through the GPU scanout handoff (#3486).
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <deque>
#include <optional>
#include <unordered_set>

namespace prosper::frontend {

// A work serial may be reserved while an ordered submission is being recorded, but it becomes
// public only after that submission has completed. The originating image's registration makes the
// identity explicit; a pure copy keeps this pair unchanged.
struct CompletedProducer {
    uint64_t registration = 0;
    uint64_t work = 0;
    constexpr bool known() const { return registration && work; }
    constexpr bool operator==(const CompletedProducer&) const = default;
};

// Snapshot beside the exact VkImage under the renderer resource lock. The current image
// registration may differ from the producer's after a representation-only copy.
struct ProducerSource {
    uint64_t image_registration = 0;
    CompletedProducer completed;
    constexpr bool known() const { return image_registration && completed.known(); }
};

// Same-binary overhead control. Fixed at first use; disabled runs report unavailable F8 lineage
// fields. The ordinary build enables accounting without requiring an option or driver feature.
inline bool producer_lineage_enabled() {
    static const bool enabled = std::getenv("PROSPER_NO_PRODUCER_LINEAGE") == nullptr;
    return enabled;
}

inline uint64_t next_producer_identity() {
    if (!producer_lineage_enabled()) return 0;
    static std::atomic<uint64_t> next{0};
    return next.fetch_add(1, std::memory_order_relaxed) + 1;
}

enum class ProducerDelivery { New, Repeat, Unknown };

// App presentation is single-threaded. Keep an exact bounded history of delivered work, not just
// the immediately previous token: alternating stale output buffers must remain repeats. Once a
// token ages out, a later occurrence at or below the eviction watermark is unknown rather than
// falsely fresh. This conservatively loses coverage for a never-delivered older version too.
class ProducerDeliveryHistory {
public:
    static constexpr size_t kCapacity = 4096;

    ProducerDelivery classify(CompletedProducer producer) {
        if (!producer.known()) return ProducerDelivery::Unknown;
        if (seen_.contains(producer.work)) return ProducerDelivery::Repeat;
        if (producer.work <= evicted_through_) return ProducerDelivery::Unknown;
        seen_.insert(producer.work);
        order_.push_back(producer.work);
        if (order_.size() > kCapacity) {
            const uint64_t old = order_.front();
            order_.pop_front();
            seen_.erase(old);
            if (old > evicted_through_) evicted_through_ = old;
        }
        return ProducerDelivery::New;
    }

private:
    std::deque<uint64_t> order_;
    std::unordered_set<uint64_t> seen_;
    uint64_t evicted_through_ = 0;
};

struct ProducerLineageSnapshot {
    uint64_t producer_publications = 0;
    uint64_t producer_publications_known = 0;
    uint64_t producer_delivered_new = 0;
    uint64_t producer_delivered_repeat = 0;
    uint64_t producer_delivered_unknown = 0;
};

class ProducerLineageCounters {
public:
    void publication(ProducerSource source) {
        if (!producer_lineage_enabled()) return;
        publications_.fetch_add(1, std::memory_order_relaxed);
        if (source.known()) publications_known_.fetch_add(1, std::memory_order_relaxed);
    }

    // Called at the host-present attempt boundary. A skipped/failed/out-of-date attempt cannot
    // deliver an image, even if its scanout slot had already been acquired.
    std::optional<ProducerDelivery> presentation(bool successful, ProducerSource source) {
        if (!producer_lineage_enabled() || !successful) return std::nullopt;
        const ProducerDelivery result = history_.classify(source.known() ? source.completed
                                                                         : CompletedProducer{});
        switch (result) {
        case ProducerDelivery::New: delivered_new_.fetch_add(1, std::memory_order_relaxed); break;
        case ProducerDelivery::Repeat: delivered_repeat_.fetch_add(1, std::memory_order_relaxed); break;
        case ProducerDelivery::Unknown: delivered_unknown_.fetch_add(1, std::memory_order_relaxed); break;
        }
        return result;
    }

    ProducerLineageSnapshot snapshot() const {
        return {publications_.load(std::memory_order_relaxed),
                publications_known_.load(std::memory_order_relaxed),
                delivered_new_.load(std::memory_order_relaxed),
                delivered_repeat_.load(std::memory_order_relaxed),
                delivered_unknown_.load(std::memory_order_relaxed)};
    }

private:
    ProducerDeliveryHistory history_;
    std::atomic<uint64_t> publications_{0}, publications_known_{0};
    std::atomic<uint64_t> delivered_new_{0}, delivered_repeat_{0}, delivered_unknown_{0};
};

inline ProducerLineageCounters& producer_lineage_counters() {
    static ProducerLineageCounters counters;
    return counters;
}

} // namespace prosper::frontend
