#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace prosper::host {

struct GuestReadableRange {
    uint64_t begin = 0;
    uint64_t end = 0;
};

class GuestReadableRangeCache {
public:
    bool contains(uint64_t begin, uint64_t end) {
        if (begin >= end) return false;
        if (have_last_hit_ && begin >= last_hit_.begin && end <= last_hit_.end) return true;
        auto candidate = std::upper_bound(
            ranges_.begin(), ranges_.end(), begin,
            [](uint64_t value, const GuestReadableRange& range) { return value < range.begin; });
        if (candidate == ranges_.begin()) return false;
        --candidate;
        if (begin < candidate->begin || end > candidate->end) return false;
        last_hit_ = *candidate;
        have_last_hit_ = true;
        return true;
    }

    bool containing_range(uint64_t begin, uint64_t end, GuestReadableRange& out) {
        if (!contains(begin, end)) return false;
        out = last_hit_;
        return true;
    }

    void insert(uint64_t begin, uint64_t end) {
        if (begin >= end) return;
        size_t first = 0;
        while (first < ranges_.size() && ranges_[first].end < begin) ++first;
        size_t last = first;
        while (last < ranges_.size() && ranges_[last].begin <= end) {
            begin = std::min(begin, ranges_[last].begin);
            end = std::max(end, ranges_[last].end);
            ++last;
        }
        ranges_.erase(ranges_.begin() + first, ranges_.begin() + last);
        ranges_.insert(ranges_.begin() + first, {begin, end});
        last_hit_ = {begin, end};
        have_last_hit_ = true;
    }

    void erase(uint64_t begin, uint64_t end) {
        if (begin >= end) return;
        std::vector<GuestReadableRange> kept;
        kept.reserve(ranges_.size() + 1);
        for (const GuestReadableRange& range : ranges_) {
            if (range.end <= begin || range.begin >= end) {
                kept.push_back(range);
                continue;
            }
            if (range.begin < begin) kept.push_back({range.begin, begin});
            if (range.end > end) kept.push_back({end, range.end});
        }
        ranges_.swap(kept);
        last_hit_ = {};
        have_last_hit_ = false;
    }

    void sync_generation(uint64_t generation) {
        if (generation_ == generation) return;
        clear();
        generation_ = generation;
    }

    void clear() {
        ranges_.clear();
        last_hit_ = {};
        have_last_hit_ = false;
    }

    size_t range_count() const { return ranges_.size(); }
    uint64_t generation() const { return generation_; }

private:
    std::vector<GuestReadableRange> ranges_;
    GuestReadableRange last_hit_;
    uint64_t generation_ = 0;
    bool have_last_hit_ = false;
};

// Only explicitly tracked, currently readable HLE mappings are eligible for cross-submit reuse.
// Host-managed guest stacks and diagnostic mappings remain submit-local because their lifetime does
// not pass through the kernel-memory HLE.
namespace detail {
inline std::atomic<uint64_t> guest_mapping_generation{1};
inline std::mutex guest_mapping_mutex;
inline GuestReadableRangeCache guest_readable_mappings;

inline void advance_guest_mapping_generation() {
    guest_mapping_generation.fetch_add(1, std::memory_order_release);
}
}

inline uint64_t guest_mapping_generation() {
    return detail::guest_mapping_generation.load(std::memory_order_acquire);
}

inline void notify_guest_mapping_added(uint64_t begin, uint64_t size, bool readable) {
    if (!size) return;
    {
        std::lock_guard lock(detail::guest_mapping_mutex);
        if (begin > UINT64_MAX - size) {
            detail::guest_readable_mappings.clear();
            detail::advance_guest_mapping_generation();
            return;
        }
        const uint64_t end = begin + size;
        detail::guest_readable_mappings.erase(begin, end);
        if (readable) detail::guest_readable_mappings.insert(begin, end);
    }
    detail::advance_guest_mapping_generation();
}

inline void notify_guest_mapping_removed(uint64_t begin, uint64_t size) {
    if (!size) return;
    {
        std::lock_guard lock(detail::guest_mapping_mutex);
        if (begin > UINT64_MAX - size) {
            detail::guest_readable_mappings.clear();
            detail::advance_guest_mapping_generation();
            return;
        }
        detail::guest_readable_mappings.erase(begin, begin + size);
    }
    detail::advance_guest_mapping_generation();
}

inline bool guest_readable_mapping_containing(uint64_t begin, uint64_t end,
                                              GuestReadableRange& out) {
    std::lock_guard lock(detail::guest_mapping_mutex);
    return detail::guest_readable_mappings.containing_range(begin, end, out);
}

} // namespace prosper::host
