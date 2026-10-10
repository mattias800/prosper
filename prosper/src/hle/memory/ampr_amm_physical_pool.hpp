#pragma once

#include <cstdint>
#include <iterator>
#include <map>
#include <mutex>

namespace prosper {

// Physical pages donated to AMM remain allocated to AMM across Unmap. Only their individual
// carves become free; a failed host map returns its carve through the same coalescing path.
class AmprAmmPhysicalPool {
public:
    bool publish(uint64_t physical, uint64_t length) {
        std::lock_guard lock(mutex_);
        if (!length || physical > UINT64_MAX - length) return false;
        if (end_ && physical != end_) return false;
        end_ = physical + length;
        return_locked(physical, length);
        return true;
    }

    bool take(uint64_t length, uint64_t& physical) {
        std::lock_guard lock(mutex_);
        if (!length) return false;
        for (auto it = free_ranges_.begin(); it != free_ranges_.end(); ++it) {
            if (length > it->second - it->first) continue;
            physical = it->first;
            const uint64_t end = it->second;
            free_ranges_.erase(it);
            if (physical + length < end) free_ranges_.emplace(physical + length, end);
            return true;
        }
        return false;
    }

    void give_back(uint64_t physical, uint64_t length) {
        std::lock_guard lock(mutex_);
        return_locked(physical, length);
    }

private:
    void return_locked(uint64_t physical, uint64_t length) {
        uint64_t end = physical + length;
        auto next = free_ranges_.lower_bound(physical);
        if (next != free_ranges_.begin()) {
            const auto previous = std::prev(next);
            if (previous->second == physical) {
                physical = previous->first;
                free_ranges_.erase(previous);
            }
        }
        if (next != free_ranges_.end() && next->first == end) {
            end = next->second;
            free_ranges_.erase(next);
        }
        free_ranges_.emplace(physical, end);
    }

    std::mutex mutex_;
    uint64_t end_ = 0;
    std::map<uint64_t, uint64_t> free_ranges_; // sorted, coalesced physical [begin, end) ranges
};

} // namespace prosper
