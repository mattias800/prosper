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

// ===========================================================================================
// THE INVARIANT (#2393). Stated here, at the definition, because it is load-bearing for code
// in two other subsystems and used to be written down nowhere:
//
//     EVERY change to guest page protection must advance guest_mapping_generation().
//
// Two thread-local caches in src/gpu/gpu_executor.cpp memoize positive answers -- "this span is
// readable", "this span is writable" -- and drop everything when this counter moves. They are
// pure memoization of an OS query, so the counter is their ONLY staleness guard. A protection
// change that does not advance it leaves them answering the OLD permission until something
// unrelated happens to move it, and the dangerous direction is a revocation: a cached "writable"
// for a page that is now read-only turns a would-be refusal into a fault or a lost write, with
// nothing anywhere reporting a problem.
//
// Advancing it SPURIOUSLY is always safe. Every consumer treats a change as "drop the memo and
// re-probe", so an unnecessary advance costs one cache refill and can never produce a wrong
// answer. Prefer advancing when unsure. Nothing keys off the counter's VALUE -- it is never an
// index, a handle, or compared across subsystems -- so it may be advanced from anywhere.
//
// Two ways to satisfy the invariant, and which one you need depends on where you are:
//
//   * notify_guest_mapping_added() / notify_guest_mapping_removed() -- the guest-facing kernel
//     memory HLE path. They also maintain the readable-mapping registry, and they take
//     guest_mapping_mutex to do it. Use these whenever you own the mapping's lifetime.
//
//   * notify_guest_page_protection_changed() -- below. Counter only, no registry, no lock.
//     Use it from a signal handler, or when you are temporarily flipping the protection of a
//     page inside a mapping you do not own (a watchpoint), where mutating the registry would be
//     actively wrong -- see that function's comment.
// ===========================================================================================

inline uint64_t guest_mapping_generation() {
    return detail::guest_mapping_generation.load(std::memory_order_acquire);
}

// Publish "some guest page's protection just changed" without touching the mapping registry.
//
// ASYNC-SIGNAL-SAFE, and the whole reason this entry point exists. Its body is one
// `fetch_add` on a lock-free atomic: no lock, no allocation, no libc call, no errno, no TLS.
// C++ [support.signal] permits a signal handler to operate on a lock-free atomic object with
// static storage duration, which is exactly what this is, and the static_assert below turns
// that precondition into a compile-time check on every platform prosper builds for rather than
// a claim in a comment. Calling notify_guest_mapping_added/removed from a handler instead would
// take guest_mapping_mutex and can self-deadlock against a fault arriving on a thread that
// already holds it.
//
// It is also the CORRECT call for a watchpoint even off the signal path. A watch protects one
// page inside a larger mapping it does not own; notify_guest_mapping_removed(page, 0x1000)
// would punch a hole in the readable registry that nothing ever repairs, because the watch does
// not know the enclosing mapping's real protection and cannot restore the entry on disarm. That
// would trade a transient stale positive for a permanent stale negative.
//
// Ordering: call it AFTER the protection change, matching the kernel-memory HLE's own
// change-then-notify order. Neither order closes the instantaneous window in which another
// thread reads the counter, then the protection changes, then that thread consults its
// now-stale memo -- protection cannot be changed and published atomically, and the uncached
// /proc/self/maps path this replaced had the identical window between its parse and its
// caller's use. What this does close is the UNBOUNDED staleness, which is the actual defect.
inline void notify_guest_page_protection_changed() {
    static_assert(std::atomic<uint64_t>::is_always_lock_free,
                  "guest_mapping_generation must be lock-free: it is advanced from a SIGSEGV/"
                  "SIGTRAP handler, and a locking atomic there is not async-signal-safe");
    detail::advance_guest_mapping_generation();
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
