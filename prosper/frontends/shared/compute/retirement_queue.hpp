// retirement_queue.hpp -- the ordering contract for pipelined GPU work (ADR 0009, PERF-P8).
//
// A pure model, no Vulkan and no guest: it decides WHICH pending operations must retire (their
// completion waited for, their writeback applied) before something may proceed, so the rule can be
// proven by tests instead of discovered by a title that breaks. The live backend adopts it stage by
// stage; until then it is the executable statement of the invariant.
//
// The rule, from the ADR: a guest-visible effect (a label write, an end-of-pipe event, a flip, a
// guest-memory read, a write-watch invalidation) is applied only after every operation that precedes
// it in the command stream has retired, and operations retire in submission order. Everything else
// the executor waits for today is internal and may stay pending.
//
//   * An operation records the guest ranges it READS and WRITES.
//   * A new operation may be submitted without waiting if it does not conflict with any pending one:
//     read-after-write, write-after-write and write-after-read on overlapping ranges all conflict.
//     On a conflict every pending operation up to and including the LAST conflicting one retires
//     first (in order), never just the conflicting one: retiring out of order would apply a later
//     writeback before an earlier one.
//   * An observation retires either everything pending (a guest-visible effect with no range:
//     label write, EOP, flip, submit end) or everything up to the last pending writer of the range it
//     reads or overwrites (a guest-memory read, a CPU consumer).
//   * Depth is bounded (PERF-P6): admitting an operation into a full queue retires the oldest first.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

namespace prosper::frontend {

struct GuestRange {
    uint64_t addr = 0;
    uint64_t bytes = 0;

    bool empty() const { return bytes == 0; }
    bool overlaps(const GuestRange& other) const {
        return !empty() && !other.empty() && addr < other.addr + other.bytes &&
               other.addr < addr + bytes;
    }
};

struct PendingOperation {
    uint64_t id = 0;                       // submission order; strictly increasing
    std::vector<GuestRange> reads;
    std::vector<GuestRange> writes;
};

class RetirementQueue {
public:
    explicit RetirementQueue(size_t max_in_flight = 4) : max_in_flight_(max_in_flight ? max_in_flight : 1) {}

    // Ids that must retire (oldest first) before `op` may be admitted. Retiring them is the
    // caller's job; call retire() with the returned ids, then admit().
    std::vector<uint64_t> required_before_admit(const PendingOperation& op) const {
        size_t last_conflict = npos;
        for (size_t i = 0; i < pending_.size(); ++i)
            if (conflicts(pending_[i], op)) last_conflict = i;
        size_t count = last_conflict == npos ? 0 : last_conflict + 1;
        if (pending_.size() - std::min(count, pending_.size()) >= max_in_flight_)
            count = pending_.size() - max_in_flight_ + 1;   // bound the depth: retire the oldest
        return ids_prefix(count);
    }

    void admit(const PendingOperation& op) { pending_.push_back(op); }

    // A guest-visible effect with no range (label write, EOP, flip, submit end): everything retires.
    std::vector<uint64_t> required_for_effect() const { return ids_prefix(pending_.size()); }

    // A guest-memory observation of `range`: everything up to the last pending writer of it.
    // `will_write` additionally covers readers (a write-after-read), for a CPU store.
    std::vector<uint64_t> required_for_observation(const GuestRange& range,
                                                   bool will_write = false) const {
        size_t last = npos;
        for (size_t i = 0; i < pending_.size(); ++i) {
            for (const GuestRange& w : pending_[i].writes)
                if (w.overlaps(range)) last = i;
            if (will_write)
                for (const GuestRange& r : pending_[i].reads)
                    if (r.overlaps(range)) last = i;
        }
        return ids_prefix(last == npos ? 0 : last + 1);
    }

    // Retire the oldest `ids.size()` operations. The ids must be exactly the current prefix, which
    // is how an out-of-order retirement is made impossible rather than merely discouraged.
    bool retire(const std::vector<uint64_t>& ids) {
        if (ids.size() > pending_.size()) return false;
        for (size_t i = 0; i < ids.size(); ++i)
            if (pending_[i].id != ids[i]) return false;
        pending_.erase(pending_.begin(), pending_.begin() + static_cast<std::ptrdiff_t>(ids.size()));
        return true;
    }

    size_t size() const { return pending_.size(); }
    bool empty() const { return pending_.empty(); }

private:
    static constexpr size_t npos = static_cast<size_t>(-1);

    static bool any_overlap(const std::vector<GuestRange>& a, const std::vector<GuestRange>& b) {
        for (const GuestRange& x : a)
            for (const GuestRange& y : b)
                if (x.overlaps(y)) return true;
        return false;
    }
    static bool conflicts(const PendingOperation& pending, const PendingOperation& incoming) {
        return any_overlap(pending.writes, incoming.reads) ||    // read after write
               any_overlap(pending.writes, incoming.writes) ||   // write after write
               any_overlap(pending.reads, incoming.writes);      // write after read
    }
    std::vector<uint64_t> ids_prefix(size_t count) const {
        std::vector<uint64_t> ids;
        for (size_t i = 0; i < count && i < pending_.size(); ++i) ids.push_back(pending_[i].id);
        return ids;
    }

    size_t max_in_flight_;
    std::deque<PendingOperation> pending_;
};

}  // namespace prosper::frontend
