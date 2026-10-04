// exc_pending.cpp -- see exc_pending.hpp.
#include "hle/kernel/exc_pending.hpp"

#include <atomic>

namespace prosper {
namespace {

static_assert(std::atomic<uint64_t>::is_always_lock_free && std::atomic<int>::is_always_lock_free,
              "the delivery side runs in a signal handler: the table must be lock-free");

// A slot's tid is 0 (free), kClaiming (being filled by a raiser), or the owning thread. Publishing
// is two-phase -- claim, store the type, THEN publish the tid -- so a take can never match a slot
// whose type is not yet written (it would otherwise hand out the slot's previous type).
constexpr uint64_t kClaiming = ~0ull;
struct Slot {
    std::atomic<uint64_t> tid{0};
    std::atomic<int> type{0};
};
Slot g_slots[kExcPendingSlots];

} // namespace

size_t exc_pending_put(uint64_t tid, int type) {
    if (!tid || tid == kClaiming) return 0;   // reserved values
    for (size_t i = 0; i < kExcPendingSlots; ++i) {
        // Load first: a locked CAS on every occupied slot is needless bus traffic.
        if (g_slots[i].tid.load(std::memory_order_relaxed) != 0) continue;
        uint64_t free_slot = 0;
        if (g_slots[i].tid.compare_exchange_strong(free_slot, kClaiming)) {
            g_slots[i].type.store(type);
            g_slots[i].tid.store(tid);   // publish last (seq_cst: the type is visible first)
            return i + 1;
        }
    }
    return 0;
}

void exc_pending_release(size_t handle, uint64_t tid) {
    if (!handle || handle > kExcPendingSlots) return;
    uint64_t owner = tid;
    g_slots[handle - 1].tid.compare_exchange_strong(owner, 0);
}

int exc_pending_take(uint64_t tid) {
    if (!tid || tid == kClaiming) return -1;
    for (auto& s : g_slots) {
        if (s.tid.load() != tid) continue;
        const int type = s.type.load();
        uint64_t owner = tid;
        if (s.tid.compare_exchange_strong(owner, 0)) return type;
    }
    return -1;
}

} // namespace prosper
