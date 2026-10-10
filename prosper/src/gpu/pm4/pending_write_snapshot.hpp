#pragma once
#include <cstdint>
#include <optional>

namespace prosper::gpu {

// One nonblocking observation of the global pending-write queue. Counts include resource
// writes and completion labels; they are not submit counts or completed GPU work.
struct PendingWriteSnapshot {
    uint64_t queued = 0;
    uint64_t active_submits = 0;
    uint64_t inflight_batches = 0; // apply operations; one renderer batch can contain many writes
    uint64_t front_item_age_ns = 0; // includes preparation before enqueue; zero for an empty queue
    int64_t release_delay_ns = 0; // negative means the current grace deadline has passed
    uint64_t scope_begins = 0;
    uint64_t scope_ends = 0;
    uint64_t deadline_resets = 0;
    // Cumulative completed CV waits, summed across submit callers. Excludes initial mutex lock.
    // Retired subset began at a zero-active boundary with queued writes; includes drain/scheduling.
    uint64_t admission_waiters = 0; // currently blocked submit callers
    uint64_t admission_wait_count = 0;
    uint64_t admission_wait_ns = 0;
    uint64_t admission_wait_max_ns = 0;
    uint64_t admission_retired_wait_count = 0;
    uint64_t admission_retired_wait_ns = 0;
    // Actual queue-record visits and authoritative geometry observations, under the queue lock.
    // These distinguish linear incremental indexing from repeated scans without timing assertions.
    uint64_t pending_records_examined = 0;
    uint64_t pending_geometry_queries = 0;
};

// Empty means lock contention, never an empty queue. Does not drain or wait for any work.
std::optional<PendingWriteSnapshot> try_pending_write_snapshot();

} // namespace prosper::gpu
