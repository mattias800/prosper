#pragma once

#include "diagnostics/exit_reports.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

namespace prosper::frontend {

// Reuse accounting for the CPU RTT snapshot pool.
//
// The pool's whole purpose is to avoid a heap allocation plus a full copy per published render
// target, and until this census nothing counted whether it succeeded: `CpuRttSnapshot::reused` was
// read only by the pool's own unit test. A pool that never hits is invisible -- it still returns
// correct pixels, just by way of an allocation the size of a render target, every time.
//
// This is worth counting because a miss is expensive in a way the aggregate hides. `copy()`
// reuses a free buffer only on an EXACT `size()` match and retains 4 buffers by default, so a
// title cycling through several render-target extents can miss almost every time while the pool
// looks present and correct. Live profiling of one such title put
// `CpuRttSnapshotPool::copy -> vector::assign -> _M_allocate_and_copy -> __memmove_avx512` at the
// top of the busiest thread's stack.
//
// One line at end of run via register_exit_report (NOT std::atexit -- #3353).
struct CpuRttSnapshotPoolStats {
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> hit_bytes{0};
    std::atomic<uint64_t> miss_bytes{0};
};

inline CpuRttSnapshotPoolStats& cpu_rtt_snapshot_pool_stats() {
    static CpuRttSnapshotPoolStats stats;
    static const bool once = [] {
        prosper::diagnostics::register_exit_report([] {
            auto& s = cpu_rtt_snapshot_pool_stats();
            const uint64_t hits = s.hits.load(std::memory_order_relaxed);
            const uint64_t misses = s.misses.load(std::memory_order_relaxed);
            if (std::getenv("PROSPER_NO_RTT_POOL_CENSUS") || (hits + misses) == 0) return;
            const uint64_t miss_bytes = s.miss_bytes.load(std::memory_order_relaxed);
            std::fprintf(stderr,
                         "[rtt-pool] copies=%llu hit=%llu (%.1f%%) miss=%llu "
                         "allocated=%.1f MiB reused=%.1f MiB\n",
                         static_cast<unsigned long long>(hits + misses),
                         static_cast<unsigned long long>(hits),
                         100.0 * static_cast<double>(hits) / static_cast<double>(hits + misses),
                         static_cast<unsigned long long>(misses),
                         static_cast<double>(miss_bytes) / (1024.0 * 1024.0),
                         static_cast<double>(s.hit_bytes.load(std::memory_order_relaxed)) /
                             (1024.0 * 1024.0));
            std::fflush(stderr);
        });
        return true;
    }();
    (void)once;
    return stats;
}

struct CpuRttSnapshot {
    std::shared_ptr<const std::vector<uint8_t>> pixels;
    bool reused = false;
};

// CPU RTT publication must retain immutable pixels until the last graphics consumer releases
// them. The free list owns only snapshots whose shared ownership has ended, so a new compute
// dispatch can reuse mapped pages without changing an earlier draw's pixels. The state is owned
// by each snapshot's deleter as well as by the pool, allowing release on another thread or after
// the pool wrapper is destroyed. Reclamation is bounded and never needed for correctness.
class CpuRttSnapshotPool {
    struct State {
        State(size_t budget, size_t max) : budget_bytes(budget), max_retained(max) {}

        void recycle(std::unique_ptr<std::vector<uint8_t>> released) noexcept {
            const size_t charge = released->capacity();
            if (released->empty() || charge > budget_bytes || !max_retained) return;

            std::lock_guard lock(mutex);
            // Release order is LRU order. Only idle buffers are in this list; live consumers
            // retain their separate shared owners and can never be evicted by this loop.
            while (!free.empty() &&
                   (free.size() >= max_retained || retained_bytes > budget_bytes - charge)) {
                retained_bytes -= free.front().capacity();
                free.erase(free.begin());
            }
            try {
                free.push_back(std::move(*released));
                retained_bytes += charge;
            } catch (const std::bad_alloc&) {
                // Retention is optional. Shared ownership and pixel lifetime are unchanged.
            }
        }

        std::mutex mutex;
        std::vector<std::vector<uint8_t>> free;
        size_t retained_bytes = 0;
        size_t budget_bytes;
        size_t max_retained;
    };

    struct ReturnToPool {
        std::shared_ptr<State> state;
        void operator()(std::vector<uint8_t>* released) const noexcept {
            state->recycle(std::unique_ptr<std::vector<uint8_t>>(released));
        }
    };

public:
    explicit CpuRttSnapshotPool(size_t budget_bytes, size_t max_retained = 4)
        : state_(std::make_shared<State>(budget_bytes, max_retained)) {}

    CpuRttSnapshot copy(const uint8_t* source, size_t bytes) {
        if (!source || !bytes) return {};
        std::vector<uint8_t> pixels;
        bool reused = false;
        {
            std::lock_guard lock(state_->mutex);
            // Match on CAPACITY, best fit -- not on an exact `size()`. What the reuse has to
            // avoid is the reallocation, and a buffer whose capacity already holds `bytes` avoids
            // it whatever its current size: the `assign` below then copies in place. Requiring an
            // exact size match instead refused every buffer that was merely big enough, so a
            // title cycling through a handful of render-target extents missed most of the time
            // and paid a fresh render-target-sized allocation per publication. Measured on one
            // such title before this change: 14,089 publications, 38.5% reuse, 55 GiB allocated
            // and copied in a two-minute run, with the resulting memmove at the top of the
            // busiest thread's profile.
            //
            // Best fit rather than first fit so a large retained buffer is not consumed by a
            // small request while a tighter one is available; the pool's byte budget and
            // max_retained already bound what capacity is held overall.
            size_t best = state_->free.size();
            size_t best_capacity = 0;
            for (size_t index = 0; index < state_->free.size(); index++) {
                const size_t capacity = state_->free[index].capacity();
                if (capacity < bytes) continue;
                if (best == state_->free.size() || capacity < best_capacity) {
                    best = index;
                    best_capacity = capacity;
                }
            }
            if (best != state_->free.size()) {
                pixels = std::move(state_->free[best]);
                state_->retained_bytes -= best_capacity;
                state_->free.erase(state_->free.begin() + static_cast<ptrdiff_t>(best));
                reused = true;
            }
        }
        {
            auto& stats = cpu_rtt_snapshot_pool_stats();
            (reused ? stats.hits : stats.misses).fetch_add(1, std::memory_order_relaxed);
            (reused ? stats.hit_bytes : stats.miss_bytes)
                .fetch_add(bytes, std::memory_order_relaxed);
        }
        // One path for both cases. `assign` reallocates only when capacity is insufficient, which
        // the search above has already ruled out on the reuse path, and it never zero-fills. The
        // previous split existed because the reuse path was guaranteed an exact size; with
        // capacity matching, `assign` is what sets the published `size()` correctly.
        pixels.assign(source, source + bytes);

        auto allocated = std::make_unique<std::vector<uint8_t>>(std::move(pixels));
        std::unique_ptr<std::vector<uint8_t>, ReturnToPool> owned(
            allocated.release(), ReturnToPool{state_});
        std::shared_ptr<std::vector<uint8_t>> published(std::move(owned));
        return {std::move(published), reused};
    }

    size_t retained_bytes() const {
        std::lock_guard lock(state_->mutex);
        return state_->retained_bytes;
    }

    size_t retained_buffers() const {
        std::lock_guard lock(state_->mutex);
        return state_->free.size();
    }

private:
    std::shared_ptr<State> state_;
};

} // namespace prosper::frontend
