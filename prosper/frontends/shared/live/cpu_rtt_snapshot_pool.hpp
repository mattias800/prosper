#pragma once

#include "diagnostics/exit_reports.hpp"
#include "diagnostics/transfer_pressure.hpp"

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

// Prints one `[<label>]` line for `stats` at the end of the run (silent when it counted nothing).
inline void print_cpu_rtt_snapshot_pool_stats(const char* label, const CpuRttSnapshotPoolStats& s) {
    const uint64_t hits = s.hits.load(std::memory_order_relaxed);
    const uint64_t misses = s.misses.load(std::memory_order_relaxed);
    if (std::getenv("PROSPER_NO_RTT_POOL_CENSUS") || (hits + misses) == 0) return;
    const uint64_t miss_bytes = s.miss_bytes.load(std::memory_order_relaxed);
    std::fprintf(stderr,
                 "[%s] copies=%llu hit=%llu (%.1f%%) miss=%llu "
                 "allocated=%.1f MiB reused=%.1f MiB\n",
                 label, static_cast<unsigned long long>(hits + misses),
                 static_cast<unsigned long long>(hits),
                 100.0 * static_cast<double>(hits) / static_cast<double>(hits + misses),
                 static_cast<unsigned long long>(misses),
                 static_cast<double>(miss_bytes) / (1024.0 * 1024.0),
                 static_cast<double>(s.hit_bytes.load(std::memory_order_relaxed)) / (1024.0 * 1024.0));
    std::fflush(stderr);
}

inline CpuRttSnapshotPoolStats& cpu_rtt_snapshot_pool_stats() {
    static CpuRttSnapshotPoolStats stats;
    static const bool once = [] {
        prosper::diagnostics::register_exit_report(
            [] { print_cpu_rtt_snapshot_pool_stats("rtt-pool", cpu_rtt_snapshot_pool_stats()); });
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
        State(size_t budget, size_t max, CpuRttSnapshotPoolStats* counters)
            : budget_bytes(budget), max_retained(max), stats(counters) {}

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
        CpuRttSnapshotPoolStats* stats;
    };

    struct ReturnToPool {
        std::shared_ptr<State> state;
        void operator()(std::vector<uint8_t>* released) const noexcept {
            state->recycle(std::unique_ptr<std::vector<uint8_t>>(released));
        }
    };

public:
    // `stats` receives this pool's hit/miss accounting. The default is the shared object behind the
    // `[rtt-pool]` line; a pool whose traffic should be reported on its own passes another one.
    explicit CpuRttSnapshotPool(size_t budget_bytes, size_t max_retained = 4,
                                CpuRttSnapshotPoolStats* stats = nullptr)
        : state_(std::make_shared<State>(budget_bytes, max_retained,
                                         stats ? stats : &cpu_rtt_snapshot_pool_stats())) {}

    CpuRttSnapshot copy(const uint8_t* source, size_t bytes) {
        if (!source || !bytes) return {};
        bool reused = false;
        std::vector<uint8_t> pixels = take(bytes, reused);
        count(reused, bytes);
        // One path for both cases. `assign` reallocates only when capacity is insufficient, which
        // the search in take() has already ruled out on the reuse path, and it never zero-fills. The
        // previous split existed because the reuse path was guaranteed an exact size; with
        // capacity matching, `assign` is what sets the published `size()` correctly.
        prosper::diagnostics::note_transfer(
            prosper::diagnostics::Transfer::RenderTargetSnapshot, bytes);
        pixels.assign(source, source + bytes);
        return publish(std::move(pixels), reused);
    }

    // Like copy(), but the caller produces the bytes straight into the pooled buffer: `fill` is called
    // with a pointer to `bytes` writable bytes and MUST write every one of them (a recycled buffer
    // holds a previous tenant's pixels). A steady-state caller asking for the same size each time gets
    // the same buffer back with its size already right, so there is no allocation and no zero-fill --
    // `resize` only touches bytes beyond the old size. Unlike copy(), it does not feed the transfer-pressure
    // accounting (see below).
    template <typename Fill>
    CpuRttSnapshot build(size_t bytes, Fill&& fill) {
        if (!bytes) return {};
        bool reused = false;
        std::vector<uint8_t> pixels = take(bytes, reused);
        count(reused, bytes);
        // Deliberately no note_transfer(): the bytes are CONVERTED from another buffer by `fill`, not a host
        // copy of guest data, and the always-on transfer-pressure alarm reads that counter. A caller whose
        // fill is a copy should note it itself.
        pixels.resize(bytes);
        fill(pixels.data());
        return publish(std::move(pixels), reused);
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
    // Best-fit reuse of an idle buffer whose capacity holds `bytes`; an empty vector on a miss.
    std::vector<uint8_t> take(size_t bytes, bool& reused) {
        std::vector<uint8_t> pixels;
        reused = false;
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
        return pixels;
    }

    void count(bool reused, size_t bytes) const {
        {
            auto& stats = *state_->stats;
            (reused ? stats.hits : stats.misses).fetch_add(1, std::memory_order_relaxed);
            (reused ? stats.hit_bytes : stats.miss_bytes)
                .fetch_add(bytes, std::memory_order_relaxed);
        }
    }

    CpuRttSnapshot publish(std::vector<uint8_t>&& pixels, bool reused) {
        auto allocated = std::make_unique<std::vector<uint8_t>>(std::move(pixels));
        std::unique_ptr<std::vector<uint8_t>, ReturnToPool> owned(
            allocated.release(), ReturnToPool{state_});
        std::shared_ptr<std::vector<uint8_t>> published(std::move(owned));
        return {std::move(published), reused};
    }

    std::shared_ptr<State> state_;
};

// Copy `bytes` into a snapshot, through `pool` unless it is disabled.
inline CpuRttSnapshot copy_cpu_rtt_snapshot(CpuRttSnapshotPool& pool, bool pooled,
                                            const uint8_t* source, size_t bytes) {
    if (pooled) return pool.copy(source, bytes);
    CpuRttSnapshot snapshot;
    snapshot.pixels = std::make_shared<std::vector<uint8_t>>(source, source + bytes);
    return snapshot;
}

// Produce `bytes` into a snapshot through `pool` unless it is disabled, in which case a plain allocation is
// filled instead (the same no-pool behaviour copy_cpu_rtt_snapshot has).
template <typename Fill>
inline CpuRttSnapshot build_cpu_rtt_snapshot(CpuRttSnapshotPool& pool, bool pooled, size_t bytes, Fill&& fill) {
    if (pooled) return pool.build(bytes, std::forward<Fill>(fill));
    CpuRttSnapshot snapshot;
    if (!bytes) return snapshot;
    auto pixels = std::make_shared<std::vector<uint8_t>>(bytes);
    fill(pixels->data());
    snapshot.pixels = std::move(pixels);
    return snapshot;
}

} // namespace prosper::frontend
