#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace prosper::frontend {

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
public:
    explicit CpuRttSnapshotPool(size_t budget_bytes, size_t max_retained = 4,
                                bool diagnose_release = false)
        : state_(std::make_shared<State>(budget_bytes, max_retained, diagnose_release)) {}

    CpuRttSnapshot copy(const uint8_t* source, size_t bytes) {
        if (!source || !bytes) return {};
        std::vector<uint8_t> pixels;
        bool reused = false;
        {
            std::lock_guard lock(state_->mutex);
            for (size_t index = state_->free.size(); index-- > 0;) {
                if (state_->free[index].size() != bytes) continue;
                const size_t charge = state_->free[index].capacity();
                pixels = std::move(state_->free[index]);
                state_->retained_bytes -= charge;
                state_->free.erase(state_->free.begin() + index);
                reused = true;
                break;
            }
        }
        if (reused)
            std::memcpy(pixels.data(), source, bytes);
        else
            pixels.assign(source, source + bytes); // construct from input, without a zero-fill pass

        auto* owned = new std::vector<uint8_t>(std::move(pixels));
        std::shared_ptr<std::vector<uint8_t>> published(
            owned, [state = state_](std::vector<uint8_t>* released) {
                std::unique_ptr<std::vector<uint8_t>> holder(released);
                const size_t charge = released->capacity();
                if (released->empty() || charge > state->budget_bytes) {
                    if (state->diagnose_release)
                        std::fprintf(stderr,
                                     "[cpu-rtt-snapshot-release] bytes=%zu capacity=%zu "
                                     "reason=%s\n", released->size(), charge,
                                     released->empty() ? "empty" : "over-budget");
                    return;
                }
                std::lock_guard lock(state->mutex);
                if (!state->max_retained) {
                    if (state->diagnose_release)
                        std::fprintf(stderr,
                                     "[cpu-rtt-snapshot-release] bytes=%zu capacity=%zu "
                                     "retained=%zu count=%zu reason=count-limit\n",
                                     released->size(), charge, state->retained_bytes,
                                     state->free.size());
                    return;
                }
                // The route may change resolution once and never return. Old idle buffers must
                // not occupy the whole budget and prevent the new recurring extent from being
                // retained. Release order is LRU order; only free buffers can be evicted.
                size_t evicted_bytes = 0;
                size_t evicted_count = 0;
                while (!state->free.empty() &&
                       (state->free.size() >= state->max_retained ||
                        state->retained_bytes > state->budget_bytes - charge)) {
                    const size_t old_charge = state->free.front().capacity();
                    state->retained_bytes -= old_charge;
                    state->free.erase(state->free.begin());
                    evicted_bytes += old_charge;
                    ++evicted_count;
                }
                try {
                    state->free.push_back(std::move(*released));
                    state->retained_bytes += charge;
                    if (state->diagnose_release)
                        std::fprintf(stderr,
                                     "[cpu-rtt-snapshot-release] bytes=%zu capacity=%zu "
                                     "retained=%zu count=%zu evicted-bytes=%zu "
                                     "evicted-count=%zu reason=retained\n",
                                     state->free.back().size(), charge, state->retained_bytes,
                                     state->free.size(), evicted_bytes, evicted_count);
                } catch (...) {
                    // Retention is optional. Shared ownership and pixel lifetime are unchanged.
                    if (state->diagnose_release)
                        std::fprintf(stderr,
                                     "[cpu-rtt-snapshot-release] capacity=%zu "
                                     "reason=retention-error\n", charge);
                }
            });
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
    struct State {
        State(size_t budget, size_t max, bool diagnose)
            : budget_bytes(budget), max_retained(max), diagnose_release(diagnose) {}
        std::mutex mutex;
        std::vector<std::vector<uint8_t>> free;
        size_t retained_bytes = 0;
        size_t budget_bytes;
        size_t max_retained;
        bool diagnose_release;
    };
    std::shared_ptr<State> state_;
};

} // namespace prosper::frontend
