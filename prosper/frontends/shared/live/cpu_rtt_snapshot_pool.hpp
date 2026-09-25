#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
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
