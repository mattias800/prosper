#pragma once

#include <cstddef>
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
    explicit CpuRttSnapshotPool(size_t budget_bytes, size_t max_retained = 4)
        : state_(std::make_shared<State>(budget_bytes, max_retained)) {}

    CpuRttSnapshot copy(const uint8_t* source, size_t bytes) {
        if (!source || !bytes) return {};
        std::vector<uint8_t> pixels;
        bool reused = false;
        {
            std::lock_guard lock(state_->mutex);
            for (auto it = state_->free.begin(); it != state_->free.end(); ++it) {
                if (it->size() != bytes) continue;
                const size_t charge = it->capacity();
                pixels = std::move(*it);
                state_->retained_bytes -= charge;
                state_->free.erase(it);
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
                if (released->empty() || charge > state->budget_bytes) return;
                std::lock_guard lock(state->mutex);
                if (state->free.size() >= state->max_retained ||
                    state->retained_bytes > state->budget_bytes - charge)
                    return;
                try {
                    state->free.push_back(std::move(*released));
                    state->retained_bytes += charge;
                } catch (...) {
                    // Retention is optional. Shared ownership and pixel lifetime are unchanged.
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
        State(size_t budget, size_t max) : budget_bytes(budget), max_retained(max) {}
        std::mutex mutex;
        std::vector<std::vector<uint8_t>> free;
        size_t retained_bytes = 0;
        size_t budget_bytes;
        size_t max_retained;
    };
    std::shared_ptr<State> state_;
};

} // namespace prosper::frontend
