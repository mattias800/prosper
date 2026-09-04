// decode_scratch.hpp — reusable full-surface intermediates for the frontend texture materializer.
//
// Every tiled decode branch in `live_renderer.cpp` materializes a guest surface through one or two
// FULL-SIZE intermediates: `traw` (the raw tiled bytes, staged out of guest memory) and
// `hlin`/`flin`/`nlin`/`lin`/`tiled` (the detiled linear result the converter reads). Both were a
// fresh `std::vector<uint8_t>(n, 0)` per resource reference, destroyed at the end of the branch.
//
// For anything past glibc's 32 MiB dynamic mmap threshold that is not a heap allocation at all: it
// is an `mmap`, 4 KiB of page faults per page as the value-initialisation touches it, and a `munmap`
// on the way out — repeated on every callback that samples the surface. Measured on Stray
// (PPSA02101) at its title screen, whose HDR intermediate is a 3840x2176 RGBA16F surface,
// 66,846,720 tiled bytes and 66,355,200 linear:
//
//     fresh traw+hlin, copy + detile-equivalent traffic   26.00 ms
//     the two fresh zero-initialised allocations ALONE    17.84 ms
//     pooled traw+hlin, same copy and traffic              6.00 ms
//     pooled linear only, source read in place             2.83 ms
//
// So roughly two thirds of that pair's cost was the allocator and the kernel, not the decode. It is
// the same cost #3149 saw from the outside as `clear_highpages_kasan_tagged`,
// `map_anon_folio_pte_nopf`, `zap_present_ptes` and `__free_one_page` under a `do_syscall_64` at
// 22.2%, and attributed to the compute path because that is where it went looking.
//
// A lease keeps the pages mapped between references, so the second and later uses of a given size
// pay neither the faults nor the zeroing.
//
// **The zero contract is explicit here, where it used to be incidental.** `std::vector<uint8_t>(n,0)`
// gave every one of those buffers a zero tail, and several branches DEPEND on it: a short or sparse
// guest backing makes `copy_resource` return fewer bytes than requested, and the detiler and the
// converters then read the remainder. A pooled buffer arrives holding the previous surface, so a
// caller that fills it partially MUST call `Lease::zero_tail(written)`. That is one line at each
// call site and it restores exactly the bytes value-initialisation produced — but it is also the one
// way to get this wrong silently, so it is stated in the API rather than left to the reader.
//
// `PROSPER_DECODE_SCRATCH_MB` bounds what a THREAD retains between references (default 512 MiB).
// Spelled `0` it retains nothing, which is the pre-pool behaviour exactly.
//
// **A mistyped value keeps the 512 MiB default, so pooling stays ON.** This is the direction that
// matters and the first version of this comment had it backwards: it claimed a typo selected the
// safe sentinel, which is false, and the misreading was doubled by passing "0 = retain nothing" as
// `env_u64_or_default_capped`'s `default_note` -- that parameter glosses the DEFAULT, so the refusal
// printed "keeping the default (512, 0 = retain nothing)", which reads as "512 retains nothing".
//
// The consequence is specific rather than theoretical. This knob's main use is disarming the pool
// for an A/B, so the dangerous typo is one intended to turn the optimisation OFF -- `=0mb`, `=0MB`,
// `= 0` -- and it leaves the arm fully armed. `env_numeric.hpp` does the right thing (refuse loudly,
// keep the default), so the stderr line IS the signal, and it is the only one: nothing else in a run
// says which arm you got. Read it before quoting a number from an arm you believe you disarmed.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "diagnostics/env_numeric.hpp"

namespace prosper::frontend {

class DecodeScratchPool {
public:
    // A borrowed buffer. Move-only; returns itself to the pool on destruction, so an early `break`
    // or `continue` out of a decode branch cannot leak or double-book one.
    class Lease {
    public:
        Lease() = default;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept
            : pool_(other.pool_), bytes_(std::move(other.bytes_)), size_(other.size_) {
            other.pool_ = nullptr;
            other.size_ = 0;
        }
        Lease& operator=(Lease&& other) noexcept {
            if (this != &other) {
                release();
                pool_ = other.pool_;
                bytes_ = std::move(other.bytes_);
                size_ = other.size_;
                other.pool_ = nullptr;
                other.size_ = 0;
            }
            return *this;
        }
        ~Lease() { release(); }

        // The requested extent. The backing store may be LARGER (a previous, bigger surface's
        // buffer); nothing beyond `size()` is part of this lease. `data()` for a zero-extent lease
        // is UNSPECIFIED -- it may be a live pointer into a retained buffer where a fresh
        // `std::vector<uint8_t>(0)` gave nullptr, so test `size()`, never `data() != nullptr`.
        uint8_t* data() { return bytes_.data(); }
        const uint8_t* data() const { return bytes_.data(); }
        size_t size() const { return size_; }
        uint8_t& operator[](size_t i) { return bytes_[i]; }
        const uint8_t& operator[](size_t i) const { return bytes_[i]; }

        // Zero [written, size()) — the bytes a partial fill left holding the PREVIOUS tenant.
        // After this the lease holds exactly what `std::vector<uint8_t>(size(), 0)` followed by the
        // same partial fill would have held.
        void zero_tail(size_t written) {
            if (written < size_) std::memset(bytes_.data() + written, 0, size_ - written);
        }
        void zero_all() { zero_tail(0); }

    private:
        friend class DecodeScratchPool;
        Lease(DecodeScratchPool* pool, std::vector<uint8_t> bytes, size_t size)
            : pool_(pool), bytes_(std::move(bytes)), size_(size) {}
        void release() {
            if (pool_) pool_->give(std::move(bytes_));
            pool_ = nullptr;
            bytes_.clear();
            size_ = 0;
        }
        DecodeScratchPool* pool_ = nullptr;
        std::vector<uint8_t> bytes_;
        size_t size_ = 0;
    };

    // `retained_bytes` is the total capacity this pool may hold across leases that have been
    // returned; `max_retained` bounds how many buffers it keeps regardless of size. 0 for either
    // disables retention entirely (each lease then allocates and frees, as before).
    explicit DecodeScratchPool(size_t retained_bytes, size_t max_retained = 4)
        : budget_(retained_bytes), max_retained_(max_retained) {}

    // A buffer of at least `bytes`, whose first `bytes` are the lease's extent. CONTENTS ARE
    // UNSPECIFIED: overwrite them, or call `Lease::zero_tail` / `zero_all`.
    Lease take(size_t bytes) {
        // Smallest retained buffer that already fits, so a run of small textures cannot evict the
        // one big surface's pages by repeatedly claiming and re-sizing it.
        size_t best = free_.size();
        for (size_t i = 0; i < free_.size(); ++i) {
            if (free_[i].size() < bytes) continue;
            if (best == free_.size() || free_[i].size() < free_[best].size()) best = i;
        }
        if (best == free_.size()) {
            // Nothing fits: grow the LARGEST, so the pool converges on the sizes actually used
            // instead of accumulating one buffer per distinct extent.
            for (size_t i = 0; i < free_.size(); ++i)
                if (best == free_.size() || free_[i].size() > free_[best].size()) best = i;
        }
        std::vector<uint8_t> buffer;
        if (best != free_.size()) {
            buffer = std::move(free_[best]);
            free_.erase(free_.begin() + static_cast<std::ptrdiff_t>(best));
            retained_ -= std::min(retained_, buffer.size());
        }
        // Never shrink: a smaller later request reuses the mapped pages instead of returning them
        // to the kernel and faulting a fresh set in on the next big one.
        //
        // `clear()` first when the buffer must REALLOCATE, and only then. `resize` past capacity
        // copies the existing elements into the new allocation -- 66 MiB of copy to preserve
        // contents this class explicitly does not promise. Clearing makes that copy zero elements.
        // Within capacity the opposite holds: `resize` value-initialises only the delta, where
        // clearing first would memset the whole extent.
        if (buffer.capacity() < bytes) buffer.clear();
        if (buffer.size() < bytes) buffer.resize(bytes);
        return Lease(this, std::move(buffer), bytes);
    }

    size_t retained_bytes() const { return retained_; }
    size_t retained_buffers() const { return free_.size(); }

private:
    void give(std::vector<uint8_t> bytes) {
        if (bytes.empty()) return;
        retained_ += bytes.size();
        free_.push_back(std::move(bytes));
        // Over budget: drop the SMALLEST retained buffers first. The expensive pages to re-acquire
        // are the big ones, and a small buffer costs almost nothing to re-allocate.
        while (!free_.empty() && (free_.size() > max_retained_ || retained_ > budget_)) {
            size_t smallest = 0;
            for (size_t i = 1; i < free_.size(); ++i)
                if (free_[i].size() < free_[smallest].size()) smallest = i;
            retained_ -= std::min(retained_, free_[smallest].size());
            free_.erase(free_.begin() + static_cast<std::ptrdiff_t>(smallest));
        }
    }

    std::vector<std::vector<uint8_t>> free_;
    size_t retained_ = 0;
    size_t budget_ = 0;
    size_t max_retained_ = 0;
};

// Retention budget in bytes, read once. Spelled `0` it retains nothing, i.e. the pre-pool
// behaviour; a malformed value keeps the 512 MiB default, which leaves pooling ON (see the head).
inline size_t decode_scratch_budget_bytes(const char* text) {
    const uint64_t mib = prosper::diag::env_u64_or_default_capped(
        "PROSPER_DECODE_SCRATCH_MB", text, 512ull, SIZE_MAX / (1024ull * 1024ull), "MiB",
        "512 MiB retained per thread; spell 0 exactly to retain nothing");
    return static_cast<size_t>(mib * 1024ull * 1024ull);
}

// Per-thread pool. The decode paths run on the render thread(s) and never hand a lease to another
// thread, so a thread-local pool needs no lock on the hottest allocation path in the frontend.
inline DecodeScratchPool& decode_scratch_pool() {
    static const size_t budget = decode_scratch_budget_bytes(std::getenv("PROSPER_DECODE_SCRATCH_MB"));
    static thread_local DecodeScratchPool pool(budget);
    return pool;
}

}  // namespace prosper::frontend
