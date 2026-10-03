#pragma once
#include "diagnostics/env_cache.hpp"
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace prosper::diagnostics {

// Observation of an HLE wait CALL scope, not proof of kernel parking, ownership or deadlock.
// This registry deliberately has no interrupt, wake, predicate or guest-state interface.
enum class NativeHostWaitSite : uint32_t { Equeue, PthreadOnce };
enum class NativeHostWaitMode : uint32_t { Infinite, RelativeMicroseconds };
struct NativeHostWaitRecord {
    uint32_t native_id = 0;
    NativeHostWaitSite site = NativeHostWaitSite::Equeue;
    uintptr_t object = 0;
    NativeHostWaitMode mode = NativeHostWaitMode::Infinite;
    uint64_t timeout_us = 0;
    uint64_t entered_us = 0;
};
struct NativeHostWaitToken {
    size_t slot = SIZE_MAX;
    uint64_t generation = 0;
};
struct NativeHostWaitSnapshot {
    size_t found = 0; // matching stable records, including records beyond the output capacity
    size_t unstable_slots = 0; // global publishing/retiring slots; attribution is NOT observed
    uint64_t entered_run = 0;
    uint64_t dropped_run = 0;   // global failed registrations, NOT a current-thread omission count
};

// Atomic payloads and bounded scans: a sampler can suspend a writer at ANY instruction without
// waiting for its lock, allocation, TLS initialization or publication to finish. Recycled slots
// have distinct generations; a stale token cannot withdraw a new scope. Saturated slots retire.
class NativeHostWaitRegistry {
public:
    static constexpr size_t capacity = 512;
    // A fixture may pause the real publisher at its exclusive sentinel. Production passes null.
    using PublicationHook = void (*)(void*) noexcept;
    NativeHostWaitToken enter(const NativeHostWaitRecord& record,
                              PublicationHook reserved_hook = nullptr,
                              void* opaque = nullptr) noexcept;
    void leave(NativeHostWaitToken token) noexcept;
    void retire_thread(uint32_t native_id) noexcept;
    NativeHostWaitSnapshot snapshot(uint32_t native_id,
                                    std::span<NativeHostWaitRecord> output) const noexcept;

private:
    struct Slot {
        std::atomic<uint64_t> publication{0};
        uint64_t next_generation = 0;   // accessed only under the exclusive publication sentinel
        std::atomic<uint32_t> native_id{0}, site{0}, mode{0};
        std::atomic<uintptr_t> object{0};
        std::atomic<uint64_t> timeout_us{0}, entered_us{0};
    };
    std::array<Slot, capacity> slots_{};
    std::atomic<uint64_t> entered_run_{0}, dropped_run_{0};
};

inline bool native_host_wait_observation_enabled() {
#ifdef _WIN32
    // Process-start switch. Tests launch separate armed/unarmed processes, never set it later.
    return PROSPER_ENV_ON("PROSPER_HOST_WAIT_OBSERVE");
#else
    return false;
#endif
}
NativeHostWaitRegistry& native_host_wait_registry() noexcept;
NativeHostWaitToken observe_current_native_host_wait(NativeHostWaitSite site, uintptr_t object,
                                                     NativeHostWaitMode mode,
                                                     uint64_t timeout_us) noexcept;
class NativeHostWaitScope {
public:
    NativeHostWaitScope(NativeHostWaitSite site, uintptr_t object,
                        NativeHostWaitMode mode = NativeHostWaitMode::Infinite,
                        uint64_t timeout_us = 0) noexcept {
        if (native_host_wait_observation_enabled())
            token_ = observe_current_native_host_wait(site, object, mode, timeout_us);
    }
    ~NativeHostWaitScope() {
        if (token_.generation) native_host_wait_registry().leave(token_);
    }
    NativeHostWaitScope(const NativeHostWaitScope&) = delete;
    NativeHostWaitScope& operator=(const NativeHostWaitScope&) = delete;

private:
    NativeHostWaitToken token_{};
};

const char* native_host_wait_site_name(NativeHostWaitSite site) noexcept;
// The old host scan stops at six unique raw executable PROSPER candidates. Keep that bound and
// make its incomplete prefix visible, without claiming these words are a CFI/backtrace chain.
struct HostStackCandidates {
    std::array<char, 288> text{'-'};
    size_t retained = 0;
    size_t examined_words = 0;
    bool prefix_limited = false;
};
HostStackCandidates scan_host_stack_candidates(std::span<const uint64_t> words,
                                               bool (*executable)(uintptr_t),
                                               std::string (*describe)(uint64_t));
struct NativeHostWaitTraceText {
    std::array<std::array<char, 384>, 11>
        lines{};   // stack scope/candidates + wait scope + at most eight observations
};
NativeHostWaitTraceText format_native_host_wait_trace(
    uint32_t native_id, const HostStackCandidates& host_stack, size_t copied_words, bool armed,
    const NativeHostWaitSnapshot& snapshot, std::span<const NativeHostWaitRecord> records);

}   // namespace prosper::diagnostics
