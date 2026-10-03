#include "diagnostics/native_host_wait.hpp"
#include "diagnostics/diag_clock.hpp"
#include <algorithm>
#include <cstdio>
#include <limits>
#ifdef _WIN32
#include <windows.h>
#endif

namespace prosper::diagnostics {
namespace {
constexpr uint64_t publishing = std::numeric_limits<uint64_t>::max();
static_assert(std::atomic<uint64_t>::is_always_lock_free);
static_assert(std::atomic<uint32_t>::is_always_lock_free);
static_assert(std::atomic<uintptr_t>::is_always_lock_free);
// Constant-initialized, trivial, never destroyed: no lazy-static guard can be held by a target
// suspended midway through publication. All formatter work happens AFTER the sampler resumes it.
constinit NativeHostWaitRegistry registry;
}   // namespace

NativeHostWaitToken NativeHostWaitRegistry::enter(const NativeHostWaitRecord& record,
                                                  PublicationHook reserved_hook,
                                                  void* opaque) noexcept {
    if (!record.native_id) return {};
    for (size_t i = 0; i < slots_.size(); ++i) {
        Slot& slot = slots_[i];
        uint64_t expected = 0;
        if (!slot.publication.compare_exchange_strong(expected, publishing,
                                                      std::memory_order_acq_rel))
            continue;
        if (reserved_hook) reserved_hook(opaque);
        if (slot.next_generation == publishing - 1) {
            slot.publication.store(0, std::memory_order_release);
            continue; // never wrap a generation into an earlier scope's token
        }
        slot.native_id.store(record.native_id, std::memory_order_relaxed);
        slot.site.store(static_cast<uint32_t>(record.site), std::memory_order_relaxed);
        slot.object.store(record.object, std::memory_order_relaxed);
        slot.mode.store(static_cast<uint32_t>(record.mode), std::memory_order_relaxed);
        slot.timeout_us.store(record.timeout_us, std::memory_order_relaxed);
        slot.entered_us.store(record.entered_us, std::memory_order_relaxed);
        const uint64_t generation = ++slot.next_generation;
        entered_run_.fetch_add(1, std::memory_order_relaxed);
        slot.publication.store(generation, std::memory_order_release);
        return {i, generation};
    }
    dropped_run_.fetch_add(1, std::memory_order_relaxed);
    return {};
}

void NativeHostWaitRegistry::leave(NativeHostWaitToken token) noexcept {
    if (token.slot >= slots_.size() || !token.generation || token.generation == publishing) return;
    uint64_t expected = token.generation;
    if (slots_[token.slot].publication.compare_exchange_strong(expected, publishing,
                                                               std::memory_order_acq_rel))
        slots_[token.slot].publication.store(0, std::memory_order_release);
}

void NativeHostWaitRegistry::retire_thread(uint32_t native_id) noexcept {
    for (Slot& slot : slots_) {
        const uint64_t generation = slot.publication.load(std::memory_order_acquire);
        if (generation && generation != publishing &&
            slot.native_id.load(std::memory_order_relaxed) == native_id &&
            slot.publication.load(std::memory_order_acquire) == generation)
            leave({static_cast<size_t>(&slot - slots_.data()), generation});
    }
}

NativeHostWaitSnapshot
NativeHostWaitRegistry::snapshot(uint32_t native_id,
                                 std::span<NativeHostWaitRecord> output) const noexcept {
    NativeHostWaitSnapshot result{};
    for (const Slot& slot : slots_) {
        const uint64_t generation = slot.publication.load(std::memory_order_acquire);
        if (!generation) continue;
        if (generation == publishing) {
            ++result.unstable_slots;
            continue;
        }
        NativeHostWaitRecord record{
            slot.native_id.load(std::memory_order_relaxed),
            static_cast<NativeHostWaitSite>(slot.site.load(std::memory_order_relaxed)),
            slot.object.load(std::memory_order_relaxed),
            static_cast<NativeHostWaitMode>(slot.mode.load(std::memory_order_relaxed)),
            slot.timeout_us.load(std::memory_order_relaxed),
            slot.entered_us.load(std::memory_order_relaxed)};
        if (slot.publication.load(std::memory_order_acquire) != generation) {
            ++result.unstable_slots;
            continue;
        }
        if (record.native_id != native_id) continue;
        if (result.found < output.size()) output[result.found] = record;
        ++result.found;
    }
    result.entered_run = entered_run_.load(std::memory_order_relaxed);
    result.dropped_run = dropped_run_.load(std::memory_order_relaxed);
    return result; // independent relaxed totals, not a coherent partition or completeness proof
}

NativeHostWaitRegistry& native_host_wait_registry() noexcept {
    return registry;
}
NativeHostWaitToken observe_current_native_host_wait(NativeHostWaitSite site, uintptr_t object,
                                                     NativeHostWaitMode mode,
                                                     uint64_t timeout_us) noexcept {
#ifdef _WIN32
    return registry.enter({GetCurrentThreadId(), site, object, mode, timeout_us, diag_now_us()});
#else
    (void)site;
    (void)object;
    (void)mode;
    (void)timeout_us;
    return {};
#endif
}
const char* native_host_wait_site_name(NativeHostWaitSite site) noexcept {
    switch (site) {
        case NativeHostWaitSite::Equeue: return "sceKernelWaitEqueue";
        case NativeHostWaitSite::PthreadOnce: return "pthread_once/contended";
    }
    return "unknown";
}

HostStackCandidates scan_host_stack_candidates(std::span<const uint64_t> words,
                                               bool (*executable)(uintptr_t),
                                               std::string (*describe)(uint64_t)) {
    HostStackCandidates result{};
    size_t used = 0;
    for (size_t i = 0; i < words.size(); ++i) {
        if (result.retained == 6) {
            result.prefix_limited = true;
            break;
        }
        ++result.examined_words;
        const uint64_t candidate = words[i];
        if (candidate < 0x10000 || !executable(static_cast<uintptr_t>(candidate))) continue;
        const std::string described = describe(candidate);
        if (described.rfind("prosper+", 0) != 0) continue;
        bool duplicate = false;
        for (size_t j = 0; j < i; ++j) duplicate |= words[j] == candidate;
        if (duplicate) continue;
        const int appended = std::snprintf(result.text.data() + used, result.text.size() - used,
                                           result.retained ? ",%s" : "%s", described.c_str());
        if (appended <= 0 || static_cast<size_t>(appended) >= result.text.size() - used) {
            result.text[used] = '\0';
            result.prefix_limited = true;
            break;
        }
        used += static_cast<size_t>(appended);
        ++result.retained;
    }
    return result;
}

NativeHostWaitTraceText format_native_host_wait_trace(
    uint32_t native_id, const HostStackCandidates& host_stack, size_t copied_words, bool armed,
    const NativeHostWaitSnapshot& snapshot, std::span<const NativeHostWaitRecord> records) {
    NativeHostWaitTraceText result{};
    const size_t stored = armed ? std::min({snapshot.found, records.size(), size_t{8}}) : 0;
    std::snprintf(result.lines[0].data(), result.lines[0].size(),
                  "[thread-trace-host-stack] tid=%u retained=%zu examined-words=%zu/%zu "
                  "prefix-limited=%d scope=copied-window-only raw-candidates-not-cfi\n",
                  native_id, host_stack.retained, host_stack.examined_words, copied_words,
                  host_stack.prefix_limited ? 1 : 0);
    // The original combined thread line has a 512-byte sink and can truncate its trailing stack.
    // Give the actual retained candidate text its own bounded line, not just a count of it.
    std::snprintf(result.lines[1].data(), result.lines[1].size(),
                  "[thread-trace-host-stack] tid=%u candidates=%s\n", native_id,
                  host_stack.text.data());
    if (!armed) {
        std::snprintf(result.lines[2].data(), result.lines[2].size(),
                      "[thread-trace-host-wait] tid=%u armed=0 found=n/a stored=n/a "
                      "scope=equeue/once-call-boundary absent=unobserved-not-no-wait\n",
                      native_id);
        return result;
    }
    std::snprintf(result.lines[2].data(), result.lines[2].size(),
                  "[thread-trace-host-wait] tid=%u armed=%d found=%zu stored=%zu "
                  "unstable-global=%zu entered-run=%llu dropped-run=%llu "
                  "scope=equeue/once-call-boundary absent=unobserved-not-no-wait\n",
                  native_id, 1, snapshot.found, stored, snapshot.unstable_slots,
                  (unsigned long long)snapshot.entered_run,
                  (unsigned long long)snapshot.dropped_run);
    for (size_t i = 0; i < stored; ++i) {
        const auto& wait = records[i];
        std::snprintf(result.lines[i + 3].data(), result.lines[i + 3].size(),
                      "[thread-trace-host-wait] tid=%u site=%s object=0x%llx "
                      "mode=%s timeout-us=%llu entered-diag-us=%llu\n",
                      native_id, native_host_wait_site_name(wait.site),
                      (unsigned long long)wait.object,
                      wait.mode == NativeHostWaitMode::Infinite ? "infinite" : "relative",
                      (unsigned long long)wait.timeout_us, (unsigned long long)wait.entered_us);
    }
    return result;
}
}   // namespace prosper::diagnostics
