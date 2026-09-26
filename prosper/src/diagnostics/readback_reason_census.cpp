#include "diagnostics/readback_reason_census.hpp"

#include "diagnostics/exit_census.hpp"
#include "../../frontends/shared/present/readback_policy.hpp"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace prosper::diagnostics {
namespace {

constexpr size_t kCount = static_cast<size_t>(ReadbackReasonSlot::Count);

constexpr std::array<const char*, kCount> kNames{
    "not-wanted", "no-color-target", "explicit-request", "bound-non-persistent",
};

// The call site casts one enum to the other; these pin that this stays sound.
static_assert(static_cast<int>(prosper::frontend::ColorReadbackReason::NoColorTarget) ==
              static_cast<int>(ReadbackReasonSlot::NoColorTarget));
static_assert(static_cast<int>(prosper::frontend::ColorReadbackReason::ExplicitRequest) ==
              static_cast<int>(ReadbackReasonSlot::ExplicitRequest));
static_assert(static_cast<int>(prosper::frontend::ColorReadbackReason::BoundNonPersistent) ==
              static_cast<int>(ReadbackReasonSlot::BoundNonPersistent));
static_assert(static_cast<int>(prosper::frontend::ColorReadbackReason::NotWanted) ==
              static_cast<int>(ReadbackReasonSlot::NotWanted));

struct State {
    std::array<std::atomic<uint64_t>, kCount> hits{};
    std::array<std::atomic<uint64_t>, kCount> bytes{};
};

State& state() {
    static State s;
    static const bool once = [] {
        register_census("PROSPER_NO_READBACK_REASON_CENSUS", [] {
            State& v = state();
            uint64_t total = 0;
            for (size_t i = 0; i < kCount; i++) total += v.hits[i].load(std::memory_order_relaxed);
            if (!total) return false;
            std::fprintf(stderr, "[readback-reason] RUN TOTAL slots=%llu",
                         static_cast<unsigned long long>(total));
            for (size_t i = 0; i < kCount; i++) {
                const uint64_t n = v.hits[i].load(std::memory_order_relaxed);
                if (!n) continue;
                // Bytes are printed only for reasons that actually cause a copy. The first
                // version printed them for `not-wanted` too, where the extent is the size of a
                // copy that never happened -- a large, precise, entirely fictional number sitting
                // next to real ones. A reader comparing the columns would have concluded the
                // opposite of the truth about where the bytes go.
                if (static_cast<ReadbackReasonSlot>(i) == ReadbackReasonSlot::NotWanted) {
                    std::fprintf(stderr, "  %s=%llu (%.1f%%, no copy)", kNames[i],
                                 static_cast<unsigned long long>(n),
                                 100.0 * static_cast<double>(n) / static_cast<double>(total));
                } else {
                    std::fprintf(stderr, "  %s=%llu (%.1f%%, %.1f MiB copied)", kNames[i],
                                 static_cast<unsigned long long>(n),
                                 100.0 * static_cast<double>(n) / static_cast<double>(total),
                                 static_cast<double>(v.bytes[i].load(std::memory_order_relaxed)) /
                                     (1024.0 * 1024.0));
                }
            }
            std::fprintf(stderr, "\n");
            std::fflush(stderr);
            return true;
        });
        return true;
    }();
    (void)once;
    return s;
}

}  // namespace

void note_readback_reason(ReadbackReasonSlot slot, uint64_t bytes) {
    const auto i = static_cast<size_t>(slot);
    if (i >= kCount) return;
    State& s = state();
    s.hits[i].fetch_add(1, std::memory_order_relaxed);
    s.bytes[i].fetch_add(bytes, std::memory_order_relaxed);
}

}  // namespace prosper::diagnostics
