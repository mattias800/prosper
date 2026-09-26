#include "diagnostics/transfer_pressure.hpp"

#include "diagnostics/exit_census.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace prosper::diagnostics {
namespace {

constexpr size_t kCount = static_cast<size_t>(Transfer::Count);

constexpr std::array<const char*, kCount> kNames{
    "storage-materialize", "buffer-upload", "buffer-compare", "rtt-snapshot", "detile",
};
static_assert(kNames.size() == kCount, "every Transfer needs a stable name; logs are grepped");

struct State {
    std::array<std::atomic<uint64_t>, kCount> bytes{};
    std::array<std::atomic<uint64_t>, kCount> calls{};
    std::atomic<uint64_t> first_ns{0};
    std::atomic<uint64_t> last_ns{0};
};

uint64_t now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

double warn_mibps() {
    static const double threshold = [] {
        const char* value = std::getenv("PROSPER_TRANSFER_PRESSURE_WARN_MIBPS");
        if (!value || !*value) return 256.0;
        const double parsed = std::strtod(value, nullptr);
        return parsed > 0.0 ? parsed : 256.0;
    }();
    return threshold;
}

State& state() {
    static State s;
    static const bool once = [] {
        register_census("PROSPER_NO_TRANSFER_PRESSURE", [] {
            State& v = state();
            uint64_t total = 0;
            for (size_t i = 0; i < kCount; i++)
                total += v.bytes[i].load(std::memory_order_relaxed);
            if (!total) return false;
            const uint64_t first = v.first_ns.load(std::memory_order_relaxed);
            const uint64_t last = v.last_ns.load(std::memory_order_relaxed);
            const double seconds = last > first ? (last - first) / 1e9 : 0.0;
            const double MiB = 1024.0 * 1024.0;
            const double rate = seconds > 0.0 ? (total / MiB) / seconds : 0.0;
            // The rate is the finding, so it leads, and the verdict is in the first token so a
            // reader grepping one line per run does not have to interpret a number.
            std::fprintf(stderr, "[transfer-pressure] %s host-copy %.0f MiB/s (%.1f MiB over %.1fs)",
                         rate >= warn_mibps() ? "HIGH" : "ok", rate, total / MiB, seconds);
            for (size_t i = 0; i < kCount; i++) {
                const uint64_t b = v.bytes[i].load(std::memory_order_relaxed);
                if (!b) continue;
                std::fprintf(stderr, "  %s=%.1fMiB/%lluc", kNames[i], b / MiB,
                             static_cast<unsigned long long>(
                                 v.calls[i].load(std::memory_order_relaxed)));
            }
            if (rate >= warn_mibps())
                std::fprintf(stderr,
                             "  <- this much host copying is usually a residency or dirty-tracking "
                             "gap, not real work; see docs/RENDERER_ARCHITECTURE_GAPS_2026_09_25.md");
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

const char* transfer_name(Transfer category) {
    const auto i = static_cast<size_t>(category);
    return i < kCount ? kNames[i] : "unknown";
}

void note_transfer(Transfer category, uint64_t bytes) {
    const auto i = static_cast<size_t>(category);
    if (i >= kCount || !bytes) return;
    State& s = state();
    s.bytes[i].fetch_add(bytes, std::memory_order_relaxed);
    s.calls[i].fetch_add(1, std::memory_order_relaxed);
    const uint64_t now = now_ns();
    uint64_t expected = 0;
    s.first_ns.compare_exchange_strong(expected, now, std::memory_order_relaxed);
    s.last_ns.store(now, std::memory_order_relaxed);
}

uint64_t transfer_bytes(Transfer category) {
    const auto i = static_cast<size_t>(category);
    return i < kCount ? state().bytes[i].load(std::memory_order_relaxed) : 0;
}

}  // namespace prosper::diagnostics
