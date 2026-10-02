#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <thread>
#include <vector>

namespace {
constexpr unsigned rounds = 100, workers = 8;
constexpr uint64_t cookie = 0x74534c696665546dull;
std::atomic<unsigned> constructed{0}, destroyed{0}, invalid{0}, consumed{0};

struct Lifetime {
    uint64_t marker = cookie;
    std::vector<unsigned> storage;
    Lifetime() { constructed.fetch_add(1, std::memory_order_relaxed); }
    ~Lifetime() {
        if (marker != cookie) invalid.fetch_add(1, std::memory_order_relaxed);
        destroyed.fetch_add(1, std::memory_order_relaxed);
    }
};
// Exercises the real nontrivial C++ TLS lifetime; no manual cleanup, leak or opt-out.
Lifetime& slot() { thread_local Lifetime value; return value; }
}

int main() {
    std::mutex paths;
    const auto relative = std::filesystem::path("prosper") / "build-native-tls" / "test-scratch";
    for (unsigned round = 0; round < rounds; ++round) {
        std::array<std::thread, workers> threads;
        for (unsigned worker = 0; worker < workers; ++worker) threads[worker] = std::thread([&, worker] {
            auto& state = slot();
            if (state.marker != cookie) invalid.fetch_add(1, std::memory_order_relaxed);
            // Half retain a bounded idle allocation; half have empty storage, like metadata-only
            // warm shader keys. Moving it out models the active key's independent ownership.
            if (worker & 1u) state.storage.assign(32, worker + 1);
            auto active = std::move(state.storage);
            for (unsigned word : active) consumed.fetch_add(word, std::memory_order_relaxed);
            state.storage = std::move(active);
            {
                std::lock_guard lock(paths);
                // Pure path operations create allocation/reuse pressure while peers exit.
                // No directory/file is created and no external content is read.
                for (unsigned k = 0; k < 8; ++k) {
                    const auto path = std::filesystem::absolute(relative).lexically_normal();
                    if (path.filename() != "test-scratch") invalid.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
        for (auto& thread : threads) thread.join();
        const unsigned expected = (round + 1) * workers;
        if (constructed.load() != expected || destroyed.load() != expected || invalid.load() != 0) {
            std::fprintf(stderr, "[native-tls] FAIL round=%u constructed=%u destroyed=%u invalid=%u\n",
                         round, constructed.load(), destroyed.load(), invalid.load());
            return 1;
        }
    }
    const unsigned expected_words = rounds * 32 * (2 + 4 + 6 + 8);
    if (consumed.load() != expected_words) return 1;
    std::printf("native TLS lifetime: %u thread entries, %u constructors, %u destructors, "
                "%u payload sum, %u invalid\n", rounds * workers, constructed.load(),
                destroyed.load(), consumed.load(), invalid.load());
    return 0;
}
