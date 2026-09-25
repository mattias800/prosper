#include "shared/live/cpu_rtt_snapshot_pool.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <memory>
#include <thread>
#include <vector>

using prosper::frontend::CpuRttSnapshotPool;

static int failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
    ++failures; \
} } while (0)

int main() {
    // A snapshot stays immutable while a later dispatch publishes to the same guest address.
    // Only release of the LAST shared owner admits its pages back to the pool.
    CpuRttSnapshotPool pool(32, 2);
    std::array<uint8_t, 8> first_source{1, 2, 3, 4, 5, 6, 7, 8};
    const auto original_first = first_source;
    auto first = pool.copy(first_source.data(), first_source.size());
    CHECK(first.pixels && !first.reused);
    const uint8_t* first_pages = first.pixels->data();
    first_source.fill(0xEE);
    CHECK(std::equal(first.pixels->begin(), first.pixels->end(),
                     original_first.begin(), original_first.end()));

    std::array<uint8_t, 8> second_source{8, 7, 6, 5, 4, 3, 2, 1};
    auto second = pool.copy(second_source.data(), second_source.size());
    CHECK(second.pixels && !second.reused);
    CHECK(second.pixels->data() != first_pages);
    CHECK((*first.pixels)[0] == 1 && (*second.pixels)[0] == 8);

    auto held_by_renderer = first.pixels;
    first.pixels.reset();
    CHECK(pool.retained_buffers() == 0);
    held_by_renderer.reset();
    CHECK(pool.retained_buffers() == 1 && pool.retained_bytes() == 8);

    std::array<uint8_t, 8> third_source{9, 9, 9, 9, 9, 9, 9, 9};
    auto third = pool.copy(third_source.data(), third_source.size());
    CHECK(third.reused && third.pixels->data() == first_pages);
    CHECK(pool.retained_buffers() == 0 && pool.retained_bytes() == 0);
    CHECK(std::equal(third.pixels->begin(), third.pixels->end(),
                     third_source.begin(), third_source.end()));
    CHECK((*second.pixels)[0] == 8 && (*second.pixels)[7] == 1);

    // Exact size is part of the borrow contract: a smaller resolution may not retain a vector
    // whose size still names the old extent. Over-budget releases are simply discarded.
    std::array<uint8_t, 5> small_source{4, 4, 4, 4, 4};
    auto small = pool.copy(small_source.data(), small_source.size());
    CHECK(!small.reused && small.pixels->size() == small_source.size());
    CpuRttSnapshotPool no_retention(0);
    auto uncached = no_retention.copy(second_source.data(), second_source.size());
    uncached.pixels.reset();
    CHECK(no_retention.retained_buffers() == 0);
    CHECK(!pool.copy(nullptr, 8).pixels);
    CHECK(!pool.copy(second_source.data(), 0).pixels);

    {
        CpuRttSnapshotPool bounded(16, 1);
        auto one = bounded.copy(original_first.data(), original_first.size());
        auto two = bounded.copy(second_source.data(), second_source.size());
        one.pixels.reset();
        two.pixels.reset();
        CHECK(bounded.retained_buffers() == 1 && bounded.retained_bytes() <= 16);
    }

    // Destruction of the wrapper cannot invalidate a snapshot released by another thread.
    std::shared_ptr<const std::vector<uint8_t>> outliving_snapshot;
    {
        CpuRttSnapshotPool short_lived(32);
        outliving_snapshot = short_lived.copy(second_source.data(), second_source.size()).pixels;
    }
    std::thread release_on_peer([held = std::move(outliving_snapshot)]() mutable {
        CHECK((*held)[0] == 8 && (*held)[7] == 1);
        held.reset();
    });
    release_on_peer.join();
    CHECK(!outliving_snapshot);

    // Different producers may copy while the renderer releases earlier versions on other
    // threads. A vector can return to the free list only after its final reader is gone.
    CpuRttSnapshotPool concurrent(4 * 4096, 4);
    std::atomic<bool> concurrent_ok{true};
    std::array<std::thread, 4> workers;
    for (size_t worker = 0; worker < workers.size(); ++worker) {
        workers[worker] = std::thread([&, worker] {
            std::array<uint8_t, 4096> source{};
            std::array<std::shared_ptr<const std::vector<uint8_t>>, 3> held{};
            std::array<uint8_t, 3> held_values{};
            for (size_t iteration = 0; iteration < 200; ++iteration) {
                const uint8_t value = static_cast<uint8_t>(worker * 37 + iteration);
                source.fill(value);
                auto next = concurrent.copy(source.data(), source.size()).pixels;
                if (!next || next->size() != source.size() ||
                    !std::all_of(next->begin(), next->end(),
                                 [value](uint8_t byte) { return byte == value; }))
                    concurrent_ok.store(false, std::memory_order_relaxed);
                const size_t replaced = iteration % held.size();
                held[replaced] = std::move(next);
                held_values[replaced] = value;
                for (size_t slot = 0; slot < held.size(); ++slot) {
                    const auto& snapshot = held[slot];
                    if (snapshot && !std::all_of(snapshot->begin(), snapshot->end(),
                        [expected = held_values[slot]]
                        (uint8_t byte) { return byte == expected; }))
                        concurrent_ok.store(false, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& worker : workers) worker.join();
    CHECK(concurrent_ok.load(std::memory_order_relaxed));
    CHECK(concurrent.retained_buffers() <= 4 && concurrent.retained_bytes() <= 4 * 4096);

    if (failures) return 1;
    std::puts("CPU RTT snapshot pool: all checks passed");
    return 0;
}
