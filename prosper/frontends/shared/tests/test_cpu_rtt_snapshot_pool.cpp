#include "shared/live/cpu_rtt_snapshot_pool.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <memory>
#include <ranges>
#include <source_location>
#include <string_view>
#include <thread>
#include <vector>

using prosper::frontend::CpuRttSnapshotPool;

class Checks {
public:
    void expect(bool condition, std::string_view description,
                std::source_location site = std::source_location::current()) {
        if (condition) return;
        std::cerr << "FAIL " << site.file_name() << ':' << site.line() << ": "
                  << description << '\n';
        ++failures_;
    }
    [[nodiscard]] int failures() const { return failures_; }
private:
    int failures_ = 0;
};

static bool all_bytes_equal(const std::vector<uint8_t>& bytes, size_t size, uint8_t value) {
    return bytes.size() == size && std::ranges::all_of(bytes, [value](uint8_t byte) {
        return byte == value;
    });
}

static void test_immutable_versions(Checks& checks) {
    // The renderer may retain an older version of the same guest address. Reuse starts only
    // after its last owner releases it, and copying must replace every byte of a recycled vector.
    CpuRttSnapshotPool pool(32, 2);
    std::array<uint8_t, 8> source{1, 2, 3, 4, 5, 6, 7, 8};
    const auto original = source;
    auto first = pool.copy(source.data(), source.size());
    checks.expect(first.pixels && !first.reused, "first publication owns new pages");
    const uint8_t* first_pages = first.pixels->data();
    source.fill(0xEE);
    checks.expect(std::ranges::equal(*first.pixels, original), "source mutation cannot alter published pixels");

    const std::array<uint8_t, 8> second_source{8, 7, 6, 5, 4, 3, 2, 1};
    auto second = pool.copy(second_source.data(), second_source.size());
    checks.expect(second.pixels && !second.reused, "live first version cannot be recycled");
    checks.expect(second.pixels->data() != first_pages, "live versions have distinct pages");
    checks.expect((*first.pixels)[0] == 1 && (*second.pixels)[0] == 8,
                  "separate versions retain their values");

    auto held_by_renderer = first.pixels;
    first.pixels.reset();
    checks.expect(pool.retained_buffers() == 0, "another consumer still owns the first version");
    held_by_renderer.reset();
    checks.expect(pool.retained_buffers() == 1 && pool.retained_bytes() == 8,
                  "last owner returns the first version to the bounded pool");

    const std::array<uint8_t, 8> third_source{9, 9, 9, 9, 9, 9, 9, 9};
    auto third = pool.copy(third_source.data(), third_source.size());
    checks.expect(third.reused && third.pixels->data() == first_pages,
                  "same-size publication reuses released pages");
    checks.expect(pool.retained_buffers() == 0 && pool.retained_bytes() == 0,
                  "borrow removes the pages from the free budget");
    checks.expect(std::ranges::equal(*third.pixels, third_source), "recycled copy replaces every byte");
    checks.expect((*second.pixels)[0] == 8 && (*second.pixels)[7] == 1,
                  "new publication cannot change another live version");
    checks.expect(!pool.copy(nullptr, 8).pixels, "null source is refused");
    checks.expect(!pool.copy(second_source.data(), 0).pixels, "empty source is refused");
}

static void test_budget_and_resolution_change(Checks& checks) {
    const std::array<uint8_t, 8> old_source{1, 2, 3, 4, 5, 6, 7, 8};
    const std::array<uint8_t, 7> transition_source{3, 3, 3, 3, 3, 3, 3};
    const std::array<uint8_t, 5> new_source{6, 6, 6, 6, 6};
    CpuRttSnapshotPool resolution_change(16, 4);
    auto old = resolution_change.copy(old_source.data(), old_source.size());
    auto transition = resolution_change.copy(transition_source.data(), transition_source.size());
    old.pixels.reset();
    transition.pixels.reset();
    checks.expect(resolution_change.retained_bytes() == 15, "two old sizes occupy the budget");

    auto newer = resolution_change.copy(new_source.data(), new_source.size());
    checks.expect(!newer.reused, "a different size cannot borrow old pages");
    newer.pixels.reset();
    // These three arms failed on the former policy: it refused the new active size forever.
    checks.expect(resolution_change.retained_bytes() == 12 &&
                  resolution_change.retained_buffers() == 2,
                  "new-size release evicts only idle old pages within the budget");
    auto next = resolution_change.copy(new_source.data(), new_source.size());
    checks.expect(next.reused, "new recurring extent becomes reusable");
    checks.expect(std::ranges::equal(*next.pixels, new_source), "new extent retains byte-exact content");
    checks.expect(!resolution_change.copy(old_source.data(), old_source.size()).reused,
                  "evicted old extent is not reused");

    CpuRttSnapshotPool count_bounded(16, 1);
    auto one = count_bounded.copy(old_source.data(), old_source.size());
    auto two = count_bounded.copy(old_source.data(), old_source.size());
    one.pixels.reset();
    two.pixels.reset();
    checks.expect(count_bounded.retained_buffers() == 1 && count_bounded.retained_bytes() <= 16,
                  "free list obeys count and capacity budgets");
    CpuRttSnapshotPool no_budget(0);
    auto uncached = no_budget.copy(old_source.data(), old_source.size());
    uncached.pixels.reset();
    checks.expect(no_budget.retained_buffers() == 0, "zero byte budget retains nothing");
    CpuRttSnapshotPool no_entries(32, 0);
    auto discarded = no_entries.copy(old_source.data(), old_source.size());
    discarded.pixels.reset();
    checks.expect(no_entries.retained_buffers() == 0, "zero count budget retains nothing");
}

static void test_wrapper_lifetime(Checks& checks) {
    const std::array<uint8_t, 8> source{8, 7, 6, 5, 4, 3, 2, 1};
    std::shared_ptr<const std::vector<uint8_t>> surviving;
    {
        CpuRttSnapshotPool short_lived(32);
        surviving = short_lived.copy(source.data(), source.size()).pixels;
    }
    bool peer_saw_pixels = false;
    std::jthread release_on_peer([held = std::move(surviving), &peer_saw_pixels]() mutable {
        peer_saw_pixels = (*held)[0] == 8 && (*held)[7] == 1;
        held.reset();
    });
    release_on_peer.join();
    checks.expect(peer_saw_pixels && !surviving,
                  "shared release state outlives the wrapper on another thread");
}

static void stress_worker(CpuRttSnapshotPool& pool, std::atomic<bool>& ok, size_t worker) {
    std::array<uint8_t, 4096> source{};
    std::array<std::shared_ptr<const std::vector<uint8_t>>, 3> held{};
    std::array<uint8_t, 3> held_values{};
    for (const size_t iteration : std::views::iota(size_t{0}, size_t{200})) {
        const auto value = static_cast<uint8_t>(worker * 37 + iteration);
        source.fill(value);
        auto next = pool.copy(source.data(), source.size()).pixels;
        if (!next || !all_bytes_equal(*next, source.size(), value)) ok.store(false);
        const size_t slot = iteration % held.size();
        held[slot] = std::move(next);
        held_values[slot] = value;
        for (const size_t index : std::views::iota(size_t{0}, held.size())) {
            if (held[index] && !all_bytes_equal(*held[index], source.size(), held_values[index]))
                ok.store(false);
        }
    }
}

static void test_concurrent_release(Checks& checks) {
    CpuRttSnapshotPool pool(4 * 4096, 4);
    std::atomic<bool> ok{true};
    std::vector<std::jthread> workers;
    workers.reserve(4);
    for (const size_t worker : std::views::iota(size_t{0}, size_t{4})) {
        workers.emplace_back([&pool, &ok, worker] { stress_worker(pool, ok, worker); });
    }
    for (auto& worker : workers) worker.join();
    checks.expect(ok.load(), "concurrent versions never observe another worker's pixels");
    checks.expect(pool.retained_buffers() <= 4 && pool.retained_bytes() <= 4 * 4096,
                  "concurrent release obeys the free-list bounds");
}

int main() {
    Checks checks;
    test_immutable_versions(checks);
    test_budget_and_resolution_change(checks);
    test_wrapper_lifetime(checks);
    test_concurrent_release(checks);
    if (checks.failures()) return 1;
    std::cout << "CPU RTT snapshot pool: all checks passed\n";
    return 0;
}
