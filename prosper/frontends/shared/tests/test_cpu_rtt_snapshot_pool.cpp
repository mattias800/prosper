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

    // A SMALLER extent now borrows a retained buffer whose capacity already holds it. The arm
    // here used to assert the opposite ("a different size cannot borrow old pages"), which
    // recorded the old exact-size policy rather than a correctness requirement: what reuse must
    // avoid is the reallocation, and a big-enough buffer avoids it. Insisting on an exact match
    // refused every merely-sufficient buffer and cost a render-target-sized allocation per
    // publication on a title cycling through extents.
    const size_t retained_before_reuse = resolution_change.retained_bytes();
    auto newer = resolution_change.copy(new_source.data(), new_source.size());
    checks.expect(newer.reused, "a smaller extent borrows a retained buffer that can hold it");
    // The budget must be credited the borrowed buffer's CAPACITY, not the request size. Stated as
    // a strict inequality rather than a magic number so it discriminates without pinning the
    // pool's internal capacities: the buffer taken can hold `new_source` and is strictly larger
    // than it, so a correct accounting removes strictly more than the request. Mutating the
    // subtraction to `-= bytes` makes the delta exactly the request and turns this red -- the
    // suite previously had no arm that could tell those two apart.
    checks.expect(retained_before_reuse - resolution_change.retained_bytes() > new_source.size(),
                  "reuse credits the budget the borrowed buffer's capacity, not the request size");
    // THE arm that guards the risk this change introduces. Reusing a larger buffer must publish
    // the REQUESTED length, not the donor's: a consumer reads pixels->size(). If assign() were
    // ever replaced by a memcpy that left the donor's size standing, this goes red and nothing
    // else here would.
    checks.expect(newer.pixels->size() == new_source.size(),
                  "a reused larger buffer publishes the requested size, not the donor's");
    checks.expect(std::ranges::equal(*newer.pixels, new_source),
                  "a reused larger buffer publishes byte-exact content");
    newer.pixels.reset();
    auto next = resolution_change.copy(new_source.data(), new_source.size());
    checks.expect(next.reused, "new recurring extent becomes reusable");
    checks.expect(std::ranges::equal(*next.pixels, new_source), "new extent retains byte-exact content");

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
