// #4003: a drainer's earlier zero-active gate must not authorize a new submit's completions.
// This target compiles the actual CP with test-only rendezvous; the shipping object has none.
#include "gpu/pm4/command_processor.hpp"
#include "gpu/pm4/pending_write_snapshot.hpp"
#include "gpu/pm4/pm4_decode.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

using namespace prosper::gpu;

namespace {
int failures = 0;
void check(bool condition, const char* name) {
    std::printf("[%s] %s\n", condition ? "ok" : "FAIL", name);
    if (!condition) ++failures;
}
[[noreturn]] void apparatus_failure(const char* name) {
    std::fprintf(stderr, "[apparatus-failure] %s\n", name);
    std::fflush(nullptr);
    // A stuck worker can still retain the observer's references. Never unwind those captures.
    std::_Exit(2);
}
void set_environment(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
uint32_t packet_header(uint32_t length, uint32_t opcode, uint32_t reg) {
    return 0xC0000000u | (((length - 2u) & 0x3fffu) << 16u) | (opcode << 8u) |
        ((reg & (R_NUM - 1u)) << 2u);
}
void append_release(std::vector<uint32_t>& stream, uint64_t* label, uint64_t value) {
    const auto address = reinterpret_cast<uint64_t>(label);
    const uint32_t packet[] = {packet_header(7, IT_NOP, R_RELEASE_MEM),
        static_cast<uint32_t>(address), static_cast<uint32_t>(address >> 32), 2,
        static_cast<uint32_t>(value), static_cast<uint32_t>(value >> 32), 0x04};
    stream.insert(stream.end(), std::begin(packet), std::end(packet));
}
template <typename Predicate>
std::optional<PendingWriteSnapshot> observe(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    do {
        auto snapshot = try_pending_write_snapshot();
        if (snapshot && predicate(*snapshot)) return snapshot;
        std::this_thread::yield();
    } while (std::chrono::steady_clock::now() < deadline);
    return std::nullopt;
}

enum class Outcome { None, FreshGate, EarlyPublication };
struct Rendezvous {
    std::mutex fixture_mutex;
    std::condition_variable fixture_cv;
    bool fixture_a_landed = false;
    bool fixture_release_a = false;
    bool fixture_inflight_wait = false;
    bool fixture_post_cv_park = false;
    bool fixture_resume_drainer = false;
    bool fixture_release_b = false;
    bool fixture_drainer_finished = false;
    unsigned fixture_private_publications = 0;
    Outcome fixture_outcome = Outcome::None;
    std::vector<uint64_t> fixture_landing_order;
    bool fixture_origins_valid = true;

    template <typename Predicate>
    void await(Predicate predicate, const char* name) {
        std::unique_lock lock(fixture_mutex);
        if (!fixture_cv.wait_for(lock, std::chrono::seconds(2), predicate))
            apparatus_failure(name);
    }
    void signal_inflight_wait() {
        std::lock_guard lock(fixture_mutex);
        fixture_inflight_wait = true;
        fixture_cv.notify_all();
    }
    void park_after_cv() {
        std::unique_lock lock(fixture_mutex);
        if (fixture_post_cv_park) return;
        fixture_post_cv_park = true;
        fixture_cv.notify_all();
        if (!fixture_cv.wait_for(lock, std::chrono::seconds(2),
                [&] { return fixture_resume_drainer; }))
            apparatus_failure("post-CV rendezvous was not released");
    }
    void signal_fresh_gate() {
        std::lock_guard lock(fixture_mutex);
        if (fixture_post_cv_park && fixture_outcome == Outcome::None)
            fixture_outcome = Outcome::FreshGate;
        fixture_cv.notify_all();
    }
};
// Only the explicit synchronous drainer owns a context. The detached worker's checkpoints are
// no-ops, and the owning thread is joined before this pointer's target can be destroyed.
thread_local Rendezvous* fixture_drainer = nullptr;

void modern_control() {
    prosper_gpu_drain_completion_writes();
    const auto before = observe([](const auto& q) {
        return q.queued == 0 && q.inflight_batches == 0 && q.active_submits == 0;
    });
    if (!before) apparatus_failure("initial queue did not become empty");

    constexpr uint64_t a_value = 0x61000001u;
    constexpr uint64_t b1_value = 0x61000002u;
    constexpr uint64_t b2_value = 0x61000003u;
    alignas(8) uint64_t a = 0, b = 0;
    Rendezvous rendezvous;
    set_guest_gpu_write_observer([&](uint64_t address, uint64_t bytes, const char* origin) {
        if (address != reinterpret_cast<uint64_t>(&a) &&
            address != reinterpret_cast<uint64_t>(&b)) return;
        // apply_deferred_effect has already written these bytes. One inflight apply excludes
        // another writer; the callback mutex publishes this value before main inspects it.
        const uint64_t landed = address == reinterpret_cast<uint64_t>(&a) ? a : b;
        const bool active = prosper_gpu_submit_scope_active();
        std::unique_lock lock(rendezvous.fixture_mutex);
        rendezvous.fixture_landing_order.push_back(landed);
        rendezvous.fixture_origins_valid &= bytes == 8 && origin &&
            std::strcmp(origin, "RELEASE_MEM") == 0;
        if (address == reinterpret_cast<uint64_t>(&a)) {
            rendezvous.fixture_a_landed = true;
            rendezvous.fixture_cv.notify_all();
            if (!rendezvous.fixture_cv.wait_for(lock, std::chrono::seconds(2),
                    [&] { return rendezvous.fixture_release_a; }))
                apparatus_failure("worker A rendezvous was not released");
        } else if (active) {
            ++rendezvous.fixture_private_publications;
            rendezvous.fixture_outcome = Outcome::EarlyPublication;
            rendezvous.fixture_cv.notify_all();
            if (!rendezvous.fixture_cv.wait_for(lock, std::chrono::seconds(2),
                    [&] { return rendezvous.fixture_release_b; }))
                apparatus_failure("early B publication was not released for cleanup");
        }
    });

    GpuState state;
    std::vector<uint32_t> stream;
    append_release(stream, &a, a_value);
    prosper_gpu_submit_scope_begin();
    check(run_command_buffer(stream.data(), stream.size(), state) == 1,
          "A queues through the real completion packet route");
    prosper_gpu_submit_scope_end();
    rendezvous.await([&] { return rendezvous.fixture_a_landed; }, "worker did not apply A");
    const auto held = observe([](const auto& q) {
        return q.queued == 0 && q.inflight_batches == 1 && q.active_submits == 0;
    });
    check(held.has_value(), "worker A is genuinely inflight after its label write");
    if (!held) apparatus_failure("worker A state was not observable");

    std::thread drainer([&] {
        fixture_drainer = &rendezvous;
        prosper_gpu_drain_completion_writes();
        fixture_drainer = nullptr;
        {
            std::lock_guard lock(rendezvous.fixture_mutex);
            rendezvous.fixture_drainer_finished = true;
        }
        rendezvous.fixture_cv.notify_all();
    });
    rendezvous.await([&] { return rendezvous.fixture_inflight_wait; },
                     "old drainer did not enter the actual inflight CV wait");
    check(true, "old drainer passed its initial gate and entered the inflight wait");
    {
        std::lock_guard lock(rendezvous.fixture_mutex);
        rendezvous.fixture_release_a = true;
    }
    rendezvous.fixture_cv.notify_all();
    rendezvous.await([&] { return rendezvous.fixture_post_cv_park; },
                     "old drainer did not reach its post-CV park");
    const auto retired = observe([](const auto& q) {
        return q.queued == 0 && q.inflight_batches == 0 && q.active_submits == 0;
    });
    check(retired && a == a_value, "A retires before the new empty-queue submit admission");
    if (!retired) apparatus_failure("A did not retire at the forced handoff");

    prosper_gpu_submit_scope_begin();
    stream.clear();
    append_release(stream, &b, b1_value);
    append_release(stream, &b, b2_value);
    check(run_command_buffer(stream.data(), stream.size(), state) == 2,
          "new active submit queues B1 and B2 through real packets");
    const auto private_queue = observe([](const auto& q) {
        return q.queued == 2 && q.inflight_batches == 0 && q.active_submits == 1;
    });
    check(private_queue.has_value(), "B1 and B2 are queued behind the real active submit");
    if (!private_queue) apparatus_failure("new submit did not establish the intended private queue");
    {
        std::lock_guard lock(rendezvous.fixture_mutex);
        rendezvous.fixture_resume_drainer = true;
    }
    rendezvous.fixture_cv.notify_all();
    rendezvous.await([&] { return rendezvous.fixture_outcome != Outcome::None; },
                     "old drainer produced neither a fresh gate nor an actual publication");
    Outcome outcome;
    unsigned private_publications;
    {
        std::lock_guard lock(rendezvous.fixture_mutex);
        outcome = rendezvous.fixture_outcome;
        private_publications = rendezvous.fixture_private_publications;
    }
    check(outcome == Outcome::FreshGate,
          "resumed old drainer rechecks the new active submit before publication");
    check(private_publications == 0,
          "no actual completion publication occurs before the new submit returns");
    // On the bad arm B1's post-write callback holds its writer, so the ordinary label is stable.
    // On the good arm the fresh gate holds both completions in the queue. No racing read oracle.
    const auto boundary = observe([&](const auto& q) {
        return q.active_submits == 1 &&
            (outcome == Outcome::FreshGate ? q.queued == 2 && q.inflight_batches == 0 :
                                            q.queued == 1 && q.inflight_batches == 1);
    });
    if (!boundary) apparatus_failure("actual outcome did not match its queue boundary");
    check(b == 0, "new submit's label remains private at the synchronized boundary");

    prosper_gpu_submit_scope_end();
    {
        std::lock_guard lock(rendezvous.fixture_mutex);
        rendezvous.fixture_release_b = true;
    }
    rendezvous.fixture_cv.notify_all();
    rendezvous.await([&] { return rendezvous.fixture_drainer_finished; },
                     "old drainer did not finish after the submit return checkpoint");
    drainer.join();
    prosper_gpu_drain_completion_writes();
    set_guest_gpu_write_observer({});
    check(rendezvous.fixture_landing_order == std::vector<uint64_t>{a_value, b1_value, b2_value},
          "all real completion writes preserve A then B1 then B2 FIFO landing order");
    check(rendezvous.fixture_origins_valid, "all three observed writes retain exact completion origin");
    check(b == b2_value, "B2 becomes visible after the actual submit return checkpoint");
    const auto final = observe([](const auto& q) {
        return q.queued == 0 && q.inflight_batches == 0 && q.active_submits == 0;
    });
    check(final && final->scope_begins == before->scope_begins + 2 &&
          final->scope_ends == before->scope_ends + 2,
          "cleanup leaves no queued write, inflight apply or unmatched real submit scope");
}

} // namespace

extern "C" void prosper_pending_drain_inflight_wait_for_test() {
    if (fixture_drainer) fixture_drainer->signal_inflight_wait();
}
extern "C" void prosper_pending_drain_resume_for_test() {
    if (fixture_drainer) fixture_drainer->park_after_cv();
}
extern "C" void prosper_pending_drain_active_wait_for_test() {
    if (fixture_drainer) fixture_drainer->signal_fresh_gate();
}

int main() {
    set_environment("PROSPER_EOP_WRITE_SYNC", "0");
    std::printf("Actual CPU pending-drain control; default visibility; Vulkan UNRUN\n");
    modern_control();
    std::printf("== %s: %d failures ==\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
