// test_lifecycle — the cooperative stop signal (src/host/platform/lifecycle.hpp) a long-running frontend
// uses to wind the guest run-loop down on window-close. Pure, no deps.
#include "host/platform/lifecycle.hpp"
#include <gtest/gtest.h>
#include <cstdio>
#include <thread>
#include <atomic>
#include <chrono>
#include <future>

using namespace prosper;

#define CHECK(c, m) EXPECT_TRUE(c) << (m)

TEST(Lifecycle, Contract) {
    printf("== test_lifecycle ==\n");
    prosper_reset_stop();
    CHECK(!prosper_stop_requested(), "not stopped after reset");

    prosper_request_stop();
    CHECK(prosper_stop_requested(), "stop_requested true after request");

    prosper_request_stop();   // idempotent
    CHECK(prosper_stop_requested(), "request is idempotent");

    prosper_reset_stop();
    CHECK(!prosper_stop_requested(), "reset clears the flag");

    // Cross-thread: a worker requests stop, the main thread observes it.
    std::atomic<bool> started{false};
    std::thread t([&]{ started.store(true); prosper_request_stop(); });
    while (!started.load()) std::this_thread::yield();
    t.join();
    CHECK(prosper_stop_requested(), "stop set from another thread is visible");

    prosper_reset_stop();
    CHECK(!prosper_paused(), "not paused after reset");

    prosper_set_paused(true);
    CHECK(prosper_paused(), "pause is visible after request");
    auto resume_waiter = std::async(std::launch::async, [] { return prosper_wait_while_paused(); });
    CHECK(resume_waiter.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout,
          "pause blocks boundary-aware work");
    prosper_set_paused(false);
    CHECK(resume_waiter.wait_for(std::chrono::seconds(1)) == std::future_status::ready &&
              resume_waiter.get(),
          "resume releases paused work");

    prosper_set_paused(true);
    auto stop_waiter = std::async(std::launch::async, [] { return prosper_wait_while_paused(); });
    CHECK(stop_waiter.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout,
          "second waiter reaches paused state");
    prosper_request_stop();
    CHECK(stop_waiter.wait_for(std::chrono::seconds(1)) == std::future_status::ready &&
              !stop_waiter.get(),
          "stop releases paused work without resuming it");

    prosper_reset_stop();
    CHECK(!prosper_stop_requested() && !prosper_paused(),
          "reset clears stop and pause together");
}
