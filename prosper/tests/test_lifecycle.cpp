// test_lifecycle — the cooperative stop signal (src/host/lifecycle.hpp) a long-running frontend
// uses to wind the guest run-loop down on window-close. Pure, no deps.
#include "../src/host/lifecycle.hpp"
#include <cstdio>
#include <thread>
#include <atomic>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
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
    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
