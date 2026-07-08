// test_sync_delete — guards sceKernelDeleteEventFlag / sceKernelDeleteSema against blocked waiters
// (issue #104). Deleting an event flag or semaphore while a thread is parked in WaitEventFlag /
// WaitSema previously free()'d the object out from under the waiter (destroying a pthread condvar
// with waiters is UB). The fix marks the object deleted, wakes waiters (which return EACCES), and
// defers the free to the last waiter leaving. This drives it through the NID registry: park a
// waiter, delete from the main thread, and assert the waiter wakes with EACCES and nothing crashes.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <cstdint>
#include <atomic>
#include <chrono>
#include <thread>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static constexpr uint32_t kEACCES = 0x8002000Du;

int main() {
    printf("== test_sync_delete ==\n");
    register_builtin_hle();

    auto ef_create = Hle::lookup(nid_hash("sceKernelCreateEventFlag"));
    auto ef_wait   = Hle::lookup(nid_hash("sceKernelWaitEventFlag"));
    auto ef_delete = Hle::lookup(nid_hash("sceKernelDeleteEventFlag"));
    auto se_create = Hle::lookup(nid_hash("sceKernelCreateSema"));
    auto se_wait   = Hle::lookup(nid_hash("sceKernelWaitSema"));
    auto se_delete = Hle::lookup(nid_hash("sceKernelDeleteSema"));
    CHECK(ef_create && ef_wait && ef_delete && se_create && se_wait && se_delete, "ef/sema fns registered");
    if (!(ef_create && ef_wait && ef_delete && se_create && se_wait && se_delete)) { printf("== FAIL ==\n"); return 1; }

    // --- EventFlag: park a thread on an infinite wait for a bit-pattern that never gets set, then
    //     delete the flag. The waiter must wake with EACCES (not hang, not crash on freed memory). ---
    {
        void* ef = nullptr;
        ef_create((uint64_t)(uintptr_t)&ef, 0, 0, 0 /*initPattern*/, 0, 0);
        CHECK(ef != nullptr, "event flag created");
        std::atomic<uint64_t> wret{~0ull}; std::atomic<bool> done{false};
        std::thread t([&]{
            uint64_t r = ef_wait((uint64_t)(uintptr_t)ef, 0x1 /*pattern*/, 0 /*mode OR*/, 0, 0 /*infinite*/, 0);
            wret.store(r); done.store(true);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));   // let it block in the wait
        CHECK(!done.load(), "event-flag waiter is parked before delete");
        ef_delete((uint64_t)(uintptr_t)ef, 0, 0, 0, 0, 0);
        for (int i = 0; i < 200 && !done.load(); i++) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        CHECK(done.load(), "DeleteEventFlag woke the parked waiter (no hang)");
        if (done.load()) t.join(); else { t.detach(); }
        CHECK((uint32_t)wret.load() == kEACCES, "woken event-flag waiter returned EACCES");
    }

    // --- Semaphore: same shape — park on a count that never arrives, delete, expect EACCES wake. ---
    {
        void* se = nullptr;
        se_create((uint64_t)(uintptr_t)&se, 0, 0, 0 /*initCount*/, 8 /*maxCount*/, 0);
        CHECK(se != nullptr, "semaphore created");
        std::atomic<uint64_t> wret{~0ull}; std::atomic<bool> done{false};
        std::thread t([&]{
            uint64_t r = se_wait((uint64_t)(uintptr_t)se, 1 /*need*/, 0 /*infinite*/, 0, 0, 0);
            wret.store(r); done.store(true);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        CHECK(!done.load(), "semaphore waiter is parked before delete");
        se_delete((uint64_t)(uintptr_t)se, 0, 0, 0, 0, 0);
        for (int i = 0; i < 200 && !done.load(); i++) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        CHECK(done.load(), "DeleteSema woke the parked waiter (no hang)");
        if (done.load()) t.join(); else { t.detach(); }
        CHECK((uint32_t)wret.load() == kEACCES, "woken semaphore waiter returned EACCES");
    }

    // --- Delete with NO waiters must also be safe (frees immediately, no crash on a later create). ---
    {
        void* ef = nullptr; ef_create((uint64_t)(uintptr_t)&ef, 0, 0, 0, 0, 0);
        ef_delete((uint64_t)(uintptr_t)ef, 0, 0, 0, 0, 0);
        void* se = nullptr; se_create((uint64_t)(uintptr_t)&se, 0, 0, 0, 8, 0);
        se_delete((uint64_t)(uintptr_t)se, 0, 0, 0, 0, 0);
        CHECK(true, "delete with no waiters frees cleanly");
    }

    if (fails) { printf("== FAIL: %d check(s) ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
