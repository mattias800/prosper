// test_sync_delete — guards sceKernelDeleteEventFlag / sceKernelDeleteSema against blocked waiters
// (issue #104). Deleting an event flag or semaphore while a thread is parked in WaitEventFlag /
// WaitSema previously free()'d the object out from under the waiter (destroying a pthread condvar
// with waiters is UB). The fix marks the object deleted, wakes waiters (which return EACCES), and
// defers the free to the last waiter leaving. This drives it through the NID registry: park a
// waiter, delete from the main thread, and assert the waiter wakes with EACCES and nothing crashes.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include "../src/hle/sce_errno.hpp"   // kSceKernelErrorEINVAL (#1963)
#include <cstdio>
#include <cstdint>
#include <cinttypes>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <pthread.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static constexpr uint32_t kEACCES = 0x8002000Du;
static constexpr uint32_t kEBUSY = 0x80020010u;
static constexpr uint32_t kETIMEDOUT = 0x8002003Cu;

static bool set_test_env(const char* name, const char* value) {
#ifdef _WIN32
    return _putenv_s(name, value) == 0;
#else
    return setenv(name, value, 1) == 0;
#endif
}

static int file_descriptor(FILE* file) {
#ifdef _WIN32
    return _fileno(file);
#else
    return fileno(file);
#endif
}

static int duplicate_descriptor(int fd) {
#ifdef _WIN32
    return _dup(fd);
#else
    return dup(fd);
#endif
}

static int replace_descriptor(int from, int to) {
#ifdef _WIN32
    return _dup2(from, to);
#else
    return dup2(from, to);
#endif
}

static void close_descriptor(int fd) {
#ifdef _WIN32
    _close(fd);
#else
    close(fd);
#endif
}

int main() {
    printf("== test_sync_delete ==\n");
    char formatted_tid[32]{};
    snprintf(formatted_tid, sizeof(formatted_tid), "%" PRIu64,
             sync_trace_tid_value(0x80000000u));
    CHECK(strcmp(formatted_tid, "2147483648") == 0,
          "sync trace formats an unsigned high-bit native TID");

    char pthread_filter[32]{};
    snprintf(pthread_filter, sizeof(pthread_filter), "%llu",
             (unsigned long long)(uintptr_t)pthread_self());
    CHECK(set_test_env("PROSPER_SYNCLOG", "1"), "sync trace enabled for regression");
    CHECK(set_test_env("PROSPER_SYNCLOG_SEMA_ONLY", "1"), "semaphore-only trace enabled");
    CHECK(set_test_env("PROSPER_SYNCLOG_DELAY_MS", "20"), "sync trace delay configured");
    CHECK(set_test_env("PROSPER_SYNCLOG_PTHREAD", pthread_filter), "sync trace pthread filter configured");

    FILE* trace_file = tmpfile();
    int saved_stderr = -1;
    bool capturing = false;
    if (trace_file) {
        fflush(stderr);
        saved_stderr = duplicate_descriptor(file_descriptor(stderr));
        capturing = saved_stderr >= 0 &&
                    replace_descriptor(file_descriptor(trace_file), file_descriptor(stderr)) >= 0;
    }
    CHECK(capturing, "stderr trace capture initialized");

    register_builtin_hle();

    auto ef_create = Hle::lookup(nid_hash("sceKernelCreateEventFlag"));
    auto ef_poll   = Hle::lookup(nid_hash("sceKernelPollEventFlag"));
    auto ef_wait   = Hle::lookup(nid_hash("sceKernelWaitEventFlag"));
    auto ef_delete = Hle::lookup(nid_hash("sceKernelDeleteEventFlag"));
    auto se_create = Hle::lookup(nid_hash("sceKernelCreateSema"));
    auto se_wait   = Hle::lookup(nid_hash("sceKernelWaitSema"));
    auto se_delete = Hle::lookup(nid_hash("sceKernelDeleteSema"));
    auto se_signal = Hle::lookup(nid_hash("sceKernelSignalSema"));
    CHECK(ef_create && ef_poll && ef_wait && ef_delete && se_create && se_wait && se_delete && se_signal,
          "ef/sema fns registered");
    if (!(ef_create && ef_poll && ef_wait && ef_delete && se_create && se_wait && se_delete && se_signal)) {
        printf("== FAIL ==\n"); return 1;
    }

    // --- #1963: sceKernelCreateEventFlag must VALIDATE its out-pointer, not merely test it
    //     non-null. 0x80 is the exact value Little Nightmares III passes (a null audio-thread
    //     singleton plus the field offset `this + 0x80`); it is non-zero, so the old `if (a0)`
    //     accepted it and prosper faulted writing to address 0x80 — inside its own code, with a
    //     report naming neither the API nor a usable guest frame.
    //
    //     This arm is only meaningful because the value is UNMAPPED-but-non-null. A plain null
    //     would be caught by the old code too, so an arm using 0 would pass without the fix and
    //     prove nothing. Reaching the handler at all is the assertion: if the guard regresses, this
    //     test does not fail — it CRASHES, which is still a red result and is exactly the failure
    //     the fix exists to remove.
    {
        const uint64_t bad_out_ptr = 0x80;
        const uint64_t rc = ef_create(bad_out_ptr, 0, 0x20, 0, 0, 0);
        CHECK(rc == prosper::hle::kSceKernelErrorEFAULT,
              "CreateEventFlag refuses a non-null but unwritable out-pointer with EFAULT");
        // A refusal must not have published anything: the handler returns before allocating, so
        // there is no object to leak and nothing was written anywhere.
    }
    // The success path must still work — a guard that refuses everything would pass the arm above.
    {
        void* slot = nullptr;
        const uint64_t rc = ef_create((uint64_t)(uintptr_t)&slot, 0, 0x20, 0x5a5a, 0, 0);
        CHECK(rc == 0 && slot != nullptr,
              "CreateEventFlag still succeeds through a writable out-pointer");
        if (slot) ef_delete((uint64_t)(uintptr_t)slot, 0, 0, 0, 0, 0);
    }

    // --- EventFlag: park a thread on an infinite wait for a bit-pattern that never gets set, then
    //     delete the flag. The waiter must wake with EACCES (not hang, not crash on freed memory). ---
    {
        void* ef = nullptr;
        ef_create((uint64_t)(uintptr_t)&ef, 0, 0, 0 /*initPattern*/, 0, 0);
        CHECK(ef != nullptr, "event flag created");
        uint64_t observed_pattern = ~0ull;
        const uint64_t poll_result = ef_poll((uint64_t)(uintptr_t)ef, 0x1 /*pattern*/,
                                             0 /*mode OR*/,
                                             (uint64_t)(uintptr_t)&observed_pattern, 0, 0);
        CHECK((uint32_t)poll_result == kEBUSY,
              "unmatched event-flag poll returned SCE_KERNEL_ERROR_EBUSY");
        CHECK(observed_pattern == 0, "event-flag poll reported the current pattern");
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
        static const char delay_origin_name[] = "delay-origin";
        se_create((uint64_t)(uintptr_t)&se, (uint64_t)(uintptr_t)delay_origin_name, 0,
                  0 /*initCount*/, 8 /*maxCount*/, 0);
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

    // Focus is an object-lifetime identity, not just an address. Focus A from the filtered thread,
    // delete it, require allocator reuse for B, then signal B from an excluded thread. A stale
    // pointer-only focus would incorrectly retain that signal as causal traffic for A.
    {
        void* focused = nullptr;
        se_create((uint64_t)(uintptr_t)&focused, (uint64_t)(uintptr_t)"focus-a", 0, 0, 8, 0);
        CHECK(focused != nullptr, "focused semaphore created");
        const uintptr_t focused_address = (uintptr_t)focused;
        uint32_t timeout = 1;
        const uint64_t wait_result = se_wait((uint64_t)focused_address, 1,
                                             (uint64_t)(uintptr_t)&timeout, 0, 0, 0);
        CHECK((uint32_t)wait_result == kETIMEDOUT, "focused semaphore wait timed out");
        se_delete((uint64_t)focused_address, 0, 0, 0, 0, 0);

#ifdef _WIN32
        void* reused = nullptr;
        std::vector<void*> held;
        for (int i = 0; i < 256 && !reused; ++i) {
            void* candidate = nullptr;
            se_create((uint64_t)(uintptr_t)&candidate, (uint64_t)(uintptr_t)"focus-b", 0, 0, 8, 0);
            if ((uintptr_t)candidate == focused_address) reused = candidate;
            else held.push_back(candidate);
        }
        CHECK(reused != nullptr, "allocator reused the deleted focused semaphore address");
        if (reused) {
            std::thread signaler([&] { se_signal((uint64_t)(uintptr_t)reused, 1, 0, 0, 0, 0); });
            signaler.join();
            se_delete((uint64_t)(uintptr_t)reused, 0, 0, 0, 0, 0);
        }
        for (void* candidate : held)
            se_delete((uint64_t)(uintptr_t)candidate, 0, 0, 0, 0, 0);
#else
        CHECK(true, "deleted-focus address-reuse regression is Windows-specific");
#endif
    }

    // --- Delete with NO waiters must also be safe (frees immediately, no crash on a later create). ---
    {
        void* ef = nullptr; ef_create((uint64_t)(uintptr_t)&ef, 0, 0, 0, 0, 0);
        ef_delete((uint64_t)(uintptr_t)ef, 0, 0, 0, 0, 0);
        void* se = nullptr; se_create((uint64_t)(uintptr_t)&se, 0, 0, 0, 8, 0);
        se_delete((uint64_t)(uintptr_t)se, 0, 0, 0, 0, 0);
        CHECK(true, "delete with no waiters frees cleanly");
    }

    std::string trace;
    if (capturing) {
        fflush(stderr);
        replace_descriptor(saved_stderr, file_descriptor(stderr));
        close_descriptor(saved_stderr);
        saved_stderr = -1;
        rewind(trace_file);
        char buffer[4096];
        while (size_t count = fread(buffer, 1, sizeof(buffer), trace_file))
            trace.append(buffer, count);
    }
    if (trace_file) fclose(trace_file);
    CHECK(trace.find("name='delay-origin'") != std::string::npos,
          "excluded event-flag traffic anchors the common trace delay");
    CHECK(trace.find("SEMA.signal") == std::string::npos,
          "deleted semaphore focus does not retain a reused object's signal");

    if (fails) { printf("== FAIL: %d check(s) ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
