// test_guest_thread_handle — the value scePthreadCreate/scePthreadSelf hand a guest is a readable
// thread object, the same value everywhere, and anything else is rejected.
//
// THE DEFECT. Assassin's Creed Black Flag Resynced creates a thread and then reads the first dword
// THROUGH the handle it was given (`mov rax,[handle]; mov ecx,[rax]` at eboot+0x161ed91). prosper
// wrote the raw host id; on MinGW winpthreads that is a small integer index, so the guest read
// address 3 and its main thread died 1.6 s in. Linux/macOS never showed it because a glibc
// `pthread_t` is a readable pointer.
//
// WHAT EACH ARM KILLS (mutations that must turn this test red on Windows):
//   M1  k_pthread_create writes the raw `tid`                      -> §1 "readable object" and
//                                                                      "first dword is a thread id"
//   M2  k_pthread_self returns raw pthread_self()                  -> §2 identity arm, every round
//   M3  guest_thread_handle_create skips get-or-create (always     -> §2 identity arm, under the
//       allocates), so a child that adopted first diverges            race the 64 rounds provoke
//   M4  resolve accepts any value (casts it back)                  -> §4 garbage / stale arms
//   M5  join never calls guest_thread_handle_joined                -> §5 leak arm
//   M6  detach never calls guest_thread_handle_detached, or the    -> §5 leak arm (detached path)
//       exit notification is dropped
//   M7  adopted handles are joinable/detachable                    -> §6 adopted arms
//   M8  guest_thread_spawn leaves its (guest-invisible) handle     -> not reachable from here; it is
//       registered                                                    covered by §5's live-count
//                                                                     baseline only for create/join
//
// On Linux/macOS the handle is the raw pthread_t by design (identity helpers), so only the platform-
// neutral arms (§2 identity, §3 equality, §5 join) assert there; the readability, validation and
// leak arms are Windows-only because that is the only platform where the translation exists.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/kernel/guest_thread_handle.hpp"
#include "hle/kernel/sce_errno.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

namespace {

// Written as literals so a mutated sce_errno.hpp cannot make the arms agree with itself.
// scePthreadRename is aliased (encoded); scePthreadGetname is bare ON PURPOSE (see hle_kernel.cpp).
constexpr uint64_t kEncodedESRCH  = 0x80020003ull;
constexpr uint64_t kEncodedEINVAL = 0x80020016ull;
constexpr uint64_t kBareESRCH     = 3;
constexpr uint64_t kBareEINVAL    = 22;

std::atomic<uint64_t> g_child_self{0};
std::atomic<bool> g_child_started{false};
std::atomic<bool> g_child_release{false};

// Body outside the ABI shim and noinline: see test_pthread_names.cpp for the WinLibs assembler bug
// this works around (#2142).
__attribute__((noinline)) void* child_body(bool wait_for_release) {
    HleFn self = Hle::lookup(nid_hash("scePthreadSelf"));
    g_child_self.store(self(0, 0, 0, 0, 0, 0), std::memory_order_release);
    g_child_started.store(true, std::memory_order_release);
    while (wait_for_release && !g_child_release.load(std::memory_order_acquire))
        std::this_thread::yield();
    return (void*)0x77;
}
#ifdef _WIN32
extern "C" __attribute__((sysv_abi)) void* child_wait(void*) { return child_body(true); }
extern "C" __attribute__((sysv_abi)) void* child_quick(void*) { return child_body(false); }
#else
void* child_wait(void*) { return child_body(true); }
void* child_quick(void*) { return child_body(false); }
#endif

void reset_child() {
    g_child_self.store(0);
    g_child_started.store(false);
    g_child_release.store(false);
}

void wait_started() {
    for (int i = 0; i < 20000 && !g_child_started.load(std::memory_order_acquire); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

}   // namespace

int main() {
    printf("== test_guest_thread_handle ==\n");
    register_builtin_hle();

    HleFn create = Hle::lookup(nid_hash("scePthreadCreate"));
    HleFn self = Hle::lookup(nid_hash("scePthreadSelf"));
    HleFn equal = Hle::lookup(nid_hash("scePthreadEqual"));
    HleFn join = Hle::lookup(nid_hash("scePthreadJoin"));
    HleFn detach = Hle::lookup(nid_hash("scePthreadDetach"));
    HleFn rename = Hle::lookup(nid_hash("scePthreadRename"));
    HleFn getname = Hle::lookup(nid_hash("scePthreadGetname"));
    HleFn posix_join = Hle::lookup(nid_hash("pthread_join"));
    CHECK(create && self && equal && join && detach && rename && getname && posix_join,
          "thread entry points are registered");
    if (!create || !self || !equal || !join || !detach || !rename || !getname || !posix_join)
        return 1;

    const uint64_t baseline = hle::guest_thread_handle_live_count();

    // --- §1 the handle is a readable object whose first dword is the thread id --------------------
    reset_child();
    uint64_t handle = 0;
    CHECK(create((uint64_t)(uintptr_t)&handle, 0, (uint64_t)(uintptr_t)child_wait, 0,
                 (uint64_t)(uintptr_t) "handle-test", 0) == 0,
          "scePthreadCreate succeeds");
    CHECK(handle != 0, "scePthreadCreate publishes a non-null handle");
#ifdef _WIN32
    CHECK(handle > 0x10000, "the handle is an object address, not a small host index (M1)");
    uint32_t first_dword = 0;
    if (handle > 0x10000) memcpy(&first_dword, (const void*)(uintptr_t)handle, sizeof first_dword);
    CHECK(first_dword != 0, "the guest can read a non-zero thread id through the handle (M1)");
#endif

    // --- §2 identity: Self inside the child is the handle Create wrote ---------------------------
    wait_started();
    CHECK(g_child_self.load() == handle,
          "scePthreadSelf in the child equals the handle scePthreadCreate wrote (M2)");
    const uint64_t main_self = self(0, 0, 0, 0, 0, 0);
    CHECK(main_self != handle, "the creating thread's own handle differs from the child's");
    CHECK(self(0, 0, 0, 0, 0, 0) == main_self, "scePthreadSelf is stable for one thread");
    CHECK(equal(handle, g_child_self.load(), 0, 0, 0, 0) != 0, "scePthreadEqual: same thread");
    CHECK(equal(handle, main_self, 0, 0, 0, 0) == 0, "scePthreadEqual: different threads");

    // --- §3 name through the handle ---------------------------------------------------------------
    CHECK(rename(handle, (uint64_t)(uintptr_t) "renamed", 0, 0, 0, 0) == 0,
          "scePthreadRename resolves a handle to its thread");
    char name[40]{};
    CHECK(getname(handle, (uint64_t)(uintptr_t)name, 0, 0, 0, 0) == 0 && strcmp(name, "renamed") == 0,
          "scePthreadGetname reads it back through the handle");

    // --- §5 join through the handle, then it is spent ---------------------------------------------
    g_child_release.store(true);
    void* exit_value = nullptr;
    CHECK(join(handle, (uint64_t)(uintptr_t)&exit_value, 0, 0, 0, 0) == 0 &&
              exit_value == (void*)0x77,
          "scePthreadJoin through the handle returns the thread's exit value");
#ifdef _WIN32
    CHECK(join(handle, 0, 0, 0, 0, 0) == kEncodedESRCH,
          "a joined handle is stale: Sony join reports ESRCH (M4)");
    CHECK(posix_join(handle, 0, 0, 0, 0, 0) == kBareESRCH,
          "a joined handle is stale: POSIX join reports bare ESRCH (M4)");
#endif

    // --- §4 values that are not handles are refused, never cast ----------------------------------
#ifdef _WIN32
    const uint64_t junk[] = {1, 3, 0x1234, 0xdeadbeef, ~0ull};
    bool all_esrch = true;
    for (const uint64_t j : junk) {
        const uint64_t jr = join(j, 0, 0, 0, 0, 0), dr = detach(j, 0, 0, 0, 0, 0);
        const uint64_t rr = rename(j, (uint64_t)(uintptr_t) "x", 0, 0, 0, 0);
        const uint64_t gr = getname(j, (uint64_t)(uintptr_t)name, 0, 0, 0, 0);
        if (jr != kEncodedESRCH || dr != kEncodedESRCH || rr != kEncodedESRCH || gr != kBareESRCH) {
            printf("    junk 0x%llx: join=0x%llx detach=0x%llx rename=0x%llx getname=0x%llx\n",
                   (unsigned long long)j, (unsigned long long)jr, (unsigned long long)dr,
                   (unsigned long long)rr, (unsigned long long)gr);
            all_esrch = false;
        }
    }
    CHECK(all_esrch, "join/detach/rename/getname refuse non-handle values with ESRCH (M4)");
#endif

    // --- §6 a thread scePthreadCreate did not make is adopted and not joinable --------------------
#ifdef _WIN32
    CHECK(detach(main_self, 0, 0, 0, 0, 0) == kEncodedEINVAL,
          "detach on an adopted handle is EINVAL (M7)");
    uint64_t adopted_from_thread = 0;
    uint64_t adopted_join_rc = 0;
    std::thread host_thread([&] {
        adopted_from_thread = self(0, 0, 0, 0, 0, 0);
        adopted_join_rc = join(main_self, 0, 0, 0, 0, 0);   // another thread joining main
    });
    host_thread.join();
    CHECK(adopted_from_thread != 0 && adopted_from_thread != main_self,
          "a plain host thread gets its own adopted handle");
    CHECK(adopted_join_rc == kEncodedEINVAL, "joining an adopted thread is EINVAL (M7)");
    (void)kBareEINVAL;
#endif

    // --- §2b identity under the start-up race: many rounds, child runs Self immediately -----------
    int mismatches = 0;
    for (int round = 0; round < 64; ++round) {
        reset_child();
        uint64_t h = 0;
        if (create((uint64_t)(uintptr_t)&h, 0, (uint64_t)(uintptr_t)child_quick, 0, 0, 0) != 0) {
            ++mismatches;
            continue;
        }
        wait_started();
        if (g_child_self.load() != h) ++mismatches;
        join(h, 0, 0, 0, 0, 0);
    }
    CHECK(mismatches == 0, "Create's handle equals the child's Self for 64 racing rounds (M3)");

    // --- §5b nothing leaks per thread: joined and detached threads both release their object -----
    for (int round = 0; round < 40; ++round) {
        reset_child();
        uint64_t h = 0;
        create((uint64_t)(uintptr_t)&h, 0, (uint64_t)(uintptr_t)child_quick, 0, 0, 0);
        wait_started();
        if (round % 2 == 0) join(h, 0, 0, 0, 0, 0);
        else detach(h, 0, 0, 0, 0, 0);
    }
    uint64_t live = 0;
    for (int i = 0; i < 2000; ++i) {   // detached threads release on exit, asynchronously
        live = hle::guest_thread_handle_live_count();
#ifdef _WIN32
        // main_self and the adopted objects of finished plain threads may remain; the 40 created
        // objects must not.
        if (live <= baseline + 3) break;
#else
        break;
#endif
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
#ifdef _WIN32
    CHECK(live <= baseline + 3, "joined and detached threads release their handle objects (M5, M6)");
#endif

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
