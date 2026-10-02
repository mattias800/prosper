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
//   Review guards: publication held before registration, all five allocation rollback seams,
//   a created worker's AttrGet host-key lookup, detached-at-create id/retirement, adopted host exit.
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
std::atomic<uintptr_t> g_child_stack_marker{0};
#ifdef _WIN32
bool g_entry_before_publication = false;
void hold_creation_before_registration() {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(150);
    while (!g_child_started.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    g_entry_before_publication = g_child_started.load(std::memory_order_acquire);
}
#endif

// Body outside the ABI shim and noinline: see test_pthread_names.cpp for the WinLibs assembler bug
// this works around (#2142).
__attribute__((noinline)) void* child_body(bool wait_for_release) {
    volatile uint64_t stack_marker = 0;
    g_child_stack_marker.store((uintptr_t)&stack_marker, std::memory_order_release);
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
    g_child_stack_marker.store(0);
}

void wait_started() {
    for (int i = 0; i < 2000 && !g_child_started.load(std::memory_order_acquire); ++i)
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

#ifdef _WIN32
    // Before the first successful creation, exercise every pool-growth and map insertion failure.
    // Guest entry must not run, no handle may be published, and partial map/live state rolls back.
    bool rollback_ok = true;
    for (unsigned point = 1; point <= 5; ++point) {
        reset_child();
        uint64_t refused = 0x13579bdf;
        hle::guest_thread_handle_fail_allocation_for_test(point);
        const uint64_t rc = create((uint64_t)(uintptr_t)&refused, 0,
                                   (uint64_t)(uintptr_t)child_quick, 0, 0, 0);
        rollback_ok &= rc == 0x8002000cull && refused == 0x13579bdf &&
                       !g_child_started.load() && hle::guest_thread_handle_live_count() == baseline;
    }
    hle::guest_thread_handle_fail_allocation_for_test(0);
    CHECK(rollback_ok, "all five allocation failures return ENOMEM without output/entry/live leaks");

    reset_child();
    uint64_t rendezvous = 0;
    hle::guest_thread_handle_before_create_for_test(hold_creation_before_registration);
    const uint64_t rendezvous_rc = create((uint64_t)(uintptr_t)&rendezvous, 0,
                                          (uint64_t)(uintptr_t)child_quick, 0, 0, 0);
    hle::guest_thread_handle_before_create_for_test(nullptr);
    wait_started();
    CHECK(rendezvous_rc == 0 && !g_entry_before_publication &&
          g_child_self.load() == rendezvous,
          "guest entry cannot pass the gate while registration is held (publication control)");
    if (rendezvous_rc == 0) CHECK(join(rendezvous, 0, 0, 0, 0, 0) == 0,
                                "publication-control worker joins");
#endif

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
    CHECK(g_child_started.load(), "created worker reached guest entry within the bound");
    CHECK(g_child_self.load() == handle,
          "scePthreadSelf in the child equals the handle scePthreadCreate wrote (M2)");
    const uint64_t main_self = self(0, 0, 0, 0, 0, 0);
    CHECK(main_self != handle, "the creating thread's own handle differs from the child's");
    CHECK(self(0, 0, 0, 0, 0, 0) == main_self, "scePthreadSelf is stable for one thread");
    CHECK(equal(handle, g_child_self.load(), 0, 0, 0, 0) != 0, "scePthreadEqual: same thread");
    CHECK(equal(handle, main_self, 0, 0, 0, 0) == 0, "scePthreadEqual: different threads");

#ifdef _WIN32
    CHECK(join(main_self, 0, 0, 0, 0, 0) == 0x8002000bull,
          "an adopted thread's self-join preserves encoded FreeBSD EDEADLK");

    HleFn attr_init = Hle::lookup(nid_hash("scePthreadAttrInit"));
    HleFn attr_get = Hle::lookup(nid_hash("scePthreadAttrGet"));
    HleFn attr_base = Hle::lookup(nid_hash("scePthreadAttrGetstackaddr"));
    HleFn attr_size = Hle::lookup(nid_hash("scePthreadAttrGetstacksize"));
    HleFn attr_destroy = Hle::lookup(nid_hash("scePthreadAttrDestroy"));
    HleFn attr_detach = Hle::lookup(nid_hash("scePthreadAttrSetdetachstate"));
    CHECK(attr_init && attr_get && attr_base && attr_size && attr_destroy && attr_detach,
          "thread attribute entry points are registered");
    if (!attr_init || !attr_get || !attr_base || !attr_size || !attr_destroy || !attr_detach) {
        g_child_release.store(true);
        join(handle, 0, 0, 0, 0, 0);
        return 1;
    }
    void* attr = nullptr;
    CHECK(attr_init((uint64_t)(uintptr_t)&attr, 0, 0, 0, 0, 0) == 0 && attr,
          "query attributes initialize");
    CHECK(attr_get(handle, (uint64_t)(uintptr_t)&attr, 0, 0, 0, 0) == 0,
          "AttrGet resolves the guest object to the worker's host stack key");
    void* stack_base = nullptr;
    size_t stack_size = 0;
    attr_base((uint64_t)(uintptr_t)&attr, (uint64_t)(uintptr_t)&stack_base, 0, 0, 0, 0);
    attr_size((uint64_t)(uintptr_t)&attr, (uint64_t)(uintptr_t)&stack_size, 0, 0, 0, 0);
    CHECK(stack_base && stack_size, "created worker query reports real nonempty stack bounds");
    const uintptr_t marker = g_child_stack_marker.load(std::memory_order_acquire);
    const uintptr_t low = (uintptr_t)stack_base;
    CHECK(marker >= low && marker - low < stack_size && marker != 0,
          "queried worker bounds contain its independently observed stack-local address");
    CHECK(attr_get(3, (uint64_t)(uintptr_t)&attr, 0, 0, 0, 0) == kBareESRCH,
          "AttrGet refuses an invalid guest handle with bare ESRCH");
    void* preserved_base = nullptr;
    size_t preserved_size = 0;
    attr_base((uint64_t)(uintptr_t)&attr, (uint64_t)(uintptr_t)&preserved_base, 0, 0, 0, 0);
    attr_size((uint64_t)(uintptr_t)&attr, (uint64_t)(uintptr_t)&preserved_size, 0, 0, 0, 0);
    CHECK(preserved_base == stack_base && preserved_size == stack_size,
          "refused AttrGet preserves the previously established attribute bounds");
    attr_destroy((uint64_t)(uintptr_t)&attr, 0, 0, 0, 0, 0);
#endif

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
    CHECK(join(adopted_from_thread, 0, 0, 0, 0, 0) == kEncodedESRCH,
          "a normally returning adopted host worker retires its guest handle");
    const uint64_t adopted_baseline = hle::guest_thread_handle_live_count();
    for (int i = 0; i < 32; ++i) {
        std::thread adopted_worker([&] { (void)self(0, 0, 0, 0, 0, 0); });
        adopted_worker.join();
    }
    CHECK(hle::guest_thread_handle_live_count() == adopted_baseline,
          "32 adopted host workers return without per-thread registry leaks");
    (void)kBareEINVAL;
#endif

#ifdef _WIN32
    // Native detached-at-create normally discards its HANDLE before Create returns. Guest creation
    // retains the host until id publication, then detaches before the child may enter guest code.
    const uint64_t detached_baseline = hle::guest_thread_handle_live_count();
    void* detached_attr = nullptr;
    attr_init((uint64_t)(uintptr_t)&detached_attr, 0, 0, 0, 0, 0);
    CHECK(attr_detach((uint64_t)(uintptr_t)&detached_attr, 1 /* guest DETACHED */, 0, 0, 0, 0) == 0,
          "detached-at-create attributes configure");
    reset_child();
    uint64_t detached_handle = 0;
    const uint64_t detached_rc = create((uint64_t)(uintptr_t)&detached_handle,
                                        (uint64_t)(uintptr_t)&detached_attr,
                                        (uint64_t)(uintptr_t)child_wait, 0, 0, 0);
    attr_destroy((uint64_t)(uintptr_t)&detached_attr, 0, 0, 0, 0, 0);
    uint32_t detached_id = 0;
    if (detached_handle > 0x10000)
        memcpy(&detached_id, (const void*)(uintptr_t)detached_handle, sizeof detached_id);
    CHECK(detached_rc == 0 && detached_id != 0,
          "detached-at-create handle is readable/nonzero before waiting for guest entry");
    wait_started();
    CHECK(g_child_self.load() == detached_handle, "detached-at-create Self retains published identity");
    CHECK(join(detached_handle, 0, 0, 0, 0, 0) == kEncodedEINVAL,
          "joining a running detached-at-create guest thread returns EINVAL");
    g_child_release.store(true, std::memory_order_release);
    for (int i = 0; i < 2000 && hle::guest_thread_handle_live_count() != detached_baseline; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    CHECK(hle::guest_thread_handle_live_count() == detached_baseline,
          "detached-at-create exit releases its guest object");
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
        // Only this still-running main thread is adopted; every ended host/guest worker retires.
        if (live == baseline + 1) break;
#else
        break;
#endif
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
#ifdef _WIN32
    CHECK(live == baseline + 1, "joined and detached threads release their handle objects (M5, M6)");
#endif

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
