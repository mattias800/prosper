// test_fiber_ult_misc — the Fiber/Ult remainder: context-size checks, mutex try-lock, condvar
// signal-all, ulthread try-join and yield, and the join claim order they share with Join.
//
// Each export below was unregistered, so the dispatcher answered `0`: a try-lock that never
// tried, a signal-all that woke nobody, a try-join that never joined. Every TEST drives the
// real NIDs against real objects — a held mutex makes try-lock answer EAGAIN (not success),
// a gated ulthread makes try-join answer EAGAIN until it finishes, and destroyed/never-created
// objects are refused. A no-op acknowledgement would produce success in all three places, which
// is exactly what these assertions forbid.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/kernel/sce_errno.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

using namespace prosper;

static uint64_t addr(const void* p) { return (uint64_t)(uintptr_t)p; }

static uint64_t call_nid(const char* nid, uint64_t a0 = 0, uint64_t a1 = 0, uint64_t a2 = 0,
                         uint64_t a3 = 0, uint64_t a4 = 0, uint64_t a5 = 0) {
    HleFn fn = Hle::lookup(nid);
    EXPECT_NE(fn, nullptr) << "NID " << nid << " is not registered";
    if (!fn) return ~0ull;
    return fn(a0, a1, a2, a3, a4, a5);
}
using HleFn7 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
static uint64_t call7_nid(const char* nid, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                          uint64_t a4, uint64_t a5, uint64_t a6) {
    HleFn fn = Hle::lookup(nid);
    EXPECT_NE(fn, nullptr) << "NID " << nid << " is not registered";
    if (!fn) return ~0ull;
    return ((HleFn7)fn)(a0, a1, a2, a3, a4, a5, a6);
}
using HleFn9 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                            uint64_t, uint64_t);
static uint64_t call9_nid(const char* nid, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                          uint64_t a4, uint64_t a5, uint64_t a6, uint64_t a7, uint64_t a8) {
    HleFn fn = Hle::lookup(nid);
    EXPECT_NE(fn, nullptr) << "NID " << nid << " is not registered";
    if (!fn) return ~0ull;
    return ((HleFn9)fn)(a0, a1, a2, a3, a4, a5, a6, a7, a8);
}

TEST(FiberUltMisc, AllNidsBound) {
    register_builtin_hle();
    static const char* table[] = {
        "sceFiberStartContextSizeCheck", "sceFiberStopContextSizeCheck",
        "sceUltMutexTryLock", "sceUltConditionVariableSignalAll", "sceUltUlthreadTryJoin",
        "sceUltUlthreadYield",
    };
    static_assert(sizeof(table) / sizeof(table[0]) == 6, "the 6 remainder exports");
    for (const char* name : table) {
        EXPECT_NE(Hle::lookup(nid_hash(name)), nullptr) << name << " is not registered";
    }
    // Deliberately unbound: prosper cannot produce the saved sceFiberRun frame pointer it returns.
    EXPECT_EQ(Hle::lookup(nid_hash("_sceFiberGetThreadFramePointerAddress")), nullptr);
}

TEST(FiberUltMisc, NidsResolveToStubValues) {
    // All seven NIDs reproduce the stub table verbatim.
    EXPECT_EQ(nid_hash("sceFiberStartContextSizeCheck"), "Lcqty+QNWFc");
    EXPECT_EQ(nid_hash("sceFiberStopContextSizeCheck"), "Kj4nXMpnM8Y");
    EXPECT_EQ(nid_hash("_sceFiberGetThreadFramePointerAddress"), "0dy4JtMUcMQ");
    EXPECT_EQ(nid_hash("sceUltMutexTryLock"), "jOsUG0BJI-Y");
    EXPECT_EQ(nid_hash("sceUltConditionVariableSignalAll"), "byiceqcMvV0");
    EXPECT_EQ(nid_hash("sceUltUlthreadTryJoin"), "DsW+3FTXL0Q");
    EXPECT_EQ(nid_hash("sceUltUlthreadYield"), "HFd-lpjGxJA");
    EXPECT_NE(nid_hash("sceUltMutexTryLock"), "AAAAAAAAAAA")
        << "positive control: the discriminator rejects a wrong NID";
}

TEST(FiberUltMisc, ContextSizeChecksAck) {
    register_builtin_hle();
    EXPECT_EQ(call_nid("Lcqty+QNWFc", 0), 0u) << "StartContextSizeCheck acknowledges";
    EXPECT_EQ(call_nid("Kj4nXMpnM8Y"), 0u) << "StopContextSizeCheck acknowledges";
}

// Sony's Ult objects are 256-byte caller-owned blobs; match the shape the semantics test
// proves (16-byte header, 0x100 spacing) so the registry lands where it does in the guest.
struct alignas(16) UltBlob { unsigned char bytes[256]; };
static UltBlob g_pool, g_runtime, g_mutex, g_cond;

static void make_pool_and_mutex() {
    ASSERT_EQ(call_nid("hZIg1EWGsHM"), 0u);
    const uint64_t pool_bytes = call_nid("WIWV1Qd7PFU", 16, 16);
    ASSERT_GT(pool_bytes, 0u);
    ASSERT_LT(pool_bytes, 1024u * 1024u);
    std::vector<unsigned char> work((size_t)pool_bytes, 0);
    ASSERT_EQ(call7_nid("YiHujOG9vXY", addr(&g_pool), 0, 16, 16, addr(work.data()), 0,
                        0x12000000ull),
              0u);
    ASSERT_EQ(call_nid("mmt8Sa6tL6c", addr(&g_mutex), 0, addr(&g_pool), 0, 0x12000000ull), 0u);
}

TEST(FiberUltMisc, MutexTryLock) {
    register_builtin_hle();
    std::memset(&g_pool, 0, sizeof(g_pool));
    std::memset(&g_mutex, 0, sizeof(g_mutex));
    make_pool_and_mutex();

    const uint64_t m = addr(&g_mutex);
    EXPECT_EQ(call_nid("jOsUG0BJI-Y", m), 0u) << "try-lock on a free mutex succeeds";
    EXPECT_EQ(call_nid("jOsUG0BJI-Y", m), hle::kSceKernelErrorEDEADLK)
        << "self-relock of the non-recursive mutex is diagnosed, not hung on";
    EXPECT_EQ(call_nid("h0XebKiMBtk", m), 0u);

    // Held by ANOTHER thread: the try must refuse with EAGAIN rather than wait or succeed.
    std::atomic<bool> held{false}, release{false};
    std::thread holder([&] {
        EXPECT_EQ(call_nid("8hEGkR1pfr8", m), 0u);
        held.store(true);
        while (!release.load()) std::this_thread::yield();
        EXPECT_EQ(call_nid("h0XebKiMBtk", m), 0u);
    });
    while (!held.load()) std::this_thread::yield();
    EXPECT_EQ(call_nid("jOsUG0BJI-Y", m), hle::kSceKernelErrorEAGAIN)
        << "try-lock on a held mutex answers EAGAIN instead of blocking";
    release.store(true);
    holder.join();
    EXPECT_EQ(call_nid("jOsUG0BJI-Y", m), 0u) << "try-lock succeeds once released";
    EXPECT_EQ(call_nid("h0XebKiMBtk", m), 0u);

    UltBlob never_created;
    std::memset(&never_created, 0, sizeof(never_created));
    EXPECT_EQ(call_nid("jOsUG0BJI-Y", addr(&never_created)), hle::kSceKernelErrorESRCH)
        << "try-lock on a never-created mutex is refused";
}

TEST(FiberUltMisc, CondSignalAll) {
    register_builtin_hle();
    std::memset(&g_pool, 0, sizeof(g_pool));
    std::memset(&g_mutex, 0, sizeof(g_mutex));
    std::memset(&g_cond, 0, sizeof(g_cond));
    make_pool_and_mutex();
    ASSERT_EQ(call_nid("jnKaHGkrxZ4", addr(&g_cond), 0, addr(&g_mutex), 0, 0x12000000ull), 0u);

    EXPECT_EQ(call_nid("byiceqcMvV0", addr(&g_cond)), 0u)
        << "signal-all with nobody waiting still succeeds";
    UltBlob never_created;
    std::memset(&never_created, 0, sizeof(never_created));
    EXPECT_EQ(call_nid("byiceqcMvV0", addr(&never_created)), hle::kSceKernelErrorESRCH)
        << "signal-all on a never-created condvar is refused";

    // Two waiters, one SignalAll: both must be released (a do-nothing or single-wake SignalAll
    // leaves at least one blocked). Each waiter holds the bound mutex until it is inside Wait, so
    // once the main thread can take the mutex after both have announced, both are waiting.
    const uint64_t m = addr(&g_mutex), c = addr(&g_cond);
    std::atomic<int> announced{0}, released{0};
    auto waiter = [&] {
        EXPECT_EQ(call_nid("8hEGkR1pfr8", m), 0u);
        announced.fetch_add(1);
        EXPECT_EQ(call_nid("5xGAHCxA8M0", c), 0u);
        released.fetch_add(1);
        EXPECT_EQ(call_nid("h0XebKiMBtk", m), 0u);
    };
    std::thread w1(waiter);
    while (announced.load() < 1) std::this_thread::yield();
    std::thread w2(waiter);
    while (announced.load() < 2) std::this_thread::yield();
    ASSERT_EQ(call_nid("8hEGkR1pfr8", m), 0u);   // both waiters are now inside Wait
    ASSERT_EQ(call_nid("h0XebKiMBtk", m), 0u);
    EXPECT_EQ(call_nid("byiceqcMvV0", c), 0u);
    for (int i = 0; i < 200 && released.load() < 2; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_EQ(released.load(), 2) << "SignalAll must release every waiter";
    while (released.load() < 2) {   // unblock a stranded waiter so the test can finish
        call_nid("JTw1cAVkuc0", c);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    w1.join();
    w2.join();
}

#if defined(__x86_64__) || defined(_M_X64)
#define FIBER_ULT_GUEST_ABI __attribute__((sysv_abi))
#else
#define FIBER_ULT_GUEST_ABI
#endif

static std::atomic<int> g_gate{0};
static FIBER_ULT_GUEST_ABI int32_t gated_probe_entry(uint64_t arg) noexcept {
    for (uint64_t spins = 0; g_gate.load() == 0 && spins < 4000000000ull; ++spins) {
    }
    return (int32_t)arg;
}

TEST(FiberUltMisc, UlthreadTryJoinAndYield) {
    register_builtin_hle();
    EXPECT_EQ(call_nid("HFd-lpjGxJA"), 0u) << "yield surrenders the timeslice";

    std::memset(&g_pool, 0, sizeof(g_pool));
    std::memset(&g_runtime, 0, sizeof(g_runtime));
    ASSERT_EQ(call_nid("hZIg1EWGsHM"), 0u);
    const uint64_t pool_bytes = call_nid("WIWV1Qd7PFU", 16, 16);
    const uint64_t rt_bytes = call_nid("grs2pbc2awM", 16, 3);
    ASSERT_GT(pool_bytes, 0u);
    ASSERT_GT(rt_bytes, 0u);
    std::vector<unsigned char> pool_work((size_t)pool_bytes, 0);
    std::vector<unsigned char> rt_work((size_t)rt_bytes, 0);
    ASSERT_EQ(call7_nid("YiHujOG9vXY", addr(&g_pool), 0, 16, 16, addr(pool_work.data()), 0,
                        0x12000000ull),
              0u);
    ASSERT_EQ(call7_nid("jw9FkZBXo-g", addr(&g_runtime), 0, 16, 3, addr(rt_work.data()), 0,
                        0x12000000ull),
              0u);

    g_gate.store(0);
    UltBlob ult;
    std::memset(&ult, 0, sizeof(ult));
    std::vector<unsigned char> ctx(64 * 1024);
    ASSERT_EQ(call9_nid("znI3q8S7KQ4", addr(&ult), 0, addr((const void*)&gated_probe_entry),
                        0x51, addr(ctx.data()), (uint64_t)ctx.size(), addr(&g_runtime), 0,
                        0x12000000ull),
              0u);

    // The entry spins on the gate, so the first try-join must observe a live ulthread.
    EXPECT_EQ(call_nid("DsW+3FTXL0Q", addr(&ult), 0), hle::kSceKernelErrorEAGAIN)
        << "try-join on a running ulthread answers EAGAIN instead of blocking";
    g_gate.store(1);
    int32_t status = -1;
    uint64_t rc = hle::kSceKernelErrorEAGAIN;
    for (int i = 0; i < 500 && rc == hle::kSceKernelErrorEAGAIN; ++i) {
        rc = call_nid("DsW+3FTXL0Q", addr(&ult), addr(&status));
        if (rc == hle::kSceKernelErrorEAGAIN) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    EXPECT_EQ(rc, 0u) << "try-join completes once the ulthread finishes";
    EXPECT_EQ(status, 0x51) << "try-join writes the entry's int32 return into *status";
    EXPECT_EQ(call_nid("DsW+3FTXL0Q", addr(&ult), 0), hle::kSceKernelErrorESRCH)
        << "a second try-join finds the unpublished object refused, not joined twice";
}

// A second Join on an ulthread that another thread is already joining must be refused at once,
// not wait for the ulthread and pthread_join the same host thread twice (undefined under POSIX).
// The claim happens before the wait, so the refusal arrives while the ulthread is still gated.
TEST(FiberUltMisc, SecondJoinIsRefusedWhileTheFirstWaits) {
    register_builtin_hle();
    std::memset(&g_pool, 0, sizeof(g_pool));
    std::memset(&g_runtime, 0, sizeof(g_runtime));
    ASSERT_EQ(call_nid("hZIg1EWGsHM"), 0u);
    const uint64_t pool_bytes = call_nid("WIWV1Qd7PFU", 16, 16);
    const uint64_t rt_bytes = call_nid("grs2pbc2awM", 16, 3);
    std::vector<unsigned char> pool_work((size_t)pool_bytes, 0);
    std::vector<unsigned char> rt_work((size_t)rt_bytes, 0);
    ASSERT_EQ(call7_nid("YiHujOG9vXY", addr(&g_pool), 0, 16, 16, addr(pool_work.data()), 0,
                        0x12000000ull),
              0u);
    ASSERT_EQ(call7_nid("jw9FkZBXo-g", addr(&g_runtime), 0, 16, 3, addr(rt_work.data()), 0,
                        0x12000000ull),
              0u);

    g_gate.store(0);
    UltBlob ult;
    std::memset(&ult, 0, sizeof(ult));
    std::vector<unsigned char> ctx(64 * 1024);
    ASSERT_EQ(call9_nid("znI3q8S7KQ4", addr(&ult), 0, addr((const void*)&gated_probe_entry),
                        0x52, addr(ctx.data()), (uint64_t)ctx.size(), addr(&g_runtime), 0,
                        0x12000000ull),
              0u);

    int32_t status = -1;
    std::atomic<uint64_t> first_rc{~0ull};
    std::thread first([&] { first_rc.store(call_nid("gCeAI57LGgI", addr(&ult), addr(&status))); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));   // first Join is now waiting

    std::atomic<uint64_t> second_rc{~0ull};
    std::thread second([&] { second_rc.store(call_nid("gCeAI57LGgI", addr(&ult), 0)); });
    for (int i = 0; i < 100 && second_rc.load() == ~0ull; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_EQ(second_rc.load(), hle::kSceKernelErrorEINVAL)
        << "the second Join must be refused while the ulthread is still running";
    g_gate.store(1);
    first.join();
    second.join();
    EXPECT_EQ(first_rc.load(), 0u);
    EXPECT_EQ(status, 0x52);
}

TEST(FiberUltMisc, PoolDestroyRetiresThePool) {
    register_builtin_hle();
    EXPECT_NE(Hle::lookup(nid_hash("sceUltWaitingQueueResourcePoolDestroy")), nullptr);
    EXPECT_EQ(nid_hash("sceUltWaitingQueueResourcePoolDestroy"), "or55417wcDk");
    std::memset(&g_pool, 0, sizeof(g_pool));
    std::memset(&g_mutex, 0, sizeof(g_mutex));
    make_pool_and_mutex();

    // A pool with a bound mutex still destroys; the mutex keeps working afterwards.
    EXPECT_EQ(call_nid("or55417wcDk", addr(&g_pool)), 0u) << "pool destroy succeeds";
    EXPECT_EQ(call_nid("or55417wcDk", addr(&g_pool)), hle::kSceKernelErrorESRCH)
        << "destroy frees exactly once";
    EXPECT_EQ(call_nid("8hEGkR1pfr8", addr(&g_mutex)), 0u)
        << "a mutex bound from the dead pool still locks";
    EXPECT_EQ(call_nid("h0XebKiMBtk", addr(&g_mutex)), 0u);
    UltBlob never_created;
    std::memset(&never_created, 0, sizeof(never_created));
    EXPECT_EQ(call_nid("or55417wcDk", addr(&never_created)), hle::kSceKernelErrorESRCH)
        << "destroy on a never-created pool is refused";
}

TEST(FiberUltMisc, RuntimeDestroyReleasesLiveUlthreads) {
    register_builtin_hle();
    EXPECT_NE(Hle::lookup(nid_hash("sceUltUlthreadRuntimeDestroy")), nullptr);
    EXPECT_EQ(nid_hash("sceUltUlthreadRuntimeDestroy"), "-gxcs521SvA");
    std::memset(&g_pool, 0, sizeof(g_pool));
    std::memset(&g_runtime, 0, sizeof(g_runtime));
    ASSERT_EQ(call_nid("hZIg1EWGsHM"), 0u);
    const uint64_t pool_bytes = call_nid("WIWV1Qd7PFU", 16, 16);
    const uint64_t rt_bytes = call_nid("grs2pbc2awM", 16, 3);
    std::vector<unsigned char> pool_work((size_t)pool_bytes, 0);
    std::vector<unsigned char> rt_work((size_t)rt_bytes, 0);
    ASSERT_EQ(call7_nid("YiHujOG9vXY", addr(&g_pool), 0, 16, 16, addr(pool_work.data()), 0,
                        0x12000000ull),
              0u);
    ASSERT_EQ(call7_nid("jw9FkZBXo-g", addr(&g_runtime), 0, 16, 3, addr(rt_work.data()), 0,
                        0x12000000ull),
              0u);

    // A runtime with a RUNNING ulthread still destroys; the join afterwards must find the
    // ulthread (not the dead runtime) and complete normally instead of faulting on it.
    g_gate.store(0);
    UltBlob ult;
    std::memset(&ult, 0, sizeof(ult));
    std::vector<unsigned char> ctx(64 * 1024);
    ASSERT_EQ(call9_nid("znI3q8S7KQ4", addr(&ult), 0, addr((const void*)&gated_probe_entry), 0x53,
                        addr(ctx.data()), (uint64_t)ctx.size(), addr(&g_runtime), 0, 0x12000000ull),
              0u);
    EXPECT_EQ(call_nid("-gxcs521SvA", addr(&g_runtime)), 0u)
        << "runtime destroy succeeds with a live ulthread";
    EXPECT_EQ(call_nid("-gxcs521SvA", addr(&g_runtime)), hle::kSceKernelErrorESRCH)
        << "destroy frees exactly once";
    g_gate.store(1);
    int32_t status = -1;
    EXPECT_EQ(call_nid("gCeAI57LGgI", addr(&ult), addr(&status)), 0u)
        << "join completes after its runtime is gone";
    EXPECT_EQ(status, 0x53);
    UltBlob never_created;
    std::memset(&never_created, 0, sizeof(never_created));
    EXPECT_EQ(call_nid("-gxcs521SvA", addr(&never_created)), hle::kSceKernelErrorESRCH)
        << "destroy on a never-created runtime is refused";
}
