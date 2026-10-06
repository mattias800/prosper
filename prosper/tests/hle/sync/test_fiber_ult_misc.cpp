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

TEST(FiberUltMisc, RuntimeDestroyRefusesWhileUlthreadsAreUnjoined) {
    // Firmware contract (libSceUlt.sprx, body 0x2bcb0): runtime+0x30 counts created-and-not-yet-
    // joined ulthreads, and a destroy while it is above zero returns busy (0x80810006 at 0x2be65)
    // and leaves the runtime untouched. Join decrements it, after which destroy succeeds.
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

    g_gate.store(0);
    UltBlob ult;
    std::memset(&ult, 0, sizeof(ult));
    std::vector<unsigned char> ctx(64 * 1024);
    ASSERT_EQ(call9_nid("znI3q8S7KQ4", addr(&ult), 0, addr((const void*)&gated_probe_entry), 0x53,
                        addr(ctx.data()), (uint64_t)ctx.size(), addr(&g_runtime), 0, 0x12000000ull),
              0u);
    EXPECT_EQ(call_nid("-gxcs521SvA", addr(&g_runtime)), hle::kSceKernelErrorEBUSY)
        << "destroy with a live (unjoined) ulthread is refused busy";

    // Join drops the runtime's unjoined count to zero. (The join alone would succeed even with the
    // runtime gone; the destroy-returns-0 assertion below is what proves the refusal left it alive.)
    g_gate.store(1);
    int32_t status = -1;
    EXPECT_EQ(call_nid("gCeAI57LGgI", addr(&ult), addr(&status)), 0u);
    EXPECT_EQ(status, 0x53);

    EXPECT_EQ(call_nid("-gxcs521SvA", addr(&g_runtime)), 0u)
        << "destroy succeeds once every ulthread is joined";
    EXPECT_EQ(call_nid("-gxcs521SvA", addr(&g_runtime)), hle::kSceKernelErrorESRCH)
        << "a second destroy is refused";
    UltBlob never_created;
    std::memset(&never_created, 0, sizeof(never_created));
    EXPECT_EQ(call_nid("-gxcs521SvA", addr(&never_created)), hle::kSceKernelErrorESRCH)
        << "destroy on a never-created runtime is refused";
}

TEST(FiberUltMisc, QueueSemNidsBound) {
    register_builtin_hle();
    static const char* table[] = {
        "_sceUltQueueDataResourcePoolCreate",
        "sceUltQueueDataResourcePoolGetWorkAreaSize",
        "sceUltQueueDataResourcePoolDestroy",
        "_sceUltQueueCreate",
        "sceUltQueuePush",
        "sceUltQueueTryPush",
        "sceUltQueuePop",
        "sceUltQueueTryPop",
        "sceUltQueueDestroy",
        "_sceUltSemaphoreCreate",
        "sceUltSemaphoreAcquire",
        "sceUltSemaphoreTryAcquire",
        "sceUltSemaphoreRelease",
        "sceUltSemaphoreDestroy",
    };
    static_assert(sizeof(table) / sizeof(table[0]) == 14, "the 14 queue/semaphore exports");
    for (const char* name : table) {
        EXPECT_NE(Hle::lookup(nid_hash(name)), nullptr) << name << " is not registered";
    }
    EXPECT_EQ(nid_hash("sceUltQueuePush"), "dUwpX3e5NDE");
    EXPECT_EQ(nid_hash("sceUltSemaphoreAcquire"), "QAH1ofI97vU");
    EXPECT_EQ(nid_hash("_sceUltSemaphoreCreate"), "h5QlIYj+Ro8");
    EXPECT_NE(nid_hash("sceUltQueuePush"), "AAAAAAAAAAA")
        << "positive control: the discriminator rejects a wrong NID";
}

static UltBlob g_dpool, g_queue, g_sem;
constexpr uint64_t kDataPoolNumData = 32;   // the data pool's declared item budget

static void make_data_pool_and_queue(uint64_t item_bytes) {
    ASSERT_EQ(call_nid("hZIg1EWGsHM"), 0u);
    const uint64_t pool_bytes = call_nid("WIWV1Qd7PFU", 16, 16);
    ASSERT_GT(pool_bytes, 0u);
    // The work areas outlive the objects built on them, as a guest's would.
    static std::vector<unsigned char> pool_work;
    pool_work.assign((size_t)pool_bytes, 0);
    ASSERT_EQ(call7_nid("YiHujOG9vXY", addr(&g_pool), 0, 16, 16, addr(pool_work.data()), 0,
                        0x12000000ull),
              0u);
    const uint64_t dp_bytes = call_nid("evj9YPkS8s4", kDataPoolNumData, item_bytes, 4);
    ASSERT_GT(dp_bytes, 0u);
    ASSERT_LT(dp_bytes, 1024u * 1024u);
    static std::vector<unsigned char> dp_work;
    dp_work.assign((size_t)dp_bytes, 0);
    ASSERT_EQ(call9_nid("TFHm6-N6vks", addr(&g_dpool), 0, kDataPoolNumData, item_bytes, 4,
                        addr(&g_pool), addr(dp_work.data()), 0, 0x12000000ull),
              0u);
    ASSERT_EQ(call7_nid("9Y5keOvb6ok", addr(&g_queue), 0, item_bytes, addr(&g_pool), addr(&g_dpool),
                        0, 0x12000000ull),
              0u);
}

TEST(FiberUltMisc, QueueDataPoolLifecycle) {
    register_builtin_hle();
    std::memset(&g_pool, 0, sizeof(g_pool));
    std::memset(&g_dpool, 0, sizeof(g_dpool));
    make_data_pool_and_queue(16);
    EXPECT_EQ(call_nid("dh11uAUWNyM", addr(&g_dpool)), 0u) << "data pool destroy succeeds";
    EXPECT_EQ(call_nid("dh11uAUWNyM", addr(&g_dpool)), hle::kSceKernelErrorESRCH)
        << "destroy frees exactly once";
    UltBlob never_created;
    std::memset(&never_created, 0, sizeof(never_created));
    EXPECT_EQ(call_nid("dh11uAUWNyM", addr(&never_created)), hle::kSceKernelErrorESRCH)
        << "destroy on a never-created pool is refused";
    // The size answer follows the documented formula: 128 + queues * 64.
    EXPECT_EQ(call_nid("evj9YPkS8s4", 0, 16, 4), 128u + 4u * 64u)
        << "size query answers prosper's own requirement";
}

TEST(FiberUltMisc, QueuePushPopRoundTrip) {
    register_builtin_hle();
    std::memset(&g_pool, 0, sizeof(g_pool));
    std::memset(&g_dpool, 0, sizeof(g_dpool));
    std::memset(&g_queue, 0, sizeof(g_queue));
    make_data_pool_and_queue(16);
    const uint64_t q = addr(&g_queue);

    uint8_t out[16];
    std::memset(out, 0xAB, sizeof(out));
    EXPECT_EQ(call_nid("uZz3ci7XYqc", q, addr(out)), hle::kSceKernelErrorEAGAIN)
        << "try-pop on an empty queue answers EAGAIN instead of blocking";
    uint8_t in[16];
    for (int i = 0; i < 16; ++i) in[i] = (uint8_t)(0xC0 + i);
    EXPECT_EQ(call_nid("dUwpX3e5NDE", q, addr(in)), 0u) << "push succeeds";
    EXPECT_EQ(call_nid("6Mc2Xs7pI1I", q, addr(in)), 0u) << "try-push succeeds";
    EXPECT_EQ(call_nid("RVSq2tsm2yw", q, addr(out)), 0u) << "pop succeeds";
    EXPECT_EQ(std::memcmp(out, in, sizeof(out)), 0)
        << "pop delivers exactly the pushed bytes, in order";
    EXPECT_EQ(call_nid("RVSq2tsm2yw", q, addr(out)), 0u) << "second pop takes the second item";
    EXPECT_EQ(call_nid("uZz3ci7XYqc", q, addr(out)), hle::kSceKernelErrorEAGAIN)
        << "queue is empty again afterwards";
    EXPECT_EQ(call_nid("dUwpX3e5NDE", q, 0), hle::kSceKernelErrorEINVAL)
        << "push through a null pointer is refused";
    EXPECT_EQ(call_nid("RVSq2tsm2yw", q, 0), hle::kSceKernelErrorEINVAL)
        << "pop through a null pointer is refused";
    UltBlob never_created;
    std::memset(&never_created, 0, sizeof(never_created));
    EXPECT_EQ(call_nid("dUwpX3e5NDE", addr(&never_created), addr(in)), hle::kSceKernelErrorESRCH)
        << "push on a never-created queue is refused";
    EXPECT_EQ(call_nid("PP9nZxpSKLY", q), 0u) << "queue destroy succeeds";
    EXPECT_EQ(call_nid("RVSq2tsm2yw", q, addr(out)), hle::kSceKernelErrorESRCH)
        << "pop on a destroyed queue is refused";
}

TEST(FiberUltMisc, QueuePopBlocksUntilPush) {
    register_builtin_hle();
    std::memset(&g_pool, 0, sizeof(g_pool));
    std::memset(&g_dpool, 0, sizeof(g_dpool));
    std::memset(&g_queue, 0, sizeof(g_queue));
    make_data_pool_and_queue(8);
    const uint64_t q = addr(&g_queue);

    uint8_t got[8];
    std::memset(got, 0, sizeof(got));
    std::atomic<uint64_t> pop_rc{~0ull};
    std::thread popper([&] { pop_rc.store(call_nid("RVSq2tsm2yw", q, addr(got))); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));   // popper is now waiting
    EXPECT_EQ(pop_rc.load(), ~0ull) << "pop on an empty queue must still be waiting";
    uint8_t sent[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT_EQ(call_nid("dUwpX3e5NDE", q, addr(sent)), 0u);
    for (int i = 0; i < 100 && pop_rc.load() == ~0ull; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    popper.join();
    EXPECT_EQ(pop_rc.load(), 0u) << "the blocked pop completes once an item arrives";
    EXPECT_EQ(std::memcmp(got, sent, sizeof(got)), 0) << "it delivers that item";
}

TEST(FiberUltMisc, SemaphoreCountsResources) {
    register_builtin_hle();
    std::memset(&g_pool, 0, sizeof(g_pool));
    std::memset(&g_sem, 0, sizeof(g_sem));
    ASSERT_EQ(call_nid("hZIg1EWGsHM"), 0u);
    const uint64_t pool_bytes = call_nid("WIWV1Qd7PFU", 16, 16);
    std::vector<unsigned char> pool_work((size_t)pool_bytes, 0);
    ASSERT_EQ(call7_nid("YiHujOG9vXY", addr(&g_pool), 0, 16, 16, addr(pool_work.data()), 0,
                        0x12000000ull),
              0u);
    const uint64_t s = addr(&g_sem);
    ASSERT_EQ(call_nid("h5QlIYj+Ro8", s, 0, 3, addr(&g_pool), 0, 0x12000000ull), 0u);

    EXPECT_EQ(call_nid("HA1Ldbi3lPY", s, 2), 0u) << "try-acquire within the count succeeds";
    EXPECT_EQ(call_nid("HA1Ldbi3lPY", s, 2), hle::kSceKernelErrorEAGAIN)
        << "try-acquire past the count answers EAGAIN";
    EXPECT_EQ(call_nid("QAH1ofI97vU", s, 1), 0u) << "blocking acquire takes the last unit";
    EXPECT_EQ(call_nid("HA1Ldbi3lPY", s, 1), hle::kSceKernelErrorEAGAIN)
        << "empty semaphore refuses";
    EXPECT_EQ(call_nid("QAH1ofI97vU", s, 0), hle::kSceKernelErrorEINVAL)
        << "acquiring zero units is refused";
    EXPECT_EQ(call_nid("lbtk5X1mecw", s, 2), 0u) << "release restores";
    EXPECT_EQ(call_nid("HA1Ldbi3lPY", s, 2), 0u) << "the released units are acquirable";
    EXPECT_EQ(call_nid("lbtk5X1mecw", s, 0), hle::kSceKernelErrorEINVAL)
        << "releasing zero units is refused";
    EXPECT_EQ(call_nid("izXyehpoZGo", s), 0u) << "semaphore destroy succeeds";
    EXPECT_EQ(call_nid("QAH1ofI97vU", s, 1), hle::kSceKernelErrorESRCH)
        << "acquire on a destroyed semaphore is refused";
    UltBlob never_created;
    std::memset(&never_created, 0, sizeof(never_created));
    EXPECT_EQ(call_nid("lbtk5X1mecw", addr(&never_created), 1), hle::kSceKernelErrorESRCH)
        << "release on a never-created semaphore is refused";
}

TEST(FiberUltMisc, SemaphoreAcquireBlocksUntilRelease) {
    register_builtin_hle();
    std::memset(&g_pool, 0, sizeof(g_pool));
    std::memset(&g_sem, 0, sizeof(g_sem));
    ASSERT_EQ(call_nid("hZIg1EWGsHM"), 0u);
    const uint64_t pool_bytes = call_nid("WIWV1Qd7PFU", 16, 16);
    std::vector<unsigned char> pool_work((size_t)pool_bytes, 0);
    ASSERT_EQ(call7_nid("YiHujOG9vXY", addr(&g_pool), 0, 16, 16, addr(pool_work.data()), 0,
                        0x12000000ull),
              0u);
    const uint64_t s = addr(&g_sem);
    ASSERT_EQ(call_nid("h5QlIYj+Ro8", s, 0, 0, addr(&g_pool), 0, 0x12000000ull), 0u);

    std::atomic<uint64_t> acq_rc{~0ull};
    std::thread waiter([&] { acq_rc.store(call_nid("QAH1ofI97vU", s, 1)); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));   // waiter is now parked
    EXPECT_EQ(acq_rc.load(), ~0ull) << "acquire on an empty semaphore must still be waiting";
    EXPECT_EQ(call_nid("lbtk5X1mecw", s, 1), 0u);
    for (int i = 0; i < 100 && acq_rc.load() == ~0ull; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    waiter.join();
    EXPECT_EQ(acq_rc.load(), 0u) << "the blocked acquire completes once released";
}

// The data pool's numData is the guest's own budget for the items its queues hold. TryPush on a
// full queue must answer EAGAIN (an unbounded queue never does, so a "push until TryPush fails"
// loop never ends), and a pop must make room again.
TEST(FiberUltMisc, QueueTryPushRefusesWhenFull) {
    register_builtin_hle();
    std::memset(&g_pool, 0, sizeof(g_pool));
    std::memset(&g_dpool, 0, sizeof(g_dpool));
    std::memset(&g_queue, 0, sizeof(g_queue));
    make_data_pool_and_queue(8);
    const uint64_t q = addr(&g_queue);
    uint8_t item[8] = {};
    for (uint64_t i = 0; i < kDataPoolNumData; ++i) {
        item[0] = (uint8_t)i;
        ASSERT_EQ(call_nid("6Mc2Xs7pI1I", q, addr(item)), 0u) << "try-push " << i << " fits";
    }
    EXPECT_EQ(call_nid("6Mc2Xs7pI1I", q, addr(item)), hle::kSceKernelErrorEAGAIN)
        << "try-push past numData answers EAGAIN";
    uint8_t out[8] = {};
    EXPECT_EQ(call_nid("uZz3ci7XYqc", q, addr(out)), 0u);
    EXPECT_EQ(out[0], 0u) << "the oldest item leaves first";
    EXPECT_EQ(call_nid("6Mc2Xs7pI1I", q, addr(item)), 0u) << "a pop makes room for one push";
    EXPECT_EQ(call_nid("6Mc2Xs7pI1I", q, addr(item)), hle::kSceKernelErrorEAGAIN);
}

TEST(FiberUltMisc, QueuePushBlocksUntilPop) {
    register_builtin_hle();
    std::memset(&g_pool, 0, sizeof(g_pool));
    std::memset(&g_dpool, 0, sizeof(g_dpool));
    std::memset(&g_queue, 0, sizeof(g_queue));
    make_data_pool_and_queue(8);
    const uint64_t q = addr(&g_queue);
    uint8_t item[8] = {};
    for (uint64_t i = 0; i < kDataPoolNumData; ++i)
        ASSERT_EQ(call_nid("dUwpX3e5NDE", q, addr(item)), 0u);
    uint8_t last[8] = {9, 9, 9, 9, 9, 9, 9, 9};
    std::atomic<uint64_t> push_rc{~0ull};
    std::thread pusher([&] { push_rc.store(call_nid("dUwpX3e5NDE", q, addr(last))); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_EQ(push_rc.load(), ~0ull) << "push on a full queue waits for room";
    uint8_t out[8] = {};
    EXPECT_EQ(call_nid("RVSq2tsm2yw", q, addr(out)), 0u);
    // The pop must WAKE the pusher. Without that the pusher still finishes, but only when its
    // first bounded wait (PROSPER_ULT_BLOCK_WARN_MS, 5 s) times out and re-checks; 2 s separates them.
    for (int i = 0; i < 200 && push_rc.load() == ~0ull; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_EQ(push_rc.load(), 0u) << "the pop wakes the waiting push, which then completes";
    pusher.join();
    // 31 earlier items remain ahead of the one the blocked push delivered: 32 pops in all.
    for (uint64_t i = 0; i < kDataPoolNumData; ++i)
        ASSERT_EQ(call_nid("uZz3ci7XYqc", q, addr(out)), 0u);
    EXPECT_EQ(std::memcmp(out, last, sizeof(out)), 0) << "the blocked item arrived last";
    EXPECT_EQ(call_nid("uZz3ci7XYqc", q, addr(out)), hle::kSceKernelErrorEAGAIN);
}

TEST(FiberUltMisc, DestroyReleasesBlockedPopAndAcquire) {
    register_builtin_hle();
    std::memset(&g_pool, 0, sizeof(g_pool));
    std::memset(&g_dpool, 0, sizeof(g_dpool));
    std::memset(&g_queue, 0, sizeof(g_queue));
    std::memset(&g_sem, 0, sizeof(g_sem));
    make_data_pool_and_queue(8);
    const uint64_t q = addr(&g_queue), s = addr(&g_sem);
    ASSERT_EQ(call_nid("h5QlIYj+Ro8", s, 0, 0, addr(&g_pool), 0, 0x12000000ull), 0u);

    uint8_t got[8] = {};
    std::atomic<uint64_t> pop_rc{~0ull}, acq_rc{~0ull};
    std::thread popper([&] { pop_rc.store(call_nid("RVSq2tsm2yw", q, addr(got))); });
    std::thread acquirer([&] { acq_rc.store(call_nid("QAH1ofI97vU", s, 1)); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    ASSERT_EQ(pop_rc.load(), ~0ull);
    ASSERT_EQ(acq_rc.load(), ~0ull);
    EXPECT_EQ(call_nid("PP9nZxpSKLY", q), 0u);
    EXPECT_EQ(call_nid("izXyehpoZGo", s), 0u);
    popper.join();
    acquirer.join();
    EXPECT_EQ(pop_rc.load(), hle::kSceKernelErrorESRCH) << "destroy releases a blocked pop";
    EXPECT_EQ(acq_rc.load(), hle::kSceKernelErrorESRCH) << "destroy releases a blocked acquire";
}

// A waiter woken by destroy may reacquire the object's mutex only after a Create has reused the
// slot. It must still report the destroy, and must not take the new queue's item or the new
// semaphore's units. The interleaving is a race, so it is run many times.
TEST(FiberUltMisc, StaleWaiterDoesNotJoinTheNextIncarnation) {
    register_builtin_hle();
    std::memset(&g_pool, 0, sizeof(g_pool));
    std::memset(&g_dpool, 0, sizeof(g_dpool));
    std::memset(&g_queue, 0, sizeof(g_queue));
    make_data_pool_and_queue(8);
    ASSERT_EQ(call_nid("PP9nZxpSKLY", addr(&g_queue)), 0u);
    constexpr int kRounds = 40;
    int stolen_items = 0, stolen_units = 0;
    for (int round = 0; round < kRounds; ++round) {
        UltBlob first, second, sem_a, sem_b;
        std::memset(&first, 0, sizeof(first));
        std::memset(&second, 0, sizeof(second));
        std::memset(&sem_a, 0, sizeof(sem_a));
        std::memset(&sem_b, 0, sizeof(sem_b));
        ASSERT_EQ(call7_nid("9Y5keOvb6ok", addr(&first), 0, 8, addr(&g_pool), addr(&g_dpool), 0,
                            0x12000000ull),
                  0u);
        ASSERT_EQ(call_nid("h5QlIYj+Ro8", addr(&sem_a), 0, 0, addr(&g_pool), 0, 0x12000000ull), 0u);
        uint8_t got[8] = {};
        std::atomic<uint64_t> pop_rc{~0ull}, acq_rc{~0ull};
        std::thread popper([&] { pop_rc.store(call_nid("RVSq2tsm2yw", addr(&first), addr(got))); });
        std::thread acquirer([&] { acq_rc.store(call_nid("QAH1ofI97vU", addr(&sem_a), 1)); });
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        // Destroy, then immediately reuse the slot (the only dead object of each type).
        ASSERT_EQ(call_nid("PP9nZxpSKLY", addr(&first)), 0u);
        ASSERT_EQ(call7_nid("9Y5keOvb6ok", addr(&second), 0, 8, addr(&g_pool), addr(&g_dpool), 0,
                            0x12000000ull),
                  0u);
        uint8_t sent[8] = {7, 7, 7, 7, 7, 7, 7, 7};
        ASSERT_EQ(call_nid("dUwpX3e5NDE", addr(&second), addr(sent)), 0u);
        ASSERT_EQ(call_nid("izXyehpoZGo", addr(&sem_a)), 0u);
        ASSERT_EQ(call_nid("h5QlIYj+Ro8", addr(&sem_b), 0, 1, addr(&g_pool), 0, 0x12000000ull), 0u);
        popper.join();
        acquirer.join();
        EXPECT_EQ(pop_rc.load(), hle::kSceKernelErrorESRCH) << "round " << round;
        EXPECT_EQ(acq_rc.load(), hle::kSceKernelErrorESRCH) << "round " << round;
        uint8_t out[8] = {};
        if (call_nid("uZz3ci7XYqc", addr(&second), addr(out)) != 0u) ++stolen_items;
        if (call_nid("HA1Ldbi3lPY", addr(&sem_b), 1) != 0u) ++stolen_units;
        ASSERT_EQ(call_nid("PP9nZxpSKLY", addr(&second)), 0u);
        ASSERT_EQ(call_nid("izXyehpoZGo", addr(&sem_b)), 0u);
    }
    EXPECT_EQ(stolen_items, 0) << "a stale popper took the next queue's item";
    EXPECT_EQ(stolen_units, 0) << "a stale acquirer took the next semaphore's unit";
}
