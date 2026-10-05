// test_fiber_ult_misc — the Fiber/Ult remainder: context-size checks, thread frame-pointer
// address, mutex try-lock, condvar signal-all, ulthread try-join and yield.
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
        "_sceFiberGetThreadFramePointerAddress", "sceUltMutexTryLock",
        "sceUltConditionVariableSignalAll", "sceUltUlthreadTryJoin", "sceUltUlthreadYield",
    };
    static_assert(sizeof(table) / sizeof(table[0]) == 7, "the 7 remainder exports");
    for (const char* name : table) {
        EXPECT_NE(Hle::lookup(nid_hash(name)), nullptr) << name << " is not registered";
    }
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

TEST(FiberUltMisc, ThreadFramePointerAddress) {
    register_builtin_hle();
    EXPECT_EQ(call_nid("0dy4JtMUcMQ", 0), 0x80590001u) << "null out-param is refused";
    uint64_t fp = 0xDEADBEEFull;
    EXPECT_EQ(call_nid("0dy4JtMUcMQ", addr(&fp)), 0u);
    EXPECT_EQ(fp, 0u) << "no frame address is known, so zero is written, not garbage";
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
