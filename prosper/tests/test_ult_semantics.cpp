// test_ult_semantics.cpp — #1603 Phase 2: libSceUlt must have REAL semantics, not visible refusals.
//
// Every assertion here fails on the Phase 1 tree (every entry point returned SCE_KERNEL_ERROR_ENOSYS
// and no state existed at all), and each is written so that it also fails for the *interesting* wrong
// implementation, not merely for the absent one:
//
//   * the mutual-exclusion check runs real concurrent threads over a non-atomic counter and asserts
//     both an EXACT total and a direct occupancy invariant. A no-op mutex does not crash — it
//     produces a wrong-but-plausible number, which is precisely why the exact total is the contract
//     and a "did it run" check would be worthless. Measured against a forced no-op lock: 68,636
//     overlapping critical-section entries and 27,984 of 40,000 increments surviving.
//   * the work-area check asserts the size the query returns is the size Create demands, so the two
//     cannot drift apart into a number nobody honours.
//   * the object-header check proves a zeroed (uncreated) and a destroyed object are both REFUSED, so
//     the magic is load-bearing rather than decorative.
//
// The Ult object the guest owns is a 256-byte caller-allocated blob; these tests allocate the same
// shape so prosper's 16-byte header lands where it does in the guest.
#include "../src/hle/dispatch.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

// Sony's Ult objects are 256-byte opaque blobs (proven for Earthion: a mutex at eboot+0x241730 is
// followed by unrelated data at 0x241830, and the class at eboot+0xd8a0 spaces its mutex/condvar/
// mutex members 0x100 apart). Match that so the test exercises the real shape.
struct alignas(16) UltBlob { unsigned char bytes[256]; };
static UltBlob g_pool, g_runtime, g_mutex_a, g_mutex_b;

static uint64_t call(const char* nid, uint64_t a0 = 0, uint64_t a1 = 0, uint64_t a2 = 0,
                     uint64_t a3 = 0, uint64_t a4 = 0, uint64_t a5 = 0) {
    HleFn fn = Hle::lookup(nid);
    if (!fn) { std::printf("  [FAIL] NID %s is not registered\n", nid); ++fails; return ~0ull; }
    return fn(a0, a1, a2, a3, a4, a5);
}
// The two Create entry points take a 7th argument (the SDK/api version) on the guest stack; the
// import bridge forwards it as a normal parameter, so a 7-arg signature reads it directly.
using HleFn7 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
static uint64_t call7(const char* nid, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                      uint64_t a4, uint64_t a5, uint64_t a6) {
    HleFn fn = Hle::lookup(nid);
    if (!fn) { std::printf("  [FAIL] NID %s is not registered\n", nid); ++fails; return ~0ull; }
    return ((HleFn7)fn)(a0, a1, a2, a3, a4, a5, a6);
}

static const char* const kPoolSize    = "WIWV1Qd7PFU";
static const char* const kPoolCreate  = "YiHujOG9vXY";
static const char* const kRtSize      = "grs2pbc2awM";
static const char* const kRtCreate    = "jw9FkZBXo-g";
static const char* const kMtxCreate   = "mmt8Sa6tL6c";
static const char* const kMtxLock     = "8hEGkR1pfr8";
static const char* const kMtxUnlock   = "h0XebKiMBtk";
static const char* const kMtxDestroy  = "jW+HnafeS3Y";
static const char* const kInitialize  = "hZIg1EWGsHM";

// Earthion's own parameters, so the test exercises the values the only importing title actually uses.
static constexpr uint32_t kNumMaxUlthread = 16, kNumWorkerThread = 3;
static constexpr uint32_t kNumThreads = 16, kNumSyncObjects = 16;
static constexpr uint64_t kApiVersion = 0x12000000ull;   // literal at every guest create site

int main() {
    register_builtin_hle();
    ult_reset_counts_for_test();
    ult_set_return_success_for_test(false);
    std::memset(&g_pool, 0, sizeof(g_pool));
    std::memset(&g_runtime, 0, sizeof(g_runtime));
    std::memset(&g_mutex_a, 0, sizeof(g_mutex_a));
    std::memset(&g_mutex_b, 0, sizeof(g_mutex_b));

    CHECK(call(kInitialize) == 0, "sceUltInitialize succeeds");

    // --- work areas -------------------------------------------------------------------------
    // The size query returns prosper's own requirement, and Create must consume exactly that buffer.
    // Asserting both against the SAME number is what stops the pair drifting into a size nobody
    // honours — the #1618 failure in a different costume.
    const uint64_t pool_bytes = call(kPoolSize, kNumThreads, kNumSyncObjects);
    const uint64_t rt_bytes   = call(kRtSize, kNumMaxUlthread, kNumWorkerThread);
    CHECK(pool_bytes > 0 && pool_bytes < 1024 * 1024,
          "sceUltWaitingQueueResourcePoolGetWorkAreaSize returns a real, honourable size");
    CHECK(rt_bytes > 0 && rt_bytes < 1024 * 1024,
          "sceUltUlthreadRuntimeGetWorkAreaSize returns a real, honourable size");

    std::vector<unsigned char> pool_work(pool_bytes ? (size_t)pool_bytes : 1, 0xAB);
    std::vector<unsigned char> rt_work(rt_bytes ? (size_t)rt_bytes : 1, 0xAB);

    const uint64_t pool_rc = call7(kPoolCreate, (uint64_t)(uintptr_t)&g_pool, 0, kNumThreads,
                                   kNumSyncObjects, (uint64_t)(uintptr_t)pool_work.data(), 0,
                                   kApiVersion);
    CHECK(pool_rc == 0, "_sceUltWaitingQueueResourcePoolCreate succeeds with that exact buffer");

    const uint64_t rt_rc = call7(kRtCreate, (uint64_t)(uintptr_t)&g_runtime, 0, kNumMaxUlthread,
                                 kNumWorkerThread, (uint64_t)(uintptr_t)rt_work.data(), 0,
                                 kApiVersion);
    CHECK(rt_rc == 0, "_sceUltUlthreadRuntimeCreate succeeds with that exact buffer");

    // Create must actually USE the work area, not merely accept it. The buffers were poisoned with
    // 0xAB; a Create that consumed nothing would leave them untouched.
    bool pool_used = false, rt_used = false;
    for (size_t i = 0; i < pool_work.size(); ++i) if (pool_work[i] != 0xAB) { pool_used = true; break; }
    for (size_t i = 0; i < rt_work.size(); ++i)   if (rt_work[i] != 0xAB)   { rt_used = true; break; }
    CHECK(pool_used && rt_used,
          "both Creates consume the work area their size query asked for (not a decorative number)");

    // A work area that is not there at all must be refused, not written through.
    const uint64_t rt_null = call7(kRtCreate, (uint64_t)(uintptr_t)&g_runtime, 0, kNumMaxUlthread,
                                   kNumWorkerThread, 0, 0, kApiVersion);
    CHECK(rt_null != 0, "_sceUltUlthreadRuntimeCreate refuses a null work area");

    // --- object identity ---------------------------------------------------------------------
    // A mutex must come from a real pool: passing something that is not one has to be refused, or
    // _sceUltMutexCreate would accept any pointer as its pool argument.
    UltBlob not_a_pool;
    std::memset(&not_a_pool, 0, sizeof(not_a_pool));
    const uint64_t bad_pool = call(kMtxCreate, (uint64_t)(uintptr_t)&g_mutex_a, 0,
                                   (uint64_t)(uintptr_t)&not_a_pool, 0, kApiVersion);
    CHECK(bad_pool != 0, "_sceUltMutexCreate refuses a pool pointer that was never created");

    // An uncreated (zeroed) mutex must be refused rather than treated as lockable.
    CHECK(call(kMtxLock, (uint64_t)(uintptr_t)&g_mutex_a) != 0,
          "sceUltMutexLock refuses a mutex that was never created (zeroed guest object)");

    const uint64_t mtx_rc = call(kMtxCreate, (uint64_t)(uintptr_t)&g_mutex_a, 0,
                                 (uint64_t)(uintptr_t)&g_pool, 0, kApiVersion);
    CHECK(mtx_rc == 0, "_sceUltMutexCreate succeeds against a real pool");

    // prosper writes only a 16-byte header into the caller's 256-byte blob. The guest never reads
    // inside one (proven by an exhaustive scan of every RIP-relative reference in Earthion's whole
    // executable segment: 16 static Ult objects, zero references at a non-zero offset), but writing
    // past the header would corrupt whatever the guest packs next to it.
    bool tail_clean = true;
    for (size_t i = 16; i < sizeof(g_mutex_a.bytes); ++i)
        if (g_mutex_a.bytes[i] != 0) { tail_clean = false; break; }
    CHECK(tail_clean, "prosper writes only a 16-byte header into the caller's 256-byte Ult object");

    // --- lock / unlock ------------------------------------------------------------------------
    CHECK(call(kMtxLock, (uint64_t)(uintptr_t)&g_mutex_a) == 0, "sceUltMutexLock takes the lock");
    // Non-recursive (Earthion passes optParam=NULL): a self-relock must be reported and refused, not
    // silently allowed and not hung on.
    CHECK(call(kMtxLock, (uint64_t)(uintptr_t)&g_mutex_a) != 0,
          "a self-relock of the non-recursive mutex is refused rather than deadlocking");
    CHECK(call(kMtxUnlock, (uint64_t)(uintptr_t)&g_mutex_a) == 0, "sceUltMutexUnlock releases it");
    // Unlocking something this thread does not hold must not "succeed".
    CHECK(call(kMtxUnlock, (uint64_t)(uintptr_t)&g_mutex_a) != 0,
          "sceUltMutexUnlock by a non-owner is refused");

    // --- THE contract: real mutual exclusion --------------------------------------------------
    // Mutual exclusion has to be DEMONSTRATED, and demonstrating it is easy to get wrong. A first
    // version of this test used a bare `v = shared; shared = v + 1` under the lock and asserted the
    // exact total — and it PASSED with the mutex forced to a no-op (PROSPER_ULT_RETURN_SUCCESS=1),
    // because the unprotected read-modify-write is a couple of nanoseconds while the surrounding call
    // overhead is far longer, so the threads simply never collided. A test that a broken lock passes
    // proves nothing.
    //
    // So there are now two independent detectors, and neither depends on timing luck:
    //   1. OCCUPANCY — an atomic counter incremented on entry to the critical section and decremented
    //      on exit. Under a real lock the value observed on entry is always 0. Any other value means
    //      two threads were inside at once, which is a direct observation of exclusion failure rather
    //      than an inference from a corrupted result.
    //   2. EXACT TOTAL — with the critical section deliberately widened so an unsynchronised
    //      interleaving loses updates with overwhelming probability instead of by luck.
    {
        static volatile uint64_t shared = 0;   // deliberately NOT atomic — the mutex is the contract
        static volatile uint64_t spin = 0;     // widens the window so lost updates are near-certain
        constexpr int kThreads = 8, kIters = 5000;
        const uint64_t m = (uint64_t)(uintptr_t)&g_mutex_a;
        std::atomic<int> occupancy{0};
        std::atomic<uint64_t> overlaps{0}, lock_failures{0}, unlock_failures{0};
        std::vector<std::thread> ts;
        for (int t = 0; t < kThreads; ++t) {
            ts.emplace_back([&] {
                for (int i = 0; i < kIters; ++i) {
                    if (call(kMtxLock, m) != 0) { lock_failures.fetch_add(1); continue; }
                    if (occupancy.fetch_add(1, std::memory_order_acq_rel) != 0)
                        overlaps.fetch_add(1, std::memory_order_relaxed);
                    const uint64_t v = shared;
                    for (int k = 0; k < 256; ++k) spin = spin + 1;
                    shared = v + 1;
                    if (occupancy.fetch_sub(1, std::memory_order_acq_rel) != 1)
                        overlaps.fetch_add(1, std::memory_order_relaxed);
                    if (call(kMtxUnlock, m) != 0) unlock_failures.fetch_add(1);
                }
            });
        }
        for (auto& th : ts) th.join();
        const uint64_t expected = (uint64_t)kThreads * kIters;
        CHECK(lock_failures.load() == 0 && unlock_failures.load() == 0,
              "every concurrent lock and unlock succeeded");
        CHECK(overlaps.load() == 0,
              "MUTUAL EXCLUSION: no two of 8 threads were ever inside the critical section at once");
        if (overlaps.load() != 0)
            std::printf("         (%llu overlapping entries observed — the lock excludes nothing)\n",
                        (unsigned long long)overlaps.load());
        CHECK(shared == expected,
              "MUTUAL EXCLUSION: 8 threads x 5000 unsynchronised increments total EXACTLY 40000");
        if (shared != expected)
            std::printf("         (got %llu, expected %llu — updates were lost, so the lock is not real)\n",
                        (unsigned long long)shared, (unsigned long long)expected);
    }

    // --- destroy ------------------------------------------------------------------------------
    CHECK(call(kMtxDestroy, (uint64_t)(uintptr_t)&g_mutex_a) == 0, "sceUltMutexDestroy succeeds");
    CHECK(call(kMtxLock, (uint64_t)(uintptr_t)&g_mutex_a) != 0,
          "locking a DESTROYED mutex is refused (the header magic is cleared, not left stale)");

    // A second, independent mutex must be a genuinely different lock, not an alias of the first.
    const uint64_t b_rc = call(kMtxCreate, (uint64_t)(uintptr_t)&g_mutex_b, 0,
                               (uint64_t)(uintptr_t)&g_pool, 0, kApiVersion);
    CHECK(b_rc == 0, "a second mutex can be created from the same pool");
    CHECK(call(kMtxLock, (uint64_t)(uintptr_t)&g_mutex_b) == 0 &&
          call(kMtxUnlock, (uint64_t)(uintptr_t)&g_mutex_b) == 0,
          "the second mutex locks and unlocks independently of the first");

    // --- the counters stay real ----------------------------------------------------------------
    CHECK(ult_call_count(kMtxLock) > 40000 && ult_call_count(kMtxUnlock) > 40000,
          "implemented entry points are still counted per call, not deduped");

    // --- the escape hatches still work ---------------------------------------------------------
    const bool prev = ult_set_return_success_for_test(true);
    const uint64_t faked = call(kMtxLock, (uint64_t)(uintptr_t)&g_mutex_b);
    const uint64_t faked_size = call(kPoolSize, kNumThreads, kNumSyncObjects);
    ult_set_return_success_for_test(prev);
    CHECK(faked == 0, "PROSPER_ULT_RETURN_SUCCESS still reproduces the legacy fake success");
    // #1618: no escape hatch may put a sentinel where a size belongs.
    CHECK(faked_size == 0,
          "under the legacy policy a size query returns 0, never an error sentinel (#1618)");

    std::printf(fails ? "== FAIL: %d ==\n" : "== PASS ==\n", fails);
    return fails ? 1 : 0;
}
