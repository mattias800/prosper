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
#include <chrono>
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
static const char* const kUltCreate   = "znI3q8S7KQ4";
static const char* const kUltJoin     = "gCeAI57LGgI";
static const char* const kCvCreate    = "jnKaHGkrxZ4";
static const char* const kCvWait      = "5xGAHCxA8M0";
static const char* const kCvSignal    = "JTw1cAVkuc0";
static const char* const kCvDestroy   = "xrmmI832R4U";

// _sceUltUlthreadCreate takes 9 arguments; runtime/optParam/apiVersion arrive on the guest stack and
// the import bridge forwards them as ordinary parameters.
using HleFn9 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                            uint64_t, uint64_t);
static uint64_t call9(const char* nid, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                      uint64_t a4, uint64_t a5, uint64_t a6, uint64_t a7, uint64_t a8) {
    HleFn fn = Hle::lookup(nid);
    if (!fn) { std::printf("  [FAIL] NID %s is not registered\n", nid); ++fails; return ~0ull; }
    return ((HleFn9)fn)(a0, a1, a2, a3, a4, a5, a6, a7, a8);
}

// An ulthread entry is GUEST code, so it uses the guest's System V AMD64 convention — prosper enters
// it through prosper_call_guest_on_stack, which marshals into SysV (first argument in %rdi).
//
// On Linux and macOS the host ABI IS System V, so an ordinary host function is an accurate stand-in
// and this attribute is a no-op. On Windows the host is Microsoft x64, where the first argument
// arrives in %rcx — so an untagged host function reads a garbage argument and returns a garbage
// status, while still executing and touching its stack, which makes it look like it worked. That is
// exactly what happened: MinGW CI failed only on "join writes the entry's int32 return into *status"
// while all 45 other assertions passed. The product was never wrong — real guest code is SysV — the
// test was modelling guest code with a host-ABI function.
//
// Tagged rather than #ifdef'd around the body: the statement "this function uses the guest's calling
// convention" is true on every platform, and the attribute is simply redundant where the native ABI
// already matches. Guarded on x86-64 because that is the only architecture the attribute exists for.
#if defined(__x86_64__) || defined(_M_X64)
#define ULT_GUEST_ABI __attribute__((sysv_abi))
#else
#define ULT_GUEST_ABI
#endif

// Earthion gives each ulthread 0x2000 bytes; this test uses a larger context because the entry is
// host-compiled test code rather than the guest's own tightly-sized worker. It stays at or below the
// 64 KiB stack probe so the high-water figure is exact, which is the shape Earthion's real 8 KiB and
// 32 KiB contexts have.
static constexpr uint64_t kContextBytes = 64 * 1024;
static constexpr uint64_t kTouchBytes   = 24 * 1024;
static std::atomic<int> g_entry_ran{0};
static std::atomic<int> g_entry_gate{0};
static volatile uint64_t g_sink = 0;

// Both entries are deliberately plain leaf functions: no destructors, no exceptions, nothing that
// needs unwind data. __attribute__((sysv_abi)) conflicts with MinGW's SEH-based C++ unwinding (the
// reason PROSPER_SYSV_ABI is empty for the HLE handlers — see dispatch.hpp), and keeping these
// bodies trivial is what makes the tag safe here.

// Touches a known depth of its stack so the context high-water measurement has a floor to clear.
// `volatile` so the writes survive optimisation — an elided frame would measure nothing.
static ULT_GUEST_ABI int32_t touch_stack_entry(uint64_t arg) noexcept {
    volatile unsigned char buf[kTouchBytes];
    for (size_t i = 0; i < kTouchBytes; i += 64) buf[i] = (unsigned char)(i ^ 0x5A);
    for (size_t i = 0; i < kTouchBytes; i += 64) g_sink += buf[i];
    g_entry_ran.fetch_add(1);
    return (int32_t)(arg & 0x7fffffff);
}

// Announces itself and then spins until released, so several ulthreads are provably running at once.
// A bare atomic spin rather than std::this_thread::yield(): a leaf body keeps the SysV tag safe on
// MinGW. The bound turns a gate that is never released into a test failure instead of a CI timeout.
static ULT_GUEST_ABI int32_t gated_entry(uint64_t arg) noexcept {
    g_entry_ran.fetch_add(1);
    for (uint64_t spins = 0; g_entry_gate.load() == 0 && spins < 4000000000ull; ++spins) {}
    return (int32_t)arg;
}

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

    // --- ulthreads ----------------------------------------------------------------------------
    // Each ulthread runs its entry on the guest-supplied context buffer, which IS its stack on
    // hardware. These assertions are all impossible on a tree where _sceUltUlthreadCreate refused:
    // the entry never ran at all, so the counter stays 0 and no status is ever produced.
    {
        g_entry_ran.store(0);
        g_entry_gate.store(0);
        UltBlob ult;
        std::memset(&ult, 0, sizeof(ult));
        std::vector<unsigned char> ctx(kContextBytes);

        const uint64_t rc = call9(kUltCreate, (uint64_t)(uintptr_t)&ult, 0,
                                  (uint64_t)(uintptr_t)&touch_stack_entry, 0x1234,
                                  (uint64_t)(uintptr_t)ctx.data(), kContextBytes,
                                  (uint64_t)(uintptr_t)&g_runtime, 0, kApiVersion);
        CHECK(rc == 0, "_sceUltUlthreadCreate starts the ulthread");

        int32_t status = -1;
        const uint64_t jrc = call(kUltJoin, (uint64_t)(uintptr_t)&ult, (uint64_t)(uintptr_t)&status);
        CHECK(jrc == 0, "sceUltUlthreadJoin returns once the ulthread has finished");
        CHECK(g_entry_ran.load() == 1, "the ulthread entry ACTUALLY RAN (it never did before)");
        // The entry is int32_t(*)(uint64_t) and returns its argument's low bits; join reports it
        // through the 4-byte out-param the guest passes at eboot+0x9f1e.
        CHECK(status == 0x1234, "sceUltUlthreadJoin writes the entry's int32 return into *status");

        // Amendment 6: the context buffer is filled at create and the untouched prefix measured at
        // exit, so an ulthread approaching its stack limit is reported rather than silently
        // overflowing. The entry deliberately touches kTouchBytes.
        uint64_t ctx_size = 0;
        const uint64_t used = ult_last_join_stack_used_for_test(&ctx_size);
        CHECK(ctx_size == kContextBytes && used >= kTouchBytes && used <= kContextBytes,
              "the guest-stack high-water is measured and lands within the context buffer");
        if (!(used >= kTouchBytes && used <= kContextBytes))
            std::printf("         (used=%llu touched=%llu context=%llu)\n",
                        (unsigned long long)used, (unsigned long long)kTouchBytes,
                        (unsigned long long)kContextBytes);

        // A joined ulthread is gone: joining again must be refused, not silently repeated.
        CHECK(call(kUltJoin, (uint64_t)(uintptr_t)&ult, 0) != 0,
              "joining an already-joined ulthread is refused");
    }

    // Amendment 2b: prosper runs ulthreads one-to-one rather than multiplexing them onto
    // numWorkerThread, so more can run at once than on hardware. That deviation must be OBSERVED,
    // not assumed — the runtime carries a high-water mark of simultaneously running ulthreads.
    {
        g_entry_ran.store(0);
        g_entry_gate.store(0);                 // entries spin here until released
        constexpr int kN = 6;                  // deliberately > numWorkerThread (3)
        UltBlob ults[kN];
        std::vector<std::vector<unsigned char>> ctxs;
        bool all_started = true;
        for (int i = 0; i < kN; ++i) {
            std::memset(&ults[i], 0, sizeof(ults[i]));
            ctxs.emplace_back(kContextBytes);
        }
        for (int i = 0; i < kN; ++i)
            all_started &= call9(kUltCreate, (uint64_t)(uintptr_t)&ults[i], 0,
                                 (uint64_t)(uintptr_t)&gated_entry, (uint64_t)i,
                                 (uint64_t)(uintptr_t)ctxs[i].data(), kContextBytes,
                                 (uint64_t)(uintptr_t)&g_runtime, 0, kApiVersion) == 0;
        CHECK(all_started, "six ulthreads start in a runtime created for numMaxUlthread=16");
        // Wait until every entry is spinning, so the high-water reflects real simultaneity.
        for (int spins = 0; g_entry_ran.load() < kN && spins < 200000; ++spins)
            std::this_thread::yield();
        const uint64_t hw = ult_runtime_high_water_for_test((uint64_t)(uintptr_t)&g_runtime);
        g_entry_gate.store(1);
        for (int i = 0; i < kN; ++i) call(kUltJoin, (uint64_t)(uintptr_t)&ults[i], 0);
        CHECK(hw > kNumWorkerThread,
              "the runtime OBSERVES that more ulthreads ran at once than numWorkerThread=3");
        if (hw <= kNumWorkerThread)
            std::printf("         (high-water %llu — the deviation counter is not measuring)\n",
                        (unsigned long long)hw);
    }

    // numMaxUlthread is the guest's own bound and is unambiguous, so it is ENFORCED rather than
    // merely reported: a runtime sized for 1 must refuse the second ulthread.
    {
        g_entry_ran.store(0);
        g_entry_gate.store(0);
        UltBlob small_rt, u1, u2;
        std::memset(&small_rt, 0, sizeof(small_rt));
        std::memset(&u1, 0, sizeof(u1));
        std::memset(&u2, 0, sizeof(u2));
        const uint64_t bytes = call(kRtSize, 1, 1);
        std::vector<unsigned char> work((size_t)bytes, 0);
        std::vector<unsigned char> c1(kContextBytes), c2(kContextBytes);
        CHECK(call7(kRtCreate, (uint64_t)(uintptr_t)&small_rt, 0, 1, 1,
                    (uint64_t)(uintptr_t)work.data(), 0, kApiVersion) == 0,
              "a runtime sized for numMaxUlthread=1 is created");
        const uint64_t first = call9(kUltCreate, (uint64_t)(uintptr_t)&u1, 0,
                                     (uint64_t)(uintptr_t)&gated_entry, 0,
                                     (uint64_t)(uintptr_t)c1.data(), kContextBytes,
                                     (uint64_t)(uintptr_t)&small_rt, 0, kApiVersion);
        const uint64_t second = call9(kUltCreate, (uint64_t)(uintptr_t)&u2, 0,
                                      (uint64_t)(uintptr_t)&gated_entry, 0,
                                      (uint64_t)(uintptr_t)c2.data(), kContextBytes,
                                      (uint64_t)(uintptr_t)&small_rt, 0, kApiVersion);
        CHECK(first == 0 && second != 0,
              "the runtime refuses an ulthread beyond the numMaxUlthread the guest asked for");
        g_entry_gate.store(1);
        call(kUltJoin, (uint64_t)(uintptr_t)&u1, 0);
    }

    // --- condition variables --------------------------------------------------------------------
    // sceUltConditionVariableWait takes ONE argument; the mutex is bound at create. The tests below
    // exercise the two facts the implementation is built on, both established from the guest's own
    // call sites: the caller HOLDS the bound mutex at Wait, and does NOT hold it at Signal.
    {
        UltBlob cv, cv_mutex;
        std::memset(&cv, 0, sizeof(cv));
        std::memset(&cv_mutex, 0, sizeof(cv_mutex));
        CHECK(call(kMtxCreate, (uint64_t)(uintptr_t)&cv_mutex, 0, (uint64_t)(uintptr_t)&g_pool, 0,
                   kApiVersion) == 0,
              "a mutex for the condvar to bind is created");
        CHECK(call(kCvCreate, (uint64_t)(uintptr_t)&cv, 0, (uint64_t)(uintptr_t)&cv_mutex, 0,
                   kApiVersion) == 0,
              "_sceUltConditionVariableCreate binds the condvar to that mutex");

        // Binding to something that is not a mutex must be refused.
        UltBlob cv2, not_a_mutex;
        std::memset(&cv2, 0, sizeof(cv2));
        std::memset(&not_a_mutex, 0, sizeof(not_a_mutex));
        CHECK(call(kCvCreate, (uint64_t)(uintptr_t)&cv2, 0, (uint64_t)(uintptr_t)&not_a_mutex, 0,
                   kApiVersion) != 0,
              "_sceUltConditionVariableCreate refuses a mutex that was never created");

        // Waiting WITHOUT the bound mutex is undefined for pthread_cond_wait, so it must be refused
        // rather than executed. (The guest never does this — all five of its Wait sites lock first —
        // but a fake implementation that ignored the mutex entirely would pass every other check.)
        CHECK(call(kCvWait, (uint64_t)(uintptr_t)&cv) != 0,
              "sceUltConditionVariableWait without holding the bound mutex is refused");

        // A Signal with nobody waiting is LOST, exactly as for a real condition variable. If it were
        // remembered, the next Wait would return immediately without ever being signalled — the
        // difference between a condvar and a semaphore, and a real source of guest logic bugs.
        CHECK(call(kCvSignal, (uint64_t)(uintptr_t)&cv) == 0, "a Signal with no waiter succeeds");

        // The real contract: a waiter blocks until signalled, and the signaller does NOT hold the
        // bound mutex when it signals — which is what the guest does at all eleven of its sites.
        std::atomic<int> waiter_state{0};   // 1 = about to wait, 2 = returned from wait
        std::thread waiter([&] {
            if (call(kMtxLock, (uint64_t)(uintptr_t)&cv_mutex) != 0) return;
            waiter_state.store(1);
            if (call(kCvWait, (uint64_t)(uintptr_t)&cv) == 0) waiter_state.store(2);
            call(kMtxUnlock, (uint64_t)(uintptr_t)&cv_mutex);
        });
        while (waiter_state.load() == 0) std::this_thread::yield();
        // Give the waiter time to actually block. If the earlier lost Signal had been remembered, or
        // if Wait returned spuriously, it would have advanced to 2 by now.
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        const bool still_waiting = waiter_state.load() == 1;
        // Signal from a thread that holds NOTHING — the guest's own pattern.
        call(kCvSignal, (uint64_t)(uintptr_t)&cv);
        waiter.join();
        CHECK(still_waiting,
              "a waiter stays blocked: an earlier signal with no waiter was lost, not remembered");
        CHECK(waiter_state.load() == 2,
              "sceUltConditionVariableSignal from a thread holding no mutex wakes the waiter");

        // Exactly one waiter per Signal, and the bound mutex is genuinely reacquired on return —
        // both threads increment a non-atomic counter while nominally holding it.
        {
            static volatile int guarded = 0;
            std::atomic<int> woken{0}, ready{0};
            std::thread w1, w2;
            auto body = [&] {
                call(kMtxLock, (uint64_t)(uintptr_t)&cv_mutex);
                ready.fetch_add(1);
                if (call(kCvWait, (uint64_t)(uintptr_t)&cv) == 0) {
                    const int v = guarded; guarded = v + 1;   // must run under the reacquired mutex
                    woken.fetch_add(1);
                }
                call(kMtxUnlock, (uint64_t)(uintptr_t)&cv_mutex);
            };
            w1 = std::thread(body);
            w2 = std::thread(body);
            while (ready.load() < 2) std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            call(kCvSignal, (uint64_t)(uintptr_t)&cv);
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            const int after_one = woken.load();
            call(kCvSignal, (uint64_t)(uintptr_t)&cv);
            w1.join(); w2.join();
            CHECK(after_one == 1, "one Signal releases exactly ONE of two waiters, not both");
            CHECK(woken.load() == 2 && guarded == 2,
                  "the second Signal releases the other, and both ran under the reacquired mutex");
        }

        CHECK(call(kCvDestroy, (uint64_t)(uintptr_t)&cv) == 0,
              "sceUltConditionVariableDestroy succeeds");
        CHECK(call(kCvSignal, (uint64_t)(uintptr_t)&cv) != 0,
              "signalling a DESTROYED condvar is refused");
    }

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
