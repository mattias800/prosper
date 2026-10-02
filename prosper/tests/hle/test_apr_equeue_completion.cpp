// test_apr_equeue_completion — APR completion events on a SHARED equeue must not be coalesced
// away when the binding tag is not a counter (#1673).
//
// Two tag dialects share the H896Pt-yB4I binding call, discriminated by the binding's id:
//
//   id == 0  IoDispatcher direct channel (#210): tag is an opaque per-request POINTER. Already
//            posted with coalesce=false — every completion is delivered. Not the subject here.
//   id != 0  FAPREventQueueListener channel (#208): tag is (ring<<58)|counter, ctor-seeded dense
//            from 1000. prosper posts the per-(eq,ring) HIGH-WATER MARK and coalesces, which is
//            exactly the kqueue "completed up to" level the listener's last+1..cnt range walk
//            implements. Correct for that consumer, and this test pins it.
//
// CRI ADX2 (cri_ware_unity.prx) uses the id != 0 branch with a tag that is a literal ZERO
// (`xor ecx,ecx` at cri+0x11b88f) and a waiter that tests only the event IDENT
// (sceKernelWaitEqueue + sceKernelGetEventId at cri+0x11ba65, NULL timeout, retry until match).
// So its "counter" never advances, every completion posts an IDENTICAL (ident, filter) event, and
// the level-style coalescing collapses N discrete completions into one delivery. The waiter that
// does not get an event blocks forever on a NULL timeout.
//
// This is the same rule prosper already applies to the two other completion sources: a completion
// is a discrete COUNT, never a level (EOP #234, APR pointer-tag #210). The counter dialect is the
// one genuine exception, because there the counter itself carries "completed up to".
//
// Live shape this reproduces (PPSA19991 Tales of Graces f Remastered, headless boot_trace,
// PROSPER_AMPRLOG=1 PROSPER_EVLOG=1): ONE equeue 0x318b75c0, 471 bound submits, of which 437 bind
// id=0 (pointer dialect) and 34 bind a nonzero id — 22 of those on id=1 alone — every one of the
// 34 posting `AprTagComplete token=0x0 (ring=0)`. Same equeue, same ident, constant zero tag.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/kernel/apr_event_dialect.hpp"
#include "hle/kernel/kernel_event_filters.hpp"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <vector>

using namespace prosper;

namespace prosper {
// Test seam (hle_kernel_mem.cpp): the dialect recorded on a command buffer's binding at bind time.
bool prosper_apr_binding_dialect_for_test(uint64_t cb, AprDialect* out);
}

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// SceKernelEvent (FreeBSD kevent layout, 0x20 bytes) — must match the backend's struct exactly.
struct KEvent { int64_t ident; int16_t filter; uint16_t flags; uint32_t fflags; int64_t data; uint64_t udata; };
static_assert(sizeof(KEvent) == 0x20, "SceKernelEvent must be 0x20 bytes");

// The APR completion filter the backend posts (SCE_KERNEL_EVFILT_AMPR, -25). Some guests read it:
// Black Flag Resynced's completion consumer dispatches on filter == 0xffe7 and ignores every other
// event, so the exact value is part of the contract (#4139). Asserting it here is also what proves
// the counted entries are APR completions and not some other source that merely happened to raise
// the queue depth.
static constexpr int16_t kAprFilter = EVFILT_AMPR;   // shared constant, pinned to -25 here
static_assert(EVFILT_AMPR == -25, "the guest dispatches on filter 0xffe7");

// Drain whatever is queued so each scenario starts from an empty queue.
static void drain(HleFn wait, uint64_t eq) {
    std::vector<KEvent> ev(128);
    int32_t out = -1; uint32_t cap = 1000;   // 1 ms
    wait(eq, (uint64_t)(uintptr_t)ev.data(), ev.size(), (uint64_t)(uintptr_t)&out,
         (uint64_t)(uintptr_t)&cap, 0);
}

int main() {
    printf("== test_apr_equeue_completion ==\n");
    register_builtin_hle();

    auto create   = Hle::lookup(nid_hash("sceKernelCreateEqueue"));
    auto addampr  = Hle::lookup(nid_hash("sceKernelAddAmprEvent"));
    auto adduser  = Hle::lookup(nid_hash("sceKernelAddUserEvent"));
    auto trigger  = Hle::lookup(nid_hash("sceKernelTriggerUserEvent"));
    auto wait     = Hle::lookup(nid_hash("sceKernelWaitEqueue"));
    auto getcount = Hle::lookup(nid_hash("sceKernelGetEventCount"));
    auto bind     = Hle::lookup("H896Pt-yB4I");   // AprCbSetEventQueue (legacy binding NID)
    auto submit   = Hle::lookup("eE4Szl8sil8");   // sceKernelAprSubmitCommandBuffer (plain, CRI's)
    CHECK(create && addampr && adduser && trigger && wait && getcount && bind && submit,
          "all APR + equeue entry points registered");
    if (!(create && addampr && adduser && trigger && wait && getcount && bind && submit)) {
        printf("== FAIL ==\n"); return 1;
    }

    uint64_t eq = 0;
    create((uint64_t)(uintptr_t)&eq, 0, 0, 0, 0, 0);
    CHECK(eq != 0, "CreateEqueue produced a handle");

    // The posts are deferred ~2 ms on a detached thread (modeled DMA latency). Wait generously so a
    // slow machine cannot turn "not yet delivered" into a false pass on the count assertions below.
    const auto settle = [] { std::this_thread::sleep_for(std::chrono::milliseconds(120)); };

    // --- Subject: the constant-tag (CRI) dialect on a SHARED equeue. -------------------------
    // Two command buffers, ONE equeue, the SAME nonzero id, tag = 0 for both — exactly the live
    // PPSA19991 shape. Two discrete completions, so two deliveries are owed.
    {
        const int64_t kId = 1;
        const uint64_t cb1 = 0x205df94000ull, cb2 = 0x205df94028ull;   // live cb addresses
        addampr(eq, (uint64_t)kId, 0, 0, 0, 0);
        bind(cb1, eq, (uint64_t)kId, /*tag=*/0, 0, 0);
        bind(cb2, eq, (uint64_t)kId, /*tag=*/0, 0, 0);
        submit(cb1, /*ring_1based=*/1, 0, 0, 0, 0);
        submit(cb2, /*ring_1based=*/1, 0, 0, 0, 0);
        settle();

        const uint64_t n = getcount(eq, 0, 0, 0, 0, 0);
        CHECK(n == 2, "two constant-tag APR completions on a shared equeue are both retained");

        // A bare count of 2 would also be satisfied by two events that never shared a coalescing
        // key at all, which is the failure this assertion exists to exclude: read them back and
        // require that both really are APR completions carrying the SAME (ident, filter). Only a
        // non-coalescing post can leave two same-key entries queued.
        std::vector<KEvent> ev(8);
        int32_t out = -1; uint32_t cap = 50000;
        wait(eq, (uint64_t)(uintptr_t)ev.data(), ev.size(), (uint64_t)(uintptr_t)&out,
             (uint64_t)(uintptr_t)&cap, 0);
        CHECK(out == 2, "WaitEqueue returns both completions");
        const bool same_key = out == 2 &&
                              ev[0].ident == kId && ev[1].ident == kId &&
                              ev[0].filter == kAprFilter && ev[1].filter == kAprFilter;
        CHECK(same_key, "both retained entries are APR completions sharing one (ident, filter)");
        drain(wait, eq);
    }

    // --- Control 1: the counter dialect must STILL coalesce to the high-water mark. -----------
    // This is the #208 listener's "completed up to" level. A fix that simply switched the whole
    // id != 0 branch to coalesce=false would break this arm, which is why it is here: it bounds
    // the change to the dialect that actually needs it.
    {
        const int64_t kId = 0x74fe;
        const uint64_t cb3 = 0x205df95000ull, cb4 = 0x205df95028ull;
        addampr(eq, (uint64_t)kId, 0, 0, 0, 0);
        // Dense ascending per-ring counters, ring 0, ctor-seeded at 1000 — the real guest shape.
        bind(cb3, eq, (uint64_t)kId, /*tag=*/1000, 0, 0);
        submit(cb3, 1, 0, 0, 0, 0);
        bind(cb4, eq, (uint64_t)kId, /*tag=*/1001, 0, 0);
        submit(cb4, 1, 0, 0, 0, 0);
        settle();

        const uint64_t n = getcount(eq, 0, 0, 0, 0, 0);
        CHECK(n == 1, "advancing counter-tag completions still coalesce to one level event");

        std::vector<KEvent> ev(8);
        int32_t out = -1; uint32_t cap = 50000;
        wait(eq, (uint64_t)(uintptr_t)ev.data(), ev.size(), (uint64_t)(uintptr_t)&out,
             (uint64_t)(uintptr_t)&cap, 0);
        CHECK(out == 1 && ev[0].data == 1001,
              "the coalesced level carries the HIGHEST counter (completed-up-to)");
        drain(wait, eq);
    }

    // --- Control 2: a genuine LEVEL source must keep coalescing. ------------------------------
    // Guards against an over-broad fix that disables coalescing in eq_post generally. The 60 Hz
    // vblank pump depends on this; without it the queue fills with stale ticks.
    {
        const int64_t kUserId = 4242;
        adduser(eq, (uint64_t)kUserId, 0xFEEDu, 0, 0, 0);
        trigger(eq, (uint64_t)kUserId, 0, 0, 0, 0);
        trigger(eq, (uint64_t)kUserId, 0, 0, 0, 0);
        trigger(eq, (uint64_t)kUserId, 0, 0, 0, 0);
        const uint64_t n = getcount(eq, 0, 0, 0, 0, 0);
        CHECK(n == 1, "repeated level-source (user event) triggers still coalesce to one");
        drain(wait, eq);
    }

    // --- Control 3: the pointer dialect (id == 0) is unchanged and still never coalesces. -----
    {
        const uint64_t cb5 = 0x205df96000ull, cb6 = 0x205df96028ull;
        addampr(eq, 0, 0, 0, 0, 0);
        bind(cb5, eq, /*id=*/0, /*tag=*/0xAAAA0000ull, 0, 0);
        bind(cb6, eq, /*id=*/0, /*tag=*/0xBBBB0000ull, 0, 0);
        submit(cb5, 1, 0, 0, 0, 0);
        submit(cb6, 1, 0, 0, 0, 0);
        settle();
        const uint64_t n = getcount(eq, 0, 0, 0, 0, 0);
        CHECK(n == 2, "pointer-dialect (id==0) completions remain individually delivered");
        drain(wait, eq);
    }

    // --- Black Flag Resynced's shape: id != 0 with a HEAP-POINTER tag. ----------------------------
    // Its IOComplete consumer reads event.data as the request pointer and filter as 0xffe7. Two
    // outstanding requests must arrive as two events, each carrying ITS OWN pointer: treating the
    // pointer as a counter would coalesce them and deliver the high-water mark (the other request).
    {
        const uint64_t cb7 = 0x205df97000ull, cb8 = 0x205df97028ull;
        constexpr uint64_t kPtrA = 0x4080041240ull, kPtrB = 0x4080052380ull;
        addampr(eq, 1, 0, 0, 0, 0);
        bind(cb7, eq, /*id=*/1, /*tag=*/kPtrA, 0, 0);
        bind(cb8, eq, /*id=*/1, /*tag=*/kPtrB, 0, 0);
        submit(cb7, 1, 0, 0, 0, 0);
        submit(cb8, 1, 0, 0, 0, 0);
        settle();
        const uint64_t n = getcount(eq, 0, 0, 0, 0, 0);
        CHECK(n == 2, "id != 0 pointer-tag completions are individually delivered, not coalesced");
        std::vector<KEvent> ev(128);
        int32_t out = -1; uint32_t cap = 1000;
        wait(eq, (uint64_t)(uintptr_t)ev.data(), ev.size(), (uint64_t)(uintptr_t)&out,
             (uint64_t)(uintptr_t)&cap, 0);
        bool a = false, b = false, filt = out == 2;
        for (int i = 0; i < out && i < 2; i++) {
            a |= (uint64_t)ev[i].data == kPtrA;
            b |= (uint64_t)ev[i].data == kPtrB;
            filt &= (uint16_t)ev[i].filter == 0xffe7;
        }
        CHECK(a && b, "each pointer-tag event carries its own request pointer in data");
        CHECK(filt, "the posted filter is 0xffe7 (-25), the value the guest dispatches on");
        drain(wait, eq);
    }

    // --- Dialect is recorded on the binding at bind time. ------------------------------------------
    // One bind per dialect, then read the stored enum. Pins the classification each live consumer
    // needs: UE4 listener counter, IoDispatcher / Black Flag request pointer, CRI constant zero.
    {
        const auto dialect_of = [&](uint64_t cb, AprDialect* d) {
            return prosper_apr_binding_dialect_for_test(cb, d);
        };
        AprDialect d = AprDialect::Counter;
        bind(0x205dfa0000ull, eq, 0x74fe + 4, (4ull << 58) | 1000, 0, 0);
        CHECK(dialect_of(0x205dfa0000ull, &d) && d == AprDialect::Counter,
              "UE4 listener bind (id 0x74fe+ring, dense counter tag) is the Counter dialect");
        bind(0x205dfa0028ull, eq, 0, 0x4080041240ull, 0, 7 | 0xf);
        CHECK(dialect_of(0x205dfa0028ull, &d) && d == AprDialect::RequestPointer,
              "IoDispatcher bind (id 0, pointer tag) is the RequestPointer dialect");
        bind(0x205dfa0050ull, eq, 1, 0x4080052380ull, 0, 0);
        CHECK(dialect_of(0x205dfa0050ull, &d) && d == AprDialect::RequestPointer,
              "Black Flag bind (id 1, heap-pointer tag) is the RequestPointer dialect");
        bind(0x205dfa0078ull, eq, 1, 0, 0, 0);
        CHECK(dialect_of(0x205dfa0078ull, &d) && d == AprDialect::ConstantZero,
              "CRI bind (id != 0, tag 0) is the ConstantZero dialect");
        // Rebinding the same command buffer re-classifies it: the dialect follows the latest bind.
        bind(0x205dfa0078ull, eq, 0x74fe, 1000, 0, 0);
        CHECK(dialect_of(0x205dfa0078ull, &d) && d == AprDialect::Counter,
              "a rebind replaces the recorded dialect");
        CHECK(!dialect_of(0x205dfaffffull, &d), "an unbound command buffer has no dialect");
        static_assert(classify_apr_dialect(0, 0) == AprDialect::RequestPointer, "id 0 is always a pointer");
        static_assert(classify_apr_dialect(5, (1ull << 32) - 1) == AprDialect::Counter, "below 2^32 is a counter");
        static_assert(classify_apr_dialect(5, 1ull << 32) == AprDialect::RequestPointer, "2^32 is a pointer");
        drain(wait, eq);
    }

    // --- A pointer-dialect tag must not poison the Counter dialect's high-water mark. --------------
    // Both bind on ONE equeue and the pointer tag's ring bits are 0, the same ring as a ring-0
    // listener counter. Feeding the pointer into the counter's high-water mark would make the next
    // counter completion deliver the POINTER as its "completed up to" value.
    {
        uint64_t eq2 = 0;
        create((uint64_t)(uintptr_t)&eq2, 0, 0, 0, 0, 0);
        constexpr uint64_t kPtr = 0x4080099000ull;
        const uint64_t cbp = 0x205dfb0000ull, cbc = 0x205dfb0028ull;
        addampr(eq2, 1, 0, 0, 0, 0);
        addampr(eq2, 0x74fe, 0, 0, 0, 0);
        bind(cbp, eq2, 1, kPtr, 0, 0);
        bind(cbc, eq2, 0x74fe, 1000, 0, 0);
        submit(cbp, 1, 0, 0, 0, 0);
        settle();
        submit(cbc, 1, 0, 0, 0, 0);
        settle();
        std::vector<KEvent> ev(8);
        int32_t out = -1; uint32_t cap = 50000;
        wait(eq2, (uint64_t)(uintptr_t)ev.data(), ev.size(), (uint64_t)(uintptr_t)&out,
             (uint64_t)(uintptr_t)&cap, 0);
        bool ptr_ok = false, ctr_ok = false;
        for (int i = 0; i < out && i < 8; i++) {
            if (ev[i].ident == 1) ptr_ok |= (uint64_t)ev[i].data == kPtr;
            if (ev[i].ident == 0x74fe) ctr_ok |= (uint64_t)ev[i].data == 1000;
        }
        CHECK(out == 2 && ptr_ok, "the pointer completion carries its own pointer");
        CHECK(ctr_ok, "the counter completion still carries its own counter, not the pointer");
    }

    printf(fails ? "== FAIL ==\n" : "== PASS ==\n");
    return fails ? 1 : 0;
}
