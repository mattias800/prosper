// test_sync_destroy_lifetime — #2042. Destroying a guest synchronization object must not hand its
// storage back to the allocator, because prosper's handlers resolve the object from a guest slot and
// then dereference it — sometimes while PARKED INSIDE it. A freed block is reused by the very next
// `calloc`, so a guest that destroys an object another thread is still using stops corrupting its
// own state and starts corrupting prosper's heap instead.
//
// The arms below call the handlers through the NID registry, exactly as the guest reaches them.
//
//   Arm 1  every *Destroy handler in the family: within the quarantine window, a destroyed object's
//          address is never issued again. Deterministic under the fix on every allocator; RED on
//          glibc because its tcache returns a just-freed block of the same size to the very next
//          allocation, which is precisely the reuse that turns the dangling pointer into corruption.
//   Arm 2  the case the issue describes, constructed BY HAND: a thread parked inside
//          `k_rwlock_rdlock` on a write-held lock, destroyed under it. Asserts the parked thread's
//          object stays live and that the next Init does not alias it.
//
// Arm 2 asserts its own preconditions before it asserts anything else (the write hold really
// excludes; the reader really did not acquire), because an arm that cannot construct the state it
// claims to test passes for the wrong reason — the harness would be testing nothing at all.
//
// THE COUNTER-ARM IS PART OF THE SUITE. `PROSPER_SYNC_RETIRE_SECONDS=0` restores the pre-#2042
// immediate free on this same binary, and CMake registers that as `sync_destroy_lifetime_counter_arm`
// with WILL_FAIL (**Linux only** — the red depends on glibc's tcache reusing a just-freed block
// immediately, which is the mechanism being demonstrated and not measured on other allocators; see
// the CMake comment), so CI keeps proving that these assertions CAN fail. Measured: 10 do — all seven
// spellings in arm 1 and three in arm 2, including `[rwlock] UNMATCHED unlock … holds=0` as the
// freed object's accounting is reused by the next Init and the parked reader is never woken. The
// retirement census below still passes under the counter-arm, and should: the objects really are
// handed over, the window is simply zero, so they are reclaimed on the spot. Rebuilding this file
// against pre-#2042 handlers instead fails 11 — the census too, since nothing is handed over at all.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include "../src/hle/sync_retire.hpp"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

namespace {

HleFn look(const char* name) { return Hle::lookup(nid_hash(name)); }

uint64_t call(HleFn f, uint64_t a0, uint64_t a1 = 0, uint64_t a2 = 0) {
    return f(a0, a1, a2, 0, 0, 0);
}

// One round of Init/Destroy through a guest slot, reporting the object the Init published.
// `init_a2` carries the one argument that differs across the family (semaphore value, barrier count).
void* init_destroy_round(HleFn init, HleFn destroy, uint64_t init_a2) {
    void* slot = nullptr;
    call(init, (uint64_t)(uintptr_t)&slot, 0, init_a2);
    void* published = slot;
    call(destroy, (uint64_t)(uintptr_t)&slot);
    return published;
}

// Arm 1 for one spelling. Rounds of Init/Destroy must never publish the same address twice.
void distinct_storage_arm(const char* label, const char* init_name, const char* destroy_name,
                          uint64_t init_a2) {
    HleFn init = look(init_name), destroy = look(destroy_name);
    const std::string registered = std::string(label) + ": both handlers registered";
    CHECK(init && destroy, registered.c_str());
    if (!init || !destroy) return;

    constexpr int kRounds = 8;
    std::vector<void*> seen;
    bool allocated_every_round = true;
    for (int i = 0; i < kRounds; ++i) {
        void* p = init_destroy_round(init, destroy, init_a2);
        if (!p) { allocated_every_round = false; break; }
        seen.push_back(p);
    }
    // Without this the arm would pass vacuously against an Init that published nothing.
    const std::string allocated = std::string(label) + ": Init published an object every round";
    CHECK(allocated_every_round && (int)seen.size() == kRounds, allocated.c_str());

    int repeats = 0;
    for (size_t i = 0; i < seen.size(); ++i)
        for (size_t j = i + 1; j < seen.size(); ++j)
            if (seen[i] == seen[j]) ++repeats;
    char msg[192];
    snprintf(msg, sizeof msg,
             "%s: no destroyed object's storage is issued again (%d repeats over %d rounds)",
             label, repeats, kRounds);
    CHECK(repeats == 0, msg);
}

}  // namespace

int main() {
    printf("== test_sync_destroy_lifetime ==\n");
    register_builtin_hle();

    const uint64_t retired_at_start = retired_sync_object_count();

    // ---- Arm 1: the whole family -------------------------------------------------------------
    // Every guest-slot sync object with a *Destroy handler. The issue named four; barrier and the
    // guest STL's own _Mtx_/_Cnd_ spellings have the identical shape and are included here.
    printf("-- arm 1: a destroyed object's storage is never reissued --\n");
    distinct_storage_arm("rwlock",  "scePthreadRwlockInit",  "scePthreadRwlockDestroy",  0);
    distinct_storage_arm("mutex",   "scePthreadMutexInit",   "scePthreadMutexDestroy",   0);
    distinct_storage_arm("cond",    "scePthreadCondInit",    "scePthreadCondDestroy",    0);
    distinct_storage_arm("sem",     "scePthreadSemInit",     "scePthreadSemDestroy",     1);
    distinct_storage_arm("barrier", "scePthreadBarrierInit", "scePthreadBarrierDestroy", 2);
    distinct_storage_arm("_Mtx",    "_Mtx_init",             "_Mtx_destroy",             0);
    distinct_storage_arm("_Cnd",    "_Cnd_init",             "_Cnd_destroy",             0);

    // 7 spellings x 8 rounds. Counting them is what separates "retired" from "the allocator merely
    // happened not to reuse the block", which is the only other way arm 1 could come out clean.
    const uint64_t retired = retired_sync_object_count() - retired_at_start;
    char count_msg[160];
    snprintf(count_msg, sizeof count_msg,
             "every Destroy retired its object (%llu retired, expected 56; %llu still quarantined)",
             (unsigned long long)retired, (unsigned long long)quarantined_sync_object_count());
    CHECK(retired == 56, count_msg);

    // ---- Arm 2: destroy under a thread parked inside the handler ------------------------------
    printf("-- arm 2: an rwlock a thread is parked inside survives its Destroy --\n");
    HleFn RWinit  = look("scePthreadRwlockInit");
    HleFn RWwr    = look("scePthreadRwlockWrlock");
    HleFn RWrd    = look("scePthreadRwlockRdlock");
    HleFn RWun    = look("scePthreadRwlockUnlock");
    HleFn RWtrywr = look("scePthreadRwlockTrywrlock");
    HleFn RWdes   = look("scePthreadRwlockDestroy");
    CHECK(RWinit && RWwr && RWrd && RWun && RWtrywr && RWdes, "rwlock handlers registered");
    if (!(RWinit && RWwr && RWrd && RWun && RWtrywr && RWdes)) { printf("== FAIL ==\n"); return 1; }

    void* slot_a = nullptr;
    call(RWinit, (uint64_t)(uintptr_t)&slot_a);
    void* parked_object = slot_a;
    CHECK(parked_object != nullptr, "rwlock A initialised");

    CHECK(call(RWwr, (uint64_t)(uintptr_t)&slot_a) == 0, "main takes the write lock on A");
    // PRECONDITION, not a result: the hold must be real, or nothing below constructs the case.
    CHECK(call(RWtrywr, (uint64_t)(uintptr_t)&slot_a) != 0,
          "precondition: the write hold really excludes (trywrlock refused)");

    std::atomic<bool> entered{false}, acquired{false};
    std::thread reader([&] {
        entered.store(true, std::memory_order_release);
        call(RWrd, (uint64_t)(uintptr_t)&slot_a);          // parks inside pthread_rwlock_rdlock
        acquired.store(true, std::memory_order_release);
    });
    while (!entered.load(std::memory_order_acquire)) std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    // PRECONDITION: it cannot have acquired — main holds the write lock — so after 200 ms past its
    // own entry flag it is inside k_rwlock_rdlock, holding `parked_object` as a raw pointer.
    CHECK(!acquired.load(std::memory_order_acquire),
          "precondition: the reader is parked INSIDE k_rwlock_rdlock on this object");

    call(RWdes, (uint64_t)(uintptr_t)&slot_a);             // the guest bug: destroy under the waiter
    CHECK(slot_a == nullptr, "Destroy cleared the guest slot (unchanged behaviour)");

    void* slot_b = nullptr;
    call(RWinit, (uint64_t)(uintptr_t)&slot_b);
    CHECK(slot_b != nullptr, "rwlock B initialised after the destroy");
    // THE ASSERTION. On master the allocator hands B the block A was freed from, so B and the
    // object the reader is parked in are the same memory: B's fresh pthread_rwlock_init resets the
    // futex state under the parked thread, and B's hold accounting is whatever that thread does next.
    CHECK(slot_b != parked_object,
          "a new rwlock does not alias the object a thread is parked in");

    // The destroy cleared slot A, so the guest can no longer reach the object it is holding — but
    // the object itself must still be a live rwlock. Reach it through a slot holding the same
    // pointer (which is exactly what an in-flight handler holds) and release the write hold: the
    // parked reader must then wake. Under the fix this is an ordinary operation on a retired object;
    // on master it is the use-after-free itself.
    void* alias = parked_object;
    CHECK(call(RWun, (uint64_t)(uintptr_t)&alias) == 0,
          "the retired object is still a live rwlock: its write hold releases");

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!acquired.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    const bool woke = acquired.load(std::memory_order_acquire);
    CHECK(woke, "the parked reader woke and completed its acquisition on the retired object");
    if (woke) reader.join();
    else      reader.detach();   // still parked: joining would hang the suite instead of failing it

    if (woke)
        CHECK(call(RWun, (uint64_t)(uintptr_t)&alias) == 0,
              "the read hold the parked reader took releases normally afterwards");

    // Lock B is untouched by any of it: nothing that happened to A reached it.
    CHECK(call(RWtrywr, (uint64_t)(uintptr_t)&slot_b) == 0,
          "rwlock B is uncontaminated: it write-acquires cleanly");
    call(RWun, (uint64_t)(uintptr_t)&slot_b);

    printf(fails ? "== FAIL ==\n" : "== PASS ==\n");
    return fails ? 1 : 0;
}
