// sync_futex.cpp — see sync_futex.hpp.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE // pthread_cond_clockwait on Linux
#endif
#include "sync_futex.hpp"
#include "dispatch.hpp"
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <climits>
#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#include <sys/syscall.h>
#include "../host/posix_shim.hpp"
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00   // WakeByAddressAll needs >= 0x0602 (Win8)
#endif
#include <windows.h>   // native futex: WakeByAddressAll (pairs with WaitOnAddress in hle_kernel_mem.cpp)
#include <pthread.h>
#include <array>
#endif

namespace prosper {
namespace {
std::atomic<int> g_waiters{0};
#ifdef _WIN32
struct WaitSlot {
    std::atomic<uint64_t> publication{0};
    uint64_t next_publication = 0; // written only while publication holds the exclusive sentinel
    std::atomic<uint64_t> pthread_id{0};
    std::atomic<uint64_t> windows_tid{0};
    std::atomic<uintptr_t> object{0};
    std::atomic<uintptr_t> source{0};
    std::atomic<GuestWaitKind> kind{GuestWaitKind::None};
};
// SuspendThread can stop a target at any instruction. A mutex-protected registry can therefore
// deadlock when the target is suspended while owning that mutex and the raising thread tries to
// look up the wait it must wake. Fixed atomic slots keep exception delivery suspension-safe.
std::array<WaitSlot, 1024> g_wait_slots;
constexpr uint64_t kWaitSlotPublishing = UINT64_MAX;

struct CondSlot {
    std::atomic<uintptr_t> cond{0};
    std::atomic<uint32_t> sequence{0};
};

// PROSPER_SYNC_RING: a lock-free ring of the last N synchronisation events.
//
// Motivating case (#2139): a title deadlocks with EVERY guest thread parked and none running. A
// thread snapshot says where everyone is stuck, but not how they got there -- and once the deadlock
// is total there is nothing left running to log anything. What is needed is the tail of history
// leading INTO the freeze: which object was last signalled, which wait was interrupted, and whether
// anybody signalled the object the stuck thread is waiting on. This is that record.
//
// Deliberately a ring of plain stores rather than PROSPER_SYNCLOG's fprintf: this sits on the
// hottest paths in the emulator, and the failure it exists to diagnose is TIMING SENSITIVE, so an
// instrument that serialises on stderr changes the answer. Off by default for the same reason.
enum class SyncTraceKind : uint32_t { WaitEnter, WaitWake, Signal, Broadcast, Interrupt,
                                      FutexWait, FutexWake, GuestWake };

struct SyncTraceEvent {
    std::atomic<uint64_t> published{0};
    uint64_t sequence = 0;
    SyncTraceKind kind = SyncTraceKind::WaitEnter;
    uint32_t windows_tid = 0;
    uint64_t pthread_id = 0;
    uintptr_t object = 0;
    uintptr_t source = 0;
    uint32_t value = 0;   // the slot's sequence counter as this event observed it
};

// Sized by the caller, because the right size is a property of the run being diagnosed rather than
// something this file can guess: Blue Prince produces ~200k events in the 100 s before it deadlocks,
// so a fixed small ring retains only the last instants and drops exactly the objects of interest --
// the ones whose last activity is OLDEST. PROSPER_SYNC_RING=<count> sets it; PROSPER_SYNC_RING=1
// (or any non-numeric value) takes the default below. 262144 events is about 14 MiB.
constexpr size_t kSyncTraceDefaultEvents = 65536;

size_t sync_trace_capacity_from_env() {
    const char* value = std::getenv("PROSPER_SYNC_RING");
    if (!value || !*value) return 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 0);
    if (end == value || parsed < 2) return kSyncTraceDefaultEvents;   // "1", "yes", ... => default
    return (size_t)parsed;
}

// Read once at startup: getenv is neither cheap nor safe to call from these paths.
const size_t g_sync_trace_capacity = sync_trace_capacity_from_env();
// new[] rather than calloc: SyncTraceEvent holds std::atomic members, which must be constructed.
SyncTraceEvent* const g_sync_trace =
    g_sync_trace_capacity ? new SyncTraceEvent[g_sync_trace_capacity] : nullptr;
std::atomic<uint64_t> g_sync_trace_sequence{0};
const bool g_sync_trace_enabled = g_sync_trace != nullptr;

void sync_trace(SyncTraceKind kind, uintptr_t object, uintptr_t source, uint32_t value) {
    if (!g_sync_trace_enabled) return;
    const uint64_t sequence = g_sync_trace_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    SyncTraceEvent& event = g_sync_trace[(sequence - 1) % g_sync_trace_capacity];
    // Publish the generation LAST, so a reader that races a writer discards a torn entry rather
    // than reporting half of one. Same contract as the exception ring in hle_kernel.cpp.
    event.published.store(0, std::memory_order_relaxed);
    event.sequence = sequence;
    event.kind = kind;
    event.windows_tid = GetCurrentThreadId();
    event.pthread_id = (uint64_t)pthread_self();
    event.object = object;
    event.source = source;
    event.value = value;
    event.published.store(sequence, std::memory_order_release);
}

const char* sync_trace_kind_name(SyncTraceKind kind) {
    switch (kind) {
        case SyncTraceKind::WaitEnter: return "wait-enter";
        case SyncTraceKind::WaitWake:  return "wait-wake";
        case SyncTraceKind::Signal:    return "signal";
        case SyncTraceKind::Broadcast: return "broadcast";
        case SyncTraceKind::Interrupt: return "interrupt";
        case SyncTraceKind::FutexWait: return "futex-wait";
        case SyncTraceKind::FutexWake: return "futex-wake";
        case SyncTraceKind::GuestWake: return "guest-wake";
    }
    return "?";
}
std::array<CondSlot, 4096> g_cond_slots;
constexpr uintptr_t kCondSlotRetiring = UINTPTR_MAX;
std::atomic<uint64_t> g_cond_wait_failure{0};

int consume_cond_wait_failure(CondWaitFailurePointForTest point) {
    uint64_t armed = g_cond_wait_failure.load(std::memory_order_acquire);
    if ((uint32_t)(armed >> 32) != (uint32_t)point) return 0; // dormant path: no locked RMW
    if (!g_cond_wait_failure.compare_exchange_strong(
            armed, 0, std::memory_order_acq_rel)) return 0;
    return (int)(uint32_t)armed;
}

// One condvar must resolve to ONE slot, for the life of that condvar.
//
// #2139: this used to search for an existing owner and claim a free slot in the SAME pass, so a free
// slot encountered before this cond's own slot was taken instead of it. That splits a waiter and its
// signaller across two slots, and the signal is then delivered to a sequence nobody is waiting on --
// a permanent, silent lost wakeup. Blue Prince deadlocked exactly this way:
//
//   guest cond 0x…910 claims slot ...ad8 (…ac8 was busy); a thread parks on ...ad8;
//   the cond that owned ...ac8 is destroyed, freeing an EARLIER slot;
//   the next lookup for 0x…910 claims ...ac8 and signals it. The waiter on ...ad8 never wakes,
//   the signaller then waits for that thread's reply, and all 70 guest threads pile up behind it.
//
// It needs slot churn plus a destroy in the window between a waiter's lookup and its signaller's, so
// it only bites deep into a long run -- which is why it read as timing sensitivity.
//
// The fix is that an existing owner ANYWHERE in the table beats any free slot: prove the key absent
// before claiming. Kept lock-free deliberately -- exception delivery can suspend a thread at any
// instruction, so a mutex here can be held by a thread that is stopped (see g_wait_slots above).
CondSlot* cond_slot_for(pthread_cond_t* cond) {
    const uintptr_t key = (uintptr_t)cond;
    if (!key) return nullptr;
    auto find_owner = [key]() -> CondSlot* {
        for (CondSlot& slot : g_cond_slots)
            if (slot.cond.load(std::memory_order_acquire) == key) return &slot;
        return nullptr;
    };
    // Bounded: each iteration either returns or releases a slot that a lower-indexed claim won, and
    // only a concurrent first-use of the SAME condvar can cause one. A spin cap keeps a pathological
    // interleaving from becoming an unbounded loop inside a guest sync primitive.
    for (int attempt = 0; attempt < 64; ++attempt) {
        if (CondSlot* owner = find_owner()) return owner;

        CondSlot* claimed = nullptr;
        for (CondSlot& slot : g_cond_slots) {
            uintptr_t expected = 0;
            if (slot.cond.compare_exchange_strong(expected, key, std::memory_order_acq_rel)) {
                claimed = &slot;
                break;
            }
        }
        if (!claimed) return nullptr;   // table full

        // Two threads can reach the claim above for the same new condvar at once and take two
        // different slots. Resolve it the same way on both sides -- lowest index wins -- so they
        // converge on one slot instead of each keeping its own.
        CondSlot* winner = find_owner();
        if (winner == claimed) return claimed;
        claimed->cond.store(0, std::memory_order_release);
    }
    return find_owner();
}

WaitSlot* register_wait(GuestWaitKind kind, uintptr_t object, uintptr_t source = 0) {
    const uint64_t windows_tid = GetCurrentThreadId();
    for (WaitSlot& slot : g_wait_slots) {
        uint64_t publication = 0;
        if (!slot.publication.compare_exchange_strong(
                publication, kWaitSlotPublishing, std::memory_order_acq_rel))
            continue;

        // Publish one unique generation last. Each nested wait owns a distinct slot: the GC callback itself
        // waits on semaphores and must not erase the interrupted outer wait when it returns.
        slot.pthread_id.store((uint64_t)pthread_self(), std::memory_order_relaxed);
        slot.windows_tid.store(windows_tid, std::memory_order_relaxed);
        slot.object.store(object, std::memory_order_relaxed);
        slot.source.store(source, std::memory_order_relaxed);
        slot.kind.store(kind, std::memory_order_relaxed);
        uint64_t token;
        do {
            token = ++slot.next_publication;
        } while (token == 0 || token == kWaitSlotPublishing);
        slot.publication.store(token, std::memory_order_release);
        return &slot;
    }
    return nullptr;
}

void unregister_wait(WaitSlot* slot) {
    if (!slot) return;
    uint64_t publication = slot->publication.load(std::memory_order_acquire);
    if (publication == 0 || publication == kWaitSlotPublishing ||
        !slot->publication.compare_exchange_strong(
            publication, kWaitSlotPublishing, std::memory_order_acq_rel))
        return;
    slot->pthread_id.store(0, std::memory_order_relaxed);
    slot->windows_tid.store(0, std::memory_order_relaxed);
    slot->object.store(0, std::memory_order_relaxed);
    slot->source.store(0, std::memory_order_relaxed);
    slot->kind.store(GuestWaitKind::None, std::memory_order_relaxed);
    slot->publication.store(0, std::memory_order_release);
}

struct WaitSlotSnapshot {
    uint64_t pthread_id = 0;
    uint64_t windows_tid = 0;
    GuestWaitSnapshot wait{};
};

bool snapshot_wait_slot(const WaitSlot& slot, WaitSlotSnapshot& snapshot) {
    const uint64_t publication = slot.publication.load(std::memory_order_acquire);
    if (publication == 0 || publication == kWaitSlotPublishing) return false;
    const uint64_t pthread_id = slot.pthread_id.load(std::memory_order_relaxed);
    const uint64_t windows_tid = slot.windows_tid.load(std::memory_order_relaxed);
    const GuestWaitKind kind = slot.kind.load(std::memory_order_relaxed);
    const uintptr_t object = slot.object.load(std::memory_order_relaxed);
    const uintptr_t source = slot.source.load(std::memory_order_relaxed);
    if (slot.publication.load(std::memory_order_acquire) != publication)
        return false;
    snapshot = {pthread_id, windows_tid, {kind, object, source}};
    return true;
}
#endif
}

#ifdef _WIN32
void win_set_cond_wait_failure_for_test(CondWaitFailurePointForTest point, int error) {
    const uint64_t armed = ((uint64_t)(uint32_t)point << 32) | (uint32_t)error;
    g_cond_wait_failure.store(armed, std::memory_order_release);
}
#endif

WaitRegistration futex_wait_enter(uint64_t addr) {
#ifdef _WIN32
    WaitSlot* registration = register_wait(GuestWaitKind::Address, (uintptr_t)addr,
                                           (uintptr_t)addr);
    // Recorded so a raw futex waiter is not indistinguishable from one whose producer never
    // existed: without this every `address` wait looks like a wait-for-graph root by construction.
    sync_trace(SyncTraceKind::FutexWait, (uintptr_t)addr, (uintptr_t)addr, 0);
    // Close queue-before-sleep: a GC stop can be published just before this wait is registered.
    // Accept it now rather than blocking forever after the raiser's wake lookup already missed us.
    dispatch_pending_guest_exception();
#else
    (void)addr;
#endif
    g_waiters.fetch_add(1, std::memory_order_seq_cst);
#ifdef _WIN32
    return registration;
#else
    return nullptr;
#endif
}
void futex_wait_exit(WaitRegistration registration) {
#ifdef _WIN32
    dispatch_pending_guest_exception();
#endif
    g_waiters.fetch_sub(1, std::memory_order_seq_cst);
#ifdef _WIN32
    unregister_wait(static_cast<WaitSlot*>(registration));
#else
    (void)registration;
#endif
}

int interruptible_cond_wait(pthread_cond_t* cond, pthread_mutex_t* mutex,
                            GuestWaitKind kind, uintptr_t source,
                            CondWaitMutexState* mutex_state,
                            const CondWaitMutexBookkeeping* bookkeeping) {
    if (mutex_state) *mutex_state = {};
#ifdef _WIN32
    if (const int injected = consume_cond_wait_failure(
            CondWaitFailurePointForTest::BeforeUnlock)) return injected;
    CondSlot* slot = cond_slot_for(cond);
    if (!slot) return ENOMEM;
    uint32_t expected = slot->sequence.load(std::memory_order_acquire);
    WaitSlot* registration = register_wait(kind, (uintptr_t)&slot->sequence,
                                           source ? source : (uintptr_t)cond);
    const bool bookkeeping_released = bookkeeping && bookkeeping->release &&
                                      bookkeeping->release(mutex);
    const int unlock_result = pthread_mutex_unlock(mutex);
    if (unlock_result != 0) {
        if (bookkeeping_released && bookkeeping->acquire) bookkeeping->acquire(mutex);
        unregister_wait(registration);
        return unlock_result;
    }
    if (mutex_state) mutex_state->unlocked = true;
    dispatch_pending_guest_exception();
    sync_trace(SyncTraceKind::WaitEnter, (uintptr_t)&slot->sequence,
               source ? source : (uintptr_t)cond, expected);
    WaitOnAddress((volatile void*)&slot->sequence, &expected, sizeof(expected), INFINITE);
    sync_trace(SyncTraceKind::WaitWake, (uintptr_t)&slot->sequence,
               source ? source : (uintptr_t)cond,
               slot->sequence.load(std::memory_order_acquire));
    dispatch_pending_guest_exception();
    if (const int injected = consume_cond_wait_failure(
            CondWaitFailurePointForTest::BeforeRelock)) {
        unregister_wait(registration);
        return injected;
    }
    const int lock_result = interruptible_mutex_lock(mutex);
    if (lock_result == 0) {
        if (mutex_state) mutex_state->reacquired = true;
        if (bookkeeping_released && bookkeeping->acquire) bookkeeping->acquire(mutex);
    }
    unregister_wait(registration);
    return lock_result;
#else
    (void)kind;
    (void)source;
    const bool bookkeeping_released = bookkeeping && bookkeeping->release &&
                                      bookkeeping->release(mutex);
    const int result = pthread_cond_wait(cond, mutex);
    if (bookkeeping_released && bookkeeping->acquire) bookkeeping->acquire(mutex);
    if (result == 0 && mutex_state) *mutex_state = {true, true};
    return result;
#endif
}

int interruptible_cond_timedwait(pthread_cond_t* cond, pthread_mutex_t* mutex,
                                 const timespec* deadline, GuestWaitKind kind,
                                 uintptr_t source, CondWaitMutexState* mutex_state,
                                 const CondWaitMutexBookkeeping* bookkeeping) {
    if (mutex_state) *mutex_state = {};
#ifdef _WIN32
    if (const int injected = consume_cond_wait_failure(
            CondWaitFailurePointForTest::BeforeUnlock)) return injected;
    CondSlot* slot = cond_slot_for(cond);
    if (!slot) return ENOMEM;
    uint32_t expected = slot->sequence.load(std::memory_order_acquire);
    WaitSlot* registration = register_wait(kind, (uintptr_t)&slot->sequence,
                                           source ? source : (uintptr_t)cond);
    const bool bookkeeping_released = bookkeeping && bookkeeping->release &&
                                      bookkeeping->release(mutex);
    const int unlock_result = pthread_mutex_unlock(mutex);
    if (unlock_result != 0) {
        if (bookkeeping_released && bookkeeping->acquire) bookkeeping->acquire(mutex);
        unregister_wait(registration);
        return unlock_result;
    }
    if (mutex_state) mutex_state->unlocked = true;
    dispatch_pending_guest_exception();

    timespec now{};
    clock_gettime(CLOCK_REALTIME, &now);
    int64_t sec = (int64_t)deadline->tv_sec - (int64_t)now.tv_sec;
    int64_t nsec = (int64_t)deadline->tv_nsec - (int64_t)now.tv_nsec;
    if (nsec < 0) { nsec += 1000000000; sec--; }
    DWORD timeout = 0;
    if (sec >= 0) {
        uint64_t ms = (uint64_t)sec * 1000 + ((uint64_t)nsec + 999999) / 1000000;
        timeout = ms >= (uint64_t)INFINITE ? INFINITE - 1 : (DWORD)ms;
    }
    const bool signaled = WaitOnAddress((volatile void*)&slot->sequence, &expected,
                                        sizeof(expected), timeout) != FALSE;
    dispatch_pending_guest_exception();
    if (const int injected = consume_cond_wait_failure(
            CondWaitFailurePointForTest::BeforeRelock)) {
        unregister_wait(registration);
        return injected;
    }
    const int lock_result = interruptible_mutex_lock(mutex);
    if (lock_result == 0) {
        if (mutex_state) mutex_state->reacquired = true;
        if (bookkeeping_released && bookkeeping->acquire) bookkeeping->acquire(mutex);
    }
    unregister_wait(registration);
    if (lock_result != 0) return lock_result;
    return signaled ? 0 : ETIMEDOUT;
#else
    (void)kind;
    (void)source;
    const bool bookkeeping_released = bookkeeping && bookkeeping->release &&
                                      bookkeeping->release(mutex);
    const int result = pthread_cond_timedwait(cond, mutex, deadline);
    if (bookkeeping_released && bookkeeping->acquire) bookkeeping->acquire(mutex);
    if ((result == 0 || result == ETIMEDOUT) && mutex_state) *mutex_state = {true, true};
    return result;
#endif
}

int interruptible_cond_timedwait_relative(pthread_cond_t* cond, pthread_mutex_t* mutex,
                                          const timespec* duration, GuestWaitKind kind,
                                          uintptr_t source, CondWaitMutexState* mutex_state,
                                          const CondWaitMutexBookkeeping* bookkeeping) {
    if (mutex_state) *mutex_state = {};
    if (!duration || duration->tv_sec < 0 || duration->tv_nsec < 0 ||
        duration->tv_nsec >= 1000000000L) return EINVAL;
#ifdef _WIN32
    if (const int injected = consume_cond_wait_failure(
            CondWaitFailurePointForTest::BeforeUnlock)) return injected;
    CondSlot* slot = cond_slot_for(cond);
    if (!slot) return ENOMEM;
    uint32_t expected = slot->sequence.load(std::memory_order_acquire);
    WaitSlot* registration = register_wait(kind, (uintptr_t)&slot->sequence,
                                           source ? source : (uintptr_t)cond);
    const bool bookkeeping_released = bookkeeping && bookkeeping->release &&
                                      bookkeeping->release(mutex);
    const int unlock_result = pthread_mutex_unlock(mutex);
    if (unlock_result != 0) {
        if (bookkeeping_released && bookkeeping->acquire) bookkeeping->acquire(mutex);
        unregister_wait(registration);
        return unlock_result;
    }
    if (mutex_state) mutex_state->unlocked = true;
    dispatch_pending_guest_exception();

    const uint64_t milliseconds = (uint64_t)duration->tv_sec * 1000ull +
                                  ((uint64_t)duration->tv_nsec + 999999ull) / 1000000ull;
    const DWORD timeout = milliseconds >= (uint64_t)INFINITE ? INFINITE - 1 : (DWORD)milliseconds;
    const bool signaled = WaitOnAddress((volatile void*)&slot->sequence, &expected,
                                        sizeof(expected), timeout) != FALSE;
    dispatch_pending_guest_exception();
    if (const int injected = consume_cond_wait_failure(
            CondWaitFailurePointForTest::BeforeRelock)) {
        unregister_wait(registration);
        return injected;
    }
    const int lock_result = interruptible_mutex_lock(mutex);
    if (lock_result == 0) {
        if (mutex_state) mutex_state->reacquired = true;
        if (bookkeeping_released && bookkeeping->acquire) bookkeeping->acquire(mutex);
    }
    unregister_wait(registration);
    if (lock_result != 0) return lock_result;
    return signaled ? 0 : ETIMEDOUT;
#elif defined(__APPLE__)
    (void)kind;
    (void)source;
    const bool bookkeeping_released = bookkeeping && bookkeeping->release &&
                                      bookkeeping->release(mutex);
    const int result = pthread_cond_timedwait_relative_np(cond, mutex, duration);
    if (bookkeeping_released && bookkeeping->acquire) bookkeeping->acquire(mutex);
    if ((result == 0 || result == ETIMEDOUT) && mutex_state) *mutex_state = {true, true};
    return result;
#else
    (void)kind;
    (void)source;
    timespec deadline{};
    if (clock_gettime(CLOCK_MONOTONIC, &deadline) != 0) return errno ? errno : EINVAL;
    deadline.tv_sec += duration->tv_sec;
    deadline.tv_nsec += duration->tv_nsec;
    if (deadline.tv_nsec >= 1000000000L) {
        ++deadline.tv_sec;
        deadline.tv_nsec -= 1000000000L;
    }
    const bool bookkeeping_released = bookkeeping && bookkeeping->release &&
                                      bookkeeping->release(mutex);
    const int result = pthread_cond_clockwait(cond, mutex, CLOCK_MONOTONIC, &deadline);
    if (bookkeeping_released && bookkeeping->acquire) bookkeeping->acquire(mutex);
    if ((result == 0 || result == ETIMEDOUT) && mutex_state) *mutex_state = {true, true};
    return result;
#endif
}

int interruptible_cond_signal(pthread_cond_t* cond) {
#ifdef _WIN32
    CondSlot* slot = cond_slot_for(cond);
    if (!slot) return ENOMEM;
    const uint32_t value = slot->sequence.fetch_add(1, std::memory_order_release) + 1;
    sync_trace(SyncTraceKind::Signal, (uintptr_t)&slot->sequence, (uintptr_t)cond, value);
    WakeByAddressSingle((PVOID)&slot->sequence);
    return 0;
#else
    return pthread_cond_signal(cond);
#endif
}

int interruptible_cond_broadcast(pthread_cond_t* cond) {
#ifdef _WIN32
    CondSlot* slot = cond_slot_for(cond);
    if (!slot) return ENOMEM;
    const uint32_t value = slot->sequence.fetch_add(1, std::memory_order_release) + 1;
    sync_trace(SyncTraceKind::Broadcast, (uintptr_t)&slot->sequence, (uintptr_t)cond, value);
    WakeByAddressAll((PVOID)&slot->sequence);
    return 0;
#else
    return pthread_cond_broadcast(cond);
#endif
}

void interruptible_cond_forget(pthread_cond_t* cond) {
#ifdef _WIN32
    const uintptr_t key = (uintptr_t)cond;
    if (!key) return;
    for (CondSlot& slot : g_cond_slots) {
        uintptr_t owner = key;
        // Retire through a sentinel so a concurrent allocator cannot claim the slot between
        // clearing its owner and resetting the generation. Destroying a condvar with live users is
        // already invalid; this only closes reuse races between successive valid lifetimes.
        if (!slot.cond.compare_exchange_strong(owner, kCondSlotRetiring,
                                               std::memory_order_acq_rel))
            continue;
        slot.sequence.store(0, std::memory_order_relaxed);
        slot.cond.store(0, std::memory_order_release);
        return;
    }
#else
    (void)cond;
#endif
}

int interruptible_mutex_lock(pthread_mutex_t* mutex) {
#ifdef _WIN32
    // Winpthreads' timed mutex wait has remained inside WaitForSingleObject after its absolute
    // deadline was already seconds in the past. Polling trylock keeps the wait cooperative: a GC
    // stop request is acknowledged even when the mutex owner is itself waiting for that thread.
    for (;;) {
        const int result = pthread_mutex_trylock(mutex);
        if (result != EBUSY) return result;
        dispatch_pending_guest_exception();
        Sleep(1);
    }
#else
    return pthread_mutex_lock(mutex);
#endif
}

bool interrupt_guest_wait(uint64_t thread) {
#ifdef _WIN32
    bool interrupted = false;
    for (WaitSlot& slot : g_wait_slots) {
        WaitSlotSnapshot snapshot{};
        if (!snapshot_wait_slot(slot, snapshot) || snapshot.pthread_id != thread) continue;
        const GuestWaitKind kind = snapshot.wait.kind;
        const uintptr_t object = snapshot.wait.object;
        if (kind == GuestWaitKind::Address && object) {
            WakeByAddressAll((PVOID)object);
            interrupted = true;
        }
        if ((kind == GuestWaitKind::ConditionSequence || kind == GuestWaitKind::EventFlag ||
             kind == GuestWaitKind::Semaphore) && object) {
            auto* sequence = reinterpret_cast<std::atomic<uint32_t>*>(object);
            const uint32_t value = sequence->fetch_add(1, std::memory_order_release) + 1;
            sync_trace(SyncTraceKind::Interrupt, object, snapshot.wait.source, value);
            WakeByAddressAll((PVOID)object);
            interrupted = true;
        }
    }
    return interrupted;
#else
    (void)thread;
#endif
    return false;
}

size_t snapshot_guest_waits(uint64_t windows_tid, GuestWaitSnapshot* snapshots, size_t capacity) {
    size_t count = 0;
#ifdef _WIN32
    for (const WaitSlot& slot : g_wait_slots) {
        WaitSlotSnapshot slot_snapshot{};
        if (!snapshot_wait_slot(slot, slot_snapshot) || slot_snapshot.windows_tid != windows_tid)
            continue;
        if (count < capacity && snapshots) snapshots[count] = slot_snapshot.wait;
        ++count;
    }
#else
    (void)windows_tid;
    (void)snapshots;
    (void)capacity;
#endif
    return count;
}

bool snapshot_guest_wait(uint64_t windows_tid, GuestWaitSnapshot& snapshot) {
    snapshot = {};
    GuestWaitSnapshot only{};
    if (snapshot_guest_waits(windows_tid, &only, 1) != 1) return false;
    snapshot = only;
    return true;
}

void snapshot_guest_wait_registry(size_t& condition_slots_used,
                                  size_t& condition_slots_capacity) {
    condition_slots_used = 0;
    condition_slots_capacity = 0;
#ifdef _WIN32
    condition_slots_capacity = g_cond_slots.size();
    for (const CondSlot& slot : g_cond_slots)
        condition_slots_used += slot.cond.load(std::memory_order_acquire) != 0;
#endif
}

void futex_wake(uint64_t addr, int n) {
#if defined(__linux__) || defined(__APPLE__)
    if (!addr) return;
    prosper_futex_wake((uint32_t*)(uintptr_t)addr, n > 1);
#elif defined(_WIN32)
    // Native Win32 futex wake, pairing with sceKernelWaitOnAddress's WaitOnAddress (hle_kernel_mem.cpp).
    // The GPU command processor calls this from RELEASE_MEM/EOP to wake a guest render/producer thread
    // parked on the label the GPU just wrote — without it the guest's WaitOnAddress never wakes on
    // Windows and rendering wedges after the first submit. Wake ALL (n is a hint; label waiters are few).
    if (!addr) return;
    (void)n;
    sync_trace(SyncTraceKind::FutexWake, (uintptr_t)addr, (uintptr_t)addr, 0);
    WakeByAddressAll((PVOID)(uintptr_t)addr);
#else
    (void)addr; (void)n;
#endif
}

void wake_label_waiters(uint64_t addr) {
    if (!addr) return;
    // The label was just stored with a PLAIN write; x86 permits store->load reordering, so without a
    // full fence this counter load could execute before that store is visible and a registering waiter
    // could then block on the stale value with nobody left to wake. Fence, THEN check.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (g_waiters.load(std::memory_order_seq_cst) == 0) return;
    futex_wake(addr, INT_MAX);
    futex_wake(addr + 4, INT_MAX);   // 64-bit labels: a waiter may block on the high dword too
}

void sync_ring_note_guest_wake(uintptr_t address, uint64_t count) {
#ifdef _WIN32
    sync_trace(SyncTraceKind::GuestWake, address, address, (uint32_t)count);
#else
    (void)address; (void)count;
#endif
}

// PROSPER_SYNC_RING reader. Prints the retained events oldest-first. Safe to call while the guest
// is deadlocked -- it only reads published entries and never takes a lock, which is the whole point:
// the situation it describes is one where no lock can be acquired because nothing is running.
void dump_guest_sync_trace(const char* path) {
#ifdef _WIN32
    FILE* out = stderr;
    FILE* opened = nullptr;
    if (path && *path) { opened = std::fopen(path, "a"); if (opened) out = opened; }
    const uint64_t total = g_sync_trace_sequence.load(std::memory_order_acquire);
    if (!g_sync_trace_enabled) {
        std::fprintf(out, "[sync-trace] disabled (set PROSPER_SYNC_RING=1 to retain events)\n");
        if (opened) std::fclose(opened);
        return;
    }
    const uint64_t capacity = (uint64_t)g_sync_trace_capacity;
    const uint64_t first = total > capacity ? total - capacity + 1 : 1;
    std::fprintf(out, "[sync-trace] %llu events total, showing %llu..%llu\n",
                 (unsigned long long)total, (unsigned long long)first,
                 (unsigned long long)total);
    for (uint64_t sequence = first; sequence <= total; ++sequence) {
        const SyncTraceEvent& event = g_sync_trace[(sequence - 1) % g_sync_trace_capacity];
        // Re-read the generation around the payload: a writer that lapped us mid-read invalidates
        // this entry, and a torn record is worse than an omitted one.
        const uint64_t before = event.published.load(std::memory_order_acquire);
        if (before != sequence) continue;
        const SyncTraceKind kind = event.kind;
        const uint32_t windows_tid = event.windows_tid;
        const uint64_t pthread_id = event.pthread_id;
        const uintptr_t object = event.object;
        const uintptr_t source = event.source;
        const uint32_t value = event.value;
        if (event.published.load(std::memory_order_acquire) != sequence) continue;
        std::fprintf(out,
                     "[sync-trace] seq=%llu kind=%-10s tid=%lu pthread=0x%llx object=0x%llx "
                     "source=0x%llx value=%lu\n",
                     (unsigned long long)sequence, sync_trace_kind_name(kind),
                     (unsigned long)windows_tid, (unsigned long long)pthread_id,
                     (unsigned long long)object, (unsigned long long)source,
                     (unsigned long)value);
    }
    if (opened) std::fclose(opened);
#else
    (void)path;
#endif
}

} // namespace prosper
