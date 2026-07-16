// sync_futex.cpp — see sync_futex.hpp.
#include "sync_futex.hpp"
#include "dispatch.hpp"
#include <atomic>
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
std::array<CondSlot, 4096> g_cond_slots;

CondSlot* cond_slot_for(pthread_cond_t* cond) {
    const uintptr_t key = (uintptr_t)cond;
    for (CondSlot& slot : g_cond_slots) {
        uintptr_t owner = slot.cond.load(std::memory_order_acquire);
        if (owner == key || (owner == 0 && slot.cond.compare_exchange_strong(
                owner, key, std::memory_order_acq_rel)))
            return &slot;
    }
    return nullptr;
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

WaitRegistration futex_wait_enter(uint64_t addr) {
#ifdef _WIN32
    WaitSlot* registration = register_wait(GuestWaitKind::Address, (uintptr_t)addr,
                                           (uintptr_t)addr);
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
                            GuestWaitKind kind, uintptr_t source) {
#ifdef _WIN32
    CondSlot* slot = cond_slot_for(cond);
    if (!slot) return ENOMEM;
    uint32_t expected = slot->sequence.load(std::memory_order_acquire);
    WaitSlot* registration = register_wait(kind, (uintptr_t)&slot->sequence,
                                           source ? source : (uintptr_t)cond);
    const int unlock_result = pthread_mutex_unlock(mutex);
    if (unlock_result != 0) {
        unregister_wait(registration);
        return unlock_result;
    }
    dispatch_pending_guest_exception();
    WaitOnAddress((volatile void*)&slot->sequence, &expected, sizeof(expected), INFINITE);
    dispatch_pending_guest_exception();
    unregister_wait(registration);
    return interruptible_mutex_lock(mutex);
#else
    (void)kind;
    (void)source;
    return pthread_cond_wait(cond, mutex);
#endif
}

int interruptible_cond_timedwait(pthread_cond_t* cond, pthread_mutex_t* mutex,
                                 const timespec* deadline, GuestWaitKind kind,
                                 uintptr_t source) {
#ifdef _WIN32
    CondSlot* slot = cond_slot_for(cond);
    if (!slot) return ENOMEM;
    uint32_t expected = slot->sequence.load(std::memory_order_acquire);
    WaitSlot* registration = register_wait(kind, (uintptr_t)&slot->sequence,
                                           source ? source : (uintptr_t)cond);
    const int unlock_result = pthread_mutex_unlock(mutex);
    if (unlock_result != 0) {
        unregister_wait(registration);
        return unlock_result;
    }
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
    unregister_wait(registration);
    const int lock_result = interruptible_mutex_lock(mutex);
    if (lock_result != 0) return lock_result;
    return signaled ? 0 : ETIMEDOUT;
#else
    (void)kind;
    (void)source;
    return pthread_cond_timedwait(cond, mutex, deadline);
#endif
}

int interruptible_cond_signal(pthread_cond_t* cond) {
#ifdef _WIN32
    CondSlot* slot = cond_slot_for(cond);
    if (!slot) return ENOMEM;
    slot->sequence.fetch_add(1, std::memory_order_release);
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
    slot->sequence.fetch_add(1, std::memory_order_release);
    WakeByAddressAll((PVOID)&slot->sequence);
    return 0;
#else
    return pthread_cond_broadcast(cond);
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
            sequence->fetch_add(1, std::memory_order_release);
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

bool snapshot_guest_wait(uint64_t windows_tid, GuestWaitSnapshot& snapshot) {
    snapshot = {};
#ifdef _WIN32
    for (const WaitSlot& slot : g_wait_slots) {
        WaitSlotSnapshot slot_snapshot{};
        if (!snapshot_wait_slot(slot, slot_snapshot) || slot_snapshot.windows_tid != windows_tid)
            continue;
        snapshot = slot_snapshot.wait;
        return true;
    }
#else
    (void)windows_tid;
#endif
    return false;
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

} // namespace prosper
