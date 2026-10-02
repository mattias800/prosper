// guest_thread_handle.cpp — Windows implementation of the guest-visible thread handle.
// See guest_thread_handle.hpp for why this exists and the identity/validation contract.
#include "hle/kernel/guest_thread_handle.hpp"

#if defined(_WIN32)

#include "host/platform/immortal.hpp"

#include <windows.h>

#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace prosper::hle {
namespace {

// The object a guest dereferences. `tid` MUST stay the first member: the title reads the first
// dword of the handle. The rest is host-private bookkeeping the guest never touches.
struct ThreadObject {
    uint32_t tid = 0;
    uint32_t reserved = 0;      // keeps `host` 8-aligned and gives the guest a defined zero dword
    pthread_t host = 0;
    bool live = false;          // in the tables; false while sitting on the free list
    bool adopted = false;       // made by scePthreadSelf, not scePthreadCreate
    bool finished = false;      // the thread has run its exit path
    bool released = false;      // joined or detached: the creator no longer owns the handle
};
static_assert(offsetof(ThreadObject, tid) == 0, "the guest reads the first dword as the thread id");

constexpr size_t kSlabObjects = 256;

struct State {
    std::mutex mutex;
    std::unordered_map<uintptr_t, ThreadObject*> by_host;          // host pthread_t -> object
    std::unordered_map<uint64_t, ThreadObject*> by_handle;         // object address -> object
    std::vector<std::unique_ptr<ThreadObject[]>> slabs;            // never freed: stale reads stay mapped
    std::vector<ThreadObject*> free_list;
    uint64_t live_count = 0;
};

// Immortal: a detached guest thread can reach this after the process begins exiting (#2613).
Immortal<State> g_state;

ThreadObject* allocate_locked(State& s) {
    if (s.free_list.empty()) {
        try {
            s.slabs.emplace_back(new ThreadObject[kSlabObjects]);
        } catch (...) {
            return nullptr;
        }
        ThreadObject* slab = s.slabs.back().get();
        // Push in reverse so the lowest address is handed out first (stable, easy to read in a log).
        for (size_t i = kSlabObjects; i-- > 0;) s.free_list.push_back(&slab[i]);
    }
    ThreadObject* o = s.free_list.back();
    s.free_list.pop_back();
    *o = ThreadObject{};
    o->live = true;
    ++s.live_count;
    return o;
}

void release_locked(State& s, ThreadObject* o) {
    s.by_host.erase((uintptr_t)o->host);
    s.by_handle.erase((uint64_t)(uintptr_t)o);
    o->tid = 0;                // a stale guest read sees zero, not a recycled thread's id
    o->live = false;
    s.free_list.push_back(o);
    --s.live_count;
}

uint32_t tid_of(pthread_t host) {
    const HANDLE h = (HANDLE)pthread_gethandle(host);
    if (!h || h == INVALID_HANDLE_VALUE) return 0;
    return (uint32_t)GetThreadId(h);
}

// Get the object for `host`, creating it when absent. `tid` is the Windows thread id when known
// (0 = ask winpthreads). An object whose recorded tid disagrees belongs to a dead thread whose
// pthread_t index was recycled, so it is retired and replaced rather than handed to the new thread.
ThreadObject* get_or_create_locked(State& s, pthread_t host, uint32_t tid, bool adopted) {
    if (tid == 0) tid = tid_of(host);
    auto it = s.by_host.find((uintptr_t)host);
    if (it != s.by_host.end()) {
        ThreadObject* existing = it->second;
        if (tid == 0 || existing->tid == 0 || existing->tid == tid) {
            if (existing->tid == 0) existing->tid = tid;
            return existing;
        }
        release_locked(s, existing);
    }
    ThreadObject* o = allocate_locked(s);
    if (!o) return nullptr;
    o->tid = tid;
    o->host = host;
    o->adopted = adopted;
    s.by_host.emplace((uintptr_t)host, o);
    s.by_handle.emplace((uint64_t)(uintptr_t)o, o);
    return o;
}

}   // namespace

uint64_t guest_thread_handle_create(pthread_t host) {
    State& s = *g_state;
    std::lock_guard<std::mutex> lock(s.mutex);
    ThreadObject* o = get_or_create_locked(s, host, 0, /*adopted=*/false);
    if (!o) return 0;
    o->adopted = false;   // the child may have adopted it first via scePthreadSelf; creation wins
    return (uint64_t)(uintptr_t)o;
}

uint64_t guest_thread_handle_self() {
    State& s = *g_state;
    std::lock_guard<std::mutex> lock(s.mutex);
    ThreadObject* o = get_or_create_locked(s, pthread_self(), (uint32_t)GetCurrentThreadId(),
                                           /*adopted=*/true);
    return o ? (uint64_t)(uintptr_t)o : 0;
}

bool guest_thread_handle_resolve(uint64_t handle, pthread_t* host, bool* adopted) {
    if (!handle || !host) return false;
    State& s = *g_state;
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto it = s.by_handle.find(handle);
    if (it == s.by_handle.end()) return false;
    *host = it->second->host;
    if (adopted) *adopted = it->second->adopted;
    return true;
}

void guest_thread_handle_exited(pthread_t host) {
    State& s = *g_state;
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto it = s.by_host.find((uintptr_t)host);
    if (it == s.by_host.end()) return;
    ThreadObject* o = it->second;
    o->finished = true;
    // A joinable thread keeps its handle until joined (the guest may join a finished thread); a
    // detached one, and an adopted one nobody can join, has no further use for it.
    if (o->released || o->adopted) release_locked(s, o);
}

void guest_thread_handle_joined(uint64_t handle) {
    State& s = *g_state;
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto it = s.by_handle.find(handle);
    if (it != s.by_handle.end()) release_locked(s, it->second);
}

void guest_thread_handle_detached(uint64_t handle) {
    State& s = *g_state;
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto it = s.by_handle.find(handle);
    if (it == s.by_handle.end()) return;
    ThreadObject* o = it->second;
    o->released = true;
    if (o->finished) release_locked(s, o);
}

uint64_t guest_thread_handle_live_count() {
    State& s = *g_state;
    std::lock_guard<std::mutex> lock(s.mutex);
    return s.live_count;
}

}   // namespace prosper::hle

#endif   // _WIN32
