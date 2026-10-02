// guest_thread_handle.hpp — the value a guest sees as a `ScePthread` / `pthread_t`.
//
// WHY THIS EXISTS. The guest treats the handle scePthreadCreate writes as a POINTER to a thread
// object and reads the first dword (the thread id) straight through it: Assassin's Creed Black Flag
// Resynced does `mov rax,[handle]; mov ecx,[rax]` at eboot+0x161ed91 right after creating its
// "SystemLogger" thread. On Linux/macOS the host `pthread_t` is a pointer to readable memory, so
// handing it over raw happens to work. On MinGW's winpthreads `pthread_t` is a `uintptr_t` INDEX
// (pthread.h: `typedef uintptr_t pthread_t`), so the guest received `3`, dereferenced address 3 and
// the main guest thread died 1.6 s into the boot. The window then sat black for 72 minutes.
//
// THE CONTRACT. Translate ONLY at the guest edge. Every HLE body that hands a thread handle to the
// guest (create, self) goes through `guest_thread_handle_*`, and every body that accepts one
// (join, detach, rename, getname, attr-get, raise-exception) resolves it back to the host `pthread_t` first.
// Internal registries (names, stacks, exception slots, duplicated HANDLEs) keep their existing host
// key, so this is not a rekeying of the kernel layer.
//
//   * Identity: the handle scePthreadCreate wrote, the handle scePthreadSelf returns inside that
//     thread, and the handle every other call accepts are the SAME value. Windows guest entry stays
//     behind the native start gate until registration and the output-handle write are complete.
//   * Readable: the object is ordinary host memory and its first dword is the thread id. Objects
//     are pooled and never returned to the OS. Free slots read zero; a slot may later be reused for
//     another thread, so a spent handle must not be used after join/detached exit.
//   * Validated: a value absent from the live table resolves to nothing (the caller returns ESRCH)
//     instead of being cast to a `pthread_t` and handed to winpthreads. Like native pthread handles,
//     a stale address reused by a new owner is not distinguishable from that new owner's handle.
//   * Adopted: scePthreadSelf on a thread scePthreadCreate never made (the guest main thread, a
//     driver worker) yields a stable handle flagged `adopted`; joining it from another thread or
//     detaching it is EINVAL. Self-join retains EDEADLK. Native key cleanup retires adopted workers.
//
// Linux and macOS keep the raw `pthread_t` (inline identity below): it is already readable there,
// and changing a working platform is not part of this fix.
#pragma once

#include <pthread.h>
#include <cstdint>

namespace prosper::hle {

#if defined(_WIN32)

// Thread-creation path: the handle for `host`, created if absent and marked joinable (not adopted).
// Called while native guest start is still gated/joinable. Returns 0 on pool/map exhaustion or if
// the native id cannot be obtained; all allocation failures roll back without publishing a handle.
uint64_t guest_thread_handle_create(pthread_t host);

// scePthreadSelf: the handle for the calling thread, lazily adopting it when scePthreadCreate did
// not make it.
uint64_t guest_thread_handle_self();

// Resolve a guest-supplied handle. False for null, garbage, and handles already joined/retired.
// Optional ownership flags let guest join/detach refuse before a detached native HANDLE is gone.
bool guest_thread_handle_resolve(uint64_t handle, pthread_t* host, bool* adopted = nullptr,
                                 bool* detached = nullptr);

// Lifecycle notifications. `exited` runs on the thread itself as the last guest-visible step;
// `joined` after a successful pthread_join; `detached` after a successful pthread_detach. An object
// is released once the thread has both finished and been joined or detached.
void guest_thread_handle_exited(pthread_t host);
void guest_thread_handle_joined(uint64_t handle);
void guest_thread_handle_detached(uint64_t handle);

// Test hook: number of live objects, so a regression can assert nothing leaks per thread.
uint64_t guest_thread_handle_live_count();
// Fault points: free-list reserve, slab-vector reserve, slab allocation, host map, handle map.
// One-shot and mutex-protected; only regression tests set a nonzero point (1..5).
void guest_thread_handle_fail_allocation_for_test(unsigned point);
// Bounded caller-side publication rendezvous; null in normal execution. Called outside all locks.
void guest_thread_handle_before_create_for_test(void (*probe)());

#else

inline uint64_t guest_thread_handle_create(pthread_t host) { return (uint64_t)(uintptr_t)host; }
inline uint64_t guest_thread_handle_self() { return (uint64_t)(uintptr_t)pthread_self(); }
inline bool guest_thread_handle_resolve(uint64_t handle, pthread_t* host, bool* adopted = nullptr,
                                      bool* detached = nullptr) {
    if (adopted) *adopted = false;
    if (detached) *detached = false;
    *host = (pthread_t)(uintptr_t)handle;
    return true;   // unchanged behaviour: the raw value is the host id on these platforms
}
inline void guest_thread_handle_exited(pthread_t) {}
inline void guest_thread_handle_joined(uint64_t) {}
inline void guest_thread_handle_detached(uint64_t) {}
inline uint64_t guest_thread_handle_live_count() { return 0; }

#endif

}   // namespace prosper::hle
