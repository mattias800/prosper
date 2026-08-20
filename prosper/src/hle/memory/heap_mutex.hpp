#pragma once
// heap_mutex.hpp — #707 (Blasphemous 2, macOS): keep hot mutexes OFF the binary's __DATA.
//
// WHY THIS EXISTS
// On macOS/libc++ a namespace-scope `std::mutex` is CONSTANT-initialized to a NON-ZERO value
// (PTHREAD_MUTEX_INITIALIZER carries a non-zero _MUTEX_SIG), so the object lands in the Mach-O
// __DATA segment. During Blasphemous 2 asset load an unattributed #707 corruptor (a mach_vm-family
// write that bypasses current page protection but honors max_protection — see
// docs/BLASPHEMOUS2_EQ_CORRUPTION.md) zeroes a page-granular region of this executable's OWN __DATA,
// wiping the pthread `__sig` of any std::mutex living in that cluster. A later `lock()` then returns
// EINVAL, `std::mutex::lock()` throws, and the process terminates. On Linux/glibc
// PTHREAD_MUTEX_INITIALIZER is all-zero, so the same mutex is .bss and untouched — which is exactly
// why #707 is latent on Linux.
//
// WHAT PROSPER_HEAP_MUTEX GIVES YOU
// A mutex that survives the corruptor on macOS and is a drop-in for `std::mutex name;`
// (works with std::lock_guard / std::unique_lock via CTAD — call sites must use the deduced form
// `std::lock_guard lk(name)`, NOT `std::lock_guard<std::mutex> lk(name)`):
//   * the guest-visible object is a trivially-constructed forwarder -> it is zero-initializable and
//     therefore lands in .bss, which PR #753 round-2 evidence showed the corruptor does NOT reach —
//     never __DATA;
//   * the real std::mutex is heap-allocated ONCE, EAGERLY, at static-init time (single-threaded,
//     before the guest ever runs). It is NEVER new/malloc'd on a lock path. This is critical: an
//     earlier lazy-allocating variant (atomic pointer + first-use `new`) deadlocked inside the
//     macOS %fs SIGSEGV handler — which itself takes one of these locks — and wedged the runtime.
//     Keep allocation eager. The heap block is intentionally never freed (process-lifetime global).
//
// On non-Apple platforms this expands to a plain `std::mutex` (already .bss; zero overhead), so the
// primary Linux/Windows paths are completely unaffected.
#include <mutex>

#ifdef __APPLE__
namespace prosper {
struct HeapMutex {
    std::mutex* p;                      // .bss (zero-init); assigned by the paired initializer below
    void lock()     { p->lock(); }
    void unlock()   { p->unlock(); }
    bool try_lock() { return p->try_lock(); }
};
}  // namespace prosper
// The initializer MUST be declared immediately after the HeapMutex so that, within a TU, the
// forwarder is zero-initialized (static-init phase, before any dynamic initializer runs) and this
// ctor then assigns its heap backing during static init — before the guest runs. Cross-TU init
// order is irrelevant because each TU initializes only its own forwarder.
#define PROSPER_HEAP_MUTEX(name)                                              \
    ::prosper::HeapMutex name;                                                \
    static struct name##_HeapMutexInit {                                      \
        name##_HeapMutexInit() { name.p = new std::mutex(); }                 \
    } name##_heap_mutex_init_
#define PROSPER_HEAP_MUTEX_EXTERN(name) extern ::prosper::HeapMutex name
#else
#define PROSPER_HEAP_MUTEX(name)        std::mutex name
#define PROSPER_HEAP_MUTEX_EXTERN(name) extern std::mutex name
#endif
