#pragma once
// Darwin (macOS) compatibility for the small set of Linux/glibc-specific APIs the emulator uses.
// Everything here is a REAL implementation (these back actual guest synchronization and fault-safe
// memory access), not a stub — correctness-first per the project charter. Included by the HLE TUs
// that use these APIs; a no-op on Linux/Windows.
//
// Provided on __APPLE__:
//   - process_vm_readv / process_vm_writev  (same-process only) via mach_vm_read_overwrite/mach_vm_write
//   - pthread_getattr_np                    via pthread_get_stackaddr_np/pthread_get_stacksize_np
//   - pthread_mutex_timedlock, pthread_rwlock_timedrdlock/timedwrlock   (trylock + sleep loop)
//   - sem_*        remapped to a mutex+cond counting semaphore (Darwin's unnamed sem_init is ENOSYS)
//   - pthread_barrier_*  remapped to a generation-counted mutex+cond barrier
//   - pthread_sigqueue   degraded to pthread_kill (sigval payload dropped — warns once; see note)
//   - prosper_mincore    signature-portable mincore wrapper (Darwin takes char*, Linux unsigned char*)
//
// Windows has no guest-execution substrate yet, and every symbol here is consumed only from
// POSIX-guarded code (exec_image_linux.cpp is __linux__/__APPLE__-only; hle_file.cpp's
// prosper_mincore use is inside #ifndef _WIN32). The whole header is therefore POSIX-only — on
// Windows it must expand to nothing, since <sys/mman.h> does not exist under MinGW.
#ifndef _WIN32

#include <sys/mman.h>
#include <fcntl.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>

// Global-asm portability: Mach-O prepends '_' to C symbols and has no ELF .type directive.
// PROSPER_ASM_TRAMPOLINE emits the shared "pass entry rsp as the 7th argument" shim used by the
// HLE callback/APR paths (see hle_file.cpp f_apr_read_submit_entry for the alignment rationale).
#ifdef __APPLE__
#define PROSPER_ASM_TRAMPOLINE(entry, target) \
    asm(".text\n" \
        ".globl _" #entry "\n" \
        "_" #entry ":\n" \
        "  movq %rsp, %rax\n" \
        "  pushq %rax\n" \
        "  call _" #target "\n" \
        "  addq $8, %rsp\n" \
        "  ret\n");
#else
#define PROSPER_ASM_TRAMPOLINE(entry, target) \
    asm(".text\n" \
        ".globl " #entry "\n" \
        ".type " #entry ",@function\n" \
        #entry ":\n" \
        "  movq %rsp, %rax\n" \
        "  pushq %rax\n" \
        "  call " #target "\n" \
        "  addq $8, %rsp\n" \
        "  ret\n");
#endif

#ifndef __APPLE__

static inline int prosper_mincore(const void* addr, size_t len, unsigned char* vec) {
#ifdef _WIN32
    (void)addr; (void)len; (void)vec; return -1;
#else
    return mincore(const_cast<void*>(addr), len, vec);
#endif
}

#else  // __APPLE__

#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <pthread.h>
#include <sys/uio.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <ctime>
#include <csignal>

static inline int prosper_mincore(const void* addr, size_t len, unsigned char* vec) {
    return mincore(const_cast<void*>(addr), len, reinterpret_cast<char*>(vec));
}

// ---- fault-safe same-process memory copy (Linux: process_vm_readv/writev) ----------------------
// The HLE uses these as an EFAULT-safe memcpy against possibly-unmapped guest addresses (pid is
// always getpid()). mach_vm_read_overwrite/mach_vm_write give the same guarantee on Darwin. The
// mach calls fail wholesale rather than partially; callers treat any short result as unreadable,
// which is the same failure class. Only the 1-iovec form the HLE uses is supported.
static inline ssize_t process_vm_readv(pid_t /*pid: self*/,
                                       const struct iovec* local, unsigned long liovcnt,
                                       const struct iovec* remote, unsigned long riovcnt,
                                       unsigned long /*flags*/) {
    if (liovcnt != 1 || riovcnt != 1 || local->iov_len != remote->iov_len) { errno = EINVAL; return -1; }
    mach_vm_size_t got = 0;
    kern_return_t kr = mach_vm_read_overwrite(mach_task_self(),
                                              (mach_vm_address_t)remote->iov_base,
                                              (mach_vm_size_t)remote->iov_len,
                                              (mach_vm_address_t)local->iov_base, &got);
    if (kr != KERN_SUCCESS) { errno = EFAULT; return -1; }
    return (ssize_t)got;
}
static inline ssize_t process_vm_writev(pid_t /*pid: self*/,
                                        const struct iovec* local, unsigned long liovcnt,
                                        const struct iovec* remote, unsigned long riovcnt,
                                        unsigned long /*flags*/) {
    if (liovcnt != 1 || riovcnt != 1 || local->iov_len != remote->iov_len) { errno = EINVAL; return -1; }
    kern_return_t kr = mach_vm_write(mach_task_self(),
                                     (mach_vm_address_t)remote->iov_base,
                                     (vm_offset_t)local->iov_base,
                                     (mach_msg_type_number_t)local->iov_len);
    if (kr != KERN_SUCCESS) { errno = EFAULT; return -1; }
    return (ssize_t)local->iov_len;
}

// ---- struct stat nanosecond timestamp field names ----------------------------------------------
#ifndef st_atim
#define st_atim st_atimespec
#define st_mtim st_mtimespec
#define st_ctim st_ctimespec
#endif

// ---- pthread_getattr_np ------------------------------------------------------------------------
// Darwin exposes the running thread's stack directly; repackage it into a pthread_attr_t so the
// existing pthread_attr_getstack() consumers work unchanged. Note pthread_get_stackaddr_np returns
// the stack TOP (highest address); POSIX attr carries the lowest address.
static inline int pthread_getattr_np(pthread_t t, pthread_attr_t* attr) {
    if (pthread_attr_init(attr) != 0) return EINVAL;
    void*  top = pthread_get_stackaddr_np(t);
    size_t sz  = pthread_get_stacksize_np(t);
    if (!top || !sz) return EINVAL;
    return pthread_attr_setstack(attr, (char*)top - sz, sz);
}

// ---- timed lock acquisition ----------------------------------------------------------------------
// Darwin has no pthread_*_timedlock family. A trylock + 500µs sleep loop against the CLOCK_REALTIME
// deadline preserves the guest-visible contract (0 only when actually locked, ETIMEDOUT on expiry);
// wake latency is up to ~0.5ms coarser than a kernel timedlock. CONFIDENCE: MED (granularity only).
static inline bool prosper_deadline_passed(const struct timespec* dl) {
    struct timespec now; clock_gettime(CLOCK_REALTIME, &now);
    return now.tv_sec > dl->tv_sec || (now.tv_sec == dl->tv_sec && now.tv_nsec >= dl->tv_nsec);
}
static inline int pthread_mutex_timedlock(pthread_mutex_t* m, const struct timespec* dl) {
    for (;;) {
        int rc = pthread_mutex_trylock(m);
        if (rc != EBUSY) return rc;
        if (prosper_deadline_passed(dl)) return ETIMEDOUT;
        usleep(500);
    }
}
static inline int pthread_rwlock_timedrdlock(pthread_rwlock_t* rw, const struct timespec* dl) {
    for (;;) {
        int rc = pthread_rwlock_tryrdlock(rw);
        if (rc != EBUSY) return rc;
        if (prosper_deadline_passed(dl)) return ETIMEDOUT;
        usleep(500);
    }
}
static inline int pthread_rwlock_timedwrlock(pthread_rwlock_t* rw, const struct timespec* dl) {
    for (;;) {
        int rc = pthread_rwlock_trywrlock(rw);
        if (rc != EBUSY) return rc;
        if (prosper_deadline_passed(dl)) return ETIMEDOUT;
        usleep(500);
    }
}

// ---- counting semaphore (Darwin sem_init is ENOSYS: unnamed POSIX semaphores unsupported) -------
struct prosper_sem_t {
    pthread_mutex_t m;
    pthread_cond_t  c;
    int             count;
};
static inline int prosper_sem_init(prosper_sem_t* s, int /*pshared*/, unsigned value) {
    pthread_mutex_init(&s->m, nullptr); pthread_cond_init(&s->c, nullptr);
    s->count = (int)value; return 0;
}
static inline int prosper_sem_wait(prosper_sem_t* s) {
    pthread_mutex_lock(&s->m);
    while (s->count <= 0) pthread_cond_wait(&s->c, &s->m);
    s->count--; pthread_mutex_unlock(&s->m); return 0;
}
static inline int prosper_sem_trywait(prosper_sem_t* s) {
    pthread_mutex_lock(&s->m);
    if (s->count <= 0) { pthread_mutex_unlock(&s->m); errno = EAGAIN; return -1; }
    s->count--; pthread_mutex_unlock(&s->m); return 0;
}
static inline int prosper_sem_timedwait(prosper_sem_t* s, const struct timespec* dl) {
    pthread_mutex_lock(&s->m);
    while (s->count <= 0) {
        int rc = pthread_cond_timedwait(&s->c, &s->m, dl);
        if (rc == ETIMEDOUT) { pthread_mutex_unlock(&s->m); errno = ETIMEDOUT; return -1; }
    }
    s->count--; pthread_mutex_unlock(&s->m); return 0;
}
static inline int prosper_sem_post(prosper_sem_t* s) {
    pthread_mutex_lock(&s->m);
    s->count++; pthread_cond_signal(&s->c); pthread_mutex_unlock(&s->m); return 0;
}
static inline int prosper_sem_getvalue(prosper_sem_t* s, int* out) {
    pthread_mutex_lock(&s->m); *out = s->count; pthread_mutex_unlock(&s->m); return 0;
}
static inline int prosper_sem_destroy(prosper_sem_t* s) {
    pthread_mutex_destroy(&s->m); pthread_cond_destroy(&s->c); return 0;
}
#define sem_t         prosper_sem_t
#define sem_init      prosper_sem_init
#define sem_wait      prosper_sem_wait
#define sem_trywait   prosper_sem_trywait
#define sem_timedwait prosper_sem_timedwait
#define sem_post      prosper_sem_post
#define sem_getvalue  prosper_sem_getvalue
#define sem_destroy   prosper_sem_destroy

// ---- barrier (Darwin has no pthread_barrier_t) ---------------------------------------------------
struct prosper_barrier_t {
    pthread_mutex_t m;
    pthread_cond_t  c;
    unsigned        threshold, waiting, generation;
};
#ifndef PTHREAD_BARRIER_SERIAL_THREAD
#define PTHREAD_BARRIER_SERIAL_THREAD (-1)
#endif
static inline int prosper_barrier_init(prosper_barrier_t* b, const void* /*attr*/, unsigned count) {
    if (!count) return EINVAL;
    pthread_mutex_init(&b->m, nullptr); pthread_cond_init(&b->c, nullptr);
    b->threshold = count; b->waiting = 0; b->generation = 0; return 0;
}
static inline int prosper_barrier_wait(prosper_barrier_t* b) {
    pthread_mutex_lock(&b->m);
    unsigned gen = b->generation;
    if (++b->waiting == b->threshold) {
        b->generation++; b->waiting = 0;
        pthread_cond_broadcast(&b->c);
        pthread_mutex_unlock(&b->m);
        return PTHREAD_BARRIER_SERIAL_THREAD;
    }
    while (gen == b->generation) pthread_cond_wait(&b->c, &b->m);
    pthread_mutex_unlock(&b->m);
    return 0;
}
static inline int prosper_barrier_destroy(prosper_barrier_t* b) {
    pthread_mutex_destroy(&b->m); pthread_cond_destroy(&b->c); return 0;
}
#define pthread_barrier_t       prosper_barrier_t
#define pthread_barrier_init    prosper_barrier_init
#define pthread_barrier_wait    prosper_barrier_wait
#define pthread_barrier_destroy prosper_barrier_destroy

// ---- pthread_sigqueue ----------------------------------------------------------------------------
// Darwin cannot queue a realtime signal with a payload to a specific thread. pthread_kill delivers
// the signal but DROPS the sigval; the exception-injection path that reads si_value will see zero.
// CONFIDENCE: LOW for that diagnostic path (warns once when first exercised) — acceptable for boot
// bring-up; revisit with a side-channel map if a title exercises guest exception injection on macOS.
static inline int pthread_sigqueue(pthread_t t, int sig, const union sigval /*value*/) {
    static bool warned = false;
    if (!warned) { warned = true; fprintf(stderr, "[shim] pthread_sigqueue: sigval dropped on Darwin (pthread_kill)\n"); }
    return pthread_kill(t, sig);
}

#endif // __APPLE__

// ---- signal-context register access (shared by exec_image and the exception delivery HLE) -------
// Linux exposes the interrupted registers as an indexable gregs[] array; Darwin as named fields in
// __ss. PROSPER_GREGS gives handler code one indexable view of both; writes through it mutate the
// mcontext exactly like the Linux array (sigreturn applies them on resume).
#ifdef __APPLE__
#include <signal.h>
#include <sys/ucontext.h>
#pragma clang diagnostic ignored "-Wdeprecated-declarations"   // syscall(2): raw traps stay async-signal-safe
enum { REG_R8 = 0, REG_R9, REG_R10, REG_R11, REG_R12, REG_R13, REG_R14, REG_R15,
       REG_RDI, REG_RSI, REG_RBP, REG_RBX, REG_RDX, REG_RAX, REG_RCX, REG_RSP,
       REG_RIP, REG_EFL };
using greg_t = int64_t;
struct prosper_gregs_view {
    ucontext_t* uc;
    uint64_t& operator[](int r) const {
        auto& ss = uc->uc_mcontext->__ss;
        switch (r) {
            case REG_R8:  return (uint64_t&)ss.__r8;   case REG_R9:  return (uint64_t&)ss.__r9;
            case REG_R10: return (uint64_t&)ss.__r10;  case REG_R11: return (uint64_t&)ss.__r11;
            case REG_R12: return (uint64_t&)ss.__r12;  case REG_R13: return (uint64_t&)ss.__r13;
            case REG_R14: return (uint64_t&)ss.__r14;  case REG_R15: return (uint64_t&)ss.__r15;
            case REG_RDI: return (uint64_t&)ss.__rdi;  case REG_RSI: return (uint64_t&)ss.__rsi;
            case REG_RBP: return (uint64_t&)ss.__rbp;  case REG_RBX: return (uint64_t&)ss.__rbx;
            case REG_RDX: return (uint64_t&)ss.__rdx;  case REG_RAX: return (uint64_t&)ss.__rax;
            case REG_RCX: return (uint64_t&)ss.__rcx;  case REG_RSP: return (uint64_t&)ss.__rsp;
            case REG_RIP: return (uint64_t&)ss.__rip;  default:      return (uint64_t&)ss.__rflags;
        }
    }
};
#define PROSPER_GREGS(u)   (prosper_gregs_view{(u)})
#define PROSPER_REG_ERR(u) ((uint64_t)(u)->uc_mcontext->__es.__err)
// si_fd (which perf fd fired) is Linux-only siginfo; on Darwin no perf fd ever exists, so -1
// routes the never-reached HWBP dispatch to its "not one of ours" paths.
#define PROSPER_SI_FD(si) (-1)
#elif defined(__linux__)
#define PROSPER_GREGS(u)   ((u)->uc_mcontext.gregs)
#define PROSPER_REG_ERR(u) ((uint64_t)(u)->uc_mcontext.gregs[REG_ERR])
#define PROSPER_SI_FD(si)  ((si)->si_fd)
#endif

// ---- substrate primitives shared by exec_image / hle_kernel_mem / sync_futex --------------------
// Real host thread id (async-signal-safe on both platforms; must NOT read %fs-based TLS because
// guest threads may run with a guest %fs — see the g_jb comment in exec_image_linux.cpp).
#ifdef __APPLE__
#include <mach/mach_time.h>
static inline uint64_t prosper_gettid() {
    uint64_t tid = 0; pthread_threadid_np(nullptr, &tid);   // reads %gs-based TSD, never %fs
    return tid;
}
#elif defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
static inline uint64_t prosper_gettid() { return (uint64_t)syscall(SYS_gettid); }
#endif

#ifndef _WIN32
// mmap that claims `addr` only if the range is entirely free (Linux MAP_FIXED_NOREPLACE).
// Darwin lacks the flag: pass the address as a hint (Darwin honors free hints) and back out if the
// kernel placed the mapping elsewhere. Not atomic against a concurrent mapper, but all in-process
// fixed-address allocation goes through the HLE/loader paths. CONFIDENCE: MED (race window only).
static inline void* prosper_mmap_noreplace(void* addr, size_t len, int prot, int flags, int fd, off_t off) {
#ifdef __linux__
    return mmap(addr, len, prot, flags | MAP_FIXED_NOREPLACE, fd, off);
#else
    void* p = mmap(addr, len, prot, flags & ~MAP_FIXED, fd, off);
    if (p == MAP_FAILED) return p;
    if (p != addr) { munmap(p, len); errno = EEXIST; return MAP_FAILED; }
    return p;
#endif
}

// Anonymous shared-memory fd usable for multi-mapping one physical range (Linux memfd_create).
// Darwin: a POSIX shm object, unlinked immediately so the fd is the only reference.
static inline int prosper_memfd_create(const char* name) {
#ifdef __linux__
    return (int)syscall(SYS_memfd_create, name, 1 /*MFD_CLOEXEC*/);
#else
    char nm[64];
    for (unsigned i = 0; i < 64; i++) {
        snprintf(nm, sizeof nm, "/%s-%d-%u", name, (int)getpid(), i);
        int fd = shm_open(nm, O_RDWR | O_CREAT | O_EXCL, 0600);
        if (fd >= 0) { shm_unlink(nm); return fd; }
        if (errno != EEXIST) break;
    }
    return -1;
#endif
}

// futex(FUTEX_WAIT/WAKE)-shaped wait-on-address. Darwin: os_sync_wait_on_address (macOS 14.4+).
// Semantics preserved for the HLE callers: wait returns 0 on wake, -1/ETIMEDOUT on expiry,
// -1/EAGAIN when *addr != expect at entry; wake returns without error when no one waits.
#ifdef __APPLE__
extern "C" int os_sync_wait_on_address(void* addr, uint64_t value, size_t size, uint32_t flags);
extern "C" int os_sync_wait_on_address_with_timeout(void* addr, uint64_t value, size_t size,
                                                    uint32_t flags, uint32_t clockid, uint64_t timeout_ns);
extern "C" int os_sync_wake_by_address_any(void* addr, size_t size, uint32_t flags);
extern "C" int os_sync_wake_by_address_all(void* addr, size_t size, uint32_t flags);
static inline int prosper_futex_wait(uint32_t* addr, uint32_t expect, const struct timespec* rel) {
    if (__atomic_load_n(addr, __ATOMIC_SEQ_CST) != expect) { errno = EAGAIN; return -1; }
    int r;
    if (rel) {
        uint64_t ns = (uint64_t)rel->tv_sec * 1000000000ull + (uint64_t)rel->tv_nsec;
        r = os_sync_wait_on_address_with_timeout(addr, expect, 4, 0, 32 /*OS_CLOCK_MACH_ABSOLUTE_TIME*/, ns);
    } else {
        r = os_sync_wait_on_address(addr, expect, 4, 0);
    }
    return r < 0 ? -1 : 0;   // errno already ETIMEDOUT on expiry
}
static inline void prosper_futex_wake(uint32_t* addr, bool all) {
    if (all) os_sync_wake_by_address_all(addr, 4, 0);
    else     os_sync_wake_by_address_any(addr, 4, 0);
}
#elif defined(__linux__)
#include <linux/futex.h>
static inline int prosper_futex_wait(uint32_t* addr, uint32_t expect, const struct timespec* rel) {
    return (int)syscall(SYS_futex, addr, FUTEX_WAIT | FUTEX_PRIVATE_FLAG, expect, rel, nullptr, 0);
}
static inline void prosper_futex_wake(uint32_t* addr, bool all) {
    syscall(SYS_futex, addr, FUTEX_WAKE | FUTEX_PRIVATE_FLAG, all ? INT32_MAX : 1, nullptr, nullptr, 0);
}
#endif
#endif // !_WIN32 (substrate primitives)

#endif // !_WIN32 (whole header: POSIX-only, no-op under MinGW)
