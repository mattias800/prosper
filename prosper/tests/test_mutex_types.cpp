// test_mutex_types — guest mutexes must honor their TYPE (#145). Sony/FreeBSD semantics:
// attr type 1=ERRORCHECK (the DEFAULT — FreeBSD PTHREAD_MUTEX_DEFAULT; Kyty's attr-init
// settype(1)), 2=RECURSIVE, 3/4=NORMAL/ADAPTIVE. Previously settype was a no-op and every
// mutex (incl. static-sentinel self-inits) was forced RECURSIVE: trylock-on-owned SUCCEEDED
// and relock never reported EDEADLK, silently diverging from the console. Guest-visible
// errnos are FreeBSD values (EBUSY=16, EDEADLK=11 — NOT Linux's 35).
//
// The two spellings carry DIFFERENT error encodings (#1945). Sony's `scePthreadMutex*` reports the
// libkernel-encoded SCE_KERNEL_ERROR_* form (0x8002_0000 | FreeBSD errno); the POSIX
// `pthread_mutex_*` aliases report the bare errno. The shipped guest libc.prx proves the Sony side:
// its `_Mtx_trylock` compares the result against 0x80020010 and its `_Mtx_lock` against 0x8002000b,
// and maps anything else non-zero to Dinkumware `_Thrd_error`, which raises an uncaught
// std::system_error. Both halves are asserted below so neither can drift into the other.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include "../src/hle/sce_errno.hpp"
#include <cstdio>
#include <cstdint>

using namespace prosper;

static constexpr uint64_t kSceEBUSY   = prosper::hle::kSceKernelErrorEBUSY;     // 0x80020010
static constexpr uint64_t kSceEDEADLK = prosper::hle::kSceKernelErrorEDEADLK;   // 0x8002000b

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_mutex_types ==\n");
    register_builtin_hle();
    auto attr_init    = Hle::lookup(nid_hash("scePthreadMutexattrInit"));
    auto attr_settype = Hle::lookup(nid_hash("scePthreadMutexattrSettype"));
    auto attr_gettype = Hle::lookup(nid_hash("scePthreadMutexattrGettype"));
    auto attr_destroy = Hle::lookup(nid_hash("scePthreadMutexattrDestroy"));
    auto m_init    = Hle::lookup(nid_hash("scePthreadMutexInit"));
    auto m_lock    = Hle::lookup(nid_hash("scePthreadMutexLock"));
    auto m_trylock = Hle::lookup(nid_hash("scePthreadMutexTrylock"));
    auto m_unlock  = Hle::lookup(nid_hash("scePthreadMutexUnlock"));
    auto m_destroy = Hle::lookup(nid_hash("scePthreadMutexDestroy"));
    CHECK(attr_init && attr_settype && attr_gettype && attr_destroy &&
              m_init && m_lock && m_trylock && m_unlock && m_destroy,
          "mutex HLE functions registered");
    if (fails) { printf("== FAIL ==\n"); return 1; }

    auto U = [](void* p) { return (uint64_t)(uintptr_t)p; };

    // 1. DEFAULT (no attr) = ERRORCHECK: trylock-on-owned fails EBUSY(16); relock reports
    //    EDEADLK as the FreeBSD value 11 (not Linux 35); unlock releases.
    void* mtx = nullptr;
    CHECK(m_init(U(&mtx), 0, 0, 0, 0, 0) == 0 && mtx, "default init creates a mutex");
    CHECK(m_lock(U(&mtx), 0, 0, 0, 0, 0) == 0, "default: first lock succeeds");
    CHECK(m_trylock(U(&mtx), 0, 0, 0, 0, 0) == kSceEBUSY,
          "default: trylock on owned fails SCE-encoded EBUSY (0x80020010)");
    CHECK(m_lock(U(&mtx), 0, 0, 0, 0, 0) == kSceEDEADLK,
          "default: relock reports SCE-encoded EDEADLK (0x8002000b)");
    CHECK(m_unlock(U(&mtx), 0, 0, 0, 0, 0) == 0, "default: unlock succeeds");
    CHECK(m_trylock(U(&mtx), 0, 0, 0, 0, 0) == 0 && m_unlock(U(&mtx), 0, 0, 0, 0, 0) == 0,
          "default: trylock after unlock succeeds");
    m_destroy(U(&mtx), 0, 0, 0, 0, 0);

    // 2. Type 2 = RECURSIVE (settype honored): relock succeeds; needs two unlocks.
    void* attr = nullptr; void* rmtx = nullptr;
    CHECK(attr_init(U(&attr), 0, 0, 0, 0, 0) == 0 && attr, "attr init");
    int attr_type = -1;
    CHECK(attr_gettype(U(&attr), U(&attr_type), 0, 0, 0, 0) == 0 && attr_type == 1,
          "fresh attr reports Sony DEFAULT/ERRORCHECK type 1");
    CHECK(attr_settype(U(&attr), 2, 0, 0, 0, 0) == 0, "settype(2 RECURSIVE) accepted");
    CHECK(attr_gettype(U(&attr), U(&attr_type), 0, 0, 0, 0) == 0 && attr_type == 2,
          "recursive attr round-trips as type 2");
    CHECK(m_init(U(&rmtx), U(&attr), 0, 0, 0, 0) == 0, "recursive init");
    CHECK(m_lock(U(&rmtx), 0, 0, 0, 0, 0) == 0 && m_lock(U(&rmtx), 0, 0, 0, 0, 0) == 0,
          "recursive: relock on owner succeeds");
    CHECK(m_trylock(U(&rmtx), 0, 0, 0, 0, 0) == 0, "recursive: trylock on owner succeeds");
    CHECK(m_unlock(U(&rmtx), 0, 0, 0, 0, 0) == 0 && m_unlock(U(&rmtx), 0, 0, 0, 0, 0) == 0 &&
          m_unlock(U(&rmtx), 0, 0, 0, 0, 0) == 0, "recursive: three unlocks release the three locks");
    m_destroy(U(&rmtx), 0, 0, 0, 0, 0);
    attr_destroy(U(&attr), 0, 0, 0, 0, 0);

    // 3. Type 3 = NORMAL: trylock-on-owned fails EBUSY.
    void* nattr = nullptr; void* nmtx = nullptr;
    attr_init(U(&nattr), 0, 0, 0, 0, 0);
    CHECK(attr_settype(U(&nattr), 3, 0, 0, 0, 0) == 0, "settype(3 NORMAL) accepted");
    attr_type = -1;
    CHECK(attr_gettype(U(&nattr), U(&attr_type), 0, 0, 0, 0) == 0 && attr_type == 3,
          "normal attr round-trips as type 3");
    m_init(U(&nmtx), U(&nattr), 0, 0, 0, 0);
    CHECK(m_lock(U(&nmtx), 0, 0, 0, 0, 0) == 0, "normal: lock");
    CHECK(m_trylock(U(&nmtx), 0, 0, 0, 0, 0) == kSceEBUSY,
          "normal: trylock on owned fails SCE-encoded EBUSY");
    CHECK(m_unlock(U(&nmtx), 0, 0, 0, 0, 0) == 0, "normal: unlock");
    m_destroy(U(&nmtx), 0, 0, 0, 0, 0);
    attr_destroy(U(&nattr), 0, 0, 0, 0, 0);

    // 4. Type 4 = ADAPTIVE_NP: retains its guest identity even though its proven host behavior is
    //    implemented with ERRORCHECK (the same host type used for Sony type 1).
    void* aattr = nullptr;
    attr_init(U(&aattr), 0, 0, 0, 0, 0);
    CHECK(attr_settype(U(&aattr), 4, 0, 0, 0, 0) == 0, "settype(4 ADAPTIVE_NP) accepted");
    attr_type = -1;
    CHECK(attr_gettype(U(&aattr), U(&attr_type), 0, 0, 0, 0) == 0 && attr_type == 4,
          "adaptive attr round-trips as type 4");
    attr_destroy(U(&aattr), 0, 0, 0, 0, 0);

    // 5. Invalid type rejected. #2178: scePthreadMutexattrSettype is fallible, so the SONY spelling
    //    reports the encoded form like every other fallible Sony entry point — it was handing back a
    //    bare 22 while scePthreadMutexTrylock two lines up handed back 0x80020010. The POSIX
    //    spelling registered on the same body still reports the bare 22, and asserting both is what
    //    distinguishes a correct alias from either spelling drifting into the other's contract.
    void* battr = nullptr;
    attr_init(U(&battr), 0, 0, 0, 0, 0);
    CHECK(attr_settype(U(&battr), 99, 0, 0, 0, 0) == prosper::hle::kSceKernelErrorEINVAL,
          "scePthreadMutexattrSettype(99) rejected with encoded EINVAL (0x80020016)");
    auto p_attr_settype = Hle::lookup(nid_hash("pthread_mutexattr_settype"));
    CHECK(p_attr_settype && p_attr_settype(U(&battr), 99, 0, 0, 0, 0) == 22,
          "pthread_mutexattr_settype(99) keeps the bare EINVAL(22)");
    attr_destroy(U(&battr), 0, 0, 0, 0, 0);

    // 6. STATIC SENTINEL self-init remains non-recursive (native POSIX default; Windows ERRORCHECK):
    //    this is the bdwgc GC_allocate_ml shape — trylock on an owned static lock MUST fail
    //    (before this fix the self-init forced RECURSIVE and trylock succeeded on the owner).
    void* smtx = nullptr;   // NULL sentinel; first use self-initializes
    CHECK(m_lock(U(&smtx), 0, 0, 0, 0, 0) == 0 && smtx, "static sentinel self-initializes on lock");
    CHECK(m_trylock(U(&smtx), 0, 0, 0, 0, 0) == kSceEBUSY,
          "static non-recursive: trylock on owned fails SCE-encoded EBUSY");
    CHECK(m_unlock(U(&smtx), 0, 0, 0, 0, 0) == 0, "static non-recursive: unlock");

    // 7. The POSIX spelling of the SAME body keeps the bare errno (#1945). This is the arm that
    //    discriminates "the Sony family is encoded" from "everything got encoded": if the split is
    //    ever collapsed in either direction, exactly one of section 1/3/6 and this one fails.
    //    NOTE: the mutex here is created through the explicit init (attr-less = ERRORCHECK, as
    //    section 1 establishes), NOT through a NULL-sentinel self-init. A self-initialized static
    //    lock is the platform default type, which on Linux is NORMAL — relocking one does not
    //    report EDEADLK, it deadlocks the test.
    auto p_init    = Hle::lookup(nid_hash("pthread_mutex_init"));
    auto p_lock    = Hle::lookup(nid_hash("pthread_mutex_lock"));
    auto p_trylock = Hle::lookup(nid_hash("pthread_mutex_trylock"));
    auto p_unlock  = Hle::lookup(nid_hash("pthread_mutex_unlock"));
    auto p_destroy = Hle::lookup(nid_hash("pthread_mutex_destroy"));
    CHECK(p_init && p_lock && p_trylock && p_unlock && p_destroy,
          "POSIX mutex spellings registered");
    if (p_init && p_lock && p_trylock && p_unlock && p_destroy) {
        void* pmtx = nullptr;
        CHECK(p_init(U(&pmtx), 0, 0, 0, 0, 0) == 0 && pmtx, "posix: init creates a mutex");
        CHECK(p_lock(U(&pmtx), 0, 0, 0, 0, 0) == 0, "posix: first lock succeeds");
        CHECK(p_trylock(U(&pmtx), 0, 0, 0, 0, 0) == 16,
              "posix: trylock on owned returns the BARE FreeBSD EBUSY(16), not the SCE encoding");
        CHECK(p_lock(U(&pmtx), 0, 0, 0, 0, 0) == 11,
              "posix: relock returns the BARE FreeBSD EDEADLK(11), not the SCE encoding");
        CHECK(p_unlock(U(&pmtx), 0, 0, 0, 0, 0) == 0, "posix: unlock succeeds");
        p_destroy(U(&pmtx), 0, 0, 0, 0, 0);
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
