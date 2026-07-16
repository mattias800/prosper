// test_mutex_types — guest mutexes must honor their TYPE (#145). Sony/FreeBSD semantics:
// attr type 1=ERRORCHECK (the DEFAULT — FreeBSD PTHREAD_MUTEX_DEFAULT; Kyty's attr-init
// settype(1)), 2=RECURSIVE, 3/4=NORMAL/ADAPTIVE. Previously settype was a no-op and every
// mutex (incl. static-sentinel self-inits) was forced RECURSIVE: trylock-on-owned SUCCEEDED
// and relock never reported EDEADLK, silently diverging from the console. Guest-visible
// errnos are FreeBSD values (EBUSY=16, EDEADLK=11 — NOT Linux's 35).
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <cstdint>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_mutex_types ==\n");
    register_builtin_hle();
    auto attr_init    = Hle::lookup(nid_hash("scePthreadMutexattrInit"));
    auto attr_settype = Hle::lookup(nid_hash("scePthreadMutexattrSettype"));
    auto attr_destroy = Hle::lookup(nid_hash("scePthreadMutexattrDestroy"));
    auto m_init    = Hle::lookup(nid_hash("scePthreadMutexInit"));
    auto m_lock    = Hle::lookup(nid_hash("scePthreadMutexLock"));
    auto m_trylock = Hle::lookup(nid_hash("scePthreadMutexTrylock"));
    auto m_unlock  = Hle::lookup(nid_hash("scePthreadMutexUnlock"));
    auto m_destroy = Hle::lookup(nid_hash("scePthreadMutexDestroy"));
    CHECK(attr_init && attr_settype && attr_destroy && m_init && m_lock && m_trylock && m_unlock && m_destroy,
          "mutex HLE functions registered");
    if (fails) { printf("== FAIL ==\n"); return 1; }

    auto U = [](void* p) { return (uint64_t)(uintptr_t)p; };

    // 1. DEFAULT (no attr) = ERRORCHECK: trylock-on-owned fails EBUSY(16); relock reports
    //    EDEADLK as the FreeBSD value 11 (not Linux 35); unlock releases.
    void* mtx = nullptr;
    CHECK(m_init(U(&mtx), 0, 0, 0, 0, 0) == 0 && mtx, "default init creates a mutex");
    CHECK(m_lock(U(&mtx), 0, 0, 0, 0, 0) == 0, "default: first lock succeeds");
    CHECK(m_trylock(U(&mtx), 0, 0, 0, 0, 0) == 16, "default: trylock on owned fails EBUSY(16)");
    CHECK(m_lock(U(&mtx), 0, 0, 0, 0, 0) == 11, "default: relock reports FreeBSD EDEADLK(11)");
    CHECK(m_unlock(U(&mtx), 0, 0, 0, 0, 0) == 0, "default: unlock succeeds");
    CHECK(m_trylock(U(&mtx), 0, 0, 0, 0, 0) == 0 && m_unlock(U(&mtx), 0, 0, 0, 0, 0) == 0,
          "default: trylock after unlock succeeds");
    m_destroy(U(&mtx), 0, 0, 0, 0, 0);

    // 2. Type 2 = RECURSIVE (settype honored): relock succeeds; needs two unlocks.
    void* attr = nullptr; void* rmtx = nullptr;
    CHECK(attr_init(U(&attr), 0, 0, 0, 0, 0) == 0 && attr, "attr init");
    CHECK(attr_settype(U(&attr), 2, 0, 0, 0, 0) == 0, "settype(2 RECURSIVE) accepted");
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
    m_init(U(&nmtx), U(&nattr), 0, 0, 0, 0);
    CHECK(m_lock(U(&nmtx), 0, 0, 0, 0, 0) == 0, "normal: lock");
    CHECK(m_trylock(U(&nmtx), 0, 0, 0, 0, 0) == 16, "normal: trylock on owned fails EBUSY(16)");
    CHECK(m_unlock(U(&nmtx), 0, 0, 0, 0, 0) == 0, "normal: unlock");
    m_destroy(U(&nmtx), 0, 0, 0, 0, 0);
    attr_destroy(U(&nattr), 0, 0, 0, 0, 0);

    // 4. Invalid type rejected.
    void* battr = nullptr;
    attr_init(U(&battr), 0, 0, 0, 0, 0);
    CHECK(attr_settype(U(&battr), 99, 0, 0, 0, 0) == 0x16, "settype(99) rejected EINVAL(22)");
    attr_destroy(U(&battr), 0, 0, 0, 0, 0);

    // 5. STATIC SENTINEL self-init remains non-recursive (native POSIX default; Windows ERRORCHECK):
    //    this is the bdwgc GC_allocate_ml shape — trylock on an owned static lock MUST fail
    //    (before this fix the self-init forced RECURSIVE and trylock succeeded on the owner).
    void* smtx = nullptr;   // NULL sentinel; first use self-initializes
    CHECK(m_lock(U(&smtx), 0, 0, 0, 0, 0) == 0 && smtx, "static sentinel self-initializes on lock");
    CHECK(m_trylock(U(&smtx), 0, 0, 0, 0, 0) == 16, "static non-recursive: trylock on owned fails EBUSY(16)");
    CHECK(m_unlock(U(&smtx), 0, 0, 0, 0, 0) == 0, "static non-recursive: unlock");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
