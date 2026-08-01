// test_select_sleep — select/pselect honour the pure-sleep shape and refuse everything else (#1660).
//
// `select(0, NULL, NULL, NULL, &tv)` is the portable sleep idiom: nfds <= 0 with all three
// descriptor sets NULL, so no descriptor is ever examined. Before this was implemented the call fell
// through to the generic unimplemented stub, which returns 0 in ~111 ns — collapsing a 3-second
// sleep into a no-op. On PPSA19244 that made one service thread spin **9.0 million times per
// second** (1,300,185,430 calls in 144 s), saturating a core through the dispatch path.
//
// The contract under test, and the reason each half matters:
//
//   1. The pure-sleep shape must WAIT. `0` is the correct return ("timed out, nothing ready") — the
//      defect was returning it immediately. So the assertion is on ELAPSED TIME, not the value: a
//      test that only checked the return value passes against the very stub this replaces.
//   2. A descriptor-set query must stay FAIL-VISIBLE. prosper has no socket backing, so answering
//      "0 = nothing ready" would be the same success-shaped lie in a new place. It must return -1.
//
// Both halves fail without the fix: (1) elapses ~0 ms against the unimplemented stub, and (2)
// returns 0 rather than -1.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <functional>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static int64_t elapsed_ms(const std::function<void()>& body) {
    const auto t0 = std::chrono::steady_clock::now();
    body();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - t0).count();
}

int main() {
    printf("test_select_sleep\n");
    register_builtin_hle();

    // The guest binds these by NID; assert the hash matches the real PS5 3.20 export so the
    // registration cannot silently stop binding (nid_hash is the same function the stub emitter
    // uses, and these are the NIDs PPSA19244's import table actually carries).
    CHECK(nid_hash("select") == "T8fER+tIGgk", "nid_hash(\"select\") matches the PS5 3.20 export");
    CHECK(nid_hash("pselect") == "ZO2nWoTAv60", "nid_hash(\"pselect\") matches the PS5 3.20 export");

    auto select_fn  = Hle::lookup(nid_hash("select"));
    auto pselect_fn = Hle::lookup(nid_hash("pselect"));
    CHECK(select_fn != nullptr, "select is registered (not left to the unimplemented stub)");
    CHECK(pselect_fn != nullptr, "pselect is registered");
    if (!select_fn || !pselect_fn) { printf("FAILED (%d)\n", ++fails); return 1; }

    // --- 1. the pure-sleep shape actually waits -------------------------------------------------
    // 150 ms is long enough to be unambiguous against scheduler noise and short enough for ctest.
    // Assert a small tolerance below the request rather than exact equality: nanosleep may round.
    {
        int64_t tv[2] = { 0, 150000 };                       // struct timeval { sec, usec }
        uint64_t rc = 0;
        const int64_t ms = elapsed_ms([&] {
            rc = select_fn(0, 0, 0, 0, (uint64_t)(uintptr_t)tv, 0);
        });
        printf("  select(0,NULL,NULL,NULL,{0,150000}) -> %llu in %lld ms\n",
               (unsigned long long)rc, (long long)ms);
        CHECK(ms >= 140, "select honours a 150 ms timeout (the stub returned in ~0 ms)");
        CHECK(rc == 0, "select returns 0 (timed out, nothing ready) after waiting");
    }
    {
        int64_t ts[2] = { 0, 150000000 };                    // struct timespec { sec, nsec }
        uint64_t rc = 0;
        const int64_t ms = elapsed_ms([&] {
            rc = pselect_fn(0, 0, 0, 0, (uint64_t)(uintptr_t)ts, 0);
        });
        printf("  pselect(0,NULL,NULL,NULL,{0,150000000}) -> %llu in %lld ms\n",
               (unsigned long long)rc, (long long)ms);
        CHECK(ms >= 140, "pselect honours a 150 ms timespec timeout");
        CHECK(rc == 0, "pselect returns 0 after waiting");
    }

    // A zero timeout is a legal poll of an empty set: it must return promptly, and must not be
    // confused with the unbounded-wait (NULL timeout) case.
    {
        int64_t tv[2] = { 0, 0 };
        uint64_t rc = 0;
        const int64_t ms = elapsed_ms([&] {
            rc = select_fn(0, 0, 0, 0, (uint64_t)(uintptr_t)tv, 0);
        });
        CHECK(ms < 100, "select with a zero timeout returns promptly");
        CHECK(rc == 0, "select with a zero timeout returns 0");
    }

    // --- 2. a descriptor-set query stays fail-visible --------------------------------------------
    // This is the half that keeps the fix honest: the sleep path must NOT be widened to cover a
    // real readiness query, because prosper has no socket backing to answer one with.
    {
        uint64_t fdset[16] = {0};                            // contents irrelevant; non-NULL is
        int64_t tv[2] = { 0, 0 };                            // what makes this a real query
        const uint64_t rc = select_fn(4, (uint64_t)(uintptr_t)fdset, 0, 0,
                                      (uint64_t)(uintptr_t)tv, 0);
        printf("  select(4, &readfds, ...) -> %lld (errno=%d)\n", (long long)(int64_t)rc, errno);
        CHECK((int64_t)rc == -1, "select with a readfds set fails visibly (-1), never a false 0");
        CHECK(errno == ENOSYS, "select sets errno=ENOSYS for the unimplemented descriptor query");
    }
    {
        uint64_t fdset[16] = {0};
        const uint64_t rc = select_fn(4, 0, (uint64_t)(uintptr_t)fdset, 0, 0, 0);
        CHECK((int64_t)rc == -1, "select with a writefds set fails visibly (-1)");
    }
    {
        uint64_t fdset[16] = {0};
        const uint64_t rc = select_fn(4, 0, 0, (uint64_t)(uintptr_t)fdset, 0, 0);
        CHECK((int64_t)rc == -1, "select with an exceptfds set fails visibly (-1)");
    }
    {
        uint64_t fdset[16] = {0};
        const uint64_t rc = pselect_fn(4, (uint64_t)(uintptr_t)fdset, 0, 0, 0, 0);
        CHECK((int64_t)rc == -1, "pselect with a readfds set fails visibly (-1)");
    }
    // Deliberate scope boundary. `nfds > 0` with every set NULL examines no descriptor either, so by
    // POSIX semantics it is *also* a sleep — but no title has been observed making that call, so the
    // implemented shape stays exactly the one the evidence supports (`nfds <= 0` AND all sets NULL)
    // and this form is refused loudly rather than assumed. Pinning it here means widening the
    // contract later has to be a deliberate edit with its own evidence, not an accident.
    {
        int64_t tv[2] = { 0, 0 };
        const uint64_t rc = select_fn(8, 0, 0, 0, (uint64_t)(uintptr_t)tv, 0);
        CHECK((int64_t)rc == -1,
              "nfds>0 with all sets NULL is outside the observed shape: refused, not assumed");
    }

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
