// test_false_success_nids — the FALSE SUCCESS class from #2081.
//
// An import prosper does not register returns the dispatcher's default 0. Where that 0 is also the
// contract's SCE_OK, the guest is told an operation succeeded that never ran, and any out-parameter
// it was supposed to fill keeps whatever was already there. Nothing crashes and no diagnostic
// fires, so these regress silently — which is exactly why they need a test rather than a boot.
//
// Every assertion below is written to die to a SPECIFIC wrong implementation, not merely to be true
// of the right one. The mutation each arm kills is named on the arm, because an assertion that
// passes for the wrong reason is the recurring failure in this project: a `CHECK(ret == 0)` on a
// fill contract passes against the very stub the change exists to remove.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

namespace {

// The guest-visible NIDs, from the PS5 3.20 export tables (verified by nid_hash round-trip in
// tools/nid_census --self-check, which re-derives all 39,158 published pairs with 0 mismatches).
constexpr const char* kRandomGetRandomNumber      = "PI7jIZj4pcE";
constexpr const char* kNpTrophy2GetTrophyInfo     = "EwNylPdWUTM";
constexpr const char* kNpTrophy2GetGroupInfo      = "DoZWauG8mu0";
constexpr const char* kNpTrophy2GetGroupInfoArray = "+PDSI6WgPRc";
constexpr const char* kNpTrophy2GetGameInfo       = "4IzqhhUQ3nk";  // already registered
constexpr const char* kNpTrophy2GetTrophyInfoArray = "y3zHpdZO6ME"; // already registered

bool all_bytes_equal(const unsigned char* p, size_t n, unsigned char v) {
    for (size_t i = 0; i < n; ++i) if (p[i] != v) return false;
    return true;
}

void test_random() {
    printf("-- libSceRandom::sceRandomGetRandomNumber --\n");
    HleFn fn = Hle::lookup(kRandomGetRandomNumber);
    // Kills: leaving the NID unregistered, which is the whole bug — the dispatcher would answer 0.
    CHECK(fn != nullptr, "sceRandomGetRandomNumber is registered");
    if (!fn) return;

    // A buffer pre-filled with a recognisable sentinel. Every "did it write?" question below is
    // asked against this pattern, so a handler that returns success without touching memory is
    // distinguishable from one that fills it — which the dispatcher default is NOT.
    constexpr unsigned char kSentinel = 0xA5;
    constexpr size_t kReq = 32, kBufSize = 64;

    unsigned char buf[kBufSize];
    memset(buf, kSentinel, sizeof(buf));
    uint64_t ret = fn((uint64_t)(uintptr_t)buf, kReq, 0, 0, 0, 0);

    // Kills: returning an error for an ordinary in-contract request.
    CHECK(ret == 0, "a 32-byte request returns SCE_OK");
    // Kills: THE BUG — success with the buffer untouched (the dispatcher's `return 0`).
    CHECK(!all_bytes_equal(buf, kReq, kSentinel), "the requested bytes were actually written");
    // Kills: a memset(buf, 0, size) "implementation" — zeros are a write, but they are not entropy,
    // and they would satisfy the previous assertion while re-creating the same predictability bug.
    CHECK(!all_bytes_equal(buf, kReq, 0), "the written bytes are not all zero");
    // Kills: writing past the requested length (a cap/length confusion in the handler).
    CHECK(all_bytes_equal(buf + kReq, kBufSize - kReq, kSentinel),
          "bytes past the requested length are untouched");

    // Kills: any deterministic filler — a constant, a counter, a fixed seed, or a buffer that is
    // filled once and cached. This is the arm that separates "wrote something" from "wrote random".
    unsigned char again[kReq];
    memset(again, kSentinel, sizeof(again));
    uint64_t ret2 = fn((uint64_t)(uintptr_t)again, kReq, 0, 0, 0, 0);
    CHECK(ret2 == 0, "a second request also returns SCE_OK");
    CHECK(memcmp(buf, again, kReq) != 0, "two successive draws differ");

    // Zero-length: nothing was asked for, so nothing may be written and the call still succeeds.
    // Kills: a handler that treats size==0 as an error, or that writes a byte anyway.
    unsigned char zbuf[8];
    memset(zbuf, kSentinel, sizeof(zbuf));
    CHECK(fn((uint64_t)(uintptr_t)zbuf, 0, 0, 0, 0, 0) == 0, "a zero-length request returns SCE_OK");
    CHECK(all_bytes_equal(zbuf, sizeof(zbuf), kSentinel), "a zero-length request writes nothing");

    // Over the published 64-byte single-request cap: must FAIL, and must not partially fill. A
    // partial fill reported as success would be the same lie this handler exists to remove.
    // Kills: silently serving an over-length request, and "reject but scribble anyway".
    unsigned char big[128];
    memset(big, kSentinel, sizeof(big));
    uint64_t over = fn((uint64_t)(uintptr_t)big, sizeof(big), 0, 0, 0, 0);
    CHECK(over != 0, "an over-cap request returns an error");
    CHECK(all_bytes_equal(big, sizeof(big), kSentinel), "an over-cap request writes nothing");

    // A null / non-pointer buffer with a real length must be refused rather than faulting or
    // succeeding. Kills: a handler that dereferences a0 without validating it.
    CHECK(fn(0, 16, 0, 0, 0, 0) != 0, "a null buffer returns an error");

    // Every failure answer must be non-zero, since the measured call sites branch on exactly that
    // (nid_gate_scan: nonzero=2 on PPSA08804 and PPSA04263, nonzero=1 on PPSA24651/PPSA13579, and
    // no call site anywhere compares against a specific constant).
    // Kills: returning a "negative" that is zero in the low 32 bits the guest actually tests.
    CHECK((uint32_t)over != 0, "the error answer is non-zero in its low 32 bits");
}

void test_nptrophy2_info_queries() {
    printf("-- libSceNpTrophy2 info queries --\n");
    // Each of these fills a caller-supplied out-struct. Unregistered, they return SCE_OK over
    // memory nothing wrote — the #213 shape, where a heap-garbage trophy count sized a 34 GB array.
    // All five must answer, not just the two a title happened to crash on.
    const struct { const char* nid; const char* name; } queries[] = {
        { kNpTrophy2GetGameInfo,        "sceNpTrophy2GetGameInfo" },
        { kNpTrophy2GetTrophyInfoArray, "sceNpTrophy2GetTrophyInfoArray" },
        { kNpTrophy2GetTrophyInfo,      "sceNpTrophy2GetTrophyInfo" },
        { kNpTrophy2GetGroupInfo,       "sceNpTrophy2GetGroupInfo" },
        { kNpTrophy2GetGroupInfoArray,  "sceNpTrophy2GetGroupInfoArray" },
    };
    for (const auto& q : queries) {
        HleFn fn = Hle::lookup(q.nid);
        char msg[160];
        snprintf(msg, sizeof(msg), "%s is registered", q.name);
        // Kills: leaving any one of the five on the unregistered path. The singular/plural pairs are
        // the trap this arm exists for — registering `…InfoArray` and not `…Info` leaves the
        // identical failure live behind a name that reads as covered.
        CHECK(fn != nullptr, msg);
        if (!fn) continue;

        // The out-struct is not written on this path, so the ONLY thing keeping the caller off
        // uninitialised memory is a non-zero return. Kills: registering these to a `return 0`
        // no-op, which would be indistinguishable from the unregistered state it replaced.
        uint64_t out[8];
        memset(out, 0x5A, sizeof(out));
        uint64_t r = fn(1, 1, (uint64_t)(uintptr_t)out, 0, 0, 0);
        snprintf(msg, sizeof(msg), "%s reports unavailable rather than success", q.name);
        CHECK(r != 0, msg);
        snprintf(msg, sizeof(msg), "%s error is non-zero in its low 32 bits", q.name);
        CHECK((uint32_t)r != 0, msg);
    }
}

} // namespace

int main() {
    printf("== test_false_success_nids ==\n");
    register_builtin_hle();
    test_random();
    test_nptrophy2_info_queries();
    if (fails) { printf("== FAIL: %d check(s) failed ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
