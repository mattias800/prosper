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
constexpr const char* kSaveDataTransferringMount    = "WAzWTZm1H+I";
constexpr const char* kSaveDataTransferringMountPs4 = "RjMlsR8EXrw";
constexpr const char* kSaveDataDirNameSearchPs4     = "X4MYzukPc3g";

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

void test_savedata_transferring_mount() {
    printf("-- libSceSaveData transferring-mount pair --\n");
    // Both entry points mount a PS4-era save for import into the PS5 title. No prosper installation
    // has a PS4 save area, so the answer is "not found" — and the FALSE SUCCESS failure here is not
    // theoretical: `RjMlsR8EXrw` unregistered told Sonic Frontiers (PPSA03831) that the mount had
    // succeeded, whereupon the title formatted a path out of the mount-point result nothing had
    // written and opened `/gamedata` at filesystem root, once per frame, forever.
    const struct { const char* nid; const char* name; } mounts[] = {
        { kSaveDataTransferringMount,    "sceSaveDataTransferringMount" },
        { kSaveDataTransferringMountPs4, "sceSaveDataTransferringMountPs4" },
    };
    uint64_t answers[2] = { 0, 0 };
    for (size_t i = 0; i < 2; ++i) {
        HleFn fn = Hle::lookup(mounts[i].nid);
        char msg[160];
        snprintf(msg, sizeof(msg), "%s is registered", mounts[i].name);
        // Kills: leaving either sibling on the unregistered path. Registering one of a `…Ps4` pair
        // and not the other is the exact state this arm was written against — the covered name
        // reads as covering both.
        CHECK(fn != nullptr, msg);
        if (!fn) return;

        // A mount-point result the caller zeroed, as the guest does. The handler must NOT write a
        // mount point on the failure path — but note that a `return 0` stub writes nothing either,
        // so this assertion is a discriminator only together with the non-zero return below: what
        // makes the pair sound is "reports unavailable AND leaves the result alone".
        unsigned char result[32];
        memset(result, 0, sizeof(result));
        uint64_t param[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        answers[i] = fn((uint64_t)(uintptr_t)param, (uint64_t)(uintptr_t)result, 0, 0, 0, 0);

        // Kills: THE BUG — the dispatcher's `return 0`, and equally a hand-written `return 0` stub.
        snprintf(msg, sizeof(msg), "%s reports unavailable rather than success", mounts[i].name);
        CHECK(answers[i] != 0, msg);
        // Kills: an error whose low 32 bits are zero, AND — the reason this is a SIGN test rather
        // than a non-zero one — any positive low dword. The guest sites gate with `test eax,eax; js`,
        // a SIGN test, so `return 1` is non-zero and still reads as SUCCESS at every one of them:
        // the bug would be reinstated with this assertion green. Sony error codes are 0x8xxxxxxx, so
        // negative-as-int32 is the property the call sites actually test.
        snprintf(msg, sizeof(msg), "%s error is NEGATIVE as int32 (the sign the call sites test)",
                 mounts[i].name);
        CHECK((int32_t)(uint32_t)answers[i] < 0, msg);
        // Kills: writing a mount point the caller would then treat as a real mounted path.
        snprintf(msg, sizeof(msg), "%s leaves the mount-point result untouched", mounts[i].name);
        CHECK(all_bytes_equal(result, sizeof(result), 0), msg);
    }
    // Kills: the two siblings drifting apart — one answering NOT_FOUND and the other some other
    // code, so the title's behaviour would depend on which entry point it happened to call.
    // NOTE this is a cross-title POLICY lock, not #1873's same-question-two-libraries case:
    // PPSA03831 does not import WAzWTZm1H+I at all. It is still worth holding, because five local
    // titles call the Ps4 form and THREE of them const-compare the code — PPSA03839 against
    // 0x809F0003, and PPSA07809 *and* PPSA08804 against 0x809F000F — so the value is not inert
    // across the corpus. (PPSA08804's is inside its error arm past the branch, which is why a gate
    // scan that stops at the first branch reads it as a plain non-zero test.)
    //
    // The strongest evidence anyone has on the precise errno, recorded so it is not re-derived: two
    // independent titles have a DEDICATED arm for 0x809F000F and none for 0x809F0008. PPSA07809
    // routes it to a distinct state; PPSA08804 returns 2 for it against 3 for everything else.
    // 0x809F000F appears nowhere in prosper and the 3.20 dump carries no constants, so it is
    // unresolved — a lead, not a conclusion. Frontiers gates on the SIGN alone, so nothing here
    // depends on it.
    CHECK(answers[0] == answers[1], "both transferring-mount entry points give the same answer");

    // sceSaveDataDirNameSearchPs4 (#2210). With the mount above, these are the ONLY two
    // PS4-namespace exports libSceSaveData has across all 275 PS5 3.20 libraries, so the pair is
    // closed here rather than left one-registered -- which is the state that reads as covered.
    {
        HleFn fn = Hle::lookup(kSaveDataDirNameSearchPs4);
        // Kills: the unregistered path, where prosper_on_unimpl returns 0 == SCE_OK for this
        // contract and the caller is told a PS4 save-data search SUCCEEDED over a result it never
        // wrote. Imported by 13 modules across 12 titles, including PPSA25009/PPSA24651/PPSA13579,
        // verified with nid_census against the local dumps.
        CHECK(fn != nullptr, "sceSaveDataDirNameSearchPs4 is registered");
        if (fn) {
            // POISONED, not zeroed, and that is the difference that makes this arm mean something.
            // The mounts block above zeroes its result, so its "untouched" assertion cannot tell a
            // handler that wrote zeros from one that wrote nothing. 0x5A can.
            unsigned char result[64];
            memset(result, 0x5A, sizeof(result));
            uint64_t param[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            const uint64_t r = fn((uint64_t)(uintptr_t)param, (uint64_t)(uintptr_t)result, 0, 0, 0, 0);

            // Kills: THE BUG, and equally a hand-written `return 0` stub replacing it.
            CHECK(r != 0, "sceSaveDataDirNameSearchPs4 reports unavailable rather than success");
            // Sign, not merely non-zero: Sony errors are 0x8xxxxxxx and guest sites gate with
            // `test eax,eax; js`, so `return 1` would reinstate the bug with a non-zero check green.
            CHECK((int32_t)(uint32_t)r < 0,
                  "sceSaveDataDirNameSearchPs4 error is NEGATIVE as int32 (the sign call sites test)");
            // Kills: inventing a written result -- a zeroed hit count over a layout nobody has
            // established. That is the exact MIRROR of the defect being fixed, and it is the more
            // tempting mistake because it looks more helpful.
            CHECK(all_bytes_equal(result, sizeof(result), 0x5A),
                  "sceSaveDataDirNameSearchPs4 writes NOTHING to the caller's result buffer");
            // Kills: the two PS4-namespace entry points drifting apart. Both derive their answer
            // from the same local-inventory fact -- prosper has no PS4 save-data store -- so a
            // title's behaviour must not depend on which of them it happened to call.
            CHECK(r == answers[1],
                  "both PS4-namespace savedata entry points give the same answer");
        }
    }
}

} // namespace

int main() {
    printf("== test_false_success_nids ==\n");
    register_builtin_hle();
    test_random();
    test_nptrophy2_info_queries();
    test_savedata_transferring_mount();
    if (fails) { printf("== FAIL: %d check(s) failed ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
