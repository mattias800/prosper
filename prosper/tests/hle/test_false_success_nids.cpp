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
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
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
            // THIS BLOCK ASSERTED THE OPPOSITE UNTIL #3124, and the reversal is deliberate.
            //
            // It required a hard error and a completely untouched buffer, and named the present
            // behaviour as the mistake to avoid: "inventing a written result -- a zeroed hit count
            // over a layout NOBODY HAS ESTABLISHED". That reasoning was right, and its condition is
            // what changed: the layout is now established, twice over.
            //
            //   - s_savedata_dirsearch (the PS5 sibling) already writes hitNum @0x00,
            //     dirNames @0x08, dirNamesNum @0x10, setNum @0x14, from live evidence in #299.
            //   - PROSPER_SVCLOG=1 on Tactics Ogre (PPSA03839) captured BOTH spellings on one boot
            //     with a byte-identical result struct: [0x00]=0, [0x08]=caller buffer, [0x10]=0x400.
            //
            // What forced the reversal is that the hard error is not inert. Tactics Ogre calls this
            // once, after sceSaveDataInitialize3, and on NOT_FOUND it submits two DCBs, draws once,
            // then stops submitting while staying alive -- one black frame for the rest of the run.
            // Bisected over 700 commits (#3124). "Registering a NID can be worse than not" cuts both
            // ways: the danger is not only a false SUCCESS, it is any answer a caller cannot proceed
            // past.
            //
            // So the contract is now the sibling's: report zero hits EXPLICITLY and succeed. The
            // original objection is preserved as the last two arms -- the handler must write the
            // two count fields and NOTHING else, so "inventing a result" is still caught.
            unsigned char result[64];
            memset(result, 0x5A, sizeof(result));
            uint64_t param[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            const uint64_t r = fn((uint64_t)(uintptr_t)param, (uint64_t)(uintptr_t)result, 0, 0, 0, 0);

            // Kills: the #2302 hard error, which stalls PPSA03839's boot outright.
            CHECK(r == 0, "sceSaveDataDirNameSearchPs4 succeeds (a hard error stalls PPSA03839)");
            // Kills: THE ORIGINAL BUG and any `return 0` stub -- success is only honest if the
            // count the caller reads was actually written. 0x5A poison, not zeros, is what makes
            // "wrote zero" distinguishable from "wrote nothing".
            CHECK(result[0x00] == 0 && result[0x01] == 0 && result[0x02] == 0 && result[0x03] == 0,
                  "hitNum @0x00 is explicitly written to zero, not left as caller residue");
            CHECK(result[0x14] == 0 && result[0x15] == 0 && result[0x16] == 0 && result[0x17] == 0,
                  "setNum @0x14 is explicitly written to zero");
            // Kills: a handler that memsets the whole struct, which would clobber the caller's own
            // dirNames pointer @0x08 and capacity @0x10 -- the fields the sibling READS. This is
            // the original block's "do not invent a result" objection, kept and made precise.
            CHECK(all_bytes_equal(result + 0x04, 0x10, 0x5A),
                  "the caller's dirNames pointer and capacity are NOT overwritten");
            CHECK(all_bytes_equal(result + 0x18, sizeof(result) - 0x18, 0x5A),
                  "nothing beyond the two count fields is written");
            // The two PS4-namespace entry points were previously required to give the SAME answer,
            // on the reasoning that both derive from one local-inventory fact (prosper has no PS4
            // save-data store). #3124 separates them, because they ask different questions and the
            // charter's same-answer rule is about one question asked through several libraries:
            //
            //   TransferringMountPs4  -- MOUNT a PS4 save area. There is none, so it FAILS, and it
            //                            must: answering SCE_OK there is what black-screened Sonic
            //                            Frontiers for four sessions (#2023).
            //   DirNameSearchPs4      -- SEARCH that area for directories. There are none, so it
            //                            SUCCEEDS WITH ZERO HITS. "Found nothing" is not an error.
            //
            // The sibling this one must agree with is the PS5 spelling of the SAME question,
            // sceSaveDataDirNameSearch, which enumerates and reports its count -- and reports zero
            // the same way when the search is empty.
            CHECK(r != answers[1],
                  "the PS4 SEARCH succeeds where the PS4 MOUNT fails -- different questions, and "
                  "conflating them is what #3124 had to undo");
        }
    }
}

// libSceHttp answers every entry point except the URI parser with the dispatcher default 0
// (#2930). For the two entry points every caller starts with, that 0 IS a valid-looking id:
// sceHttpInit's contract returns a positive library context, sceHttpCreateTemplate a positive
// template id bound to it. Six of eight surveyed titles call both.
constexpr const char* kHttpInit            = "A9cVMUtEp4Y";
constexpr const char* kHttpCreateTemplate  = "0gYjPTR-6cY";
constexpr const char* kHttpDeleteTemplate  = "4I8vEpuEhZ8";

void test_http_ids() {
    printf("-- libSceHttp::sceHttpInit / sceHttpCreateTemplate --\n");
    HleFn init_fn = Hle::lookup(kHttpInit);
    // Kills: leaving the NID unregistered -- the dispatcher default 0 is the whole bug.
    CHECK(init_fn != nullptr, "sceHttpInit is registered");
    if (!init_fn) return;
    uint64_t ctx = init_fn(0, 0, 0, 0, 0, 0);
    CHECK(ctx != 0, "sceHttpInit returns a non-zero library context id");
    // Kills: answering with a negative error instead of an id -- guest sites carry the value
    // into later calls, so any id-shaped answer must have the sign bit clear.
    CHECK((int64_t)ctx > 0, "sceHttpInit returns a POSITIVE id, not an error");

    HleFn create_fn = Hle::lookup(kHttpCreateTemplate);
    CHECK(create_fn != nullptr, "sceHttpCreateTemplate is registered");
    if (!create_fn) return;
    uint64_t tmpl = create_fn(ctx, 0, 0, 0, 0, 0);
    CHECK(tmpl != 0, "sceHttpCreateTemplate returns a non-zero template id");
    CHECK((int64_t)tmpl > 0, "sceHttpCreateTemplate returns a POSITIVE id, not an error");

    HleFn del_fn = Hle::lookup(kHttpDeleteTemplate);
    CHECK(del_fn != nullptr, "sceHttpDeleteTemplate is registered");
    if (!del_fn) return;
    // Kills three mutations at once: a delete whose body is just `return 0` (never clearing its
    // slot), a table that never frees, and an exhaustion path answering an id-shaped value. Fill
    // the table to exhaustion, free ONE live id, and the allocator must hand out a positive id
    // again.
    uint64_t probe = tmpl;
    while ((int64_t)probe > 0)
        probe = create_fn(ctx, 0, 0, 0, 0, 0);
    CHECK((int64_t)probe < 0, "an exhausted template table answers NEGATIVE, not a fake id");
    // Kills: a delete that reports failure (or a crash) on a live id.
    CHECK(del_fn(tmpl, 0, 0, 0, 0, 0) == 0, "sceHttpDeleteTemplate answers SCE_OK for a live id");
    probe = create_fn(ctx, 0, 0, 0, 0, 0);
    CHECK((int64_t)probe > 0, "a freed slot is handed out again (delete actually cleared its slot)");
}

} // namespace

int main() {
    printf("== test_false_success_nids ==\n");
    register_builtin_hle();
    test_random();
    test_nptrophy2_info_queries();
    test_savedata_transferring_mount();
    test_http_ids();
    if (fails) { printf("== FAIL: %d check(s) failed ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
