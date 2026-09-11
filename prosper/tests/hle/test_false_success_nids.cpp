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
#if !defined(_WIN32)
#include <sys/mman.h>   // the untracked host mapping test_pool_decommit must be left alone
#endif
#include "hle/dispatch/nid.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace prosper;

static constexpr uint64_t kEinvalPool = 0x80020016ull;   // SCE_KERNEL_ERROR_EINVAL

extern "C" int prosper_reserved_range_state(uint64_t addr);

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
// #3502 -- the memory-pool family and the flexible-memory size pair. All five resolved against the
// PS5 3.20 export tables and cross-checked against the live import list of the titles that call
// them (PPSA25258, PPSA21783).
constexpr const char* kMemoryPoolExpand            = "qCSfqDILlns";
constexpr const char* kMemoryPoolReserve           = "pU-QydtGcGY";
constexpr const char* kMemoryPoolCommit            = "Vzl66WmfLvk";
constexpr const char* kConfiguredFlexibleMemorySize = "n1-v6FgU7MQ";
constexpr const char* kAvailableFlexibleMemorySize  = "aNz11fnnzi4";
constexpr const char* kGetPageTableStats            = "tZ2yplY8MBY";

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

void test_kernel_memory_pool() {
    printf("-- libkernel memory pool + flexible-memory sizes (#3502) --\n");
    // Unregistered, these answered the dispatcher's 0 = SCE_OK and wrote no out-parameter, so the
    // guest read its own uninitialised stack as an address or a size. NINJA GAIDEN 4 (PPSA25258)
    // took Reserve's unwritten result straight into Commit and dereferenced null at +0x118.
    //
    // Boots cannot police this. The one title observed calling GetPageTableStats (PPSA21783) still
    // aborts for an unrelated reason with or without a correct answer, so every assertion about it
    // has to live here or nowhere.

    // ---- the flexible-memory size pair ----
    HleFn cfg   = Hle::lookup(kConfiguredFlexibleMemorySize);
    HleFn avail = Hle::lookup(kAvailableFlexibleMemorySize);
    CHECK(cfg != nullptr,   "sceKernelConfiguredFlexibleMemorySize is registered");
    CHECK(avail != nullptr, "sceKernelAvailableFlexibleMemorySize is registered");
    if (cfg && avail) {
        uint64_t configured = 0x5A5A5A5A5A5A5A5Aull;
        uint64_t available  = 0x5A5A5A5A5A5A5A5Aull;
        CHECK(cfg((uint64_t)(uintptr_t)&configured, 0, 0, 0, 0, 0) == 0,
              "Configured returns SCE_OK");
        // Kills: registering it to a `return 0` no-op. That passes a ret==0 assertion while
        // leaving the poison in place -- the exact stub this change removes.
        CHECK(configured != 0x5A5A5A5A5A5A5A5Aull, "Configured writes its out-parameter");
        CHECK(configured != 0, "Configured does not report a zero budget");
        CHECK(avail((uint64_t)(uintptr_t)&available, 0, 0, 0, 0, 0) == 0,
              "Available returns SCE_OK");
        // Kills: re-introducing a second literal instead of the shared kFlexibleMemoryPoolBytes,
        // which is how the pair silently drifts apart.
        CHECK(configured == available,
              "Configured and Available answer from ONE constant");
        // Kills: dropping the null guard. Without it this call dereferences 0 instead of
        // returning EINVAL, so the arm below is a crash test as much as a value test.
        CHECK(cfg(0, 0, 0, 0, 0, 0) != 0, "Configured rejects a null out-parameter");
    }

    // ---- sceKernelGetPageTableStats: FOUR 32-bit out-parameters ----
    HleFn pts = Hle::lookup(kGetPageTableStats);
    CHECK(pts != nullptr, "sceKernelGetPageTableStats is registered");
    if (pts) {
        // Eight bytes of poison per slot; the contract writes only the low four. The width claim
        // rests on live probe evidence (four consecutive stack addresses four bytes apart), and
        // this is the only thing that holds it.
        uint64_t slot[4];
        memset(slot, 0x5A, sizeof(slot));
        CHECK(pts((uint64_t)(uintptr_t)&slot[0], (uint64_t)(uintptr_t)&slot[1],
                  (uint64_t)(uintptr_t)&slot[2], (uint64_t)(uintptr_t)&slot[3], 0, 0) == 0,
              "GetPageTableStats returns SCE_OK");
        for (int i = 0; i < 4; ++i) {
            char msg[160];
            snprintf(msg, sizeof(msg), "GetPageTableStats writes out%d", i);
            CHECK((uint32_t)(slot[i] & 0xffffffffull) == 0u, msg);
            // Kills: widening the store to uint64_t. That corrupts four bytes of guest stack
            // beyond the contract at every slot -- sixteen bytes in total -- and NO boot can see
            // it, because the only title observed calling this aborts either way.
            snprintf(msg, sizeof(msg), "GetPageTableStats leaves the upper 4 bytes of out%d alone", i);
            CHECK((uint32_t)(slot[i] >> 32) == 0x5A5A5A5Au, msg);
        }
        // Kills: dropping the null guard on ANY ONE of the four pointers -- a plausible edit,
        // since only the first is obviously an out-parameter at a glance.
        //
        // Each iteration nulls exactly ONE slot and leaves the other three valid, which is the
        // whole point. The first version of this arm passed a single configuration -- one valid
        // pointer and three nulls -- and asserted it covered "any" slot. It did not: deleting the
        // `!a2` or `!a3` disjunct still left `!a1` to catch that call, so the arm stayed green
        // against the very mutation it named. An arm that sits NEXT to the gap it claims reads
        // exactly like one that covers it. Raised in review of #3505.
        for (int null_slot = 0; null_slot < 4; ++null_slot) {
            uint64_t sink[4] = {0, 0, 0, 0};
            uint64_t arg[4];
            for (int j = 0; j < 4; ++j) arg[j] = (uint64_t)(uintptr_t)&sink[j];
            arg[null_slot] = 0;
            char msg[160];
            snprintf(msg, sizeof(msg), "GetPageTableStats rejects a null in slot %d ALONE", null_slot);
            CHECK(pts(arg[0], arg[1], arg[2], arg[3], 0, 0) != 0, msg);
        }
    }

    // ---- sceKernelMemoryPoolReserve: the call whose unwritten output crashed PPSA25258 ----
    HleFn reserve = Hle::lookup(kMemoryPoolReserve);
    CHECK(reserve != nullptr, "sceKernelMemoryPoolReserve is registered");
    CHECK(Hle::lookup(kMemoryPoolExpand) != nullptr, "sceKernelMemoryPoolExpand is registered");
    CHECK(Hle::lookup(kMemoryPoolCommit) != nullptr, "sceKernelMemoryPoolCommit is registered");
    if (reserve) {
        constexpr uint64_t kLen = 0x10000;   // small: this arm really does reserve address space
        uint64_t out = 0x5A5A5A5A5A5A5A5Aull;
        const uint64_t rc = reserve(0 /*no hint*/, kLen, kLen /*align*/, 0 /*flags*/,
                                    (uint64_t)(uintptr_t)&out, 0);
        CHECK(rc == 0, "Reserve succeeds for an unhinted 64 KiB range");
        // Kills: the return-0-writes-nothing stub. This is NINJA GAIDEN 4's actual bug -- the
        // guest passed this untouched value to Commit as an address.
        CHECK(out != 0x5A5A5A5A5A5A5A5Aull, "Reserve writes its addrOut");
        CHECK(out != 0, "Reserve does not hand back a null address");
        CHECK((out & (kLen - 1)) == 0, "Reserve honours the requested alignment");
        // Kills: dropping the addrOut null guard, which would fault rather than return EINVAL.
        CHECK(reserve(0, kLen, kLen, 0, 0, 0) != 0, "Reserve rejects a null addrOut");
    }
    // Expand and Commit are deliberately exercised only for registration here: Expand consumes the
    // direct-memory arena and Commit maps pages, and a unit test that quietly did either would be
    // making the suite's later arms depend on how much arena it had left. Their behaviour is
    // covered by the live-boot evidence on #3500.
}

// sceKernelAprSubmitCommandBufferAndGetId (qvMUCyyaCSI) — the third member of the APR submit
// family, and the one that was missing (#3498).
//
// The dispatcher's return-0 default is a false success in TWO ways here, and only one of them is
// about the id. The obvious one: 0 is the answer for a contract whose return value IS the id the
// guest then waits on. The one that actually stalls a title: the submit never runs, so the command
// buffer's cursor is never released and the guest's append loop — which polls GetSize minus GetUsed
// — has no room to encode its next command.
//
// THE CURSOR ARM NEEDS SOMETHING ON THE CURSOR TO BE MEANINGFUL. Written without the append below
// it reads "the cursor is 0 after submit" on a buffer whose cursor was already 0, and passes
// against a handler that invents an id and never submits — verified by that exact mutation, which
// is how this arm's first draft was caught. The append is therefore checked too: it is the arm's
// own positive control, and if it ever stops moving the cursor the reset check goes back to being
// vacuous silently.
void test_apr_submit_and_get_id() {
    HleFn submit_id   = Hle::lookup("qvMUCyyaCSI");
    HleFn set_buffer  = Hle::lookup("N-FSPA4S3nI");
    HleFn get_offset  = Hle::lookup("GnxKOHEawhk");
    HleFn append_320  = Hle::lookup("o67gODLFpls");   // appends a 0x20-byte completion command
    HleFn construct   = Hle::lookup("8aI7R7WaOlc");   // sceAmprCommandBufferConstructor
    CHECK(submit_id != nullptr,
          "sceKernelAprSubmitCommandBufferAndGetId is registered (unregistered, its id is 0)");
    if (!submit_id || !set_buffer || !get_offset || !append_320 || !construct) return;

    // An unbound command buffer: no equeue, so the submit hands out prosper's own per-ring token
    // rather than echoing a guest tag. Addresses are the same private shape the AMM test uses.
    // CONSTRUCT before SetBuffer: the cursor is only tracked for a buffer that has been constructed
    // with a capacity, which is the guest's own order and is why an attach-only fixture reports a
    // cursor of 0 forever.
    constexpr uint64_t kCb   = 0x7f0000110000ull;
    constexpr uint64_t kBuf  = 0x7f0000120000ull;
    constexpr uint64_t kSize = 0x40000ull;
    // a2 carries the capacity in the shape that makes the constructor TRACK the cursor
    // (ampr_cb_tracks_offset_arg); a1 = 0 leaves the request-shaped path alone.
    construct(kCb, /*a1=*/0, /*a2=capacity=*/0x1000, 0, 0, 0);
    set_buffer(kCb, kBuf, kSize, kBuf, 0, 3);

    // Put a command on the buffer. eq = 0 keeps the binding unbound, which is the state this arm
    // wants and also keeps the append from posting anything.
    append_320(kCb, /*eq=*/0, /*id=*/0, /*tag=*/0, 0, 0);
    const uint64_t appended = get_offset(kCb, 0, 0, 0, 0, 0);
    CHECK(appended != 0,
          "positive control: the append MOVED the command-buffer cursor (without this the reset "
          "check below cannot fail)");

    const uint64_t first = submit_id(kCb, /*ring_1based=*/6, 0, 0, 0, 0);
    CHECK(first != 0, "the returned id is NOT zero -- zero is what the missing handler answered");
    CHECK(get_offset(kCb, 0, 0, 0, 0, 0) == 0,
          "the submit RESET the command-buffer cursor (the half that stalls an append loop)");

    append_320(kCb, 0, 0, 0, 0, 0);
    const uint64_t second = submit_id(kCb, /*ring_1based=*/6, 0, 0, 0, 0);
    CHECK(second != first, "a second submit gets a DIFFERENT id (ids name submits, not the buffer)");
}

// sceKernelMemoryPoolDecommit (LXo1tpFqJGs) — the fourth member of the pool family (#3506).
//
// This is the only handler in its change that DESTROYS guest memory, so the arms are about what it
// must not do as much as what it must. #3506 asks for exactly this by name, and warns that "a
// Decommit arm that passes against 'unmap unconditionally' would be worse than none" — so each arm
// below names the mutation it exists to catch.
//
// The fixture is a real reserve + commit through the pool API, not a synthetic mapping, because the
// property under test is agreement between prosper's tracker and its own mappings.
void test_pool_decommit() {
    HleFn reserve  = Hle::lookup("pU-QydtGcGY");   // sceKernelMemoryPoolReserve
    HleFn commit   = Hle::lookup("Vzl66WmfLvk");   // sceKernelMemoryPoolCommit
    HleFn decommit = Hle::lookup("LXo1tpFqJGs");   // sceKernelMemoryPoolDecommit
    CHECK(decommit != nullptr,
          "sceKernelMemoryPoolDecommit is registered (unregistered, Commit had no inverse)");
    if (!reserve || !commit || !decommit) return;

    constexpr uint64_t kGranule = 0x10000ull;      // the lazy-commit granule decommit works in
    constexpr uint64_t kSpan    = kGranule * 4;

    uint64_t base = 0;
    CHECK(reserve(0, kSpan, kGranule, 0, (uint64_t)(uintptr_t)&base, 0) == 0 && base,
          "fixture: reserved four 64 KiB granules");
    if (!base) return;
    // Commit only the FIRST TWO granules. The gap is the point: NINJA GAIDEN 4 decommits whole pool
    // spans of which only part is committed, and refusing that produced 8.1 million refusals.
    CHECK(commit(base, kGranule * 2, 0xc, 0x3, 0, 0) == 0, "fixture: committed the first two granules");
    // prosper_reserved_range_state: 0 = untracked, 1 = tracked reservation, 2 = committed,
    // 4 = the AMM no-lazy-commit window. The positive control asserts 2 rather than "not 1",
    // because the arm below turns it back into 1 and a "not 1" control would pass on an untracked
    // range too -- which is the state a fixture that silently failed to commit would be in.
    CHECK(prosper_reserved_range_state(base) == 2,
          "fixture positive control: the committed range reads as COMMITTED");

    *(volatile uint32_t*)(uintptr_t)base = 0xA5A5A5A5u;   // faults if the commit did not back it

    // Decommit the WHOLE four-granule span, two of which were never committed.
    CHECK(decommit(base, kSpan, 0, 0, 0, 0) == 0,
          "a decommit spanning committed and uncommitted granules SUCCEEDS "
          "(mutation: refuse unless fully committed -> this fails)");

    // The reservation must survive. Without it a later Commit at the same address can land
    // elsewhere, or fail. This is the arm that catches "decommit = munmap".
    CHECK(prosper_reserved_range_state(base) == 1,
          "the RESERVATION survives the release (mutation: plain munmap -> this fails)");

    uint64_t recommitted = base;
    CHECK(commit(base, kGranule, 0xc, 0x3, 0, 0) == 0,
          "the released range can be committed again");
    CHECK(recommitted == base, "re-commit lands at the SAME address the reservation held");
    *(volatile uint32_t*)(uintptr_t)base = 0x5A5A5A5Au;   // faults if the re-commit did not back it
    CHECK(*(volatile uint32_t*)(uintptr_t)base == 0x5A5A5A5Au,
          "the re-committed page is writable and reads back");

#if !defined(_WIN32)
    // Refusal direction: a range prosper never tracked at all must touch nothing. This is a live
    // host mapping the test owns, so "touches nothing" is checkable rather than assumed — an
    // unconditional unmap would make the read below fault.
    //
    // The mapping is over-allocated and aligned UP to a whole granule. mmap guarantees PAGE
    // alignment (4 KiB), not granule alignment, so an unaligned base would leave the request with no
    // whole 64 KiB granule inside it — the inward rounding would then decline it before reaching the
    // tracker at all, and the arm would pass without ever exercising the refusal it names. Raised in
    // review of #3530: the arm was INTERMITTENTLY vacuous, which is worse than reliably so.
    void* foreign_raw = mmap(nullptr, (size_t)kGranule * 3, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(foreign_raw != MAP_FAILED, "fixture: a host mapping prosper never tracked");
    if (foreign_raw != MAP_FAILED) {
        const uint64_t foreign =
            ((uint64_t)(uintptr_t)foreign_raw + kGranule - 1) & ~(kGranule - 1);
        CHECK((foreign & (kGranule - 1)) == 0,
              "fixture positive control: the untracked mapping is granule-aligned, so the request "
              "really reaches the tracker");
        *(volatile uint32_t*)(uintptr_t)foreign = 0xC3C3C3C3u;
        decommit(foreign, kGranule, 0, 0, 0, 0);
        CHECK(*(volatile uint32_t*)(uintptr_t)foreign == 0xC3C3C3C3u,
              "decommit leaves memory prosper never committed ALONE "
              "(mutation: unmap unconditionally -> this faults or reads back changed)");
        munmap(foreign_raw, (size_t)kGranule * 3);
    }
#endif

    // Sub-granule requests release NOTHING rather than rounding outward. Rounding a destructive
    // range out would unmap bytes the guest did not name, and a hole smaller than the lazy-commit
    // granule would later be re-backed by an mmap that also replaces its live neighbours.
    uint64_t base2 = 0;
    if (reserve(0, kSpan, kGranule, 0, (uint64_t)(uintptr_t)&base2, 0) == 0 && base2 &&
        commit(base2, kSpan, 0xc, 0x3, 0, 0) == 0) {
        *(volatile uint32_t*)(uintptr_t)base2 = 0x11223344u;
        CHECK(decommit(base2 + 0x100, 0x200, 0, 0, 0, 0) == 0,
              "a sub-granule decommit succeeds without releasing anything");
        CHECK(*(volatile uint32_t*)(uintptr_t)base2 == 0x11223344u,
              "a sub-granule decommit does not release the granule containing it "
              "(mutation: round outward -> this faults)");
        decommit(base2, kSpan, 0, 0, 0, 0);
    }

    CHECK(decommit(0, kGranule, 0, 0, 0, 0) == kEinvalPool, "a null address is EINVAL");
    CHECK(decommit(base, 0, 0, 0, 0, 0) == kEinvalPool, "a zero length is EINVAL");

    // Overflow, and this arm is here because the bug was real rather than theoretical. Rounding the
    // base UP to a granule can wrap on its own, independently of whether the span wraps: with only
    // the span checked, `(-16, 8)` wraps `base` to 0 and yields a length of nearly 2^64, at which
    // point the clip returns EVERY committed mapping and the handler unmaps the whole process while
    // answering SCE_OK. Found in review of #3530.
    //
    // Both arms must be here. A span that wraps and a base whose ROUND-UP wraps are different
    // predicates, and the first version of the guard had only the first — so an arm testing just
    // `(-1, 2)` would have passed against the broken code.
    //
    // WHAT THE MUTATION LOOKS LIKE, so nobody reads a crash as a broken test: dropping the round-up
    // guard does not print `[FAIL]`, it kills the process with SIGSEGV (measured: exit 139). That is
    // the arm working — the handler unmaps the pages the test itself is running on. ctest scores it
    // as a failure either way; the distinction is only for whoever runs the mutation by hand.
    CHECK(decommit(~0ull - 15, 8, 0, 0, 0, 0) == kEinvalPool,
          "an address whose granule ROUND-UP overflows is EINVAL, not a whole-process unmap "
          "(mutation: drop the round-up guard -> this returns 0)");
    CHECK(decommit(~0ull - 1, 4, 0, 0, 0, 0) == kEinvalPool,
          "a span that itself overflows is EINVAL");
    // The process is still alive and the re-committed page above still readable — which is the
    // consequence the overflow arms exist to prevent, checked rather than assumed.
    CHECK(*(volatile uint32_t*)(uintptr_t)base == 0x5A5A5A5Au,
          "an overflowing decommit released NOTHING (mutation: drop the guard -> this faults)");
}

int main() {
    printf("== test_false_success_nids ==\n");
    register_builtin_hle();
    test_random();
    test_nptrophy2_info_queries();
    test_savedata_transferring_mount();
    test_http_ids();
    test_apr_submit_and_get_id();
    test_pool_decommit();
    test_kernel_memory_pool();
    if (fails) { printf("== FAIL: %d check(s) failed ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
