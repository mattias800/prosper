// test_ult_fail_visible.cpp — #1603: whatever libSceUlt does NOT implement must fail VISIBLY, and
// whatever it does implement must stay counted per call.
//
// Before hle_ult.cpp existed, none of these NIDs was registered, so every libSceUlt call fell to the
// generic unresolved-import path, which returns 0 (= SCE_OK for this ABI family) and logs the NID
// exactly ONCE no matter how often it is called. On Earthion that was one log line for 1,005,742
// sceUltMutexLock calls, and four "successfully created" ulthreads that never ran.
//
// This file owns the REGISTRATION and RETURN-POLICY contract. The implemented semantics (mutual
// exclusion, work areas, object identity) are tested in test_ult_semantics.cpp.
#include "../src/hle/dispatch.hpp"

#include <cstdint>
#include <cstdio>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

// SCE_KERNEL_ERROR_ENOSYS — 0x80020000 | FreeBSD errno 78. See hle_ult.cpp for why this specific
// value is a derived convention rather than an invented libSceUlt constant.
static constexpr uint64_t kEnosys = 0x8002004Eull;

// Every libSceUlt NID Earthion (PPSA28061) imports. Split by what prosper does with each one, so the
// split itself is a checked fact rather than a comment.
static const char* const kImplementedNids[] = {
    "hZIg1EWGsHM",  // sceUltInitialize
    "d-kSG2fLrvI",  // sceUltFinalize
    "jw9FkZBXo-g",  // _sceUltUlthreadRuntimeCreate
    "YiHujOG9vXY",  // _sceUltWaitingQueueResourcePoolCreate
    "mmt8Sa6tL6c",  // _sceUltMutexCreate
    "8hEGkR1pfr8",  // sceUltMutexLock
    "h0XebKiMBtk",  // sceUltMutexUnlock
    "jW+HnafeS3Y",  // sceUltMutexDestroy
};
// Registered, counted, and refused — not yet implemented. "Not seen in the measured window" is not
// "never called": these are the teardown and blocking paths.
static const char* const kUnimplementedNids[] = {
    "znI3q8S7KQ4",  // _sceUltUlthreadCreate
    "gCeAI57LGgI",  // sceUltUlthreadJoin
    "jnKaHGkrxZ4",  // _sceUltConditionVariableCreate
    "5xGAHCxA8M0",  // sceUltConditionVariableWait
    "JTw1cAVkuc0",  // sceUltConditionVariableSignal
    "xrmmI832R4U",  // sceUltConditionVariableDestroy
};
// The two entry points whose contract returns a SIZE rather than a status (#1618). A size has no
// error channel, so these may never be handed a sentinel under ANY policy.
static const char* const kSizeNids[] = {
    "grs2pbc2awM",  // sceUltUlthreadRuntimeGetWorkAreaSize
    "WIWV1Qd7PFU",  // sceUltWaitingQueueResourcePoolGetWorkAreaSize
};
// Any work-area size at or above this is not a size prosper could honour — it is a sentinel or
// garbage being read as data. Earthion's own requests are (16 ulthreads, 3 workers) and (16, 16); a
// real requirement for those is well under a kilobyte. 16 MiB is far above any plausible answer and
// far below the 2.0 GiB that SCE_KERNEL_ERROR_ENOSYS produced.
static constexpr uint64_t kSaneWorkAreaLimit = 16ull * 1024 * 1024;

int main() {
    register_builtin_hle();
    ult_reset_counts_for_test();
    // The environment must not decide whether this test passes: pin both policies explicitly.
    ult_set_return_success_for_test(false);
    ult_set_legacy_enosys_for_test(false);

    bool all_registered = true;
    for (const char* nid : kImplementedNids)   all_registered &= Hle::lookup(nid) != nullptr;
    for (const char* nid : kUnimplementedNids) all_registered &= Hle::lookup(nid) != nullptr;
    for (const char* nid : kSizeNids)          all_registered &= Hle::lookup(nid) != nullptr;
    CHECK(all_registered,
          "every libSceUlt NID Earthion imports is registered (no longer the silent generic path)");

    auto lock = Hle::lookup("8hEGkR1pfr8");     // sceUltMutexLock
    auto unlock = Hle::lookup("h0XebKiMBtk");   // sceUltMutexUnlock
    CHECK(lock && unlock, "sceUltMutexLock / sceUltMutexUnlock resolve to prosper handlers");
    if (!lock || !unlock) { std::printf("== FAIL: %d ==\n", ++fails); return 1; }

    // A lock on an object that was never created must not claim to have been taken. 0x1234 is not a
    // usable guest pointer, so this also proves the pointer is validated rather than dereferenced.
    CHECK(lock(0x1234, 0, 0, 0, 0, 0) != 0,
          "sceUltMutexLock on an uncreated object does NOT report success");

    // Per-call counting is the whole point: the generic path's first-seen dedup made a million calls
    // indistinguishable from one, and implementing an entry point must not bring the dedup back.
    const uint64_t before = ult_call_count("8hEGkR1pfr8");
    lock(0x1234, 0, 0, 0, 0, 0);
    lock(0x1234, 0, 0, 0, 0, 0);
    unlock(0x1234, 0, 0, 0, 0, 0);
    CHECK(ult_call_count("8hEGkR1pfr8") == before + 2 && ult_call_count("h0XebKiMBtk") >= 1,
          "calls are counted individually, not deduped to one first-seen entry");

    // What is not implemented must refuse, loudly and non-zero — never return 0 = SCE_OK.
    bool all_refuse = true;
    for (const char* nid : kUnimplementedNids)
        all_refuse &= Hle::lookup(nid)(0, 0, 0, 0, 0, 0) == kEnosys;
    CHECK(all_refuse, "every not-yet-implemented entry point returns SCE_KERNEL_ERROR_ENOSYS");

    // #1618. These return a size_t AS THEIR RETURN VALUE — there is no out-parameter (Earthion's call
    // site does `call …; mov [rbp-0x10],rax; mov rdi,[rbp-0x10]; call malloc`). Such a signature has
    // no error channel, so an error sentinel is read as a byte count: returning ENOSYS asked the guest
    // for a 0x8002004E-byte (2.0 GiB) allocation, which is worse than leaving the NID unregistered
    // because the dispatcher's unresolved-import default is a harmless 0.
    for (const char* nid : kSizeNids)
        CHECK(Hle::lookup(nid)(16, 16, 0, 0, 0, 0) < kSaneWorkAreaLimit,
              "a *GetWorkAreaSize query returns a real size, never an error sentinel (#1618)");

    // The A/B escape hatch restores the legacy fake success but must never restore the silence.
    {
        const uint64_t n = ult_call_count("8hEGkR1pfr8");
        const bool prev = ult_set_return_success_for_test(true);
        const uint64_t legacy = lock(0x1234, 0, 0, 0, 0, 0);
        ult_set_return_success_for_test(prev);
        CHECK(prev == false, "the default policy is real behaviour, not the legacy fake success");
        CHECK(legacy == 0 && ult_call_count("8hEGkR1pfr8") == n + 1,
              "PROSPER_ULT_RETURN_SUCCESS reproduces the legacy 0 but is still counted");
    }

    // PROSPER_ULT_LEGACY_ENOSYS restores Phase 1 wholesale, so the pre-implementation behaviour stays
    // reachable for an A/B — including for the entry points that ARE implemented.
    {
        const uint64_t n = ult_call_count("8hEGkR1pfr8");
        const bool prev = ult_set_legacy_enosys_for_test(true);
        const uint64_t refused = lock(0x1234, 0, 0, 0, 0, 0);
        const uint64_t size = Hle::lookup("WIWV1Qd7PFU")(16, 16, 0, 0, 0, 0);
        ult_set_legacy_enosys_for_test(prev);
        CHECK(prev == false, "PROSPER_ULT_LEGACY_ENOSYS is off by default");
        CHECK(refused == kEnosys && ult_call_count("8hEGkR1pfr8") == n + 1,
              "PROSPER_ULT_LEGACY_ENOSYS restores Phase 1's refusal, still counted");
        CHECK(size == 0,
              "even under PROSPER_ULT_LEGACY_ENOSYS a size query returns 0, not a sentinel (#1618)");
    }

    std::printf(fails ? "== FAIL: %d ==\n" : "== PASS ==\n", fails);
    return fails ? 1 : 0;
}
