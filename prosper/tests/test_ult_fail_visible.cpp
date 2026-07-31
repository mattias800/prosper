// test_ult_fail_visible.cpp — #1603 regression guard: libSceUlt must FAIL VISIBLY.
//
// Before hle_ult.cpp existed, none of these NIDs was registered, so every libSceUlt call fell to the
// generic unresolved-import path, which returns 0 (= SCE_OK for this ABI family) and logs the NID
// exactly ONCE no matter how often it is called. On Earthion that was one log line for 1,005,742
// sceUltMutexLock calls, and four "successfully created" ulthreads that never ran.
//
// Each assertion below fails on the pre-fix tree:
//   * Hle::lookup() returns nullptr for every Ult NID -> the first two checks fail outright;
//   * the generic path returns 0 -> the "not success" check fails;
//   * the generic path counts by first-seen import index, not per call, and is not reachable
//     without a linked import table -> the per-call counting check fails.
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

// The NIDs Earthion (PPSA28061) actually imports and actually calls, spanning every family in the
// library's exercised surface: runtime, ulthread, mutex, condvar, resource pool, lifecycle.
static const char* const kCalledNids[] = {
    "hZIg1EWGsHM",  // sceUltInitialize
    "jw9FkZBXo-g",  // _sceUltUlthreadRuntimeCreate
    "grs2pbc2awM",  // sceUltUlthreadRuntimeGetWorkAreaSize
    "znI3q8S7KQ4",  // _sceUltUlthreadCreate
    "mmt8Sa6tL6c",  // _sceUltMutexCreate
    "8hEGkR1pfr8",  // sceUltMutexLock
    "h0XebKiMBtk",  // sceUltMutexUnlock
    "jnKaHGkrxZ4",  // _sceUltConditionVariableCreate
    "JTw1cAVkuc0",  // sceUltConditionVariableSignal
    "YiHujOG9vXY",  // _sceUltWaitingQueueResourcePoolCreate
    "WIWV1Qd7PFU",  // sceUltWaitingQueueResourcePoolGetWorkAreaSize
};
// Imported but not observed in the measured window (teardown / blocking paths). Registered anyway,
// because "not seen in 118 seconds" is not "never called".
static const char* const kImportedNids[] = {
    "gCeAI57LGgI",  // sceUltUlthreadJoin
    "jW+HnafeS3Y",  // sceUltMutexDestroy
    "5xGAHCxA8M0",  // sceUltConditionVariableWait
    "xrmmI832R4U",  // sceUltConditionVariableDestroy
    "d-kSG2fLrvI",  // sceUltFinalize
};

int main() {
    register_builtin_hle();
    ult_reset_counts_for_test();
    // The environment must not decide whether this test passes: pin the default policy explicitly.
    ult_set_return_success_for_test(false);

    bool all_registered = true;
    for (const char* nid : kCalledNids)   all_registered &= Hle::lookup(nid) != nullptr;
    for (const char* nid : kImportedNids) all_registered &= Hle::lookup(nid) != nullptr;
    CHECK(all_registered,
          "every libSceUlt NID Earthion imports is registered (no longer the silent generic path)");

    auto lock = Hle::lookup("8hEGkR1pfr8");     // sceUltMutexLock
    auto unlock = Hle::lookup("h0XebKiMBtk");   // sceUltMutexUnlock
    CHECK(lock && unlock, "sceUltMutexLock / sceUltMutexUnlock resolve to prosper handlers");
    if (!lock || !unlock) { std::printf("== FAIL: %d ==\n", ++fails); return 1; }

    // A lock that is not implemented must not claim to have been taken.
    const uint64_t locked = lock(0x1234, 0, 0, 0, 0, 0);
    CHECK(locked != 0, "sceUltMutexLock does NOT report success (0 = SCE_OK would be a false lock)");
    CHECK(locked == kEnosys, "sceUltMutexLock returns SCE_KERNEL_ERROR_ENOSYS (0x8002004e)");

    // Per-call counting is the whole point: the generic path's first-seen dedup made a million
    // calls indistinguishable from one.
    lock(0x1234, 0, 0, 0, 0, 0);
    lock(0x1234, 0, 0, 0, 0, 0);
    unlock(0x1234, 0, 0, 0, 0, 0);
    CHECK(ult_call_count("8hEGkR1pfr8") == 3 && ult_call_count("h0XebKiMBtk") == 1,
          "calls are counted individually, not deduped to one first-seen entry");

    // Every registered entry point must refuse, not just the mutex pair.
    bool all_refuse = true;
    for (const char* nid : kCalledNids)   all_refuse &= Hle::lookup(nid)(0, 0, 0, 0, 0, 0) != 0;
    for (const char* nid : kImportedNids) all_refuse &= Hle::lookup(nid)(0, 0, 0, 0, 0, 0) != 0;
    CHECK(all_refuse, "no libSceUlt entry point reports success");

    // The size queries must leave the guest's out-param untouched — writing an invented size is the
    // #544/#660 failure (a "success" stub whose unwritten required-memory output drove a
    // multi-gigabyte allocation). A sentinel proves prosper wrote nothing.
    uint64_t work_area = 0xdeadbeefcafef00dull;
    const uint64_t runtime_size = Hle::lookup("grs2pbc2awM")(
        (uint64_t)(uintptr_t)&work_area, 4, 0, 0, 0, 0);
    const uint64_t pool_size = Hle::lookup("WIWV1Qd7PFU")(
        (uint64_t)(uintptr_t)&work_area, 4, 0, 0, 0, 0);
    CHECK(runtime_size != 0 && pool_size != 0 && work_area == 0xdeadbeefcafef00dull,
          "the *GetWorkAreaSize queries fail and write no invented size into the out-param");

    // The A/B escape hatch restores the legacy return but must never restore the silence.
    const uint64_t before = ult_call_count("8hEGkR1pfr8");
    const bool prev = ult_set_return_success_for_test(true);
    const uint64_t legacy = lock(0x1234, 0, 0, 0, 0, 0);
    ult_set_return_success_for_test(prev);
    CHECK(prev == false, "the default policy is the error, not the legacy fake success");
    CHECK(legacy == 0 && ult_call_count("8hEGkR1pfr8") == before + 1,
          "PROSPER_ULT_RETURN_SUCCESS reproduces the legacy 0 but is still counted");

    // The default must be restored for any later assertion in this process.
    CHECK(lock(0x1234, 0, 0, 0, 0, 0) == kEnosys, "the error policy is restored after the A/B");

    std::printf(fails ? "== FAIL: %d ==\n" : "== PASS ==\n", fails);
    return fails ? 1 : 0;
}
