// hle_ult.cpp — fail-VISIBLE libSceUlt surface (#1603). This file deliberately implements NO
// user-level threading semantics; it exists so that the absence of them stops being silent.
//
// libSceUlt is Sony's user-level threading library: cooperative "ulthreads" plus the mutexes,
// condition variables, queues, semaphores and reader/writer locks that schedule on top of them.
// prosper implements none of it. Before this file existed, every call fell through to the generic
// unresolved-import path (dispatch.cpp `prosper_on_unimpl`), which logs the NID **once** and
// returns **0** — and for this ABI family 0 is SCE_OK, i.e. success. The guest was therefore told:
//
//   * `_sceUltUlthreadCreate`  -> "your thread was created"   ... it never runs;
//   * `sceUltMutexLock/Unlock` -> "the lock is held/released" ... there is no mutual exclusion;
//   * the `*GetWorkAreaSize` queries -> "success" with the size out-param left UNWRITTEN, i.e.
//     whatever stack garbage the caller had there (the #544/#660 class that made Dead Cells
//     allocate multiple gigabytes).
//
// Nothing crashed. The guest proceeded on false premises and the damage surfaced far from its
// cause. That is strictly worse than an unimplemented call that fails loudly, which is what this
// file makes it: every entry point is counted per call and reported, and by default returns a real
// error instead of a fake success.
//
// Measured on Earthion (PPSA28061 — the ONLY dump in the library that imports libSceUlt, 16
// functions per `self_dump --symbols`): a 118 s CPU-only boot called `sceUltMutexLock` and
// `sceUltMutexUnlock` 1,005,742 times **each**, and the deduped unimplemented-import log printed
// exactly one line for each. Four ulthreads were reported created and none of them ever ran.
//
// SCOPE: Phase 1 of #1603 only — visibility. Implementing ulthreads, the mutex/condvar semantics
// and the scheduler is Phase 2 and must be derived from live title evidence and the guest's own
// disassembly, not assumed. Do not grow this file into a fake implementation.
#include "dispatch.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace prosper {
namespace {

// The registered surface. Driven by what titles ACTUALLY import, not by the library's full export
// list: the PS5 3.20 firmware database exports 64 `sceUlt*` symbols, and a byte-scan of every dump
// plus prosper's own `self_dump --symbols` agree that exactly these 16 are imported, by exactly one
// title (Earthion / PPSA28061). Any other Ult NID a future title imports still falls through to the
// generic unimplemented-import path, which logs it — add it here when that happens.
// CONFIDENCE: HIGH (two independent instruments agree on the NID<->name mapping and the import set).
struct UltEntry { const char* nid; const char* name; };
constexpr UltEntry kUlt[] = {
    { "jw9FkZBXo-g", "_sceUltUlthreadRuntimeCreate" },
    { "grs2pbc2awM", "sceUltUlthreadRuntimeGetWorkAreaSize" },
    { "znI3q8S7KQ4", "_sceUltUlthreadCreate" },
    { "gCeAI57LGgI", "sceUltUlthreadJoin" },
    { "mmt8Sa6tL6c", "_sceUltMutexCreate" },
    { "8hEGkR1pfr8", "sceUltMutexLock" },
    { "h0XebKiMBtk", "sceUltMutexUnlock" },
    { "jW+HnafeS3Y", "sceUltMutexDestroy" },
    { "jnKaHGkrxZ4", "_sceUltConditionVariableCreate" },
    { "5xGAHCxA8M0", "sceUltConditionVariableWait" },
    { "JTw1cAVkuc0", "sceUltConditionVariableSignal" },
    { "xrmmI832R4U", "sceUltConditionVariableDestroy" },
    { "YiHujOG9vXY", "_sceUltWaitingQueueResourcePoolCreate" },
    { "WIWV1Qd7PFU", "sceUltWaitingQueueResourcePoolGetWorkAreaSize" },
    { "hZIg1EWGsHM", "sceUltInitialize" },
    { "d-kSG2fLrvI", "sceUltFinalize" },
};
constexpr size_t kUltCount = sizeof(kUlt) / sizeof(kUlt[0]);

// Return value for every entry point.
//
// prosper has NO primary evidence for libSceUlt's own `SCE_ULT_ERROR_*` numbering — no live capture
// of a real console's return, no published constant, and the firmware symbol database gives names
// and NIDs only, never values. Fabricating a libSceUlt-shaped constant would look authoritative
// while being a guess, so this deliberately does not do that. Instead it returns the libkernel
// errno-family code that states exactly the truth — "function not implemented" — using the
// documented `0x80020000 | freebsd_errno` convention already used across hle_file.cpp. ENOSYS is
// 78 (0x4E) on the FreeBSD-derived PS5 kernel (it is 38 on Linux; do not copy that value here).
//
// CONFIDENCE: HIGH that 0 is wrong — every Sony API in this family treats 0 as SCE_OK, so a guest
//             that checks `ret != SCE_OK` MUST see a failure here.
// CONFIDENCE: LOW  that 0x8002004E is the specific value the real libSceUlt would return. A guest
//             that compares against a particular SCE_ULT_ERROR_* constant will not recognise it and
//             should fall into its generic error path. That is still the correct outcome: the call
//             genuinely failed.
constexpr uint64_t kUltNotImplemented = 0x8002004Eull;   // SCE_KERNEL_ERROR_ENOSYS

std::atomic<uint64_t> g_calls[kUltCount];
std::atomic<uint64_t> g_next_report[kUltCount];   // 0 = "report the next call" (i.e. the first)
std::atomic<bool>     g_banner_printed{false};

// PROSPER_ULT_RETURN_SUCCESS=1 restores the pre-#1603 `return 0` for A/B measurement of what the
// fake success was buying a title. It is OFF by default and it does NOT silence anything: the
// counting and the log lines below are emitted either way, so no configuration of prosper is ever
// silently wrong about Ult again.
std::atomic<bool> g_return_success{false};
bool env_return_success() {
    const char* e = getenv("PROSPER_ULT_RETURN_SUCCESS");
    return e && *e && std::strcmp(e, "0") != 0;
}

void print_banner() {
    if (g_banner_printed.exchange(true)) return;
    fprintf(stderr,
        "[prosper] ==== libSceUlt IS NOT IMPLEMENTED (#1603) ====================================\n"
        "[prosper]   libSceUlt is Sony's USER-LEVEL THREADING library (ulthreads, mutexes,\n"
        "[prosper]   condition variables, queues). prosper implements NONE of its semantics.\n"
        "[prosper]   Every ulthread this title creates NEVER RUNS, and any guest state it guards\n"
        "[prosper]   with an Ult mutex or condvar is COMPLETELY UNSYNCHRONISED.\n"
        "[prosper]   Calls below are reported per function at call 1 and then at each power of ten,\n"
        "[prosper]   so a polling loop cannot hide behind a single deduped line.\n"
        "[prosper] ==============================================================================\n");
}

// Report the call and hand back the policy's return value. Counting is per call (an atomic
// increment), NOT per first-seen NID: the whole point of this file is that the generic path's
// first-seen dedup reported one line for a million calls.
uint64_t ult_report(size_t index) {
    const uint64_t n = g_calls[index].fetch_add(1, std::memory_order_relaxed) + 1;
    const bool success = g_return_success.load(std::memory_order_relaxed);
    const uint64_t ret = success ? 0ull : kUltNotImplemented;
    // Log at 1, 10, 100, 1e3, ... — bounded (a 1e6-call spin costs 7 lines) but never silent.
    // The threshold is a plain load/store rather than a CAS: guest threads racing here can at worst
    // emit the SAME milestone line twice, never suppress one. A duplicate diagnostic line is
    // harmless; a lock taken to avoid it would sit on a path the guest hits a million times.
    uint64_t due = g_next_report[index].load(std::memory_order_relaxed);
    if (n >= due) {
        uint64_t next = due ? due * 10 : 10;
        while (next <= n && next <= (uint64_t)1e18) next *= 10;
        g_next_report[index].store(next, std::memory_order_relaxed);
        print_banner();
        // The magnitude has to be readable at a glance. A skimmed log must show "this is being
        // hammered a million times", not a bare count the reader has to notice and compare: on
        // Earthion the deduped generic path reported ONE line for 1,005,742 sceUltMutexLock calls.
        const char* scale = n >= 1000000 ? "  *** >=1,000,000 CALLS — the guest is SPINNING on an "
                                           "unimplemented user-level lock ***"
                          : n >= 1000    ? "  *** >=1,000 CALLS — hot path, not a one-off ***"
                                         : "";
        fprintf(stderr, "[prosper] libSceUlt UNIMPLEMENTED: %s (%s) call #%llu -> returning 0x%llx%s%s\n",
                kUlt[index].name, kUlt[index].nid, (unsigned long long)n,
                (unsigned long long)ret,
                success ? "  [PROSPER_ULT_RETURN_SUCCESS: FAKE SUCCESS, guest is being lied to]"
                        : " (SCE_KERNEL_ERROR_ENOSYS)",
                scale);
    }
    return ret;
}

// One distinct host function per NID so the handler knows which entry point was called.
//
// The guest arguments are read by NOTHING here, and — the deliberate part — no guest out-param is
// ever written. That decision is load-bearing for the two size queries in the table,
// `sceUltUlthreadRuntimeGetWorkAreaSize` and `sceUltWaitingQueueResourcePoolGetWorkAreaSize`, which
// exist to fill a `size_t*` the guest then allocates from. prosper has no evidence of what those
// sizes should be, and writing a plausible-looking number would be exactly the #544/#660 defect in
// a new costume: an AGC "success" stub that left its required-memory output unwritten made Dead
// Cells allocate multiple gigabytes. Returning a real error and touching nothing is the honest
// pairing — a guest that checks the return sees failure and must not read the buffer; a guest that
// ignores the failure and consumes uninitialised memory is now displaying ITS OWN bug, on a call
// that also just logged loudly, instead of silently inheriting a fabricated value from us.
// CONFIDENCE: HIGH (writing an invented size is strictly worse than writing nothing).
template <size_t Index>
PROSPER_SYSV_ABI uint64_t ult_stub(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t) {
    return ult_report(Index);
}

template <size_t... Index>
void register_all(std::index_sequence<Index...>) {
    (Hle::register_fn(kUlt[Index].nid, (HleFn)ult_stub<Index>, kUlt[Index].name), ...);
}

}  // namespace

void register_ult_hle() {
    g_return_success.store(env_return_success(), std::memory_order_relaxed);
    register_all(std::make_index_sequence<kUltCount>{});
}

uint64_t ult_call_count(const char* nid) {
    if (!nid) return 0;
    for (size_t i = 0; i < kUltCount; ++i)
        if (std::strcmp(nid, kUlt[i].nid) == 0) return g_calls[i].load(std::memory_order_relaxed);
    return 0;
}

bool ult_set_return_success_for_test(bool return_success) {
    return g_return_success.exchange(return_success, std::memory_order_relaxed);
}

void ult_reset_counts_for_test() {
    for (size_t i = 0; i < kUltCount; ++i) {
        g_calls[i].store(0, std::memory_order_relaxed);
        g_next_report[i].store(0, std::memory_order_relaxed);
    }
    g_banner_printed.store(false, std::memory_order_relaxed);
}

void ult_dump_call_log(FILE* f) {
    uint64_t total = 0;
    for (size_t i = 0; i < kUltCount; ++i) total += g_calls[i].load(std::memory_order_relaxed);
    if (!total) return;
    fprintf(f, "\n=== libSceUlt calls (UNIMPLEMENTED — user-level threading, #1603) ===\n");
    for (size_t i = 0; i < kUltCount; ++i) {
        const uint64_t n = g_calls[i].load(std::memory_order_relaxed);
        if (!n) continue;
        fprintf(f, "  %10llu x  %-24s %s\n", (unsigned long long)n, kUlt[i].nid, kUlt[i].name);
    }
    fprintf(f, "  (%llu total calls; ulthreads created by this title never ran and Ult-guarded\n"
               "   guest state was never synchronised)\n", (unsigned long long)total);
}

}  // namespace prosper
