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
// `returns_size` marks the entry points whose contract returns a VALUE (a byte count) rather than a
// STATUS. Those must never be handed an error sentinel — see kUltSizeUnknown below (#1618).
struct UltEntry { const char* nid; const char* name; bool returns_size; };
constexpr UltEntry kUlt[] = {
    { "jw9FkZBXo-g", "_sceUltUlthreadRuntimeCreate",                  false },
    { "grs2pbc2awM", "sceUltUlthreadRuntimeGetWorkAreaSize",          true  },
    { "znI3q8S7KQ4", "_sceUltUlthreadCreate",                         false },
    { "gCeAI57LGgI", "sceUltUlthreadJoin",                            false },
    { "mmt8Sa6tL6c", "_sceUltMutexCreate",                            false },
    { "8hEGkR1pfr8", "sceUltMutexLock",                               false },
    { "h0XebKiMBtk", "sceUltMutexUnlock",                             false },
    { "jW+HnafeS3Y", "sceUltMutexDestroy",                            false },
    { "jnKaHGkrxZ4", "_sceUltConditionVariableCreate",                false },
    { "5xGAHCxA8M0", "sceUltConditionVariableWait",                   false },
    { "JTw1cAVkuc0", "sceUltConditionVariableSignal",                 false },
    { "xrmmI832R4U", "sceUltConditionVariableDestroy",                false },
    { "YiHujOG9vXY", "_sceUltWaitingQueueResourcePoolCreate",         false },
    { "WIWV1Qd7PFU", "sceUltWaitingQueueResourcePoolGetWorkAreaSize", true  },
    { "hZIg1EWGsHM", "sceUltInitialize",                              false },
    { "d-kSG2fLrvI", "sceUltFinalize",                                false },
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

// ...but ONLY for entry points whose contract returns a STATUS. Two of the sixteen do not (#1618).
//
// `sceUltUlthreadRuntimeGetWorkAreaSize` and `sceUltWaitingQueueResourcePoolGetWorkAreaSize` return a
// `size_t` AS THEIR RETURN VALUE. #1603 asserted they were size queries with an out-parameter; the
// guest's own call site disproves that — there is no out-param, and the returned value is handed
// straight to malloc:
//
//   9c55:  call sceUltWaitingQueueResourcePoolGetWorkAreaSize
//   9c5a:  mov  QWORD PTR [rbp-0x10],rax     <- the size IS the return value
//   9c62:  call malloc                       <- fed directly to malloc
//
// A signature whose return type is a size HAS NO ERROR CHANNEL, so any value returned is read as data.
// Returning SCE_KERNEL_ERROR_ENOSYS from them therefore asked the guest for a 0x8002004E-byte
// (2.0 GiB) allocation — twice — which is the #544/#660 failure class this file was written to prevent.
//
// It was also strictly worse than not registering the NIDs at all: the dispatcher's unresolved-import
// default is `return 0` (dispatch.cpp prosper_on_unimpl), so before registration the guest got 0,
// called malloc(0), received a valid small pointer and carried on harmlessly.
//
// The rule this file now follows: an unimplemented entry point may only return an error sentinel when
// its contract returns a STATUS. When the contract returns a VALUE, the visibility belongs in the log
// line — never in the return. prosper consumes no work area yet, so its own requirement is genuinely
// zero bytes and 0 is the honest answer as well as the safe one.
// CONFIDENCE: HIGH (the call site is unambiguous, and 0 restores the pre-registration behaviour).
constexpr uint64_t kUltSizeUnknown = 0ull;

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
    // A size-returning contract has no error channel: the sentinel would be read as a byte count and
    // handed to malloc (#1618). Such an entry point reports through the log line only.
    const uint64_t ret = kUlt[index].returns_size ? kUltSizeUnknown
                       : success                  ? 0ull
                                                  : kUltNotImplemented;
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
        const char* policy = kUlt[index].returns_size
                                 ? " (a SIZE, not a status — this contract has no error channel, #1618)"
                             : success
                                 ? "  [PROSPER_ULT_RETURN_SUCCESS: FAKE SUCCESS, guest is being lied to]"
                                 : " (SCE_KERNEL_ERROR_ENOSYS)";
        fprintf(stderr, "[prosper] libSceUlt UNIMPLEMENTED: %s (%s) call #%llu -> returning 0x%llx%s%s\n",
                kUlt[index].name, kUlt[index].nid, (unsigned long long)n,
                (unsigned long long)ret, policy, scale);
    }
    return ret;
}

// One distinct host function per NID so the handler knows which entry point was called.
//
// No guest memory is read or written here. #1603 justified that partly on the belief that the two
// size queries fill a `size_t*` out-parameter; they do not — they return the size in rax and the
// guest passes it straight to malloc, which is what made an error sentinel actively harmful (#1618).
// The remaining reason still holds and is the real one: prosper has no evidence of what those sizes
// should be, and writing (or returning) a plausible-looking number would be the #544/#660 defect in a
// new costume — an AGC "success" stub whose unwritten required-memory output made Dead Cells allocate
// multiple gigabytes. Zero is the honest answer for as long as prosper consumes no work area, and it
// is also the value the dispatcher's unresolved-import path would have produced.
// CONFIDENCE: HIGH (inventing a size is strictly worse than reporting the one prosper actually needs).
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
