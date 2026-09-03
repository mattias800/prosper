// probe_win_varargs — run #3246's real code on a real Microsoft x64 host.
//
// NOT part of the build and NOT a ctest case: it is the reproduction recipe for the one claim the
// Linux suite cannot make, kept beside the change so the next person does not have to rebuild it.
//
// BE PRECISE ABOUT WHAT THIS SHARES WITH PRODUCTION, because "it runs the production code" would be
// an overclaim. It compiles the production TRANSLATION sources — `guest_varargs.cpp` and
// `sysv_ms_bridge.cpp`, the two files that decide argument classes, read the guest's System V list,
// write the Microsoft one and emit the stub bytes — and it exercises them through the real MinGW CRT
// on a real Microsoft x64 ABI. What it does NOT link is `hle_libc.cpp`: `guest_snprintf` below
// RE-SPELLS the guest-ABI shim's shape rather than being it, because linking the HLE would drag in
// the whole emulator. So the ABI, the CRT and the translation are the real ones; the four-line shim
// around them is a copy, and a change to the real shim's shape has to be mirrored here by hand.
//
//   distrobox enter ps5ys -- bash -lc '
//     x86_64-w64-mingw32-g++ -std=c++20 -O2 -static -I prosper/src \
//         -o /tmp/probe.exe prosper/tools/probe_win_varargs.cpp \
//         prosper/src/host/abi/guest_varargs.cpp prosper/src/host/abi/sysv_ms_bridge.cpp
//     WINEDEBUG=-all wine /tmp/probe.exe'
//
// Vary -O0/-O1/-O2/-O3/-Og/-Os: the guest-ABI shim shape has to ASSEMBLE at all of them, because
// MinGW cannot emit SEH unwind data for a sysv_abi frame and inlining decides whether one is needed.
#include "host/abi/guest_varargs.hpp"
#include "host/abi/sysv_ms_bridge.hpp"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <windows.h>

using namespace prosper::abi;

static int g_fail = 0;
static void check(const char* what, bool ok, const char* detail) {
    printf("%-52s %s%s%s\n", what, ok ? "PASS" : "FAIL", detail && *detail ? " -- " : "",
           detail ? detail : "");
    if (!ok) ++g_fail;
}

// --- the fix, exactly as hle_libc.cpp spells it ------------------------------------------------
__attribute__((noinline))
static int host_snprintf(char* buf, size_t n, const char* fmt, const SysvVaList& ap) noexcept {
    MsVarargCall call(fmt, ap, FormatGrammar::Printf);
    return vsnprintf(buf, n, call.format(), (va_list)(char*)call.va_list_image());
}

static __attribute__((sysv_abi)) int guest_snprintf(char* buf, size_t n, const char* fmt, ...) {
    __builtin_sysv_va_list ap;
    __builtin_sysv_va_start(ap, fmt);
    SysvVaList captured;
    memcpy(&captured, &ap, sizeof captured);
    __builtin_sysv_va_end(ap);
    return host_snprintf(buf, n, fmt, captured);
}

// --- the defect: an MS-ABI variadic reached through the signature-blind bridge -------------------
// This is what a Windows build did before #3246, byte for byte: the fixed integer shuffle, then a
// host variadic that reads its arguments by the Microsoft rules.
static char g_legacy_buf[256];
static uint64_t legacy_target(void* buf, uint64_t n, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    const int r = vsnprintf((char*)buf, (size_t)n, fmt, ap);
    va_end(ap);
    return (uint64_t)r;
}
static void legacy_checkpoint() {}

int main() {
    char buf[256];

    // 1. A mixed list, including arguments past both System V register files.
    guest_snprintf(buf, sizeof buf, "%s|%d|%.2f|%s|%.2f|%d|%.2f|%s|%.2f|%lld|%.2f|%s",
                   "A", 1, 1.5, "B", 2.5, 2, 3.5, "C", 4.5, (long long)7, 5.5, "D");
    check("mixed 12-argument call", strcmp(buf, "A|1|1.50|B|2.50|2|3.50|C|4.50|7|5.50|D") == 0, buf);

    // 2. Ten doubles: two of them past xmm0..xmm7, in the System V overflow area.
    guest_snprintf(buf, sizeof buf, "%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f",
                   0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5);
    check("ten doubles (two past the SSE file)",
          strcmp(buf, "0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5") == 0, buf);

    // 3. A format the model refuses formats its safe prefix and stops.
    guest_snprintf(buf, sizeof buf, "a=%d b=%f c=%1$s", 7, 6.5, "x");
    check("unmodellable format truncates safely", strcmp(buf, "a=7 b=6.500000 c=") == 0, buf);

    // 4. THE DEFECT, executed. Same call through the pre-#3246 stub: the bridge with no declared
    //    signature, which is what a variadic handler got. If this reproduced the correct string the
    //    probe would be proving nothing.
    uint8_t staged[kMaxBridgeBytes];
    BridgeParams params;
    params.handler = (uint64_t)(uintptr_t)&legacy_target;
    params.checkpoint = (uint64_t)(uintptr_t)&legacy_checkpoint;
    const size_t n = emit_sysv_to_ms_bridge(staged, params);
    auto* code = (uint8_t*)VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE,
                                        PAGE_EXECUTE_READWRITE);
    if (!code) { printf("VirtualAlloc failed\n"); return 2; }
    memcpy(code, staged, n);
    using LegacyGuest = __attribute__((sysv_abi)) uint64_t (*)(void*, uint64_t, const char*, ...);
    memset(g_legacy_buf, 0, sizeof g_legacy_buf);
    // Deliberately all-numeric. With a `%s` behind the float this call does not merely print the
    // wrong text — the displaced argument is read as a POINTER and the CRT dereferences it, which
    // faulted outright under wine at -O2 (`Unhandled page fault on read access to FFFFFFFFFFFFFFFF`)
    // and printed the stub's own machine code as a string at -O0. That is the real severity of the
    // defect, and it is also why the control here avoids reproducing it: a probe that crashes cannot
    // report its own result.
    ((LegacyGuest)code)(g_legacy_buf, sizeof g_legacy_buf, "%d|%.2f|%d", 1, 1.5, 2);
    // Assert the SHAPE of the corruption, not merely that it differs. `!=` would also be satisfied by
    // a truncated buffer or an empty one, i.e. by the probe being broken. What actually happens is
    // deterministic for the first two fields: the first integer still arrives (both conventions place
    // it identically), and the float is read from the integer register holding the NEXT argument, a
    // tiny denormal that prints as 0.00. Only the third field is stack-dependent.
    check("signature-blind stub does NOT deliver the float",
          strncmp(g_legacy_buf, "1|0.00|", 7) == 0, g_legacy_buf);

    printf(g_fail ? "\nprobe_win_varargs: %d failure(s)\n" : "\nprobe_win_varargs: all cases passed\n",
           g_fail);
    return g_fail != 0;
}
