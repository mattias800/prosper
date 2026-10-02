// test_printf — guards the variadic libc formatters (hle_libc.cpp, issue #122). The old handlers
// forwarded only 3 integer registers, so %f (XMM), stack-spilled args, and %s past the 4th argument
// produced garbage or crashed. The fix makes them real C variadic functions. This looks up the
// registered function pointer and calls it through its TRUE variadic signature (same SysV ABI the
// guest uses via the tail-jump stub), exercising the register + XMM + stack capture end to end.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "host/abi/sysv_ms_bridge.hpp"
#include "../fixtures/test_scratch.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#if defined(_WIN32)
#include <windows.h>
#include <io.h>
#else
#include <sys/mman.h>
#endif

using namespace prosper;

static int fails = 0;
static int checks = 0;
#define CHECK(c, m) do { ++checks; if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// PROSPER_GUEST_ABI is what makes the comment above TRUE rather than aspirational. These handlers
// are compiled in the GUEST's convention (#3246), which on Windows is not the host's -- so an
// untagged pointer type would place the arguments by Microsoft x64 rules for a callee reading them
// by System V, and the first `%s` would dereference whatever landed in `rdi`. It did: this test
// SEGFAULTed on the Windows MinGW job the moment the handlers were tagged.
//
// This file used to spell those tagged types itself. It no longer does: `Hle::lookup_guest_abi`
// constructs them (#3272), so the convention is spelled in one place instead of at every call site
// that wants one of these handlers.

struct TailJumps {
    std::array<void*, 3> pages{};
    ~TailJumps() {
        for (void* page : pages) if (page) {
#if defined(_WIN32)
            VirtualFree(page, 0, MEM_RELEASE);
#else
            munmap(page, 4096);
#endif
        }
    }
    template<class F> F add(unsigned index, F handler) {
        uint8_t bytes[prosper::abi::kMaxBridgeBytes];
        prosper::abi::BridgeParams parameters;
        parameters.handler = reinterpret_cast<uintptr_t>(handler);
        parameters.guest_abi = true;
        const size_t size = prosper::abi::emit_sysv_to_ms_bridge(bytes, parameters);
        if (size > sizeof bytes) return nullptr;
#if defined(_WIN32)
        pages[index] = VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
#else
        pages[index] = mmap(nullptr, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (pages[index] == MAP_FAILED) pages[index] = nullptr;
#endif
        if (!pages[index]) return nullptr;
        memcpy(pages[index], bytes, size);
#if defined(_WIN32)
        if (!FlushInstructionCache(GetCurrentProcess(), pages[index], size)) return nullptr;
#endif
        return reinterpret_cast<F>(pages[index]);
    }
};

// Redirect only the observed printf call. All assertions run after stdout is restored.
struct StdoutCapture {
    FILE* file = nullptr;
    int saved = -1;
    StdoutCapture() {
        const auto path = prosper_test::test_scratch_file("printf-stub.txt");
        file = fopen(path.c_str(), "w+b");
        if (!file) return;
        fflush(stdout);
#if defined(_WIN32)
        saved = _dup(_fileno(stdout));
        if (saved >= 0 && _dup2(_fileno(file), _fileno(stdout)) != 0) restore();
#else
        saved = dup(fileno(stdout));
        if (saved >= 0 && dup2(fileno(file), fileno(stdout)) < 0) restore();
#endif
    }
    bool armed() const { return file && saved >= 0; }
    void restore() {
        if (saved < 0) return;
        fflush(stdout);
#if defined(_WIN32)
        _dup2(saved, _fileno(stdout)); _close(saved);
#else
        dup2(saved, fileno(stdout)); close(saved);
#endif
        saved = -1;
    }
    ~StdoutCapture() { restore(); if (file) fclose(file); }
};

int main() {
    printf("== test_printf ==\n");
    register_builtin_hle();

    // `lookup_guest_abi` builds the PROSPER_GUEST_ABI pointer type itself (#3272), so this file no
    // longer spells the calling convention and cannot get it wrong -- which matters because an
    // untagged type compiles and behaves identically everywhere except the MinGW job.
    auto snf  = Hle::lookup_guest_abi<int, char*, size_t, const char*>(nid_hash("snprintf"));
    auto spf  = Hle::lookup_guest_abi<int, char*, const char*>(nid_hash("sprintf"));
    auto snfs = Hle::lookup_guest_abi<int, char*, size_t, const char*>(nid_hash("snprintf_s"));
    CHECK(snf && spf && snfs, "printf-family fns registered");
    if (!(snf && spf && snfs)) { printf("== FAIL ==\n"); return 1; }

    char buf[256];

    // Mixed args that hit GP registers, XMM (the %f that was totally dropped), AND the stack:
    // GP varargs: "val"(rcx), 42(r8), 0xabcd(r9), 5(stack), 6(stack) — 5 GP args spill past 3 regs.
    // FP varargs: 3.25(xmm0). This is the exact shape the old register-only forward mangled.
    int r = snf(buf, sizeof buf, "%s=%d f=%.2f x=%x a5=%d a6=%lld",
                "val", 42, 3.25, 0xabcd, 5, (long long)6);
    CHECK(r > 0, "snprintf returned a positive length");
    CHECK(strcmp(buf, "val=42 f=3.25 x=abcd a5=5 a6=6") == 0, "snprintf formats %s/%d/%f/%x + stack args");

    // Float-heavy: multiple XMM args must all land correctly.
    snf(buf, sizeof buf, "%.1f %.1f %.1f %.1f", 1.5, 2.5, 3.5, 4.5);
    CHECK(strcmp(buf, "1.5 2.5 3.5 4.5") == 0, "snprintf handles multiple %f (XMM) args");

    // %s past the 4th argument (was garbage under the register-only forward).
    snf(buf, sizeof buf, "%s %s %s %s %s", "a", "b", "c", "d", "e");
    CHECK(strcmp(buf, "a b c d e") == 0, "snprintf handles %s past the 4th argument");

    // snprintf honors the size bound (truncates, returns the would-be length).
    char small[8];
    int wr = snf(small, sizeof small, "%s", "0123456789");
    CHECK(wr == 10 && strcmp(small, "0123456") == 0, "snprintf truncates to the buffer bound");

    // snprintf_s maps to the same variadic path.
    snfs(buf, sizeof buf, "%d-%.1f", 7, 2.0);
    CHECK(strcmp(buf, "7-2.0") == 0, "snprintf_s formats via the variadic path");

    // sprintf (unbounded) into a wide buffer.
    spf(buf, "%s#%d/%.3f", "k", 9, 0.125);
    CHECK(strcmp(buf, "k#9/0.125") == 0, "sprintf formats mixed args");

    // These execute the actual emitted import tail-jump, not a hand-spelled wrapper.
    // Ten GP and ten FP arguments spill past BOTH SysV register files for every variant.
    auto pf = Hle::lookup_guest_abi<int, const char*>(nid_hash("printf"));
    CHECK(pf != nullptr, "printf registered with the true guest signature");
    if (!pf) return 1;
    TailJumps jumps;
    auto stub_sn = jumps.add(0, snf);
    auto stub_sp = jumps.add(1, spf);
    auto stub_pf = jumps.add(2, pf);
    CHECK(stub_sn && stub_sp && stub_pf, "all three production tail-jump stubs installed");
    if (!(stub_sn && stub_sp && stub_pf)) return 1;
    const char* fmt = "%d/%.1f|%d/%.1f|%d/%.1f|%d/%.1f|%d/%.1f|%d/%.1f|%d/%.1f|%d/%.1f|%d/%.1f|%d/%.1f";
    const char* expected = "1/0.5|2/1.5|3/2.5|4/3.5|5/4.5|6/5.5|7/6.5|8/7.5|9/8.5|10/9.5";
#define MIXED_VALUES 1, 0.5, 2, 1.5, 3, 2.5, 4, 3.5, 5, 4.5, 6, 5.5, 7, 6.5, 8, 7.5, 9, 8.5, 10, 9.5
    r = stub_sn(buf, sizeof buf, fmt, MIXED_VALUES);
    CHECK(r == (int)strlen(expected), "snprintf stub returns exact mixed-overflow length");
    CHECK(strcmp(buf, expected) == 0, "snprintf stub delivers GP and FP overflow");
    r = stub_sp(buf, fmt, MIXED_VALUES);
    CHECK(r == (int)strlen(expected), "sprintf stub returns exact mixed-overflow length");
    CHECK(strcmp(buf, expected) == 0, "sprintf stub delivers GP and FP overflow");
    char observed[256]{};
    bool captured = false;
    {
        StdoutCapture capture;
        if (capture.armed()) {
            r = stub_pf(fmt, MIXED_VALUES);
            capture.restore();
            if (fseek(capture.file, 0, SEEK_SET) == 0) {
                const size_t bytes = fread(observed, 1, sizeof observed - 1, capture.file);
                captured = bytes == strlen(expected) && !ferror(capture.file);
            }
        }
    }
#undef MIXED_VALUES
    CHECK(captured, "printf stub output captured in process-private scratch");
    CHECK(r == (int)strlen(expected), "printf stub returns exact mixed-overflow length");
    CHECK(strcmp(observed, expected) == 0, "printf stub delivers GP and FP overflow");
    CHECK(checks == 18, "all direct and emitted-stub assertions executed");

    if (fails) { printf("== FAIL: %d check(s) ==\n", fails); return 1; }
    printf("== PASS: %d checks ==\n", checks);
    return 0;
}
