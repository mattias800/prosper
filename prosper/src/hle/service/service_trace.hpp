#pragma once

// Shared service-call helpers, lifted out of hle_service.cpp so the per-library files split
// out of it can reach them: guest-memory read/write for marshalling in and out of a Sony API, and
// the PROSPER_SVCLOG call trace they all log through.
// The copy/write helpers are guest-memory access rather than anything service-specific, and this
// is one of FOURTEEN files that open-code process_vm_readv/ReadProcessMemory for the same job. This
// header is not that consolidation -- it is scoped to src/hle/service -- but it is where the service
// half of it would land.

// The generated preamble was hle_service.cpp's own -- forty includes, for six helpers that need
// eight. Trimmed by hand to what these definitions actually name, because a header included by
// every per-library file is exactly where an over-broad preamble compounds.
#include "host/platform/posix_shim.hpp"   // Darwin process_vm_readv shim + asm portability
#include <cinttypes>    // PRIx64 in the call trace
#include <cstddef>      // size_t
#include <cstdint>      // uint64_t, UINT64_MAX
#include <cstdio>       // fprintf, stderr
#include <cstdlib>      // getenv
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>    // ReadProcessMemory / WriteProcessMemory
#else
#include <sys/uio.h>    // iovec, process_vm_readv/writev
#include <unistd.h>     // getpid
#endif

namespace prosper {

inline bool svclog() { static int v = getenv("PROSPER_SVCLOG") ? 1 : 0; return v; }
inline bool svc_ptrish(uint64_t v) { return v >= 0x10000 && v < 0x7fffffffffffull; }
inline bool svc_copy_bytes(uint64_t src, void* dst, size_t bytes) {
    if (!src || !dst || !bytes || src > UINT64_MAX - (bytes - 1)) return false;
#ifdef _WIN32
    SIZE_T copied = 0;
    return ReadProcessMemory(GetCurrentProcess(), (const void*)(uintptr_t)src, dst, bytes, &copied) &&
           copied == bytes;
#else
    iovec local{dst, bytes};
    iovec remote{(void*)(uintptr_t)src, bytes};
    return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == (ssize_t)bytes;
#endif
}
inline bool svc_write_bytes(uint64_t dst, const void* src, size_t bytes) {
    if (!dst || !src || !bytes || dst > UINT64_MAX - (bytes - 1)) return false;
#ifdef _WIN32
    SIZE_T copied = 0;
    return WriteProcessMemory(GetCurrentProcess(), (void*)(uintptr_t)dst, src, bytes, &copied) &&
           copied == bytes;
#else
    iovec local{const_cast<void*>(src), bytes};
    iovec remote{(void*)(uintptr_t)dst, bytes};
    return process_vm_writev(getpid(), &local, 1, &remote, 1, 0) == (ssize_t)bytes;
#endif
}
inline size_t svc_copy_words(uint64_t src, uint64_t* dst, size_t words) {
    const size_t bytes = words * sizeof(uint64_t);
#ifdef _WIN32
    SIZE_T copied = 0;
    ReadProcessMemory(GetCurrentProcess(), (const void*)(uintptr_t)src, dst, bytes, &copied);
    return (size_t)copied / sizeof(uint64_t);
#else
    iovec local{dst, bytes};
    iovec remote{(void*)(uintptr_t)src, bytes};
    const ssize_t copied = process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
    return copied > 0 ? (size_t)copied / sizeof(uint64_t) : 0;
#endif
}
inline void svc_log(const char* fn, uint64_t a0, uint64_t a1, uint64_t a2,
             uint64_t a3, uint64_t a4, uint64_t a5, int dump_words = 8) {
    if (!svclog()) return;
    fprintf(stderr, "[svc] %s(%#" PRIx64 ", %#" PRIx64 ", %#" PRIx64
                    ", %#" PRIx64 ", %#" PRIx64 ", %#" PRIx64 ")\n",
            fn, a0, a1, a2, a3, a4, a5);
    const uint64_t args[6] = { a0, a1, a2, a3, a4, a5 };
    for (int i = 0; i < 6; i++) {
        if (!svc_ptrish(args[i])) continue;
        // Never read across the arg's 4 KiB page end: an out-param can be a tiny heap block whose
        // page neighbor is unmapped, and a diagnostic must not be able to fault the boot.
        uint64_t page_left = 0x1000 - (args[i] & 0xfff);
        int words = (int)(page_left / 8); if (words > dump_words) words = dump_words;
        if (words > 8) words = 8;
        uint64_t q[8]{};
        words = (int)svc_copy_words(args[i], q, (size_t)words);
        if (!words) continue;
        fprintf(stderr, "[svc]   a%d ->", i);
        for (int w = 0; w < words; w++) fprintf(stderr, " %016" PRIx64, q[w]);
        fprintf(stderr, "\n");
    }
}

}  // namespace prosper
