#pragma once

// Shared service-call helpers, lifted out of hle_service.cpp so the per-library files split
// out of it can reach them: guest-memory read/write for marshalling in and out of a Sony API, and
// the PROSPER_SVCLOG call trace they all log through.
// The copy/write helpers are guest-memory access rather than anything service-specific. Since #3734
// they forward to host/memory/guest_memory_copy.hpp, which owns the one bounds policy; the names
// stay so the per-library files still being split out of hle_service.cpp (#3735) do not all have to
// change at once. New code should call host::guest_read_exact & co. directly.

// The generated preamble was hle_service.cpp's own -- forty includes, for six helpers that need
// eight. Trimmed by hand to what these definitions actually name, because a header included by
// every per-library file is exactly where an over-broad preamble compounds.
#include "host/platform/posix_shim.hpp"   // Darwin process_vm_readv shim + asm portability
#include "host/memory/guest_memory_copy.hpp"   // the one guest<->host copy policy (#3734)
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
// All-or-nothing marshalling of a guest structure. (Before #3734 these also refused bytes == 0;
// every caller passes a nonzero sizeof or a length it has already checked, so the shared policy's
// "zero bytes trivially succeeds" is not observable from any of them.)
inline bool svc_copy_bytes(uint64_t src, void* dst, size_t bytes) {
    return host::guest_read_exact(src, dst, bytes);
}
inline bool svc_write_bytes(uint64_t dst, const void* src, size_t bytes) {
    return host::guest_write_exact(dst, src, bytes);
}
// Whole words of the readable prefix -- the call trace's best-effort dump. It used to have no
// overflow check at all; the shared prefix form clamps at the top of the address space.
inline size_t svc_copy_words(uint64_t src, uint64_t* dst, size_t words) {
    return host::guest_read_prefix(src, dst, words * sizeof(uint64_t)) / sizeof(uint64_t);
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
