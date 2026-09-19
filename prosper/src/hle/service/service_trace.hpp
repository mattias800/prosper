#pragma once

// Shared service-call helpers, lifted out of hle_service.cpp so the per-library files split
// out of it can reach them: guest-memory read/write for marshalling in and out of a Sony API, and
// the PROSPER_SVCLOG call trace they all log through.
// 
// The copy/write helpers are guest-memory access rather than anything service-specific, and this
// is one of FOURTEEN files that open-code process_vm_readv/ReadProcessMemory for the same job. This
// header is not that consolidation -- it is scoped to src/hle/service -- but it is where the service
// half of it would land.

// hle_service.cpp — HLE of PS5 system services (user, NP/online, mouse, app content,
// dialogs). Bring-up policy: openers return a valid positive handle; queries zero their
// output struct and report a sane "not signed in / no device" state and success, so the
// game gets consistent values instead of uninitialized memory.
// (Game-controller input — libScePad — moved to hle_pad.cpp with a real host backend.)
#include "hle/dispatch/dispatch.hpp"
#include "hle/service/hle_addcontent.hpp"
#include "hle/fs/save_paths.hpp"   // per-title save roots (#2734)
#include "hle/fs/save_param.hpp"   // the save's parameter block (#2786)
#include "hle/util/hle_json2.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/kernel/sce_errno.hpp"   // libkernel error encoding (libSceRandom reject arms)
#include "diagnostics/env_numeric.hpp"   // #3267: a typo must not unregister a default-ON NID family
#include "hle/dispatch/callback_fs.hpp"
#include "hle/input/ime_input.hpp"
#include "hle/service/platform_ui.hpp"
#include "hle/video/video_backend.hpp"   // sceAvPlayer -> host hardware-decode backend (#705)
#include "hle/video/h264_sps.hpp"        // SPS/VUI extraction for GetPictureInfo (#2898)
#include "gpu/texture/guest_texture_layout.hpp" // exact HLE-produced sampled-linear layouts
#include "host/platform/posix_shim.hpp"   // Darwin process_vm_readv shim + asm portability
#include "host/image/boot_program.hpp"  // guest_module_name: is a callback target guest code?
#include "host/platform/lifecycle.hpp"  // cooperative stop when the guest reports its own crash (#3119)
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <atomic>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <thread>
#include <deque>
#include <filesystem>
#include <mutex>
#include <new>          // std::bad_alloc — the guest file-replacement buffer is guest-sized (#1955)
#include <set>
#include <unordered_map>
#include <vector>
#include <string>
#ifdef _WIN32
#include <direct.h>     // _mkdir (SaveDataMemory persistence dir)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/stat.h>   // mkdir
#include <sys/uio.h>    // process_vm_readv: fault-contained diagnostic snapshots
#include <sys/random.h> // getentropy: the host CSPRNG behind sceRandomGetRandomNumber
#include <unistd.h>

// Closing 1 conditional(s) the copied preamble left open; the
// original closes them further down, outside any preamble region.
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
