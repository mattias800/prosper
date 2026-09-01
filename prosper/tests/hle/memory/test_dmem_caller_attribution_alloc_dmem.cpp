// #3057: test_dmem_caller_chain.cpp's arms (added by #3055) cover
// format_dmem_caller_chain_definition -- the FORMATTER -- and never drive an allocation through the
// real dispatch table. Deleting the `attribute_dmem_allocation(...)` call that #3055 added to
// k_alloc_dmem's success path (hle_kernel_mem.cpp) removes the whole behaviour change -- the entire
// reason #3055/#2998 exist, publishing a caller chain from sceKernelAllocateDirectMemory, not only
// the Main variant -- and left that suite green, because nothing in it ever calls k_alloc_dmem.
//
// This test does. Following the recipe from #3057's issue body: set PROSPER_DMEM_CALLER=1 before
// the first dmem call (dmem_caller_log() caches its getenv() read in a function-local static, so
// ordering matters), redirect stderr to a capture stream, call sceKernelAllocateDirectMemory
// through the real NID dispatch table exactly as a guest does, and require the one-shot
// "[dmem-caller] caller-chain=unknown alloc_dmem ..." line that attribute_dmem_allocation prints
// even with no guest frames on the stack (a bare host stack has no return address inside a fixed
// guest module range, so the chain always interns as Unknown here -- see dmem_caller_chain.hpp).
//
// Without the fix this allocator publishes no record at all, so the line is never printed and the
// assertion below fails -- verified by mutation as part of this PR (delete the call site, rerun,
// confirm red; restore, rerun, confirm green).
//
// A DEDICATED BINARY, not an arm folded into tests/host/test_dmem.cpp or this repo's other dmem
// coverage: DmemCallerChainInterner's "Unknown" state is a ONE-SHOT PER PROCESS (see
// attribute_dmem_allocation in hle_kernel_mem.cpp -- it returns before printing anything once
// `correlation.first` is false). Any other dmem allocation earlier in the same process that also
// interns as Unknown -- including one made through sceKernelAllocateMainDirectMemory, covered by
// the companion test_dmem_caller_attribution_alloc_main_dmem binary -- would silently consume the
// one shot, and this assertion would find an empty capture even with the fix present (flagged in
// the issue's review follow-up comment). Keeping the two allocators in separate processes gives
// each its own first Unknown chain to observe.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef _WIN32
#include <io.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace prosper;

int main() {
    // Must happen before ANY dmem allocation in this process: dmem_caller_log() is
    // `static int v = getenv(...) ? 1 : 0;` -- read once, on the first call.
#ifdef _WIN32
    _putenv_s("PROSPER_DMEM_CALLER", "1");
#else
    setenv("PROSPER_DMEM_CALLER", "1", 1);
#endif

    register_builtin_hle();
    auto alloc = Hle::lookup(nid_hash("sceKernelAllocateDirectMemory"));
    if (!alloc) {
        std::fprintf(stderr, "sceKernelAllocateDirectMemory was not registered\n");
        return 1;
    }

    FILE* capture = std::tmpfile();
    if (!capture) {
        std::fprintf(stderr, "tmpfile() failed\n");
        return 1;
    }
    if (std::fflush(stderr) != 0) {
        std::fclose(capture);
        return 1;
    }

#ifdef _WIN32
    const int stderr_fd = _fileno(stderr);
    const int saved_stderr = _dup(stderr_fd);
    const bool redirected = saved_stderr >= 0 && _dup2(_fileno(capture), stderr_fd) == 0;
#else
    const int stderr_fd = fileno(stderr);
    const int saved_stderr = dup(stderr_fd);
    const bool redirected = saved_stderr >= 0 && dup2(fileno(capture), stderr_fd) >= 0;
#endif
    if (!redirected) {
        if (saved_stderr >= 0) {
#ifdef _WIN32
            _close(saved_stderr);
#else
            close(saved_stderr);
#endif
        }
        std::fclose(capture);
        std::fprintf(stderr, "stderr redirection failed\n");
        return 1;
    }

    // sceKernelAllocateDirectMemory(off_t start, off_t end, size_t len, size_t align, int memType,
    // off_t* physOut). A 16 KiB-granular, self-aligned request inside the pool's default window,
    // mirroring a request that succeeds in tests/host/test_dmem.cpp. This process's direct-memory
    // pool starts empty, so it must succeed.
    uint64_t phys = 0;
    constexpr uint64_t kBase = 0x10000ull;
    constexpr uint64_t kEnd = kBase + 16ull * 1024 * 1024 * 1024;
    const uint64_t rc = alloc(0, kEnd, 0x4000, 0x4000, 0, (uint64_t)(uintptr_t)&phys);

    std::fflush(stderr);
#ifdef _WIN32
    const bool restored = _dup2(saved_stderr, stderr_fd) == 0;
    _close(saved_stderr);
#else
    const bool restored = dup2(saved_stderr, stderr_fd) >= 0;
    close(saved_stderr);
#endif

    char output[1024]{};
    std::rewind(capture);
    const size_t bytes = std::fread(output, 1, sizeof(output) - 1, capture);
    std::fclose(capture);
    output[bytes] = '\0';

    const bool allocated = rc == 0 && phys != 0;
    // The exact label k_alloc_dmem passes to attribute_dmem_allocation. This is the call site
    // #3055 added and #3057 says is untested -- deleting it removes this line entirely.
    const bool attributed =
        std::strstr(output, "[dmem-caller] caller-chain=unknown alloc_dmem len=0x") != nullptr;

    const bool ok = redirected && restored && allocated && attributed;
    if (!ok) {
        std::fprintf(stderr,
            "FAIL: rc=0x%llx phys=0x%llx restored=%d allocated=%d attributed=%d\n"
            "captured stderr (%zu bytes):\n%s\n",
            (unsigned long long)rc, (unsigned long long)phys, restored, allocated, attributed,
            bytes, output);
    } else {
        std::printf("OK: sceKernelAllocateDirectMemory published a [dmem-caller] record naming"
                    " alloc_dmem\n");
    }
    return ok ? 0 : 1;
}
