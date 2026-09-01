// #3057 (companion to test_dmem_caller_attribution_alloc_dmem.cpp): drives
// sceKernelAllocateMainDirectMemory through the real NID dispatch table and requires the
// "[dmem-caller] caller-chain=... alloc_main_dmem ..." record k_alloc_main_dmem publishes -- see
// the companion file's header comment for why the assertion is agnostic to whether the chain
// resolves to "unknown" or a (possibly coincidental) module+offset: guest_va_in_module_code checks
// only an address RANGE, not whether anything real is mapped there, so on some runners an ordinary
// garbage stack word resolves into a fixed guest module window purely by chance. That is
// environmental, not part of the contract under test.
//
// sceKernelAllocateMainDirectMemory's attribution predates #3055 and is not the regression #3057
// is about -- but the issue asks for it too ("worth covering the alloc_main_dmem path in the same
// test so a regression that swaps the labels cannot pass either"), and this binary is what makes
// that reachable: it names its OWN first-seen chain (see the companion file's header comment for
// why the two allocators cannot share a process). If a future change ever passed the wrong literal
// to attribute_dmem_allocation here -- "alloc_dmem" instead of "alloc_main_dmem", or vice versa in
// the sibling binary -- this record would name the wrong allocator and the assertion below would
// catch it.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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

namespace {
// See the companion file (test_dmem_caller_attribution_alloc_dmem.cpp) for the full rationale.
// True when `output` contains a "[dmem-caller] caller-chain=<token> <api> len=0x..." record, where
// <token> is either "unknown" or a run of decimal digits -- whichever the stack scan happened to
// resolve is environmental, not part of the contract under test.
bool dmem_caller_attribution_present(const char* output, const char* api) {
    const char* const kPrefix = "[dmem-caller] caller-chain=";
    const char* pos = std::strstr(output, kPrefix);
    if (!pos) return false;
    pos += std::strlen(kPrefix);
    if (std::strncmp(pos, "unknown", 7) == 0) {
        pos += 7;
    } else {
        const char* digits_start = pos;
        while (*pos >= '0' && *pos <= '9') ++pos;
        if (pos == digits_start) return false;   // neither "unknown" nor a numeric id
    }
    if (*pos != ' ') return false;
    ++pos;
    const std::string expect = std::string(api) + " len=0x";
    return std::strncmp(pos, expect.c_str(), expect.size()) == 0;
}
} // namespace

int main() {
#ifdef _WIN32
    _putenv_s("PROSPER_DMEM_CALLER", "1");
#else
    setenv("PROSPER_DMEM_CALLER", "1", 1);
#endif

    register_builtin_hle();
    auto alloc_main = Hle::lookup(nid_hash("sceKernelAllocateMainDirectMemory"));
    if (!alloc_main) {
        std::fprintf(stderr, "sceKernelAllocateMainDirectMemory was not registered\n");
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

    // sceKernelAllocateMainDirectMemory(size_t len, size_t align, int memType, off_t* physOut) --
    // physOut is arg3 here, NOT arg5 (a different signature from AllocateDirectMemory). This
    // process's direct-memory pool starts empty, so a fresh 16 KiB-granular request must succeed.
    uint64_t phys = 0;
    const uint64_t rc = alloc_main(0x4000, 0x4000, 0, (uint64_t)(uintptr_t)&phys, 0, 0);

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
    const bool attributed = dmem_caller_attribution_present(output, "alloc_main_dmem");

    const bool ok = redirected && restored && allocated && attributed;
    if (!ok) {
        std::fprintf(stderr,
            "FAIL: rc=0x%llx phys=0x%llx restored=%d allocated=%d attributed=%d\n"
            "captured stderr (%zu bytes):\n%s\n",
            (unsigned long long)rc, (unsigned long long)phys, restored, allocated, attributed,
            bytes, output);
    } else {
        std::printf("OK: sceKernelAllocateMainDirectMemory published a [dmem-caller] record"
                    " naming alloc_main_dmem\n");
    }
    return ok ? 0 : 1;
}
