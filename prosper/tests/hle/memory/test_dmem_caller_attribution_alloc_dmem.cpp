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
// "[dmem-caller] caller-chain=..." line that attribute_dmem_allocation prints on the first
// allocation this process makes.
//
// Without the fix this allocator publishes no record at all, so the line is never printed and the
// assertion below fails -- verified by mutation as part of this PR (delete the call site, rerun,
// confirm red; restore, rerun, confirm green).
//
// WHAT THE ASSERTION CHECKS, AND WHY IT DOES NOT PIN THE "unknown" FORM (review correction, #3178):
// an earlier version of this test required the exact text "caller-chain=unknown alloc_dmem ...",
// reasoning that a bare host stack (no guest image is loaded here) has no return address inside a
// fixed guest module VA range, so the scan would always come up empty and the chain would always
// intern as Unknown. That reasoning was environment-dependent, not a property of the call site:
// guest_va_in_module_code (boot_program.hpp) tests only whether a VALUE FALLS IN A FIXED ADDRESS
// RANGE, never whether anything is actually mapped there. On CI's Linux runner an ordinary garbage
// word already sitting on the host stack landed inside the BOOT_SAVEDATA range by coincidence, so
// the scan resolved a (bogus) "SaveData.prx+0x10000002" and the chain interned as Known -- and the
// exact-"unknown" assertion failed even though attribution had genuinely happened. That is the same
// failure shape #3057 exists to fix, just relocated one level down: the arm was bound to "nothing on
// this machine's stack happens to look like a guest address" rather than to the call site.
// attribute_dmem_allocation's OWN comment says as much: the scan is a heuristic over live stack
// content, "a stale spill slot looks exactly like a return address". Forcing the Unknown state
// deterministically would mean controlling that same uninitialized memory, which is exactly the
// thing the scan is deliberately tolerant of and cannot be pinned by a portable test.
//
// So the check below is resolve-agnostic: it accepts EITHER "caller-chain=unknown" (no resolvable
// frame) OR "caller-chain=<digits>" (a frame -- real or coincidental -- resolved), and requires
// that whichever form printed, it is immediately followed by " alloc_dmem len=0x" -- i.e. that
// *some* attribution record was published, naming the right allocator. That is the actual contract
// of the call site under test; what the stack scan happens to resolve to is environmental.
//
// A DEDICATED BINARY, not an arm folded into tests/host/test_dmem.cpp or this repo's other dmem
// coverage: DmemCallerChainInterner's "first" bookkeeping (both the Unknown-announced-once state
// and the Known per-key first-seen state) is PER PROCESS (see attribute_dmem_allocation in
// hle_kernel_mem.cpp -- it returns before printing anything once `correlation.first` is false). Any
// earlier dmem allocation in the same process -- including one made through
// sceKernelAllocateMainDirectMemory, covered by the companion test_dmem_caller_attribution_alloc_main_dmem
// binary -- could consume that process's one shot for its own key and leave this assertion looking
// at an empty capture even with the fix present (flagged in the issue's review follow-up comment).
// Keeping the two allocators in separate processes gives each its own first observation.
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
// True when `output` contains a "[dmem-caller] caller-chain=<token> <api> len=0x..." record,
// where <token> is either "unknown" or a run of decimal digits (a resolved chain id). Deliberately
// agnostic to WHICH token appears -- that is environmental (see the file header comment) -- and
// deliberately anchored so the api label must immediately follow the token, not merely appear
// anywhere in the captured text.
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
    // #3055 added and #3057 says is untested -- deleting it removes this line entirely, regardless
    // of whether the chain resolves to "unknown" or a (possibly coincidental) module+offset.
    const bool attributed = dmem_caller_attribution_present(output, "alloc_dmem");

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
