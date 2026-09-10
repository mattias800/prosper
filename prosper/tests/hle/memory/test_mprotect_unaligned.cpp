// sceKernelMprotect must accept a range that is not guest-page aligned (#3498).
//
// Protection is page-granular on every platform prosper targets, so a sub-page request can only
// mean "the pages containing this span". prosper's own sceKernelMtypeprotect has always read it
// that way; sceKernelMprotect did not, and passed the guest's address to the host verbatim -- where
// mprotect(2) rejects an unaligned base with EINVAL.
//
// The evidence that the console rounds is a shipping title. FINAL FANTASY TACTICS - The Ivalice
// Chronicles (PPSA21783) calls sceKernelMprotect(eboot+0xf999e0, 0x1da20, 0xc3) during allocator
// init. Refused, it returns -1 up four frames, its framework constructor yields NULL, main()
// returns, and the CRT raises sceKernelDebugRaiseExceptionOnReleaseMode(0xa0020001) about a second
// into the boot. The title runs on hardware, so the hardware accepted that call.
//
// WHY THE ARMS ARE SHAPED THIS WAY. "Returns 0" is not the property under test -- `return 0;`
// satisfies it, and so does any implementation that answers success without protecting anything.
// What is asserted is that the PAGES CHANGED, read back from the kernel's own view in
// /proc/self/maps, and that the pages OUTSIDE the normalized span did NOT. Those two together
// separate the fix from both trivial mutations: answering blind, and widening to the whole mapping.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); fails++; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

namespace {
constexpr uint64_t kGuestPage = 0x4000ull;      // the PS5's 16 KiB page, as hle_kernel_mem.cpp has it
constexpr uint64_t kSceReadWrite = 0x3ull;      // SCE_KERNEL_PROT_CPU_READ | _CPU_WRITE
constexpr uint64_t kSceRead      = 0x1ull;      // SCE_KERNEL_PROT_CPU_READ

#if defined(__linux__)
// Is every host page in [addr, addr+len) writable, according to the KERNEL rather than according to
// the call that was supposed to make it so? Reading back from /proc/self/maps is the point: a
// handler that returns 0 without calling mprotect cannot fake this.
bool range_is_writable(uint64_t addr, uint64_t len, bool& known) {
    known = false;
    std::FILE* f = std::fopen("/proc/self/maps", "r");
    if (!f) return false;
    char line[512];
    uint64_t covered = 0;
    bool all_writable = true;
    while (std::fgets(line, sizeof line, f)) {
        unsigned long long lo = 0, hi = 0;
        char perms[8] = {0};
        if (std::sscanf(line, "%llx-%llx %7s", &lo, &hi, perms) != 3) continue;
        if (hi <= addr || lo >= addr + len) continue;
        const uint64_t overlap_lo = lo > addr ? lo : addr;
        const uint64_t overlap_hi = hi < addr + len ? hi : addr + len;
        covered += overlap_hi - overlap_lo;
        if (perms[1] != 'w') all_writable = false;
    }
    std::fclose(f);
    known = (covered == len);        // an uncovered byte means the answer is unknown, not "no"
    return all_writable;
}
#endif
}  // namespace

int main() {
    std::printf("== test_mprotect_unaligned ==\n");
    register_builtin_hle();

    // Resolved through nid_hash, the same way the registration itself names it, so the test cannot
    // drift from the handler by transcribing a NID.
    HleFn mprot = Hle::lookup(nid_hash("sceKernelMprotect"));
    CHECK(mprot != nullptr, "sceKernelMprotect is registered");
    if (!mprot) { std::printf("FAILED\n"); return 1; }

#if defined(__linux__)
    // Four guest pages, read-only to start, with a fifth kept read-only as the untouched neighbour
    // the widening mutation would trample.
    const size_t span = (size_t)(kGuestPage * 8);
    void* raw = mmap(nullptr, span, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(raw != MAP_FAILED, "test fixture: reserved a read-only span");
    if (raw == MAP_FAILED) { std::printf("FAILED\n"); return 1; }
    // Align the working base up to a guest page so the offsets below are the only misalignment.
    const uint64_t base = ((uint64_t)(uintptr_t)raw + kGuestPage - 1) & ~(kGuestPage - 1);

    // The shape FINAL FANTASY TACTICS uses: a base that is not page aligned and a length that is
    // not a multiple of the page size, together spanning two pages.
    const uint64_t unaligned_addr = base + 0x9e0;
    const uint64_t unaligned_len  = kGuestPage + 0x240;      // crosses into the second page
    const uint64_t expected_lo    = base;                    // rounded down
    const uint64_t expected_hi    = base + 2 * kGuestPage;   // rounded up

    const uint64_t rc = mprot(unaligned_addr, unaligned_len, kSceReadWrite, 0, 0, 0);
    CHECK(rc == 0, "an unaligned sceKernelMprotect SUCCEEDS (pre-fix it returned EINVAL 0x80020016)");

    bool known = false;
    const bool writable = range_is_writable(expected_lo, expected_hi - expected_lo, known);
    CHECK(known, "the containing pages are described by /proc/self/maps");
    CHECK(known && writable,
          "the CONTAINING pages really became writable -- not merely reported as protected");

    // The widening mutation: rounding to the whole mapping instead of to the requested pages would
    // make this neighbour writable too.
    bool neighbour_known = false;
    const bool neighbour_writable =
        range_is_writable(expected_hi + kGuestPage, kGuestPage, neighbour_known);
    CHECK(neighbour_known, "the neighbour page is described by /proc/self/maps");
    CHECK(neighbour_known && !neighbour_writable,
          "a page OUTSIDE the requested span is left alone (rounding is to the span, not the mapping)");

    // Rounding must not turn a genuinely bad range into a success. An address the process has never
    // mapped still fails, and a null address is still EINVAL.
    CHECK(mprot(0, kGuestPage, kSceReadWrite, 0, 0, 0) == 0x80020016ull,
          "a null address is still EINVAL");
    const uint64_t unmapped = 0x7ffff0000000ull;   // reserved-looking, and not mapped by this test
    const uint64_t bad = mprot(unmapped + 0x37, kGuestPage, kSceRead, 0, 0, 0);
    CHECK(bad != 0, "an unmapped range still FAILS after rounding (rounding is not permission)");

    munmap(raw, span);
#else
    std::printf("  [skip] the page-permission readback is /proc/self/maps-based (Linux only); the "
                "normalization itself is shared with the Windows half\n");
#endif

    std::printf("%s\n", fails ? "FAILED" : "PASSED");
    return fails ? 1 : 0;
}
