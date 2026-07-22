// Regression test for #1154: the PROSPER_AMPRLOG arg dump (ampr_arglog in hle_kernel_mem.cpp) used to
// raw-dereference guest pointers `*(uint64_t*)p` for any p that was aligned and > 0xffff. GTA V passes
// baQO9ez2gL4 with a2=0x302200 — aligned, > 0xffff, but UNMAPPED — so the deref SIGSEGV'd and killed the
// run the diagnostic was meant to observe. The fix reads via process_vm_readv, which returns EFAULT
// instead of faulting. This test exercises that exact fault-safe read contract on the failure address
// class: a mapped 16-byte pair reads back correctly; an unmapped/aligned/>0xffff address reports
// unreadable WITHOUT crashing the process. A raw deref of the unmapped case would crash this test.
//
// POSIX-only: ampr_arglog lives in the `#if defined(__linux__) || defined(__APPLE__)` half of
// hle_kernel_mem.cpp, and process_vm_readv is the POSIX fault-safe primitive (Darwin via mach in
// posix_shim.hpp). The Windows half has no equivalent diagnostic.

#include "../src/host/posix_shim.hpp"   // process_vm_readv (Linux native / Darwin mach)

#include <sys/mman.h>
#include <sys/uio.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

// Mirrors the exact read the fix performs: 16 bytes (two qwords) at `p`, fault-safe. Returns true iff
// the whole pair was readable.
static bool try_read_pair(uint64_t p, uint64_t out[2]) {
    struct iovec l { out, sizeof(uint64_t) * 2 }, r { (void*)(uintptr_t)p, sizeof(uint64_t) * 2 };
    return process_vm_readv(getpid(), &l, 1, &r, 1, 0) == (ssize_t)(sizeof(uint64_t) * 2);
}

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

int main() {
    // 1) A genuinely mapped, aligned pair reads back its bytes (proves the read path still works).
    long pagesz = sysconf(_SC_PAGESIZE);
    void* page = mmap(nullptr, (size_t)pagesz, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(page != MAP_FAILED);
    if (page != MAP_FAILED) {
        uint64_t* words = (uint64_t*)page;
        words[0] = 0x1122334455667788ull;
        words[1] = 0x99aabbccddeeff00ull;
        uint64_t got[2] = {0, 0};
        CHECK(try_read_pair((uint64_t)(uintptr_t)page, got));
        CHECK(got[0] == 0x1122334455667788ull);
        CHECK(got[1] == 0x99aabbccddeeff00ull);

        // 2) After unmapping, the same aligned address is unreadable — reported, not a crash.
        munmap(page, (size_t)pagesz);
        uint64_t junk[2] = {0xdead, 0xbeef};
        CHECK(!try_read_pair((uint64_t)(uintptr_t)page, junk));
    }

    // 3) GTA V's exact failure address: aligned, > 0xffff, never mapped. The old raw deref crashed here;
    //    the fault-safe read must report it unreadable without faulting.
    const uint64_t kGtaUnmapped = 0x302200ull;
    CHECK((kGtaUnmapped & 7) == 0 && kGtaUnmapped > 0xffff);   // clears ampr_arglog's guards
    uint64_t out[2] = {1, 2};
    CHECK(!try_read_pair(kGtaUnmapped, out));

    if (failures == 0) std::printf("amprlog_safe_read: OK\n");
    return failures == 0 ? 0 : 1;
}
