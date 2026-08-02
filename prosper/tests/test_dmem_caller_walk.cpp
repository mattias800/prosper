// #1755: PROSPER_DMEM_CALLER's stack walk must never read past the end of the mapped region its
// frame lives in. The walk scans up to 160 slots ABOVE the frame, so a frame near the top of its
// thread stack used to run straight off the last mapped page. The resulting SIGSEGV presents as a
// guest fault, which sends the reader hunting a phantom bug in the title rather than at the
// diagnostic that actually crashed.
//
// The decisive case is a frame placed a few slots below a PROT_NONE guard page: before the fix the
// walk reads through the guard and the process dies by SIGSEGV; after it, the scan clamps to the
// slots remaining in the mapped page. That half runs in a forked child so a regression reports as
// a failed test instead of taking the test binary down with it.

#include <cstdint>
#include <cstdio>

#if defined(_WIN32)
int main() { printf("skip: POSIX-only (needs fork + mprotect)\nOK\n"); return 0; }
#else
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

namespace prosper {
int dmem_caller_scan_slots_for_test(const volatile uint64_t* frame, int want);
}

static int failures = 0;
static void check(bool ok, const char* name) {
    printf("%s: %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) failures++;
}

constexpr int kScan = 160;   // the walk's own scan width

// A frame in main() is NOT deep: argv, env and auxv sit above it, so it lands roughly 1 KB below
// the top of the stack and ASLR varies that gap run to run. Measuring there asserted headroom the
// platform does not guarantee and flaked 2 runs in 10, reporting e.g. "got 149 of 160" — the clamp
// was right and the assertion was wrong. Recurse a few padded frames so the measuring frame is
// unambiguously deep. noinline + volatile keep the frames from being merged away.
__attribute__((noinline))
static int deep_scan(int depth, int want) {
    volatile uint64_t pad[64];          // 512 B of frame, per level
    pad[0] = (uint64_t)depth;
    if (depth > 0) {
        const int r = deep_scan(depth - 1, want);
        pad[63] = (uint64_t)r;          // keep pad live across the call
        return r;
    }
    volatile uint64_t here = 0;
    return prosper::dmem_caller_scan_slots_for_test(&here, want);
}

int main() {
    // 1. A frame deep in the real thread stack permits the caller's full scan width.
    const int deep = deep_scan(32, kScan);     // ~16 KB below main's frame
    check(deep == kScan, "deep thread-stack frame permits the full 160-slot scan");
    if (deep != kScan) printf("       got %d of %d slots\n", deep, kScan);

    // 1b. The above does NOT by itself prove the platform stack-bounds query ran: the page
    //     fallback also clears 160 slots (1,280 B) whenever the frame happens to sit early in its
    //     page, which is most of the time. Asking for MORE THAN A PAGE separates them for certain —
    //     the fallback tops out at 512 slots (4,096 / 8) and can never satisfy 1,024, while a real
    //     stack top tens of KB above trivially can. Without this, losing the query is a coin flip
    //     the suite would pass roughly two runs in three.
    constexpr int kOverPage = 1024;            // 8 KB > any 4 KB page fallback
    const int wide = deep_scan(32, kOverPage);
    check(wide == kOverPage, "a scan wider than a page proves the stack-bounds query ran");
    if (wide != kOverPage) printf("       got %d of %d slots (<=512 means the page fallback)\n",
                                  wide, kOverPage);

    // 2. A frame 4 slots below a PROT_NONE page must clamp to those 4 rather than read the guard.
    const long pg = sysconf(_SC_PAGESIZE);
    const pid_t pid = fork();
    if (pid == 0) {
        char* p = (char*)mmap(nullptr, 2 * pg, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) _exit(3);
        if (mprotect(p + pg, pg, PROT_NONE) != 0) _exit(4);
        auto* frame = (volatile uint64_t*)(p + pg - 4 * (long)sizeof(uint64_t));
        const int slots = prosper::dmem_caller_scan_slots_for_test(frame, kScan);
        // Perform the reads the walk itself would, BEFORE judging the count: an over-wide clamp
        // must fail by reproducing the real #1755 SIGSEGV on the guard page, not by tripping this
        // test's own arithmetic. The count check below is then only a backstop for a clamp that is
        // too wide yet still short of the guard.
        for (int i = 1; i < slots; i++) (void)frame[i];
        _exit(slots > 4 ? 5 : 0);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    check(WIFEXITED(st) && WEXITSTATUS(st) == 0,
          "walk beside a guard page clamps instead of faulting");
    if (WIFSIGNALED(st))
        printf("       child died by signal %d — this is the #1755 crash\n", WTERMSIG(st));
    else if (WIFEXITED(st) && WEXITSTATUS(st) != 0)
        printf("       child exit=%d (3=mmap 4=mprotect 5=clamp too wide)\n", WEXITSTATUS(st));

    printf("%s\n", failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}
#endif
