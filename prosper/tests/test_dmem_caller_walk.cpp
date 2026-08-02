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

int main() {
    constexpr int kScan = 160;   // the walk's own scan width

    // 1. A frame deep in the real thread stack must permit the FULL scan. This is what proves the
    //    platform stack-bounds query returned a usable top: if it ever fails to identify the stack,
    //    the helper falls back to the end of the current page and this count drops below 160.
    volatile uint64_t here = 0;
    const int deep = prosper::dmem_caller_scan_slots_for_test(&here, kScan);
    check(deep == kScan, "deep thread-stack frame permits the full 160-slot scan");
    if (deep != kScan) printf("       got %d of %d slots\n", deep, kScan);

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
