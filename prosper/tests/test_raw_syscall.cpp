// raw_mmap_fixed_ro: the %fs-safe fixed read-only anonymous mapping used by the Linux
// fault_handler's PROSPER_NULL_PAGE backing (issue #1071). The handler runs on guest threads
// whose %fs base is guest TLS, so neither glibc mmap() nor glibc syscall() may be used there:
// both store errno through %fs on failure, which faults inside the signal handler and kills
// the process. This helper must return -errno in-register and leave host errno/TLS untouched.
//
// raw_fmt_len / raw_write_fmt (#2050): the fault-handler report sites format with snprintf and
// then write the result. snprintf returns the length it WOULD have written, so a truncating
// format used to hand raw_write a length past the end of the caller's stack buffer — an
// out-of-bounds read in the one code path that only runs when the process is already broken.
// The assertions below read what the PIPE received, never the helper's own return, so a helper
// that computed the right number and wrote the wrong one would still fail.
//
// raw_fmt_advance (#2161): the WRITE half of the same class. The same report sites accumulate
// those snprintf returns into a running cursor, so a truncating chain left the cursor past the
// end of the buffer and handed the NEXT call `sizeof b - n` (size_t: underflows to ~2^64) at
// `b + n` (already past the end) — an out-of-bounds write onto the signal stack. Those arms put
// the buffer against a PROT_NONE page so an overflowing append faults the test.
#include "../src/host/raw_syscall.hpp"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; \
    fprintf(stderr, "FAIL %s:%d: %s\n  ", __FILE__, __LINE__, #cond); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

// --- raw_fmt_len: pure arithmetic, so it is checked on every host ---------------------------
// The clamp contract: snprintf's return `n` against a destination of `cap` bytes yields the
// number of bytes that ACTUALLY landed there. Never more than cap-1, because on truncation
// snprintf writes cap-1 characters plus the terminating NUL.
static void check_fmt_len() {
    CHECK(prosper::raw_fmt_len(0, 16) == 0, "empty format");
    CHECK(prosper::raw_fmt_len(5, 16) == 5, "fits: must be exact, not clamped");
    CHECK(prosper::raw_fmt_len(15, 16) == 15, "largest untruncated length must survive intact");
    // n == cap is already truncation: snprintf needed cap+1 bytes counting the NUL.
    CHECK(prosper::raw_fmt_len(16, 16) == 15, "n == cap is truncation, not an exact fit");
    CHECK(prosper::raw_fmt_len(4000, 16) == 15, "would-be length far past the buffer");
    // Negative return = encoding error; nothing usable was written, and an unclamped value would
    // convert to a ~2^64 length.
    CHECK(prosper::raw_fmt_len(-1, 16) == 0, "encoding error must yield 0, not a huge size_t");
    CHECK(prosper::raw_fmt_len(-2147483647 - 1, 16) == 0, "INT_MIN must yield 0");
    CHECK(prosper::raw_fmt_len(0, 0) == 0, "cap 0 must not underflow to SIZE_MAX");
    CHECK(prosper::raw_fmt_len(9, 0) == 0, "cap 0 must not underflow to SIZE_MAX");
    CHECK(prosper::raw_fmt_len(0, 1) == 0, "cap 1 holds only the NUL");
    CHECK(prosper::raw_fmt_len(7, 1) == 0, "cap 1 holds only the NUL");
}

#if !defined(__linux__) || !defined(__x86_64__)
int main() {
    check_fmt_len();
    if (failures) { fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    printf("raw_syscall: raw_fmt_len checked; syscall paths skipped (Linux x86-64 only)\n");
    return 0;
}
#else

// ---------------------------------------------------------------------------------------------
// #2161: accumulated snprintf offsets underflow their remaining capacity — an OOB WRITE.
//
// A report site builds a line by accumulating snprintf() returns. snprintf returns the WOULD-BE
// length, so once any call truncates the cursor exceeds the buffer; the next call in the chain
// then gets `cap - cursor` (size_t: underflows to ~2^64) at `cap + cursor` (already past the
// end). These arms put the buffer flush against the end of a mapped page with the next page
// PROT_NONE, so "writes past the end" is a fault rather than an assertion about silent damage.
// ---------------------------------------------------------------------------------------------

static const size_t kPage = 4096;

// A `cap`-byte buffer whose LAST byte is the last byte of a mapped page; the next page is
// PROT_NONE. `buf[cap - 1]` is the last legal store, `buf + cap` is one-past-the-end (a pointer
// C permits forming) and every byte from there on faults.
static char* guard_backed(size_t cap, void** region) {
    char* r = (char*)mmap(nullptr, 2 * kPage, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (r == MAP_FAILED) return nullptr;
    if (mprotect(r + kPage, kPage, PROT_NONE) != 0) { munmap(r, 2 * kPage); return nullptr; }
    *region = r;
    return r + kPage - cap;
}

// The chain under test, lifted from the [mb3watch] guest-stack dump in
// src/host/exec_image_linux.cpp: no cursor guard at all, and its append embeds a variable-length
// %s (guest_module_name returns names up to 25 characters), so ten captured frames format to
// ~470 characters into 320 bytes. `cap` stands in for the `sizeof sb` of the real array; the
// arithmetic is otherwise identical, including `cap - (size_t)sn` — which is exactly what
// `sizeof sb - sn` computes, an int promoted into size_t.
static const char* kModName = "AkVorbisHwAccelerator.prx";   // 25 chars: the longest real one
static const unsigned long long kOff = 0x1122334455667788ull;  // 16 hex digits
static const int kFrames = 10;                                 // the site's `found < 10` cap

// MASTER'S SHAPE, verbatim — the red arm. Not a mutation written for this test: this is the
// accumulation as it stands in the fault handler today.
__attribute__((noinline)) static int chain_unclamped(char* sb, size_t cap) {
    int sn = snprintf(sb, cap, "[mb3watch]   guest-stack:");
    for (int i = 0; i < kFrames; i++)
        sn += snprintf(sb + sn, cap - (size_t)sn, " %s+0x%llx", kModName, kOff);
    sn += snprintf(sb + sn, cap - (size_t)sn, "\n");
    return sn;
}

// The same chain with the fix: every cursor advance goes through raw_fmt_advance.
__attribute__((noinline)) static int chain_clamped(char* sb, size_t cap) {
    int sn = prosper::raw_fmt_advance(0, snprintf(sb, cap, "[mb3watch]   guest-stack:"), cap);
    for (int i = 0; i < kFrames; i++)
        sn = prosper::raw_fmt_advance(
            sn, snprintf(sb + sn, cap - (size_t)sn, " %s+0x%llx", kModName, kOff), cap);
    sn = prosper::raw_fmt_advance(sn, snprintf(sb + sn, cap - (size_t)sn, "\n"), cap);
    return sn;
}

// Run `fn` in a child and report how it died. Returns the terminating signal, 0 for a clean
// exit, or -1 if the child could not be created/reaped.
static int signal_from_child(int (*fn)(char*, size_t), size_t cap) {
    fflush(nullptr);
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        void* region = nullptr;
        char* buf = guard_backed(cap, &region);
        if (!buf) _exit(70);
        int r = fn(buf, cap);
        _exit(r == 0 ? 71 : 0);       // survived: the OOB write did not happen
    }
    int st = 0;
    if (waitpid(pid, &st, 0) != pid) return -1;
    return WIFSIGNALED(st) ? WTERMSIG(st) : 0;
}

// Run `body` with fd 2 redirected into a pipe, and return what the pipe received. The assertions
// that matter are on `out.n` — the byte count the kernel actually accepted — so a regression that
// passes an over-long length is caught by the bytes that arrive, not by any value the helper
// hands back. fd 2 is restored and the write end closed before reading, so the reader always sees
// EOF and a regressed build cannot hang here.
struct Captured { size_t n; char buf[8192]; };
template <typename F>
static Captured capture_stderr(F body) {
    Captured out{};
    int fds[2];
    if (pipe(fds) != 0) { failures++; fprintf(stderr, "FAIL pipe: %s\n", strerror(errno)); return out; }
    const int saved = dup(2);
    dup2(fds[1], 2);
    body();
    fflush(nullptr);
    dup2(saved, 2);
    close(saved);
    close(fds[1]);                       // EOF for the reader
    ssize_t got;
    while (out.n < sizeof out.buf - 1 &&
           (got = read(fds[0], out.buf + out.n, sizeof out.buf - 1 - out.n)) > 0)
        out.n += (size_t)got;
    close(fds[0]);
    return out;
}

int main() {
    check_fmt_len();

    // --- success: map a fixed read-only zero page at a known-free high address -------------
    // Reserve an address range via the normal allocator, then release it and re-map it raw.
    void* probe = mmap(nullptr, 0x2000, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(probe != MAP_FAILED, "probe reservation failed: %s", strerror(errno));
    uint64_t addr = (uint64_t)probe;
    munmap(probe, 0x2000);

    errno = 999;                                   // sentinel: raw path must not touch errno
    long r = prosper::raw_mmap_fixed_ro(addr, 0x1000);
    CHECK(r == 0, "expected success mapping at 0x%llx, got %ld", (unsigned long long)addr, r);
    CHECK(errno == 999, "errno clobbered on success path: %d", errno);
    if (r == 0) {
        const volatile uint8_t* p = (const uint8_t*)addr;
        uint8_t acc = 0;
        for (int i = 0; i < 0x1000; i++) acc |= p[i];
        CHECK(acc == 0, "mapped page not zero-filled (acc=0x%x)", acc);
        munmap((void*)addr, 0x1000);
    }

    // --- failure: invalid length must report -EINVAL in-register, errno untouched ----------
    errno = 999;
    r = prosper::raw_mmap_fixed_ro(addr, 0);
    CHECK(r == -EINVAL, "expected -EINVAL for len=0, got %ld", r);
    CHECK(errno == 999, "errno clobbered on failure path: %d", errno);

    // --- failure: unaligned fixed address must report -EINVAL, errno untouched -------------
    errno = 999;
    r = prosper::raw_mmap_fixed_ro(addr + 1, 0x1000);
    CHECK(r == -EINVAL, "expected -EINVAL for unaligned addr, got %ld", r);
    CHECK(errno == 999, "errno clobbered on failure path: %d", errno);

    // --- raw_write: success delivers the bytes, returns the count, errno untouched ---------
    int fds[2];
    CHECK(pipe(fds) == 0, "pipe failed: %s", strerror(errno));
    errno = 999;
    r = prosper::raw_write(fds[1], "abc", 3);
    CHECK(errno == 999, "errno clobbered on raw_write success: %d", errno);
    CHECK(r == 3, "expected 3 bytes written, got %ld", r);
    if (r == 3) {                          // only read what was actually written — a blocking
        char rb[4] = {0};                  // read on an empty pipe would hang a regressed build
        CHECK(read(fds[0], rb, 3) == 3 && memcmp(rb, "abc", 3) == 0,
              "pipe did not deliver raw_write bytes (got '%s')", rb);
    }
    close(fds[0]); close(fds[1]);

    // --- raw_write: bad fd reports -EBADF in-register, errno untouched ---------------------
    errno = 999;
    r = prosper::raw_write(-1, "x", 1);
    CHECK(r == -EBADF, "expected -EBADF for bad fd, got %ld", r);
    CHECK(errno == 999, "errno clobbered on raw_write failure: %d", errno);

    // --- raw_write_fmt: the #2050 class, against a real page boundary ----------------------
    // Put a 32-byte buffer flush against the end of a mapped page and make the next page
    // PROT_NONE, so "one byte past the buffer" is genuinely unreadable rather than merely
    // uninteresting. Formatting 63 characters into it makes snprintf return 63 while only 31
    // landed; handing that 63 to a write length reads 31 bytes into the guard page.
    static const char kMark[] = "...[prosper: diagnostic line truncated]\n";
    const size_t kMarkLen = sizeof kMark - 1;
    const size_t pg = (size_t)sysconf(_SC_PAGESIZE);
    char* region = (char*)mmap(nullptr, pg * 2, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(region != MAP_FAILED, "guard-page reservation failed: %s", strerror(errno));
    if (region != MAP_FAILED) {
        CHECK(mprotect(region + pg, pg, PROT_NONE) == 0, "guard mprotect failed: %s", strerror(errno));
        const size_t cap = 32;
        char* b = region + pg - cap;              // last `cap` bytes of the readable page
        const char* long63 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcde";
        int n = snprintf(b, cap, "%s", long63);
        CHECK(n == 63, "premise: snprintf must report the would-be length, got %d", n);

        // Positive control for the APPARATUS, not for the fix: prove the guard page is armed and
        // that the unclamped length really does leave the buffer. Without this, "31 bytes arrived"
        // could be true for reasons that have nothing to do with the clamp. Linux may either fault
        // the whole write or deliver the readable prefix, so assert only that it cannot honour the
        // length it was given.
        long bad_ret = 0;
        Captured bad = capture_stderr([&] { bad_ret = prosper::raw_write(2, b, (uint64_t)n); });
        CHECK(bad_ret < 0 || bad.n < (size_t)n,
              "apparatus: the unclamped over-read should not have delivered %d bytes "
              "(ret=%ld, delivered=%zu) — the guard page is not doing its job", n, bad_ret, bad.n);

        // The fix: at most the buffer, and the loss is announced.
        Captured c = capture_stderr([&] { prosper::raw_write_fmt(2, b, cap, n); });
        CHECK(c.n == cap - 1 + kMarkLen,
              "truncated line: expected %zu content + %zu marker bytes, got %zu",
              cap - 1, kMarkLen, c.n);
        CHECK(c.n >= cap - 1 && memcmp(c.buf, long63, cap - 1) == 0,
              "truncated line: content is not the %zu characters snprintf actually wrote", cap - 1);
        CHECK(c.n == cap - 1 + kMarkLen && memcmp(c.buf + cap - 1, kMark, kMarkLen) == 0,
              "truncated line: truncation must be visible, not silent");

        // Untruncated: exact, and NO marker. This is the arm that kills a blanket "always clamp to
        // cap-1" or "always print the marker" — both would pass the truncation arm alone.
        int m = snprintf(b, cap, "%s", "short");
        CHECK(m == 5, "premise: short format, got %d", m);
        Captured s = capture_stderr([&] { prosper::raw_write_fmt(2, b, cap, m); });
        CHECK(s.n == 5 && memcmp(s.buf, "short", 5) == 0,
              "untruncated line must be delivered whole with no marker (got %zu bytes)", s.n);

        // The exact boundary, both sides. cap-1 characters fit; cap does not, because snprintf
        // needed one more byte for the NUL. An off-by-one clamp passes one of these, not both.
        char fill[cap];
        memset(fill, 'x', sizeof fill);
        int e = snprintf(b, cap, "%.*s", (int)cap - 1, fill);
        CHECK(e == (int)cap - 1, "premise: cap-1 characters must not truncate, got %d", e);
        Captured edge = capture_stderr([&] { prosper::raw_write_fmt(2, b, cap, e); });
        CHECK(edge.n == cap - 1, "largest untruncated line must be delivered whole, got %zu", edge.n);

        int o = snprintf(b, cap, "%.*s", (int)cap, fill);
        CHECK(o == (int)cap, "premise: cap characters must truncate, got %d", o);
        Captured over = capture_stderr([&] { prosper::raw_write_fmt(2, b, cap, o); });
        CHECK(over.n == cap - 1 + kMarkLen,
              "n == cap is truncation: expected %zu bytes, got %zu", cap - 1 + kMarkLen, over.n);

        // Encoding error: nothing usable was written, and an unclamped negative would have become
        // a ~2^64 length.
        Captured neg = capture_stderr([&] { prosper::raw_write_fmt(2, b, cap, -1); });
        CHECK(neg.n == kMarkLen && memcmp(neg.buf, kMark, kMarkLen) == 0,
              "negative return must write no content and flag the loss (got %zu bytes)", neg.n);

        munmap(region, pg * 2);
    }

    // === #2161: the accumulation clamp ========================================================
    {
        const size_t cap = 320;                    // the real `char sb[320]`

        // --- ARM 1 (RED, master's shape): the unclamped chain writes past the end -----------
        // Cursor walk: 25, then +45 per frame -> 295 after six. The seventh is handed size 25
        // for a 45-character append, so it truncates and returns 45: the cursor becomes 340,
        // twenty bytes into the guard page. The eighth iteration stores there. This arm is
        // master's code, not a mutation of it (the charter's positive-control rule) — it is the
        // reason the fix exists, and it must die.
        int sig = signal_from_child(chain_unclamped, cap);
        CHECK(sig == SIGSEGV,
              "unclamped accumulation did not fault: child signal %d (expected SIGSEGV=%d). "
              "If this stops faulting the guard page is not where the test thinks it is.",
              sig, SIGSEGV);
        // The control for arm 1, and it is deliberately NOT drawn from the same place as the
        // null: the identical unclamped chain over a buffer big enough for the whole line must
        // exit cleanly. Without it, "the child died" is also consistent with a broken harness —
        // a mismapped guard page, or a chain that faults for some reason other than capacity.
        CHECK(signal_from_child(chain_unclamped, 1024) == 0,
              "unclamped chain faulted even with room to spare — arm 1's fault is not the overrun");

        // --- ARM 2 (GREEN): the clamped chain survives the same input ------------------------
        CHECK(signal_from_child(chain_clamped, cap) == 0,
              "clamped accumulation faulted or exited non-zero");

        // --- ARM 3: what actually LANDED in the buffer, not what the helper returned ---------
        // Run it in-process against the same guard page and read the bytes back.
        void* guard_region = nullptr;
        char* buf = guard_backed(cap, &guard_region);
        CHECK(buf != nullptr, "guard-backed mapping failed: %s", strerror(errno));
        if (buf) {
            memset(buf, 0, cap);
            int sn = chain_clamped(buf, cap);

            // The untruncated line, built independently, is the oracle for the content.
            char ref[1024];
            int rn = snprintf(ref, sizeof ref, "[mb3watch]   guest-stack:");
            for (int i = 0; i < kFrames; i++)
                rn += snprintf(ref + rn, sizeof ref - (size_t)rn, " %s+0x%llx", kModName, kOff);
            CHECK((size_t)rn > cap, "oracle line (%d bytes) must exceed cap %zu or arm 3 is vacuous",
                  rn, cap);

            CHECK(strlen(buf) == cap - 1,
                  "truncated buffer should hold exactly cap-1 = %zu characters, holds %zu",
                  cap - 1, strlen(buf));
            CHECK(memcmp(buf, ref, cap - 1) == 0,
                  "truncated content is not the prefix of the full line");
            // The cursor SATURATES at cap — it does not stop at cap-1. That is the signal the
            // flush reads as "this line truncated" (raw_fmt_len/raw_write_fmt, #2050): clamping
            // to cap-1 instead would be memory-safe and SILENT.
            CHECK(sn == (int)cap, "cursor should saturate at cap=%zu, got %d", cap, sn);

            // --- ARM 4: saturation is sticky AND does not eat the last content byte ----------
            // This is what separates the fix from the obvious alternative. With a cap-1 clamp,
            // each further append gets size 1 and writes its NUL over buf[cap-1] — memory-safe,
            // but it destroys the last character of the report and still reports no truncation.
            // With the cap clamp the size is 0, so snprintf writes nothing at all.
            char before[cap];
            memcpy(before, buf, cap);
            for (int i = 0; i < 3; i++)
                sn = prosper::raw_fmt_advance(
                    sn, snprintf(buf + sn, cap - (size_t)sn, " %s+0x%llx", kModName, kOff), cap);
            CHECK(sn == (int)cap, "cursor must stay saturated after further appends, got %d", sn);
            CHECK(memcmp(before, buf, cap) == 0,
                  "appends after saturation modified the buffer — the report lost its tail");
            munmap(guard_region, 2 * kPage);
        }
    }
    {
        // --- ARM 5: a chain that FITS must not be clamped ------------------------------------
        // Without this, "always return cap-1" passes every arm above. The cursor has to be the
        // true byte count so a complete line is written complete and reports no truncation.
        const size_t cap = 64;
        void* guard_region = nullptr;
        char* buf = guard_backed(cap, &guard_region);
        CHECK(buf != nullptr, "guard-backed mapping failed: %s", strerror(errno));
        if (buf) {
            int n = prosper::raw_fmt_advance(0, snprintf(buf, cap, "[fits]"), cap);
            n = prosper::raw_fmt_advance(n, snprintf(buf + n, cap - (size_t)n, " a=%d", 7), cap);
            n = prosper::raw_fmt_advance(n, snprintf(buf + n, cap - (size_t)n, " b=%d\n", 9), cap);
            CHECK(n == (int)strlen(buf), "fitting chain: cursor %d != bytes written %zu",
                  n, strlen(buf));
            CHECK((size_t)n < cap, "fitting chain must not report truncation (n=%d cap=%zu)",
                  n, cap);
            CHECK(strcmp(buf, "[fits] a=7 b=9\n") == 0, "fitting chain content wrong: '%s'", buf);

            // --- ARM 6: the exact-fit boundary --------------------------------------------
            // cap-1 content bytes is a COMPLETE line (no truncation); one more is not. An
            // off-by-one in the saturation test (`>` for `>=`) fails exactly here.
            char fill[cap];
            memset(fill, 'x', sizeof fill); fill[cap - 1] = '\0';        // 63 characters
            int e = prosper::raw_fmt_advance(0, snprintf(buf, cap, "%s", fill), cap);
            CHECK(e == (int)cap - 1, "exact fit should give cap-1=%zu, got %d", cap - 1, e);
            CHECK((size_t)e < cap, "an exactly-filling line must not report truncation");
            e = prosper::raw_fmt_advance(e, snprintf(buf + e, cap - (size_t)e, "y"), cap);
            CHECK(e == (int)cap, "one byte past an exact fit must saturate, got %d", e);
            munmap(guard_region, 2 * kPage);
        }
    }
    // --- ARM 7: degenerate inputs are absorbed, never propagated -----------------------------
    // A negative snprintf return (encoding error, contents indeterminate) saturates: fail-visible
    // beats a line that silently drops an append. A zero cap has nowhere to write.
    CHECK(prosper::raw_fmt_advance(10, -1, 64) == 64, "negative snprintf return must saturate");
    CHECK(prosper::raw_fmt_advance(0, 5, 0) == 0, "cap 0 must stay at 0");
    CHECK(prosper::raw_fmt_advance(-5, 3, 64) == 3, "a negative cursor must be treated as 0");
    CHECK(prosper::raw_fmt_advance(64, 5, 64) == 64, "an already-saturated cursor must not move");

    if (failures) { fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    printf("raw_syscall: all checks passed\n");
    return 0;
}
#endif
