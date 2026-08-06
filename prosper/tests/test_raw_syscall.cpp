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
#include "../src/host/raw_syscall.hpp"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
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

    if (failures) { fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    printf("raw_syscall: all checks passed\n");
    return 0;
}
#endif
