// Positive control for memcpy_probe: a KNOWN number of copies from three distinguishable sites,
// checked rather than eyeballed. An empty or mis-bucketed census is indistinguishable from a
// working one without this -- the trap tools/getenv_probe's README already records.
//
// THE SIZES ARE RUNTIME VALUES ON PURPOSE. The first version of this control used constants, and
// gcc expanded the 64-byte copy inline: 300,000 calls simply never reached libc and the probe was
// "missing" a third of them. That is correct behaviour by both, and it is the caveat to carry --
// a small constant-size copy is invisible to this probe AND absent from perf's memmove samples,
// because no call happens. Only a copy that actually calls libc is in scope for either instrument.
#define _GNU_SOURCE   // dladdr, to refuse a self-measuring calibration
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>
#include <dlfcn.h>
#include <stdint.h>

#define A_CALLS 200000u
#define B_CALLS 100000u
#define C_CALLS  50000u
#define D_CALLS  70000u

static unsigned char* src;
static unsigned char* dst;
static volatile size_t a_bytes = 64, b_bytes = 4096, c_bytes = 65536, d_bytes = 8192;

__attribute__((noinline)) static void site_a(void) { memcpy(dst, src, a_bytes); }
__attribute__((noinline)) static void site_b(void) { memcpy(dst, src, b_bytes); }
__attribute__((noinline)) static void site_c(void) { memmove(dst, src, c_bytes); }
// A COMPARISON site. Without it "0 CMP rows" is indistinguishable from an interposer that never
// fires -- the same trap the byte counts above exist to close, one function over.
static volatile int sink;
__attribute__((noinline)) static void site_d(void) { sink = memcmp(dst, src, d_bytes); }

// CALIBRATE THE CYCLE COLUMN. Counts and bytes above are exact; cycles are not, and the column the
// README tells you to rank by is the inexact one. The probe brackets every call with two rdtsc, so
// each call carries a fixed instrument cost that is negligible against a megabyte and larger than
// the work itself against 64 bytes. Printing it is the only way a reader can tell which rows of a
// report are measurement and which are the program.
//
// Also measured here: the delta is ELAPSED tsc, so anything that deschedules the thread mid-copy is
// counted as copy time.
static unsigned char cal_a[1 << 22], cal_b[1 << 22];
static volatile size_t cal_size;

static double probed_cycles(size_t n, int iters) {
    cal_size = n;
    unsigned long long acc = 0;
    for (int i = 0; i < iters; i++) {
        const unsigned long long t0 = __rdtsc();
        memcpy(cal_b, cal_a, cal_size);
        const unsigned long long t1 = __rdtsc();
        acc += t1 - t0;
    }
    return (double)acc / iters;
}

static double amortised_cycles(size_t n, int iters) {
    cal_size = n;
    const unsigned long long t0 = __rdtsc();
    for (int i = 0; i < iters; i++) memcpy(cal_b, cal_a, cal_size);
    const unsigned long long t1 = __rdtsc();
    return (double)(t1 - t0) / iters;
}

static void calibrate(void) {
    // Run this WITHOUT the preload. Under it, both arms below are themselves interposed, so the
    // comparison measures the residual cost of one more rdtsc pair on top of an already-bracketed
    // call -- which comes out near zero or negative and looks like "no overhead". That reading is
    // an artifact of measuring the instrument with itself, so refuse rather than print it.
    // `&memcpy` is this binary's PLT stub and names this binary whatever is loaded. Ask the
    // global search order instead -- which a preload heads -- and see which object answers.
    //
    // FAIL CLOSED: calibrate only when memcpy resolves inside libc. Matching the shim by NAME was
    // the first attempt and it was worse than useless: the README builds the shim as `probe.so`,
    // so a reader following the recipe verbatim got no refusal and a plausible NEGATIVE overhead --
    // the exact wrong reading this refusal exists to prevent, delivered to the one person who
    // followed the instructions. Anything that is not libc is either the shim under a different
    // name or something else this control cannot reason about; both must refuse.
    Dl_info info;
    void* resolved = dlsym(RTLD_DEFAULT, "memcpy");
    const char* from = resolved && dladdr(resolved, &info) && info.dli_fname ? info.dli_fname : NULL;
    const char* base = from ? strrchr(from, '/') : NULL;
    const char* leaf = base ? base + 1 : from;
    if (!leaf || strncmp(leaf, "libc.", 5) != 0) {
        printf("\nCYCLE-COLUMN CALIBRATION SKIPPED: memcpy resolves to %s, not libc.\n"
               "  Re-run this control WITHOUT LD_PRELOAD to calibrate; under one, both arms are\n"
               "  interposed and the comparison measures the instrument with itself -- which prints\n"
               "  a plausible near-zero overhead rather than an obvious error.\n",
               from ? from : "an object this control could not name");
        return;
    }
    unsigned long long pair = 0;
    for (int i = 0; i < 200000; i++) {
        const unsigned long long t0 = __rdtsc();
        const unsigned long long t1 = __rdtsc();
        pair += t1 - t0;
    }
    printf("\nCYCLE-COLUMN CALIBRATION (this process, not under the preload)\n");
    printf("  an rdtsc pair alone costs %.2f cycles\n", pair / 200000.0);
    printf("  %10s %12s %12s %10s\n", "bytes", "bracketed", "amortised", "inflation");
    const size_t sizes[] = {64, 1024, 65536, 1048576, 3145728};
    for (unsigned i = 0; i < sizeof sizes / sizeof *sizes; i++) {
        const int iters = sizes[i] >= (1u << 20) ? 2000 : 200000;
        const double p = probed_cycles(sizes[i], iters);
        const double q = amortised_cycles(sizes[i], iters);
        printf("  %10zu %12.2f %12.2f %9.1f%%\n", sizes[i], p, q, 100.0 * (p - q) / q);
    }
    printf("  => rank LARGE sites by the cycle column; treat sub-KiB rows as mostly instrument.\n");
}

int main(void) {
    src = calloc(1, 1u << 17); dst = calloc(1, 1u << 17);
    if (!src || !dst) return 1;
    for (unsigned i = 0; i < A_CALLS; i++) site_a();
    for (unsigned i = 0; i < B_CALLS; i++) site_b();
    for (unsigned i = 0; i < C_CALLS; i++) site_c();
    for (unsigned i = 0; i < D_CALLS; i++) site_d();
    const unsigned long long calls = (unsigned long long)A_CALLS + B_CALLS + C_CALLS;
    const unsigned long long bytes = (unsigned long long)A_CALLS * 64 +
                                     (unsigned long long)B_CALLS * 4096 +
                                     (unsigned long long)C_CALLS * 65536;
    printf("EXPECT total_calls=%llu total_bytes=%llu overflow_calls=0\n",
           calls + D_CALLS, bytes + (unsigned long long)D_CALLS * 8192);
    printf("EXPECT CMP rows summing to %u calls of 8192 bytes\n", D_CALLS);
    printf("EXPECT at least 3 distinct return addresses; a site may appear as more than one when\n"
           "       the compiler duplicates the call, so compare the SUM per size, not the row count\n");
    calibrate();
    return 0;
}
