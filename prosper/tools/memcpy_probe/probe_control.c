// Positive control for memcpy_probe: a KNOWN number of copies from three distinguishable sites,
// checked rather than eyeballed. An empty or mis-bucketed census is indistinguishable from a
// working one without this -- the trap tools/getenv_probe's README already records.
//
// THE SIZES ARE RUNTIME VALUES ON PURPOSE. The first version of this control used constants, and
// gcc expanded the 64-byte copy inline: 300,000 calls simply never reached libc and the probe was
// "missing" a third of them. That is correct behaviour by both, and it is the caveat to carry --
// a small constant-size copy is invisible to this probe AND absent from perf's memmove samples,
// because no call happens. Only a copy that actually calls libc is in scope for either instrument.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    return 0;
}
