// memcpy_probe -- attribute memcpy/memmove to their exact CALL SITES, by cycles as well as bytes.
//
// WHY THIS EXISTS. `perf` cannot attribute these copies on this machine, measured twice:
// `--call-graph dwarf` resolves a callchain for 1 of 807 `__memmove_avx512_unaligned_erms` samples
// with an 8 KiB user-stack dump and 0 of 964 with a 32 KiB one, while resolving prosper's own
// frames fine. (Count samples: "children == self" in `perf report -g` is a tautology for a leaf
// function and establishes nothing.) `--call-graph fp` against a
// -fno-omit-frame-pointer build does produce chains, but they are wrong at the first hop: glibc's
// AVX-512 copy never pushes %rbp, so the unwinder reads the CALLER's frame base and silently skips
// the direct caller. Stacks like `__memmove_avx512 -> __clone3` are the visible symptom.
//
// `__builtin_return_address(0)` inside an interposer sidesteps unwinding entirely: it is the exact
// return address of the call, from which addr2line names the site.
//
// IT RECORDS CYCLES, not only bytes. Bytes copied do not establish time spent -- a large aligned
// copy and a pile of small unaligned ones are different costs for the same volume -- so the ranking
// that decides what to fix has to be the cycle column. Bytes and counts are kept beside it because
// the ratio is what says WHICH of the two a site is.
//
// IT PERTURBS TIME. Two rdtsc plus a hash probe per copy is small but not nothing, and a run under
// it is an ATTRIBUTION run, never a timing run. Never quote a frame rate from one.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <x86intrin.h>
#include <stdio.h>
#include <execinfo.h>

#define SLOTS 8192u

// Plain types, not _Atomic: every access here goes through a __atomic_* builtin, and clang
// REFUSES to pass an _Atomic-qualified object to those builtins (gcc allows it). Measured with
// the clang in this project's own container at -std=c17 and -std=c23.
struct site { uint64_t ra, calls, bytes, cycles; };
static struct site g_sites[SLOTS];
static uint64_t g_calls, g_bytes, g_cycles, g_overflow;

// A return address names the CALL, not the code that wanted the copy. The biggest site in the
// first GTA run was `libstdc++.so.6+0x523b6` -- a local symbol inside the C++ runtime, which says
// a std:: container copied something and nothing about which prosper code owns it.
//
// So for LARGE copies only, capture a short backtrace. Large is where the cycles are (60,836 calls
// of ~2.9 MB carried 61% of them), and at ~250/s a backtrace costs nothing measurable, whereas
// doing it for all 105 M calls would cost more than the copies. The threshold is the knob:
// PROSPER_MEMCPY_PROBE_BIG_BYTES, default 1 MiB.
#define BIG_FRAMES 6u
#define BIG_SLOTS 512u
struct big { uint64_t calls, bytes, cycles; void* frames[BIG_FRAMES]; int used; };
static struct big g_big[BIG_SLOTS];
static size_t g_big_threshold = 1u << 20;
static uint64_t g_big_dropped;

static void record_big(size_t n, uint64_t cycles) {
    void* fr[BIG_FRAMES + 2];
    const int got = backtrace(fr, (int)(BIG_FRAMES + 2));
    if (got <= 2) { __atomic_fetch_add(&g_big_dropped, 1, __ATOMIC_RELAXED); return; }
    void* const* chain = fr + 2;                    // skip backtrace() and this interposer
    const int depth = got - 2 > (int)BIG_FRAMES ? (int)BIG_FRAMES : got - 2;
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < depth; i++) { h ^= (uint64_t)(uintptr_t)chain[i]; h *= 1099511628211ull; }
    for (unsigned i = 0; i < 64; i++) {
        struct big* b = &g_big[(h + i) & (BIG_SLOTS - 1)];
        int used = __atomic_load_n(&b->used, __ATOMIC_ACQUIRE);
        if (!used) {
            int expect = 0;
            if (__atomic_compare_exchange_n(&b->used, &expect, -1, 0,
                                            __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
                for (int k = 0; k < depth; k++) b->frames[k] = chain[k];
                __atomic_store_n(&b->used, 1, __ATOMIC_RELEASE);
            } else { continue; }
        } else if (used < 0) { continue; }
        else {
            int same = 1;
            for (int k = 0; k < depth && same; k++) same = b->frames[k] == chain[k];
            if (!same) continue;
        }
        __atomic_fetch_add(&b->calls, 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&b->bytes, n, __ATOMIC_RELAXED);
        __atomic_fetch_add(&b->cycles, cycles, __ATOMIC_RELAXED);
        return;
    }
    __atomic_fetch_add(&g_big_dropped, 1, __ATOMIC_RELAXED);
}

// Resolved lazily. Before resolution -- the dynamic loader itself copies -- a byte loop stands in.
// It must never recurse into the interposer, so it is written out rather than calling memcpy.
static void* (*real_memcpy)(void*, const void*, size_t);
static void* (*real_memmove)(void*, const void*, size_t);
static int (*real_memcmp)(const void*, const void*, size_t);
static __thread int in_probe;

static void* bootstrap_copy(void* d, const void* s, size_t n) {
    unsigned char* dst = (unsigned char*)d;
    const unsigned char* src = (const unsigned char*)s;
    if (dst < src) { for (size_t i = 0; i < n; i++) dst[i] = src[i]; }
    else           { for (size_t i = n; i-- > 0;)  dst[i] = src[i]; }
    return d;
}

static void record(uint64_t ra, size_t n, uint64_t cycles) {
    __atomic_fetch_add(&g_calls, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&g_bytes, n, __ATOMIC_RELAXED);
    __atomic_fetch_add(&g_cycles, cycles, __ATOMIC_RELAXED);
    // Open addressing, fixed capacity, no allocation on the hot path. A full table drops the sample
    // into g_overflow rather than evicting, so the report can say how much it did not see instead of
    // quietly renormalising -- an unattributed remainder must stay visible.
    uint64_t h = (ra * 0x9e3779b97f4a7c15ull) >> 40;
    for (unsigned i = 0; i < 64; i++) {
        struct site* s = &g_sites[(h + i) & (SLOTS - 1)];
        uint64_t cur = __atomic_load_n(&s->ra, __ATOMIC_RELAXED);
        if (cur == 0) {
            uint64_t expect = 0;
            if (!__atomic_compare_exchange_n(&s->ra, &expect, ra, 0,
                                             __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                if (expect != ra) continue;
            }
        } else if (cur != ra) continue;
        __atomic_fetch_add(&s->calls, 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&s->bytes, n, __ATOMIC_RELAXED);
        __atomic_fetch_add(&s->cycles, cycles, __ATOMIC_RELAXED);
        return;
    }
    __atomic_fetch_add(&g_overflow, 1, __ATOMIC_RELAXED);
}

static void dump(void) {
    // Write a temporary and rename. `fopen(path, "w")` truncates in place, so a reader polling the
    // report while the game runs can observe a ZERO-BYTE file -- which is exactly the signature of a
    // broken shim that the positive control exists to rule out. rename(2) is atomic within a
    // directory, so a reader sees either the previous report or the next one, never a partial one.
    const char* dir = getenv("PROSPER_MEMCPY_PROBE_DIR");
    if (!dir) dir = ".";
    char path[256], tmp[288];
    snprintf(path, sizeof path, "%s/memcpy-sites-%d.txt", dir, (int)getpid());
    snprintf(tmp, sizeof tmp, "%s/.memcpy-sites-%d.tmp", dir, (int)getpid());
    FILE* f = fopen(tmp, "w");
    if (!f) return;
    fprintf(f, "total_calls=%llu total_bytes=%llu total_cycles=%llu overflow_calls=%llu\n",
            (unsigned long long)g_calls, (unsigned long long)g_bytes,
            (unsigned long long)g_cycles, (unsigned long long)g_overflow);
    // Resolve here rather than offline. A raw return address is useless once the process is gone --
    // shared libraries are relocated, so `addr2line` against the executable silently prints `??` for
    // every site that is NOT in it, and the biggest site in the first run of this probe was exactly
    // that. dladdr costs one call per site per dump, never per copy.
    fprintf(f, "# return_address cycles calls bytes dso dso_offset nearest_symbol\n");
    for (unsigned i = 0; i < SLOTS; i++) {
        if (!g_sites[i].ra) continue;
        const uint64_t raw = g_sites[i].ra;
        const int is_cmp = (raw >> 63) & 1;
        const uint64_t addr = raw & ~(1ull << 63);
        Dl_info info;
        const int ok = dladdr((void*)(uintptr_t)addr, &info);
        const char* dso = ok && info.dli_fname ? info.dli_fname : "?";
        const char* sym = ok && info.dli_sname ? info.dli_sname : "?";
        unsigned long long off = 0;
        if (ok && info.dli_fbase) off = addr - (uint64_t)(uintptr_t)info.dli_fbase;
        fprintf(f, "%s%llx %llu %llu %llu %s %llx %s\n", is_cmp ? "CMP:" : "",
                (unsigned long long)addr, (unsigned long long)g_sites[i].cycles,
                (unsigned long long)g_sites[i].calls, (unsigned long long)g_sites[i].bytes,
                dso, off, sym);
    }
    fprintf(f, "# --- backtraces for copies >= %llu bytes (dropped=%llu) ---\n",
            (unsigned long long)g_big_threshold, (unsigned long long)g_big_dropped);
    for (unsigned i = 0; i < BIG_SLOTS; i++) {
        if (__atomic_load_n(&g_big[i].used, __ATOMIC_ACQUIRE) != 1) continue;
        fprintf(f, "BIG cycles=%llu calls=%llu bytes=%llu\n",
                (unsigned long long)g_big[i].cycles, (unsigned long long)g_big[i].calls,
                (unsigned long long)g_big[i].bytes);
        for (unsigned k = 0; k < BIG_FRAMES; k++) {
            void* a = g_big[i].frames[k];
            if (!a) break;
            Dl_info info;
            const int ok = dladdr(a, &info);
            unsigned long long off = ok && info.dli_fbase
                ? (unsigned long long)((uintptr_t)a - (uintptr_t)info.dli_fbase) : 0;
            fprintf(f, "   %p %s+0x%llx %s\n", a,
                    ok && info.dli_fname ? info.dli_fname : "?", off,
                    ok && info.dli_sname ? info.dli_sname : "?");
        }
    }
    fclose(f);
    rename(tmp, path);
}

// Rewritten every 10 s from a background thread rather than at exit: a run ended by `timeout` dies
// on SIGTERM and never runs a destructor, which is how the first getenv probe produced an empty
// file after a 90-second run.
static void* reporter(void* unused) {
    (void)unused;
    // First report early so a short-lived process -- the positive control, above all -- produces a
    // file at all; then on the same 10 s cadence as tools/getenv_probe.
    sleep(2); dump();
    for (;;) { sleep(10); dump(); }
    return NULL;
}

// Belt and braces. The periodic thread is what a `timeout`-killed run relies on; this is what a
// normally-exiting one relies on. Neither alone covers both, and an absent file reads exactly like
// a program that made no copies.
__attribute__((destructor)) static void dump_at_exit(void) { dump(); }

__attribute__((constructor)) static void init(void) {
    // Declared before the `if` rather than inside its condition: a declaration-as-condition is C2y,
    // and clang rejects it in C mode at -std=c17 and -std=c23 -- including the clang in this
    // project's own container. Nothing builds this file today, which is exactly why it must not
    // acquire a dialect trap for whoever first adds it to a target.
    const char* big = getenv("PROSPER_MEMCPY_PROBE_BIG_BYTES");
    if (big) {
        char* end = NULL;
        const unsigned long long v = strtoull(big, &end, 0);
        if (end && !*end && v) g_big_threshold = (size_t)v;
    }
    real_memcpy = (void* (*)(void*, const void*, size_t))dlsym(RTLD_NEXT, "memcpy");
    real_memmove = (void* (*)(void*, const void*, size_t))dlsym(RTLD_NEXT, "memmove");
    real_memcmp = (int (*)(const void*, const void*, size_t))dlsym(RTLD_NEXT, "memcmp");
    pthread_t t;
    pthread_create(&t, NULL, reporter, NULL);
    pthread_detach(t);
}

#define PROBE_BODY(REAL)                                                          \
    if (in_probe || !(REAL)) return bootstrap_copy(d, s, n);                      \
    in_probe = 1;                                                                 \
    const uint64_t t0 = __rdtsc();                                                \
    void* r = (REAL)(d, s, n);                                                    \
    const uint64_t t1 = __rdtsc();                                                \
    record((uint64_t)__builtin_return_address(0), n, t1 - t0);                    \
    if (n >= g_big_threshold) record_big(n, t1 - t0);                             \
    in_probe = 0;                                                                 \
    return r;

void* memcpy(void* d, const void* s, size_t n)  { PROBE_BODY(real_memcpy) }
void* memmove(void* d, const void* s, size_t n) { PROBE_BODY(real_memmove) }

// memcmp shares the table. Its sites are tagged by a high bit so one file answers both questions --
// which is the point, because "the memcmp share that belongs to shader keys" is exactly the figure
// that must not be assumed from a symbol name.
int memcmp(const void* a, const void* b, size_t n) {
    if (in_probe || !real_memcmp) {
        const unsigned char* x = (const unsigned char*)a;
        const unsigned char* y = (const unsigned char*)b;
        for (size_t i = 0; i < n; i++) if (x[i] != y[i]) return (int)x[i] - (int)y[i];
        return 0;
    }
    in_probe = 1;
    const uint64_t t0 = __rdtsc();
    const int r = real_memcmp(a, b, n);
    const uint64_t t1 = __rdtsc();
    record(((uint64_t)__builtin_return_address(0)) | (1ull << 63), n, t1 - t0);
    in_probe = 0;
    return r;
}
