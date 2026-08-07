// Counts getenv() calls in the host process and reports every 10 s, so a SIGTERM-terminated run
// still yields data and the rate can be seen to EVOLVE (the #2215 collapse is a transition, not a
// steady state). The call rate is platform-independent, so a Linux run substantiates or kills the
// per-frame call-count arithmetic behind the Windows hypothesis.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <time.h>
#include <pthread.h>
#include <stdatomic.h>

static atomic_ullong g_calls = 0;
static char *(*real_getenv)(const char *) = NULL;

char *getenv(const char *name) {
    if (!real_getenv) real_getenv = (char *(*)(const char *))dlsym(RTLD_NEXT, "getenv");
    atomic_fetch_add_explicit(&g_calls, 1, memory_order_relaxed);
    return real_getenv(name);
}

static void *reporter(void *unused) {
    (void)unused;
    unsigned long long prev = 0;
    for (int t = 10;; t += 10) {
        struct timespec ts = {10, 0};
        nanosleep(&ts, NULL);
        unsigned long long n = atomic_load(&g_calls);
        fprintf(stderr, "[getenv-probe] t=%ds total=%llu delta=%llu rate=%.0f/s\n",
                t, n, n - prev, (double)(n - prev) / 10.0);
        fflush(stderr);
        prev = n;
    }
    return NULL;
}

__attribute__((constructor)) static void probe_init(void) {
    pthread_t th;
    pthread_create(&th, NULL, reporter, NULL);
    pthread_detach(th);
}
