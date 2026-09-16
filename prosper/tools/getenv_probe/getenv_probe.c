// Counts getenv() calls in the host process and reports every 10 s, so a SIGTERM-terminated run
// still yields data and the rate can be seen to EVOLVE (the #2215 collapse is a transition, not a
// steady state). The call rate is platform-independent, so a Linux run substantiates or kills the
// per-frame call-count arithmetic behind the Windows hypothesis.
//
// PROSPER_GETENV_PROBE_NAMES=<dir> additionally breaks the total down BY NAME (#3407). The total
// says the machinery is hot; it cannot say which of the tree's ~850 switches to convert, and
// neither can `grep` (which finds sites, not evaluations) nor a sampling profiler (which attributes
// the cost to `getenv` itself and, at these stack depths, routinely fails to unwind to the caller).
// Measured on a 340 s routed Grand Theft Auto V run: 150,750,463 calls, of which the top twenty
// names were 145 M -- and the names it ranked first were not the ones a source-level sweep picks.
//
// The breakdown is OPT-IN because it takes a mutex per call: leaving it on would bill the subject
// for the instrument and corrupt the very rate the default mode exists to measure. Read the
// RANKING from it, never an absolute per-name cost, and do not quote a run under it as a frame rate.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <time.h>
#include <pthread.h>
#include <stdatomic.h>

static atomic_ullong g_calls = 0;
static char *(*real_getenv)(const char *) = NULL;

// By-name table: fixed open-addressed slots, no allocation, so the probe cannot perturb the
// allocator of the process it is measuring.
#define PROBE_NAME_SLOTS 4096
static struct { char name[96]; unsigned long long count; } g_names[PROBE_NAME_SLOTS];
static pthread_mutex_t g_names_mutex = PTHREAD_MUTEX_INITIALIZER;
static const char *g_names_dir = NULL;   // NULL = breakdown disabled

static void note_name(const char *name) {
    unsigned long hash = 5381;
    for (const char *p = name; *p; ++p) hash = hash * 33u + (unsigned char)*p;
    pthread_mutex_lock(&g_names_mutex);
    for (int probe = 0; probe < 64; ++probe) {
        const int slot = (int)((hash + (unsigned long)probe) % PROBE_NAME_SLOTS);
        if (!g_names[slot].count) {
            snprintf(g_names[slot].name, sizeof g_names[slot].name, "%s", name);
            g_names[slot].count = 1;
            break;
        }
        if (!strcmp(g_names[slot].name, name)) { ++g_names[slot].count; break; }
    }
    pthread_mutex_unlock(&g_names_mutex);
}

// Rewritten on every periodic report rather than at exit, for the reason the report itself exists:
// a bounded run dies on a signal and never reaches a destructor. One file per PID, because the
// preload reaches wrapper shells too and a shared name is overwritten by whichever process exits
// last -- which is never the game.
static void write_names(void) {
    char path[512];
    snprintf(path, sizeof path, "%s/getenv-names-%d.txt", g_names_dir, (int)getpid());
    FILE *out = fopen(path, "w");
    if (!out) return;
    fprintf(out, "total=%llu\n", (unsigned long long)atomic_load(&g_calls));
    pthread_mutex_lock(&g_names_mutex);
    for (int i = 0; i < PROBE_NAME_SLOTS; ++i)
        if (g_names[i].count) fprintf(out, "%12llu %s\n", g_names[i].count, g_names[i].name);
    pthread_mutex_unlock(&g_names_mutex);
    fclose(out);
}

// glibc declares getenv's argument nonnull, so gcc warns that the guard below is dead. It is not:
// this interposes on every library in the process, and a null here would fault inside the probe
// rather than inside the caller that deserves the blame.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnonnull-compare"
char *getenv(const char *name) {
    if (!real_getenv) real_getenv = (char *(*)(const char *))dlsym(RTLD_NEXT, "getenv");
    atomic_fetch_add_explicit(&g_calls, 1, memory_order_relaxed);
    if (g_names_dir && name) note_name(name);
    return real_getenv(name);
}
#pragma GCC diagnostic pop

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
        if (g_names_dir) write_names();
    }
    return NULL;
}

__attribute__((constructor)) static void probe_init(void) {
    // Read through the real getenv: the probe's own configuration must not appear in its census.
    real_getenv = (char *(*)(const char *))dlsym(RTLD_NEXT, "getenv");
    const char *dir = real_getenv ? real_getenv("PROSPER_GETENV_PROBE_NAMES") : NULL;
    if (dir) g_names_dir = *dir ? dir : ".";
    pthread_t th;
    pthread_create(&th, NULL, reporter, NULL);
    pthread_detach(th);
}
