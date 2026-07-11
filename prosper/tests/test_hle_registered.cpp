// test_hle_registered — guards that the HLE functions we rely on are actually registered.
// register_builtin_hle() binds handlers by NID; a typo'd name or a forgotten registration would
// silently leave an import as an unimplemented stub (returns 0) and regress the boot. This checks
// a representative set across every HLE module (libc, math, file, kernel, time, service, graphics).
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>

using namespace prosper;

static int fails = 0;
static void must(const char* name) {
    if (Hle::lookup(nid_hash(name)) == nullptr) { printf("  [FAIL] not registered: %s\n", name); fails++; }
}

int main() {
    printf("== test_hle_registered ==\n");
    register_builtin_hle();

    const char* names[] = {
        // libc core
        "memcpy", "memset", "strlen", "malloc", "free", "snprintf", "bcmp", "bsearch",
        "__error", "setjmp", "longjmp",
        // math (real host thunks)
        "cosf", "sinf", "sqrtf", "powf", "atan2f", "sincosf", "ldexp", "fmin", "tanh", "log1pf",
        // file / directory
        "open", "read", "stat", "sceKernelOpen", "sceKernelMkdir", "sceKernelPread", "sceKernelStat",
        // kernel: threads / sync / exceptions
        "scePthreadCreate", "scePthreadMutexLock", "scePthreadCondWait", "pthread_equal",
        "sceKernelInstallExceptionHandler", "sceKernelRaiseException",
        "sceKernelIsStack", "scePthreadGetschedparam",
        // time / event queues
        "sceKernelClockGettime", "sceKernelUsleep", "sceKernelCreateEqueue", "sceKernelWaitEqueue",
        "getpid",
        // services / dialogs
        "sceUserServiceGetInitialUser", "scePadOpen", "sceMsgDialogUpdateStatus",
        "sceSystemServiceHideSplashScreen",
        // graphics (headless bring-up)
        "sceVideoOutOpen", "sceVideoOutSubmitFlip", "sceVideoOutGetFlipStatus",
        // audio (headless / pluggable backend)
        "sceAudioOutInit", "sceAudioOutOpen", "sceAudioOutOutput", "sceAudioOutOutputs",
        "sceAudioOutSetVolume", "sceAudioOutClose", "sceAudioOutGetPortState",
    };
    for (const char* n : names) must(n);
    // sync_on_address futex is registered by raw NID (no symbol name) — check it directly.
    if (Hle::lookup("Hc4CaR6JBL0") == nullptr) { printf("  [FAIL] sceKernelWaitOnAddress raw NID\n"); fails++; }

    // __ctype_get_mb_cur_max returns the VALUE of MB_CUR_MAX (1 in the "C" locale we
    // present), not a pointer to it — guest code sizes buffers as MB_CUR_MAX*n (#141).
    if (HleFn fn = Hle::lookup(nid_hash("__ctype_get_mb_cur_max"))) {
        uint64_t v = fn(0, 0, 0, 0, 0, 0);
        if (v != 1) { printf("  [FAIL] __ctype_get_mb_cur_max returned %llu, want 1 (the value, not a pointer)\n", (unsigned long long)v); fails++; }
    } else { printf("  [FAIL] not registered: __ctype_get_mb_cur_max\n"); fails++; }

    if (HleFn fn = Hle::lookup(nid_hash("getpid"))) {
        uint64_t v = fn(0, 0, 0, 0, 0, 0);
        if (v == 0) { printf("  [FAIL] getpid returned kernel-special pid 0\n"); fails++; }
    } else { printf("  [FAIL] not registered: getpid\n"); fails++; }

    if (fails) { printf("== FAIL: %d function(s) not registered ==\n", fails); return 1; }
    printf("== PASS: all %zu checked HLE functions registered ==\n", sizeof(names)/sizeof(names[0]) + 1);
    return 0;
}
