// test_tls_dtv_purge — regression guard for #68: the __tls_get_addr per-thread DTV must be purged
// (entries erased, blocks freed) on EVERY HLE-controlled thread-exit path. glibc recycles pthread
// ids, so a stale entry hands a new guest thread landing on a recycled id the dead thread's dirty
// TLS blocks instead of the zero/tdata-initialized state the ABI guarantees (real libc.prx keeps
// errno/locale/allocator state there) — and the blocks leak per thread churn.
// Covers: (1) fresh zero+tdata init of a new block, (2) purge-then-refetch returns fresh state on
// the same thread (the recycled-id scenario, simulated deterministically), (3) the trampoline
// normal-return exit path purges, (4) the scePthreadExit path (host pthread_exit — never returns
// through the trampoline) purges, (5) main/host threads with no DTV entries purge as a no-op,
// (6) a normal-return thread leaves guest %fs before returning to the host pthread runtime (#644).
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static HleFn TLSGET, CREATE, JOIN, EXITF;

// One TLS module template: 8 tdata bytes, memsz 32 (24 tbss bytes that must come up zero).
static unsigned char g_tdata[8] = { 0xA5, 0x5A, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66 };
enum { kMemsz = 32, kFilesz = 8 };

static unsigned char* tls_block() {   // __tls_get_addr({module 0, offset 0})
    uint64_t ti[2] = { 0, 0 };
    return (unsigned char*)(uintptr_t)TLSGET((uint64_t)(uintptr_t)ti, 0, 0, 0, 0, 0);
}
static bool block_is_fresh(const unsigned char* p) {
    if (!p) return false;
    if (memcmp(p, g_tdata, kFilesz) != 0) return false;          // tdata image copied
    for (int i = kFilesz; i < kMemsz; i++) if (p[i]) return false;   // tbss zeroed
    return true;
}

// Worker body: fetch this thread's block, record whether it came up fresh, then DIRTY it — the
// purge on exit must not let those bytes survive into any later thread.
struct WorkerResult { int fresh = -1; };
static void tls_touch_and_dirty(WorkerResult* r) {
    unsigned char* blk = tls_block();
    r->fresh = block_is_fresh(blk) ? 1 : 0;
    if (blk) memset(blk, 0xEE, kMemsz);
}
static void* worker_return(void* p) {   // exit path 1: normal return through thread_trampoline
    tls_touch_and_dirty((WorkerResult*)p);
    return nullptr;
}
static void* worker_exit(void* p) {     // exit path 2: scePthreadExit (host pthread_exit, no return)
    tls_touch_and_dirty((WorkerResult*)p);
    EXITF(0, 0, 0, 0, 0, 0);
    return (void*)0xdead;               // not reached
}
static void* worker_guest_fs_return(void*) {
    return (void*)0x644;
}

int main() {
    printf("== test_tls_dtv_purge ==\n");
    register_builtin_hle();

    TLSGET = Hle::lookup(nid_hash("__tls_get_addr"));
    CREATE = Hle::lookup(nid_hash("scePthreadCreate"));
    JOIN   = Hle::lookup(nid_hash("scePthreadJoin"));
    EXITF  = Hle::lookup(nid_hash("scePthreadExit"));
    CHECK(TLSGET && CREATE && JOIN && EXITF, "__tls_get_addr + scePthreadCreate/Join/Exit registered");
    if (!(TLSGET && CREATE && JOIN && EXITF)) { printf("== FAIL ==\n"); return 1; }

    TlsModuleDesc desc; desc.init_va = (uint64_t)(uintptr_t)g_tdata; desc.filesz = kFilesz; desc.memsz = kMemsz;
    set_tls_modules(&desc, 1);

    // (5) purge on a thread with no DTV entries is a graceful no-op (main/host threads).
    tls_dtv_purge_current_thread();
    CHECK(tls_dtv_thread_count() == 0, "purge with no DTV entries is a no-op");

    // (1) first lookup allocates a fresh block: tdata copied, tbss zeroed.
    unsigned char* blk = tls_block();
    CHECK(block_is_fresh(blk), "new TLS block is tdata-initialized and tbss-zeroed");
    CHECK(tls_block() == blk, "repeat lookup on the same thread returns the same block");
    CHECK(tls_dtv_thread_count() == 1, "one thread holds a DTV entry");

    // (2) the recycled-id scenario, deterministically: dirty the block, purge (what thread exit
    // does), then look up again AS IF a new thread landed on the same id — must be fresh state,
    // never the dead thread's dirty bytes.
    memset(blk, 0xEE, kMemsz);
    tls_dtv_purge_current_thread();
    CHECK(tls_dtv_thread_count() == 0, "purge erased this thread's DTV entry");
    CHECK(block_is_fresh(tls_block()), "post-purge lookup gets fresh state, not the dirty block");
    tls_dtv_purge_current_thread();

    // (3)+(4) both real thread-exit paths purge; churn several threads and assert no DTV growth
    // (the per-thread-churn leak) and that every thread saw a fresh block (stale-TLS symptom if a
    // pthread id does get recycled across iterations).
    for (int i = 0; i < 8; i++) {
        bool via_exit = (i & 1) != 0;
        WorkerResult r;
        uint64_t tid = 0;
        uint64_t rc = CREATE((uint64_t)(uintptr_t)&tid, 0,
                             (uint64_t)(uintptr_t)(via_exit ? worker_exit : worker_return),
                             (uint64_t)(uintptr_t)&r, 0, 0);
        if (rc != 0) { printf("  [FAIL] scePthreadCreate rc=%llu (iter %d)\n", (unsigned long long)rc, i); fails++; continue; }
        JOIN(tid, 0, 0, 0, 0, 0);
        char msg[96];
        snprintf(msg, sizeof msg, "iter %d (%s): worker saw a fresh block", i, via_exit ? "scePthreadExit" : "return");
        CHECK(r.fresh == 1, msg);
        snprintf(msg, sizeof msg, "iter %d (%s): DTV entry purged on thread exit", i, via_exit ? "scePthreadExit" : "return");
        CHECK(tls_dtv_thread_count() == 0, msg);
    }

#ifdef __linux__
    // (6) The guest entry returns with guest %fs active. The trampoline must switch permanently
    // back to host %fs before it returns into glibc's start_thread cleanup. Restoring guest %fs at
    // that terminal boundary makes __res_thread_freeres read glibc TLS through the guest TCB and
    // fault at null+0x10 (#644). Index zero is reserved; no static TLS module is needed here.
    setenv("PROSPER_GUEST_FS", "1", 1);
    TlsModuleDesc reserved{};
    guest_tls_set_templates(&reserved, 1);
    CHECK(guest_tls_enabled(), "guest FS enabled for normal-return boundary test");

    uint64_t tid = 0;
    void* result = nullptr;
    uint64_t rc = CREATE((uint64_t)(uintptr_t)&tid, 0,
                         (uint64_t)(uintptr_t)worker_guest_fs_return, 0, 0, 0);
    CHECK(rc == 0, "guest-FS worker created");
    if (rc == 0) {
        JOIN(tid, (uint64_t)(uintptr_t)&result, 0, 0, 0, 0);
        CHECK(result == (void*)0x644,
              "normal-return worker reaches host pthread cleanup and preserves its result");
    }
#endif

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
