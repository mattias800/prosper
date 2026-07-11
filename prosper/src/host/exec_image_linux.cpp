// exec_image_linux.cpp — Linux host backing + HLE stubs (M2/M3). Compiles to nothing
// on non-Linux so the shared (mingw) build is unaffected.
#include "exec_image.hpp"
#include "sse4a.hpp"
#include "../hle/nid.hpp"
#include "../hle/dispatch.hpp"

#ifdef __linux__
#include <sys/mman.h>
#include <signal.h>
#include <setjmp.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cerrno>
#include <map>
#include <mutex>
#include <atomic>

namespace prosper {

namespace {
    uint64_t g_base = 0, g_stub_base = 0, g_stub_size = 0, g_nstubs = 0;
    // Real per-thread stack registry (keyed by pthread id). Each guest thread runs on a
    // stack we allocate (the main thread's mmap'd stack; workers' stacks from
    // k_pthread_create), so the GC/thread code gets accurate bounds without the fragile
    // pthread_getattr_np.
    std::map<uint64_t, std::pair<uint64_t, uint64_t>> g_stacks;   // tid -> (base, size)
    std::mutex g_smx;
    // Recovery point. Only the (single) thread that armed it — always the main thread running
    // run_entry/run_guest_inits — can be longjmp'd back; guest worker faults have no armed point and
    // terminate the process cleanly. We key "who armed" on the real kernel tid (SYS_gettid, a syscall
    // that does NOT use %fs) rather than a thread_local flag, because real libc.prx runs its guest
    // worker threads with a GUEST %fs — which makes any %fs-based thread_local (the old g_armed/g_jb)
    // read GARBAGE on those threads (a false "armed" + a garbage jmp_buf -> longjmp-to-garbage storm).
    // Plain globals + gettid sidestep %fs entirely.
    sigjmp_buf g_jb;
    volatile long g_armed_tid = 0;               // kernel tid that armed g_jb, 0 = none
    inline long cur_tid() { return (long)syscall(SYS_gettid); }
    volatile sig_atomic_t g_trap_kind = 0;   // 0 none, 2 SEGV/BUS, 3 ILL
    volatile int          g_trap_sig = 0;
    void*    g_fault_addr = nullptr;
    uint64_t g_fault_rip = 0;
    uint64_t g_rbp = 0, g_rsp = 0, g_rax = 0, g_rdi = 0, g_rsi = 0, g_rdx = 0;
    uint64_t g_rbx = 0, g_rcx = 0, g_r8 = 0, g_r9 = 0, g_r10 = 0, g_r11 = 0,
             g_r12 = 0, g_r13 = 0, g_r14 = 0, g_r15 = 0;
    NidDb*   g_nid_db = nullptr;

    // Deterministic fault-time memory dump (opt-in via PROSPER_FAULTMEM). Live gdb breakpoints are
    // unreliable in this multithreaded, signal-scheduled guest (register readouts race); at fault
    // time the faulting thread is stopped, so dumping guest memory around each register here is the
    // trustworthy way to inspect deep crashes (e.g. the null std::ctype facet at eboot+0x3b5ea6).
    bool g_faultmem = false;
    bool g_faultlog = false;
    // Probe pipe for probe_readable() below — a pipe write imports the source pages (EFAULT on
    // unmapped memory) where a /dev/null write does not. O_NONBLOCK so a full pipe can never
    // block the fault handler; drained after every probe.
    int  g_probe_pipe[2] = {-1, -1};
    // One-time init via a C++11 magic static (thread-safe): the unguarded `if (fd < 0) pipe2`
    // pattern could tear the fd pair under two concurrent first-calls (PR #61 review). Called
    // only from install-time paths (never from the signal handler itself), so the guard's
    // internal locking is safe here.
    void ensure_probe_pipe() {
        static const bool ok = pipe2(g_probe_pipe, O_CLOEXEC | O_NONBLOCK) == 0;
        if (!ok) g_probe_pipe[0] = g_probe_pipe[1] = -1;
    }
    // DIAGNOSTIC (PROSPER_SKIP_NULL_COMPANION, default off): at the Unity GfxDevice pipeline reader
    // eboot+0xba6e08 — which derefs a null GPU-companion pointer [obj+0x140] for pipelines that were
    // never processed — log the object state and redirect RIP to the reader's own skip label
    // (eboot+0xba6e40, where its type/flag-check branches already land) so processing continues as if
    // the companion weren't needed. This is a *probe* to reveal whether the null companion is the sole
    // blocker or one of a cascade — NOT a fix (the companions are still not real). eboot base is the
    // fixed 0x400000000 in this project; overridable via PROSPER_SKIP_RIP / PROSPER_SKIP_TARGET.
    bool     g_skip_null_companion = false;
    uint64_t g_skip_rip    = 0x400ba6e08ull;   // reader companion deref (mov 0x8(%rsi),%rax; rsi=null)
    uint64_t g_skip_target = 0x400ba6e40ull;   // reader skip label (continues via [obj+0x520/0x530])
    volatile sig_atomic_t g_skip_count = 0;
    bool     g_null_page = false;              // PROSPER_NULL_PAGE: back low null reads with zero page
    volatile sig_atomic_t g_null_page_count = 0;
    unsigned g_null_page_mask = 0;             // which of the 16 low pages [0,0x10000) are backed
    // PROSPER_WATCH_COMPANION: write-watchpoint on the companion slot [obj+0x140] that the reader
    // eboot+0xba6e08 finds null. Armed on the first such read (r15 known); the slot is null there, so
    // any real writer MUST run after — this catches all of them. Implemented by mprotect-ing the
    // slot's page read-only and single-stepping (trap flag) over each write, logging writes that land
    // in the 8-byte slot with the writer's PC. Answers: does ANYTHING write it, and from where.
    bool     g_watch_companion = false;
    uint64_t g_watch_addr = 0;                 // r15+0x140
    uint64_t g_watch_page = 0;
    bool     g_watch_armed = false;
    bool     g_watch_stepping = false;
    volatile sig_atomic_t g_watch_hits = 0;
    // #312 label-slot write watch (PROSPER_WATCH_LABEL=1): armed by the AGC fence builder on the
    // first heap-resident 32-bit value-1 fence label; every write into the label's 0x20-byte block
    // is logged with the writer's rip — distinguishing the game's allocator free-path (guest rip)
    // from prosper's own fence writes (host rip), i.e. catching a write-after-free in the act.
    uint64_t g_lwatch_slot = 0;
    uint64_t g_lwatch_page = 0;
    int      g_lwatch_max  = 400;   // PROSPER_WATCH_MAX (latched at arm time; getenv is not signal-safe)
    volatile sig_atomic_t g_lwatch_armed = 0;
    bool     g_lwatch_stepping = false;
    uint64_t g_lwatch_step_rip = 0;
    volatile sig_atomic_t g_lwatch_hits = 0;
    // #312 value-triggered mode (PROSPER_WATCH_SHIFT=1): instead of logging every write into the
    // watched block (drowns in benign MB3 alloc/free traffic and disarms before the rare stomp),
    // single-step SILENTLY and only emit a record when a write leaves a BYTE-SHIFTED pool pointer
    // (top 32 bits zero, (v<<8) inside the FPoolInfo region [0x2000000000,0x2100000000)) at the
    // faulting address — the primary #312 corruptor's signature (0x20015f0000 stored as 0x20015f00).
    // Never disarms on benign writes, so it survives the whole content-load burst to catch the store.
    bool     g_lwatch_shift = false;         // latched at arm time
    uint64_t g_lwatch_fa = 0;                // faulting store address captured on SIGSEGV
    uint64_t g_lwatch_stk[8] = {0};          // guest-RA call stack captured on SIGSEGV
    int      g_lwatch_stkn = 0;
    static inline bool lwatch_is_pool_shift(uint64_t v) {
        return (v >> 32) == 0 && ((v << 8) >= 0x2000000000ull && (v << 8) < 0x2100000000ull);
    }
    bool g_bp_shift = false;   // PROSPER_BP_SHIFT=1: only log a BP hit when rsi is a pool-shifted ptr
    // PROSPER_BP=0xOFFSET (guest image offset): int3 code-breakpoint that LOGS registers at each hit,
    // then steps over and re-arms. Diagnostic (never changes control flow). Used to enumerate the
    // GfxDevice drain (proved every category-{5,9,15,18,19} pipeline has a null [+0x140] companion).
    // LIMITATION: the restore-byte + single-step + re-insert dance is NOT thread-safe. It is reliable
    // only on functions reached by ONE thread (e.g. the drain 0xba6720). On a hot/multi-threaded
    // function (e.g. the GfxDevice ctor 0x95c700) concurrent execution during the byte-restored window
    // corrupts control flow (observed: wild jump). For those, read fault-time state instead of stepping.
    bool     g_bp_on = false;
    uint64_t g_bp_addr = 0;                    // guest VA of the breakpoint (0x400000000 + offset)
    uint8_t  g_bp_orig = 0;                    // original byte replaced by 0xCC
    bool     g_bp_stepping = false;            // mid single-step (orig byte restored, TF set)
    volatile sig_atomic_t g_bp_count = 0;
    int      g_bp_max = 400;                   // cap log volume
    void bp_write_byte(uint64_t addr, uint8_t val) {
        uint64_t pg = addr & ~(uint64_t)0xfff;
        mprotect((void*)pg, 0x2000, PROT_READ | PROT_WRITE | PROT_EXEC);
        *(volatile uint8_t*)addr = val;
        __builtin___clear_cache((char*)addr, (char*)addr + 1);
    }
    // PROSPER_HWBP=0xOFFSET: a RACE-FREE x86 hardware execute breakpoint (via perf_event_open, no code
    // modification) that logs registers at each hit. The int3 logger corrupts hot/multi-threaded code;
    // this doesn't touch code, so it is safe there. Stepping past a hit uses per-thread TF (also race-
    // free) with the perf event disabled during the step. Armed on the main thread (pid=0 = calling
    // thread); breakpoints are per-thread, so this observes main-thread execution of the target.
    bool     g_hwbp_on = false;
    uint64_t g_hwbp_addr = 0;
    int      g_hwbp_fd = -1;
    bool     g_hwbp_stepping = false;
    volatile sig_atomic_t g_hwbp_count = 0;
    int      g_hwbp_max = 200;
    // PROSPER_HWBP_ALLTHREADS=1: also arm the same execute bp on every guest worker thread (each thread
    // gets its own perf fd + owns its own SIGTRAP). Lets us observe code that runs off the main thread
    // (e.g. an async-loader worker) which a main-thread-only bp misses. Per-thread fd + stepping state.
    bool                 g_hwbp_allthreads = false;
    thread_local int     t_hwbp_fd = -1;       // this thread's own bp fd (main thread mirrors g_hwbp_fd)
    thread_local bool    t_hwbp_stepping = false;
    // PROSPER_HWBP_R15=<hex>: only LOG when r15 == this value (a condition, evaluated in-process = no gdb
    // round-trip). When matched, also dump a window of memory around rax + classify rax's mapping. Built to
    // catch the one deserializer read that produces a garbage std::string length without the gdb-bp overhead
    // that times out on hot addresses. r15 is used because it carries the read length at eboot+0x7e40e1.
    uint64_t g_hwbp_r15 = 0; bool g_hwbp_r15_on = false;
    // PROSPER_HWBP_RAXMIN=<hex>: only LOG when rax >= this (a cursor-range gate). Lets a trace of a hot
    // shared reader isolate one buffer/context (e.g. the 16 MB shader pool at 0x7ffefc…) out of thousands
    // of unrelated small-string reads, so the log + the max-count apply to the context of interest.
    uint64_t g_hwbp_raxmin = 0;
    // PROSPER_HWBP_ANOM=<hex>: ring-buffer trace. Every reader hit is pushed to a ring of the last
    // RING entries {rip_off, cursor, [cursor]-as-u32}. When a hit reads a value >= this threshold
    // (an anomalous string {count}/{length} — the deser fault reads ~16 EiB), dump the whole ring
    // oldest->newest ONCE, so we see the exact cursor walk that over-advanced INTO the over-read
    // without needing to know the dynamic buffer base or isolate the crash shader up front.
    uint64_t g_hwbp_anom = 0; bool g_hwbp_anom_on = false;
    static const int HWBP_RING = 256;
    struct HwbpRingEnt { unsigned long long rip_off, cur, rax; unsigned val;
                         unsigned long long r8, f8, f10, r14, rcx; };
    HwbpRingEnt g_hwbp_ring[HWBP_RING];
    volatile int g_hwbp_ring_pos = 0;
    volatile bool g_hwbp_ring_dumped = false;
    // PROSPER_HWBP_NODE=1: at the bp (set it to the typetree-Transfer node site eboot+0xd4cec0),
    // ring-capture {r8, [r8+8], [r8+0x10], r14} per hit and dump the ring on the WORKER fault, so the
    // last entries are the crash node's typetree metadata (byteOffset/byteSize/cursor) — reveals whether
    // a wrong typetree node (bad offset or garbage r8 from worker guest-fs TLS) mis-positions the cursor.
    bool g_hwbp_node_on = false;
    // PROSPER_STEPWIN=1 + PROSPER_STEPWIN_AFTER=N: on the Nth hit of the driver bp (0x1612c70), enter a
    // WINDOWED single-step until the NEXT driver hit — the one record whose parse drifts. On each step,
    // scan the GP registers for a pointer inside the reader buffer [base,end] (the live read cursor) and
    // ring-log rip+cursor when the cursor changes; dump the ring at the next driver hit / crash. Pins the
    // exact field read that advances the cursor by 12 instead of 16 for MatrixParameter/VectorParameter.
    bool g_stepwin_on = false; int g_stepwin_after = 5;
    volatile bool g_stepwin_active = false;
    uint64_t g_stepwin_base = 0, g_stepwin_end = 0;
    unsigned long long g_stepwin_prevcur = 0; long g_stepwin_steps = 0; long g_stepwin_max = 4000000;
    static const int STEPWIN_RING = 512;
    struct StepEnt { unsigned long long rip_off, cur; };
    StepEnt g_stepwin_ring[STEPWIN_RING]; int g_stepwin_pos = 0;
    // PROSPER_HWBP_BUFDUMP=1: at the driver bp (rbx = reader), write the reader window [rbx+0x40 .. rbx+0x48]
    // to /tmp/prosper_buf_<hit>.bin per hit — captures the EXACT decompressed object buffer the deserializer
    // reads, so it can be byte-diffed against the reference (UnityPy) object (load-corruption vs dispatch).
    bool g_hwbp_bufdump = false;
    // PROSPER_HWBP_DIVCAP=1: at the bp (set to eboot+0xd4cf6c, the `call *0x88` array-read site — hit once
    // per driver call, ~6x, minimal perturbation), log the typetree-vector transfer's computed
    // count=byteSize/elemSize from the caller frame: [rbp-0xf0]=elemSize, [rbp-0xe8]=byteSize,
    // [rbp-0xf8]=count. Compares the crash record (#6) to the succeeding ones (#4/#5) to see if a wrong
    // element size (from the `call *0x38` vtable) yields a bogus count on correct data.
    bool g_hwbp_divcap = false;
    // PROSPER_HWBP_STRDUMP=1: at the bp, print any ASCII strings pointed to by the arg registers
    // (rdi/rsi/rdx/rcx/r8/r9). For the SafeBinaryRead name-mismatch log site (eboot+0xd58f4f) this
    // captures the mismatching field name + oldBaseTypeName — i.e. exactly where the generated typetree
    // disagrees with the stream, pinning the wrong-typetree/dispatch divergence.
    bool g_hwbp_strdump = false;
    // PROSPER_HWBP_KLASS=1: at the bp, treat rbx/rdi/rax/r14 as candidate IL2CPP object pointers and print
    // each one's class name. IL2CPP layout: object[0x00]=Il2CppClass* klass; Il2CppClass{ image@0x00,
    // gc_desc@0x08, const char* name@0x10, const char* namespaze@0x18 }. So klass=[obj]; name=[[obj]+0x10].
    // Used to identify the managed type whose field is null at the deser/activation getter crash.
    bool g_hwbp_klass = false;
    // Env flags used INSIDE the fault handler — latched at install time, never read via getenv() in the
    // handler (getenv walks environ + isn't async-signal-safe; #159). Mirror g_hwbp_bufdump/klass above.
    bool g_hwbp_fields = false;         // PROSPER_HWBP_FIELDS: dump rbx object fields + object-field class names
    bool g_hwbp_obj = false;            // PROSPER_HWBP_OBJ: treat rdi as an il2cpp object at a method-entry bp
    bool g_hwbp_args = false;           // PROSPER_HWBP_ARGS: print SysV arg registers (rdi/rsi/rdx/rcx/r8/r9) per hit
    const char* g_hwbp_global = nullptr;// PROSPER_HWBP_GLOBAL=0xADDR: resolve the class name at a fixed guest global
    void hwbp_dump_ring(const char* why);   // fwd decl (defined after probe_readable)
    // Optional chained DATA write-watchpoint: on the first exec-bp hit, arm a HW write watch on
    // [rax + g_hwwatch_delta] (default rax-0x80, the thread-local device slot the ctor reads). Catches
    // every writer of that slot with its RIP — reveals what sets/clears the scoped device pointer.
    bool     g_hwwatch_req = false;
    int64_t  g_hwwatch_delta = -0x80;
    char     g_hwwatch_reg[8] = "rax";   // which register the delta is relative to (PROSPER_HWWATCH_REG)
    int      g_hwwatch_fd = -1;
    uint64_t g_hwwatch_addr = 0;
    volatile sig_atomic_t g_hwwatch_count = 0;
    long perf_bp_open(uint64_t addr, uint32_t bp_type) {
        struct perf_event_attr pe; memset(&pe, 0, sizeof pe);
        pe.type = PERF_TYPE_BREAKPOINT; pe.size = sizeof pe;
        pe.bp_type = bp_type; pe.bp_addr = addr;
        pe.bp_len = (bp_type == HW_BREAKPOINT_X) ? sizeof(long) : (uint64_t)HW_BREAKPOINT_LEN_8;
        pe.sample_period = 1; pe.disabled = 1; pe.exclude_kernel = 1; pe.exclude_hv = 1;
        return syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0UL);
    }
    // Async-safe-ish: write the /proc/self/maps line containing `addr` to stderr (identifies the module
    // that a writer RIP belongs to). Fixed buffers, no malloc; open/read/write are signal-safe.
    void classify_addr(uint64_t addr) {
        int fd = open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
        if (fd < 0) return;
        char buf[8192]; ssize_t got; char line[512]; size_t li = 0;
        char pend[96]; int pn = snprintf(pend, sizeof pend, "[hwwatch]   writer 0x%llx is in: ", (unsigned long long)addr);
        while ((got = read(fd, buf, sizeof buf)) > 0) {
            for (ssize_t i = 0; i < got; i++) {
                char c = buf[i];
                if (c != '\n' && li < sizeof(line) - 1) { line[li++] = c; continue; }
                line[li] = 0;
                // parse "start-end" hex prefix
                uint64_t s = 0, e = 0; const char* p = line; bool ok = true;
                auto hx = [&](uint64_t& out) { uint64_t v = 0; int d = 0; for (; *p && *p != '-' && *p != ' '; p++) {
                    char h = *p; int nib; if (h>='0'&&h<='9') nib=h-'0'; else if (h>='a'&&h<='f') nib=h-'a'+10;
                    else { ok=false; return; } v=(v<<4)|nib; d++; } out=v; if(!d) ok=false; };
                hx(s); if (*p=='-') { p++; hx(e); }
                if (ok && addr >= s && addr < e) { syscall(SYS_write, 2,pend, pn); syscall(SYS_write, 2,line, (size_t)li);
                    syscall(SYS_write, 2,"\n", 1); close(fd); return; }
                li = 0;
            }
        }
        close(fd);
    }
    // Dump the HWBP ring (oldest->newest) once. Prints reader fields (cur/[cur]/rax) for anom traces and
    // typetree-node fields (r8/[r8+8]/[r8+0x10]/r14) for node traces. Signal-safe (fixed buffers, write()).
    void hwbp_dump_ring(const char* why) {
        if (g_hwbp_ring_dumped) return;
        g_hwbp_ring_dumped = true;
        int total = g_hwbp_ring_pos < HWBP_RING ? g_hwbp_ring_pos : HWBP_RING;
        int start = g_hwbp_ring_pos < HWBP_RING ? 0 : (g_hwbp_ring_pos % HWBP_RING);
        char hdr[96]; int hn = snprintf(hdr, sizeof hdr, "[hwbp-ring] dump (%s): last %d hits:\n", why, total);
        syscall(SYS_write, 2,hdr, hn);
        unsigned long long prev_cur = 0;
        for (int i = 0; i < total; ++i) {
            const HwbpRingEnt& e = g_hwbp_ring[(start + i) % HWBP_RING];
            long long dcur = prev_cur ? (long long)(e.cur - prev_cur) : 0;
            char lb[220];
            int ln = snprintf(lb, sizeof lb,
                "  [%3d] rip=eboot+0x%llx req/cur=0x%llx cacher/rax=0x%llx | s50/r8=0x%llx s68/f8=0x%llx f10=0x%llx r14=0x%llx  %s\n",
                i, e.rip_off, e.cur, e.rax, e.r8, e.f8, e.f10, e.r14,
                (e.cur == e.r8 || e.cur == e.f8) ? "<<< MATCH (skip fetch)" : "");
            (void)dcur;
            syscall(SYS_write, 2,lb, ln);
            prev_cur = e.cur;
        }
    }
    // Dump the step-window cursor ring (oldest->newest): each entry is a cursor ADVANCE during the
    // single-stepped drift window, with the field size = delta. Reveals the 12-vs-16 stride directly.
    void hwbp_dump_stepwin(const char* why) {
        int total = g_stepwin_pos < STEPWIN_RING ? g_stepwin_pos : STEPWIN_RING;
        int start = g_stepwin_pos < STEPWIN_RING ? 0 : (g_stepwin_pos % STEPWIN_RING);
        char hdr[128]; int hn = snprintf(hdr, sizeof hdr,
            "[stepwin] dump (%s) after %ld steps: %d cursor advances, base=0x%llx:\n",
            why, g_stepwin_steps, total, (unsigned long long)g_stepwin_base);
        syscall(SYS_write, 2,hdr, hn);
        unsigned long long prev = 0;
        for (int i = 0; i < total; ++i) {
            const StepEnt& e = g_stepwin_ring[(start + i) % STEPWIN_RING];
            long long d = prev ? (long long)(e.cur - prev) : 0;
            char lb[160];
            int ln = snprintf(lb, sizeof lb, "  [%3d] rip=eboot+0x%llx cur=0x%llx (off 0x%llx) delta=%+lld\n",
                i, e.rip_off, e.cur, (unsigned long long)(e.cur - g_stepwin_base), d);
            syscall(SYS_write, 2,lb, ln);
            prev = e.cur;
        }
    }
    void arm_hwwatch(uint64_t addr) {
        long fd = perf_bp_open(addr, HW_BREAKPOINT_W);
        char b[160];
        if (fd < 0) { int n = snprintf(b, sizeof b, "[hwwatch] perf W-watch FAILED addr=0x%llx errno=%d\n",
                          (unsigned long long)addr, errno); syscall(SYS_write, 2, b, (size_t)n);   /* raw: glibc write() reads the TCB via %fs (guest-fs unsafe in this handler) */ return; }
        g_hwwatch_fd = (int)fd; g_hwwatch_addr = addr;
        fcntl(g_hwwatch_fd, F_SETFL, O_ASYNC);
        fcntl(g_hwwatch_fd, F_SETSIG, SIGTRAP);
        struct f_owner_ex ow; ow.type = F_OWNER_TID; ow.pid = (pid_t)syscall(SYS_gettid);
        fcntl(g_hwwatch_fd, F_SETOWN_EX, &ow);
        ioctl(g_hwwatch_fd, PERF_EVENT_IOC_ENABLE, 0);
        int n = snprintf(b, sizeof b, "[hwwatch] armed W-watch at 0x%llx (fd=%d)\n",
                         (unsigned long long)addr, g_hwwatch_fd); syscall(SYS_write, 2, b, (size_t)n);   /* raw: glibc write() reads the TCB via %fs (guest-fs unsafe in this handler) */
    }
    // ---- #312 per-thread MallocBinned3 pool-descriptor HEAD watch (PROSPER_MB3WATCH) -----------
    // The per-run corruptor stomps poolArray[idx].head (struct +0x20, size-class idx=1) of the MB3
    // per-THREAD free-block cache. That cache's base is the return of pthread_getspecific(key) /
    // the value passed to pthread_setspecific(key,base) — BOTH prosper HLE calls (k_getspecific /
    // k_setspecific), so we learn the per-run base in-process. Sessions 1-9's fixed / process-wide
    // page watches lost the discover-then-arm race: the base varies per run AND the corrupting store
    // fires on a WORKER thread whose per-thread hardware debug registers a main-thread watch never
    // observes. Here each guest thread, the instant IT is handed its own cache base (in HLE, on host
    // %fs), arms a hardware WRITE-watch on base+0x20 on ITS OWN thread — before any store — so the
    // corrupting write (host or guest) traps on the owning thread with its RIP captured. One HW-DR
    // per thread (well under the 4/thread limit). A global fd->addr table lets the SIGTRAP handler
    // attribute a hit without a %fs-fragile thread_local. Default OFF (strict no-op unless armed).
    bool             g_mb3w_on = false;      // PROSPER_MB3WATCH
    int              g_mb3w_off = 0x20;      // PROSPER_MB3WATCH_OFF (head slot; idx=1 -> 0x20)
    int              g_mb3w_nslots = 1;      // PROSPER_MB3WATCH_N (consecutive 0x20-stride slots/thread)
    struct Mb3W { int fd; uint64_t addr; };
    Mb3W             g_mb3w[64];
    unsigned char    g_mb3w_seen[64] = {0};   // per-slot benign-write log budget (see handler)
    std::atomic<int> g_mb3w_cnt{0};
    int mb3w_match(int fd) {
        int n = g_mb3w_cnt.load(std::memory_order_acquire);
        for (int i = 0; i < n && i < 64; i++) if (g_mb3w[i].fd == fd) return i;
        return -1;
    }
    // Arm on the CURRENT (calling guest) thread. Called from HLE (host %fs) so glibc is safe, but we
    // use raw syscalls throughout to keep the code identical to the handler's constraints.
    void mb3w_arm_current_thread(uint64_t base) {
        if (!g_mb3w_on || !base) return;
        for (int s = 0; s < g_mb3w_nslots; s++) {
            uint64_t addr = base + (uint64_t)g_mb3w_off + (uint64_t)s * 0x20ull;
            int n = g_mb3w_cnt.load(std::memory_order_acquire);
            bool dup = false;
            for (int i = 0; i < n && i < 64; i++) if (g_mb3w[i].addr == addr) { dup = true; break; }
            if (dup || n >= 64) continue;
            long fd = perf_bp_open(addr, HW_BREAKPOINT_W);
            if (fd < 0) { char b[128]; int k = snprintf(b, sizeof b,
                "[mb3watch] arm FAILED addr=0x%llx tid=%ld errno=%d\n",
                (unsigned long long)addr, cur_tid(), errno);
                syscall(SYS_write, 2, b, (size_t)k); continue; }
            fcntl((int)fd, F_SETFL, O_ASYNC);
            fcntl((int)fd, F_SETSIG, SIGTRAP);
            struct f_owner_ex ow; ow.type = F_OWNER_TID; ow.pid = (pid_t)syscall(SYS_gettid);
            fcntl((int)fd, F_SETOWN_EX, &ow);
            ioctl((int)fd, PERF_EVENT_IOC_ENABLE, 0);
            int slot = g_mb3w_cnt.fetch_add(1, std::memory_order_acq_rel);
            if (slot < 64) { g_mb3w[slot].addr = addr; g_mb3w[slot].fd = (int)fd; }
            char b[176]; int k = snprintf(b, sizeof b,
                "[mb3watch] ARMED head watch @0x%llx (base=0x%llx off=0x%x) fd=%ld tid=%ld slot=%d\n",
                (unsigned long long)addr, (unsigned long long)base, g_mb3w_off,
                fd, cur_tid(), slot);
            syscall(SYS_write, 2, b, (size_t)k);
        }
    }
    // PROSPER_PEEK dumps offsets off registers at fault time. Supports N specs separated by ';',
    // each "reg:off,off,..."; an offset may be prefixed '*' to chase one pointer level first
    // (e.g. "r15:*0x18+0x0" = read [[r15+0x18]+0x0]).
    struct PeekSpec { char reg[4]; uint64_t off[12]; bool deref[12]; uint64_t pre[12]; int n; };
    PeekSpec g_peek[6] = {};
    int      g_peek_specs = 0;

    // Async-signal-safe readability probe. NOTE: /dev/null does NOT work for this — the kernel's
    // null_write returns count without ever touching the source buffer, so it reports EVERY
    // address "readable" (verified empirically on this project's WSL kernel: writing an unmapped
    // pointer to /dev/null returned success; the same write to a pipe returned EFAULT). A pipe
    // write actually imports the user pages. Raw syscalls: this runs inside the fault handler
    // where glibc write()/read() are guest-%fs-unsafe; we drain what we wrote immediately so the
    // pipe can never fill (probes are single small reads, well under PIPE_BUF).
    bool probe_readable(uint64_t a) {
        if (a < 0x1000 || g_probe_pipe[1] < 0) return false;
        long w = syscall(SYS_write, g_probe_pipe[1], (const void*)a, 8);
        if (w > 0) { char b[8]; syscall(SYS_read, g_probe_pipe[0], b, (size_t)w); }
        return w == 8;
    }
    // PROSPER_DUMPAT="0xADDR[,0xADDR...]" — dump 0x40 bytes at each ABSOLUTE guest address at fault
    // time (up to 6). Complements the register-relative FAULTMEM/PEEK when the address of interest
    // is a fixed VA (e.g. comparing two candidate destination buffers). Parsed at arm time.
    uint64_t g_dumpat[6] = {}; int g_dumpat_n = 0;
    void dump_at_addrs() {
        char b[256];
        for (int i = 0; i < g_dumpat_n; i++) {
            uint64_t a = g_dumpat[i];
            int n = snprintf(b, sizeof b, "[prosper] DUMPAT 0x%llx:\n", (unsigned long long)a);
            syscall(SYS_write, 2, b, (size_t)n);
            for (int r = 0; r < 8; r++) {
                uint64_t addr = a + (uint64_t)r * 8;
                if (probe_readable(addr))
                    n = snprintf(b, sizeof b, "  +0x%02x = 0x%016llx\n", r * 8, (unsigned long long)*(const uint64_t*)addr);
                else
                    n = snprintf(b, sizeof b, "  +0x%02x = <unmapped>\n", r * 8);
                syscall(SYS_write, 2, b, (size_t)n);
            }
        }
    }
    void dump_fault_mem() {
        if (g_dumpat_n) dump_at_addrs();
        if (!g_faultmem) return;
        const struct { const char* n; uint64_t v; } regs[] = {
            {"rdi", g_rdi}, {"rsi", g_rsi}, {"rdx", g_rdx}, {"rcx", g_rcx},
            {"rax", g_rax}, {"rbx", g_rbx}, {"rbp", g_rbp}, {"rsp", g_rsp},
            {"r8 ", g_r8},  {"r9 ", g_r9},  {"r12", g_r12}, {"r13", g_r13},
            {"r14", g_r14}, {"r15", g_r15},
        };
        char b[256];
        int n = snprintf(b, sizeof b, "[prosper] FAULTMEM dump (regs -> guest memory):\n");
        syscall(SYS_write, 2, b, (size_t)n);   /* raw: glibc write() reads the TCB via %fs (guest-fs unsafe in this handler) */
        for (auto& r : regs) {
            if (!probe_readable(r.v)) {
                n = snprintf(b, sizeof b, "  %s=0x%llx  (unmapped/immediate)\n",
                             r.n, (unsigned long long)r.v);
                syscall(SYS_write, 2, b, (size_t)n);   /* raw: glibc write() reads the TCB via %fs (guest-fs unsafe in this handler) */
                continue;
            }
            uint64_t q[4] = {0, 0, 0, 0};
            char part[4][24];
            for (int i = 0; i < 4; i++) {
                uint64_t addr = r.v + (uint64_t)i * 8;
                if (probe_readable(addr)) { q[i] = *(const uint64_t*)addr;
                    snprintf(part[i], sizeof part[i], "0x%llx", (unsigned long long)q[i]); }
                else snprintf(part[i], sizeof part[i], "??");
            }
            n = snprintf(b, sizeof b, "  %s=0x%llx -> [0]=%s [8]=%s [10]=%s [18]=%s\n",
                         r.n, (unsigned long long)r.v, part[0], part[1], part[2], part[3]);
            syscall(SYS_write, 2, b, (size_t)n);   /* raw: glibc write() reads the TCB via %fs (guest-fs unsafe in this handler) */
        }
        // Deep field peek (PROSPER_PEEK, parsed once at arm time — getenv is not signal-safe): read
        // specific offsets off one or more registers to classify a large object beyond the 0x20-byte
        // window above. See g_peek definition for the syntax.
        for (int sp = 0; sp < g_peek_specs; sp++) {
            const PeekSpec& ps = g_peek[sp];
            uint64_t base = 0;
            for (auto& r : regs) { bool m = true; for (int i=0;i<3;i++) if (r.n[i]!=ps.reg[i]&&!(r.n[i]==' '&&ps.reg[i]==0)) { m=false; break; } if (m) { base=r.v; break; } }
            n = snprintf(b, sizeof b, "  PEEK %s=0x%llx:\n", ps.reg, (unsigned long long)base);
            syscall(SYS_write, 2, b, (size_t)n);   /* raw: glibc write() reads the TCB via %fs (guest-fs unsafe in this handler) */
            for (int k = 0; k < ps.n; k++) {
                uint64_t obj = base;
                if (ps.deref[k]) {   // chase one pointer level: obj = [base + pre]
                    uint64_t pa = base + ps.pre[k];
                    if (!probe_readable(pa)) { n = snprintf(b, sizeof b, "    [+0x%llx]-><unmapped>\n", (unsigned long long)ps.pre[k]); write(2,b,n); continue; }
                    obj = *(const uint64_t*)pa;
                }
                uint64_t addr = obj + ps.off[k];
                const char* pfx = ps.deref[k] ? "*" : "";
                if (probe_readable(addr))
                    n = snprintf(b, sizeof b, "    %s[+0x%llx]@+0x%llx = 0x%llx\n", pfx,
                                 (unsigned long long)ps.pre[k], (unsigned long long)ps.off[k],
                                 (unsigned long long)*(const uint64_t*)addr);
                else
                    n = snprintf(b, sizeof b, "    %s[+0x%llx]@+0x%llx = <unmapped>\n", pfx,
                                 (unsigned long long)ps.pre[k], (unsigned long long)ps.off[k]);
                syscall(SYS_write, 2, b, (size_t)n);   /* raw: glibc write() reads the TCB via %fs (guest-fs unsafe in this handler) */
            }
        }
    }

    // GPU write-attribution ring scanner (src/gpu/command_processor.cpp). Weak: tools that link
    // the host exec image without the gpu lib get a null and skip the scan.
    extern "C" int prosper_gpu_write_ring_scan(uint64_t lo, uint64_t hi, char* out, size_t cap)
        __attribute__((weak));

    // Issue-#312 heap-corruption attribution dump, fired on the first PROSPER_NULL_PAGE hit (the
    // deterministic precursor of DOLL's MallocBinned3 "Canary was 0x3" fatal: a read of address
    // 0x1 at eboot+0x231012b — the game's free path following a corrupted free-block field).
    // Dumps: the faulting instruction bytes, all GPRs, a 0x60-byte window of guest memory around
    // every heap-pointer-looking register (the corrupted FFreeBlock header will be among them),
    // and any recent GPU-performed writes (EOP/WRITE_DATA ring) near those addresses — the
    // attribution question. Async-signal-safe: raw syscalls + probe_readable only.
    void nullpage_deep_dump(ucontext_t* uc, uint64_t rip) {
        char b[512]; int n;
        // instruction bytes at rip (decode offline)
        if (probe_readable(rip)) {
            const uint8_t* p = (const uint8_t*)(uintptr_t)rip;
            n = snprintf(b, sizeof b, "[nullpage]   insn @rip:"
                         " %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x\n",
                         p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15]);
            syscall(SYS_write, 2, b, (size_t)n);
        }
        auto& g = uc->uc_mcontext.gregs;
        const struct { const char* nm; uint64_t v; } regs[] = {
            {"rax", (uint64_t)g[REG_RAX]}, {"rbx", (uint64_t)g[REG_RBX]},
            {"rcx", (uint64_t)g[REG_RCX]}, {"rdx", (uint64_t)g[REG_RDX]},
            {"rsi", (uint64_t)g[REG_RSI]}, {"rdi", (uint64_t)g[REG_RDI]},
            {"rbp", (uint64_t)g[REG_RBP]}, {"rsp", (uint64_t)g[REG_RSP]},
            {"r8 ", (uint64_t)g[REG_R8]},  {"r9 ", (uint64_t)g[REG_R9]},
            {"r10", (uint64_t)g[REG_R10]}, {"r11", (uint64_t)g[REG_R11]},
            {"r12", (uint64_t)g[REG_R12]}, {"r13", (uint64_t)g[REG_R13]},
            {"r14", (uint64_t)g[REG_R14]}, {"r15", (uint64_t)g[REG_R15]},
        };
        n = 0;
        for (int i = 0; i < 16; i++) {
            n += snprintf(b + n, sizeof b - (size_t)n, "%s%s=0x%llx", (i % 4) ? " " : "[nullpage]   ",
                          regs[i].nm, (unsigned long long)regs[i].v);
            if (i % 4 == 3) { b[n++] = '\n'; syscall(SYS_write, 2, b, (size_t)n); n = 0; }
        }
        for (int i = 0; i < 16; i++) {
            uint64_t v = regs[i].v;
            if (v < 0x10000 || !probe_readable(v)) continue;
            // rbp: the canary-walk keeps its locals at rbp-0x78..-0x48 (pool-table ptr, chain
            // heads) — start the window at rbp-0x80 so they're captured.
            uint64_t base = (v & ~0xfull) - (i == 6 /*rbp*/ ? 0x80 : 0x20);
            if (!probe_readable(base)) base = v & ~0xfull;
            int nw = (i == 6) ? 20 : 12;
            n = snprintf(b, sizeof b, "[nullpage]   %s=0x%llx mem@0x%llx:", regs[i].nm,
                         (unsigned long long)v, (unsigned long long)base);
            for (int w = 0; w < nw; w++) {
                uint64_t addr = base + (uint64_t)w * 8;
                if (probe_readable(addr))
                    n += snprintf(b + n, sizeof b - (size_t)n, " %016llx",
                                  (unsigned long long)*(const uint64_t*)addr);
                else
                    n += snprintf(b + n, sizeof b - (size_t)n, " ????????????????");
            }
            n += snprintf(b + n, sizeof b - (size_t)n, "\n");
            syscall(SYS_write, 2, b, (size_t)n);
            // Corrupted-header hunt: the canary walk faulted following a NextFreeBlock == 0x1, so
            // the stomped FFreeBlock header is a qword equal to 0x1 somewhere in the current slab
            // (a 64 KiB-aligned guest-heap register, r13 in the observed walk). Scan the slab for
            // qword==1 candidates, dump each header, and tight-match the GPU write ring at exactly
            // that address — if the ring has a 4-byte value-1 write there, OUR ReleaseMem stomped it.
            if (v >= 0x1000000000ull && v < 0x1200000000ull && (v & 0xffff) == 0) {
                // Slab-wide ring scan first (the corrupted header may no longer read as ==1 by
                // dump time — the walk pops nodes as it goes), then the qword==1 candidates.
                if (prosper_gpu_write_ring_scan) {
                    static char slab_out[16384];
                    if (prosper_gpu_write_ring_scan(v, v + 0x10000, slab_out, sizeof slab_out) > 0)
                        syscall(SYS_write, 2, slab_out, strlen(slab_out));
                }
                int hits = 0;
                for (uint64_t o = 0; o < 0x10000 && hits < 24; o += 8) {
                    uint64_t addr = v + o;
                    if (!probe_readable(addr)) break;
                    if (*(const uint64_t*)addr != 0x1ull) continue;
                    hits++;
                    const uint64_t* q = (const uint64_t*)(addr - 8);
                    n = snprintf(b, sizeof b,
                                 "[nullpage]   slab+0x%04llx qword==1 @0x%llx: [-8]=%016llx [0]=%016llx [+8]=%016llx [+10]=%016llx\n",
                                 (unsigned long long)o, (unsigned long long)addr,
                                 (unsigned long long)q[0], (unsigned long long)q[1],
                                 (unsigned long long)q[2], (unsigned long long)q[3]);
                    syscall(SYS_write, 2, b, (size_t)n);
                    if (prosper_gpu_write_ring_scan) {
                        static char hit_out[1024];
                        if (prosper_gpu_write_ring_scan(addr - 8, addr + 16, hit_out, sizeof hit_out) > 0)
                            syscall(SYS_write, 2, hit_out, strlen(hit_out));
                    }
                }
            } else if (prosper_gpu_write_ring_scan) {
                // recent GPU writes near this register's address (attribution ring, gpu lib; weak —
                // tools that link exec_image without the gpu lib simply skip this)
                static char ring_out[4096];
                if (prosper_gpu_write_ring_scan(v - 0x100, v + 0x100, ring_out, sizeof ring_out) > 0)
                    syscall(SYS_write, 2, ring_out, strlen(ring_out));
            }
        }
    }

    // GPU virtual-address window that we back lazily. The PS5 has unified CPU/GPU memory, so GPU
    // VAs are real RAM; the guest builds GPU structures at these high addresses. We map a real
    // zeroed page on first touch (a faithful memory model). NOTE: contents are zero until the
    // AGC/driver layer is real — this is a documented bring-up placeholder, not faked output. Low
    // addresses (genuine null derefs) are deliberately excluded so real bugs still fault.
    static constexpr uint64_t GPU_VA_LO = 0x100000000ull;   // 4 GiB
    static constexpr uint64_t GPU_VA_HI = 0x1000000000ull;  // 64 GiB
    volatile sig_atomic_t g_lazy_pages = 0;                 // count (diagnostic)
}
// Tracked-mapping state probe for the lazy-commit fault path (hle_kernel_mem.cpp).
extern "C" int prosper_reserved_range_state(uint64_t addr);
namespace {

    // no_stack_protector: this handler can be entered while the faulting thread runs on the GUEST %fs and
    // then switches to the HOST %fs mid-function (guest_fs_enter_host_for_signal at line ~759). A
    // -fstack-protector prologue reads its canary from %fs:0x28 at entry (guest TCB) and the epilogue
    // re-reads it (host TCB); if a thread's guest-TCB canary copy ever diverges from the host canary, the
    // epilogue sees a spurious mismatch and aborts via __stack_chk_fail ("stack smashing detected") even
    // though no buffer overflowed — a fatal false positive that intermittently killed long runs. Disabling
    // the canary on exactly this %fs-switching handler removes the false positive at its source (more
    // robust than copying the canary into every guest TCB, which is per-thread-fragile).
    __attribute__((no_stack_protector))
    // AMD SSE4a (insertq/extrq) emulation. PS5 guest code is compiled for Zen2 (AMD) and emits SSE4a
    // bitfield instructions that #UD (SIGILL) on Intel hosts. Decode the faulting instruction from the
    // signal context, emulate it against the guest XMM registers, advance RIP, and let sigreturn resume.
    // Register-operand forms only (the compiler-emitted ones) — returns false for anything else so the
    // real fatal path still handles genuine SIGILLs. Async-signal-safe: touches only the ucontext and
    // the mapped, executable instruction bytes at RIP.
    //   INSERTQ (F2 0F 78 /r ib ib  and  F2 0F 79 /r): insert low `len` bits of src into dst at bit `idx`.
    //   EXTRQ   (66 0F 78 /0 ib ib  and  66 0F 79 /r): extract `len` bits at `idx`, zero-extend into low 64.
    // Control for the /78 imm forms is the two trailing imm8s; for the /79 reg forms it comes from the
    // source xmm (EXTRQ: rm[13:0]; INSERTQ: rm[77:64]).  len==0 means 64. CONFIDENCE: HIGH (Intel SDM).
    volatile unsigned long g_sse4a_emulated = 0;
    const bool g_sse4a_stat = getenv("PROSPER_SSE4A_STAT") != nullptr;
    bool try_emulate_sse4a(ucontext_t* uc) {
        auto& g = uc->uc_mcontext.gregs;
        uint64_t rip = (uint64_t)g[REG_RIP];
        const uint8_t* p = (const uint8_t*)rip;
        size_t i = 0;
        uint8_t pfx = p[i];
        if (pfx != 0xF2 && pfx != 0x66) return false;   // INSERTQ=F2, EXTRQ=66
        i++;
        uint8_t rex = 0;
        if ((p[i] & 0xF0) == 0x40) rex = p[i++];
        if (p[i++] != 0x0F) return false;
        uint8_t op = p[i++];
        if (op != 0x78 && op != 0x79) return false;
        uint8_t modrm = p[i++];
        if ((modrm >> 6) != 3) return false;            // register operands only
        int reg = ((modrm >> 3) & 7) | ((rex & 4) ? 8 : 0);
        int rm  = (modrm & 7)       | ((rex & 1) ? 8 : 0);
        auto* fp = uc->uc_mcontext.fpregs;
        if (!fp) return false;
        auto lo  = [&](int n) { const uint32_t* e = fp->_xmm[n].element; return (uint64_t)e[0] | ((uint64_t)e[1] << 32); };
        auto hi  = [&](int n) { const uint32_t* e = fp->_xmm[n].element; return (uint64_t)e[2] | ((uint64_t)e[3] << 32); };
        auto set = [&](int n, uint64_t v) { uint32_t* e = fp->_xmm[n].element; e[0] = (uint32_t)v; e[1] = (uint32_t)(v >> 32); };
        bool insertq = (pfx == 0xF2);
        uint32_t len, idx;
        if (op == 0x78)       { len = p[i++]; idx = p[i++]; }
        else if (insertq)     { uint64_t c = hi(rm); len = (uint32_t)(c & 0x3f); idx = (uint32_t)((c >> 8) & 0x3f); }
        else                  { uint64_t c = lo(rm); len = (uint32_t)(c & 0x3f); idx = (uint32_t)((c >> 8) & 0x3f); }
        len &= 0x3f; idx &= 0x3f;                       // 6-bit fields; sse4a_* treat len==0 as 64
        if (insertq)                                    // INSERTQ dst=reg, field source=rm
            set(reg, sse4a_insertq(lo(reg), lo(rm), len, idx));
        else {                                          // EXTRQ: imm form in-place on rm, reg form on reg
            int dr = (op == 0x78) ? rm : reg;
            set(dr, sse4a_extrq(lo(dr), len, idx));
        }
        g[REG_RIP] = (greg_t)(rip + i);
        g_sse4a_emulated++;
        // PROSPER_SSE4A_STAT: async-safe rate probe — every 2^20 emulations, write the count so a
        // timed run reveals whether the SIGILL round-trip is the throughput wall. Diagnostic only.
        if (g_sse4a_stat && (g_sse4a_emulated & 0xFFFFF) == 0) {
            char b[64]; int n = snprintf(b, sizeof b, "[sse4a] %lu emulated\n", g_sse4a_emulated);
            if (n > 0) { ssize_t w = syscall(SYS_write, 2,b, (size_t)n); (void)w; }
        }
        return true;
    }

    // NO stack-protector here: the canary lives at %fs:0x28, and this handler runs on whatever %fs the
    // faulting GUEST thread had — under PROSPER_GUEST_FS that is the guest TP, whose [fs+0x28] is plain
    // guest data. Sony libc's abort stub (libc.prx+0x48d0: `mov $stopcode, %fs:0x28 ; int $0x45`) even
    // WRITES its stop code into exactly that slot right before trapping into this handler, and
    // guest_fs_enter_host_for_signal() switches %fs mid-function — so a compiler-inserted canary check
    // compares values read from two DIFFERENT TLS blocks and aborts the process with a spurious
    // "*** stack smashing detected ***" (SIGABRT + core) instead of the clean _exit(90) report path.
    __attribute__((no_stack_protector))
    void fault_handler(int sig, siginfo_t* si, void* uctx) {
        // Emulate AMD-only SSE4a bitfield ops (insertq/extrq) that #UD on Intel hosts, then resume.
        if (sig == SIGILL && try_emulate_sse4a((ucontext_t*)uctx)) return;
        // A SIGILL we did NOT emulate: log the faulting instruction bytes so the offending opcode can be
        // identified (another AMD-only ISA extension, or a decode miss in try_emulate_sse4a). #163-progress.
        if (sig == SIGILL) {
            uint64_t rip = (uint64_t)((ucontext_t*)uctx)->uc_mcontext.gregs[REG_RIP];
            const uint8_t* p = (const uint8_t*)rip;
            char b[96]; int n = snprintf(b, sizeof b, "[sigill] rip=0x%llx bytes:", (unsigned long long)rip);
            for (int k = 0; k < 16 && n < (int)sizeof b - 4; k++) n += snprintf(b + n, sizeof b - n, " %02x", p[k]);
            if (n < (int)sizeof b - 1) b[n++] = '\n';
            ssize_t w = syscall(SYS_write, 2, b, (size_t)n); (void)w;
        }
        // #312 per-thread MB3 head watch: a hardware WRITE-watch on a poolArray head slot fired.
        // Write-watchpoints trap AFTER the store, so RIP already points past it — just log the
        // writer RIP + the value now in the slot and return (the watch stays armed for the next
        // write). A byte-shifted pool pointer (0x20015f0000 stored as 0x20015f00) IS the corruptor;
        // dump its full register + guest-stack context. Raw syscalls only (may run on a guest-%fs
        // worker); globals + fd table, no thread_local.
        if (sig == SIGTRAP && g_mb3w_cnt.load(std::memory_order_acquire) > 0) {
            int mi = mb3w_match(si->si_fd);
            if (mi >= 0) {
                auto& gr2 = ((ucontext_t*)uctx)->uc_mcontext.gregs;
                uint64_t addr = g_mb3w[mi].addr;
                unsigned long long v = *(volatile uint64_t*)addr;
                uint64_t wr = (uint64_t)gr2[REG_RIP];
                bool in_eboot = (wr >= 0x400000000ull && wr < 0x420000000ull);
                bool shifted = lwatch_is_pool_shift(v);
                // base+0x20 is a HOT free-list head (every idx-1 malloc/free touches it), so logging
                // every benign write floods I/O and stalls the run before the t~60s corruption burst.
                // Log only the corruptor (a byte-shifted pool pointer) + the first couple of hits per
                // slot (confirmation that the watch is live). g_mb3w_hits packs a per-slot small count.
                if (!shifted && g_mb3w[mi].fd >= 0) {
                    if (g_mb3w_seen[mi] >= 3) return;   // silent steady-state
                    g_mb3w_seen[mi]++;
                }
                char b[256]; int n = snprintf(b, sizeof b,
                    "[mb3watch] WRITE [0x%llx]=0x%llx by rip=%s0x%llx tid=%ld%s\n",
                    (unsigned long long)addr, v, in_eboot ? "eboot+" : "host:",
                    (unsigned long long)(in_eboot ? wr - 0x400000000ull : wr), cur_tid(),
                    shifted ? "  <<<<< POOLSHIFT CORRUPTOR" : "");
                syscall(SYS_write, 2, b, (size_t)n);
                if (!in_eboot) classify_addr(wr);
                if (shifted) {
                    char rb[420]; int rn = snprintf(rb, sizeof rb,
                        "[mb3watch]   regs rax=0x%llx rbx=0x%llx rcx=0x%llx rdx=0x%llx rsi=0x%llx "
                        "rdi=0x%llx r8=0x%llx r9=0x%llx r10=0x%llx r11=0x%llx rbp=0x%llx rsp=0x%llx\n",
                        (unsigned long long)gr2[REG_RAX], (unsigned long long)gr2[REG_RBX],
                        (unsigned long long)gr2[REG_RCX], (unsigned long long)gr2[REG_RDX],
                        (unsigned long long)gr2[REG_RSI], (unsigned long long)gr2[REG_RDI],
                        (unsigned long long)gr2[REG_R8],  (unsigned long long)gr2[REG_R9],
                        (unsigned long long)gr2[REG_R10], (unsigned long long)gr2[REG_R11],
                        (unsigned long long)gr2[REG_RBP], (unsigned long long)gr2[REG_RSP]);
                    syscall(SYS_write, 2, rb, (size_t)rn);
                    // Guest call stack: scan [rsp, rsp+0x200) for eboot-range return addresses.
                    uint64_t rsp = (uint64_t)gr2[REG_RSP];
                    char sb[320]; int sn = snprintf(sb, sizeof sb, "[mb3watch]   guest-stack:");
                    int found = 0;
                    for (uint64_t p = rsp; p < rsp + 0x200 && found < 10; p += 8) {
                        if (!probe_readable(p)) break;
                        uint64_t ra = *(const uint64_t*)p;
                        if (ra >= 0x400000000ull && ra < 0x420000000ull) {
                            sn += snprintf(sb + sn, sizeof sb - sn, " eboot+0x%llx",
                                           (unsigned long long)(ra - 0x400000000ull));
                            found++;
                        }
                    }
                    sn += snprintf(sb + sn, sizeof sb - sn, "\n");
                    syscall(SYS_write, 2, sb, (size_t)sn);
                }
                return;
            }
        }
        // PROSPER_HWBP race-free hardware breakpoint. The perf event is disabled while single-stepping,
        // so a SIGTRAP while g_hwbp_stepping is the step-completion (TF) -> re-enable + clear TF. Any
        // other SIGTRAP while armed is a HW-breakpoint hit -> log registers, then disable + single-step
        // over it (TF), re-enabling on the step trap. No code bytes are touched (unlike int3), so this is
        // safe on hot/multi-threaded functions.
        if (sig == SIGTRAP && ((g_hwbp_on && g_hwbp_fd >= 0) || g_hwwatch_fd >= 0)) {
            auto* uc = (ucontext_t*)uctx;
            // Step-window: continuous single-step from driver hit N to the next driver hit. On each step,
            // find the live read cursor (a GP reg pointing inside [base,end]) and ring-log advances.
            if (g_stepwin_active && si->si_code == TRAP_TRACE) {
                auto& gs = uc->uc_mcontext.gregs;
                uint64_t rip = (uint64_t)gs[REG_RIP];
                g_stepwin_steps++;
                static const int idx[] = { REG_RAX,REG_RBX,REG_RCX,REG_RDX,REG_RSI,REG_RDI,REG_R8,
                                           REG_R9,REG_R10,REG_R11,REG_R12,REG_R13,REG_R14,REG_R15 };
                unsigned long long cur = 0;
                for (int i = 0; i < 14; i++) { uint64_t v = (uint64_t)gs[idx[i]];
                    if (v >= g_stepwin_base && v < g_stepwin_end) { cur = v; break; } }
                if (cur && cur != g_stepwin_prevcur) {
                    g_stepwin_ring[g_stepwin_pos % STEPWIN_RING] =
                        { (unsigned long long)(rip - 0x400000000ull), cur };
                    g_stepwin_pos++; g_stepwin_prevcur = cur;
                }
                bool at_driver = (rip == g_hwbp_addr);
                if (at_driver || g_stepwin_steps > g_stepwin_max) {
                    g_stepwin_active = false;
                    gs[REG_EFL] &= ~0x100ll;                       // clear TF; leave the driver bp DISABLED
                    // (we're AT the bp address on at_driver; re-enabling would re-trigger. The crash
                    // follows immediately at hit #6, which is the behavior we want to observe.)
                    uint64_t sfs = guest_fs_to_host_scoped();
                    hwbp_dump_stepwin(at_driver ? "next-driver-hit" : "step-cap");
                    guest_fs_restore_scoped(sfs);
                } else {
                    gs[REG_EFL] |= 0x100ll;                        // keep single-stepping
                }
                return;
            }
            if ((t_hwbp_stepping || g_hwbp_stepping) && si->si_code == TRAP_TRACE) {
                ioctl(t_hwbp_fd >= 0 ? t_hwbp_fd : g_hwbp_fd, PERF_EVENT_IOC_ENABLE, 0);
                uc->uc_mcontext.gregs[REG_EFL] &= ~0x100ll;
                t_hwbp_stepping = false; g_hwbp_stepping = false;
                return;
            }
            // Chained DATA write-watchpoint hit (a write to the watched slot completed): log the writer
            // RIP + the value just stored. Write-watchpoints trap AFTER the store, so no step is needed.
            if (g_hwwatch_fd >= 0 && si->si_fd == g_hwwatch_fd) {
                auto& gr2 = uc->uc_mcontext.gregs;
                unsigned long long v = 0; { auto p=(volatile uint64_t*)g_hwwatch_addr; v=*p; }
                // Always log an ANOMALOUS store (not a plausible guest heap pointer 0x1000000000..
                // 0x1720000000, and nonzero) even past the count cap — this is how a corrupt value
                // (e.g. a magic constant landing where a pointer belongs) is caught among the churn.
                bool anomalous = v && (v < 0x1000000000ull || v >= 0x1730000000ull);
                if (g_hwwatch_count < 60 || anomalous) {
                    g_hwwatch_count = g_hwwatch_count + 1;
                    char b[200];
                    uint64_t wr = (uint64_t)gr2[REG_RIP];
                    bool in_eboot = (wr >= 0x400000000ull && wr < 0x420000000ull);
                    int n = snprintf(b, sizeof b, "[hwwatch] #%d WRITE [0x%llx]=0x%llx by rip=%s0x%llx tid=%ld\n",
                        (int)g_hwwatch_count, (unsigned long long)g_hwwatch_addr, v,
                        in_eboot ? "eboot+" : "", (unsigned long long)(in_eboot ? wr - 0x400000000ull : wr), cur_tid());
                    syscall(SYS_write, 2, b, (size_t)n);   /* raw: glibc write() reads the TCB via %fs (guest-fs unsafe in this handler) */
                    if (!in_eboot) classify_addr(wr);
                }
                return;
            }
            auto& gr = uc->uc_mcontext.gregs;
            uint64_t rdi = (uint64_t)gr[REG_RDI], rsi = (uint64_t)gr[REG_RSI];
            uint64_t rax = (uint64_t)gr[REG_RAX], r14 = (uint64_t)gr[REG_R14], r15 = (uint64_t)gr[REG_R15];
            // This handler returns to the guest, which may be on the guest %fs — swap to host %fs for the
            // host-libc logging below, restore before returning. No-op off the guest-fs path.
            uint64_t saved_fs = guest_fs_to_host_scoped();
            bool cond_ok = (!g_hwbp_r15_on || (r15 == g_hwbp_r15)) && (rax >= g_hwbp_raxmin);
            // PROSPER_HWBP_ANOM ring trace: push every gated hit; when a read yields a value >= the
            // anomaly threshold, dump the ring (the cursor walk into the over-read) exactly once.
            if (g_hwbp_anom_on && cond_ok && !g_hwbp_ring_dumped) {
                uint64_t rbxr = (uint64_t)gr[REG_RBX];
                unsigned val = probe_readable(rax) ? *(const uint32_t*)rax : 0xBADBADu;
                unsigned long long cur = probe_readable(rbxr + 0x38) ? *(const uint64_t*)(rbxr + 0x38) : 0;
                int p = g_hwbp_ring_pos % HWBP_RING;
                g_hwbp_ring[p] = { (unsigned long long)((uint64_t)gr[REG_RIP] - 0x400000000ull),
                                   cur, (unsigned long long)rax, val, 0, 0, 0, (unsigned long long)r14, 0 };
                g_hwbp_ring_pos = g_hwbp_ring_pos + 1;
                if ((uint64_t)val >= g_hwbp_anom) hwbp_dump_ring("anom");
            }
            // PROSPER_HWBP_NODE: capture typetree-node metadata {r8,[r8+8],[r8+0x10],r14} at eboot+0xd4cec0;
            // the ring is dumped on the worker fault, so its last entry is the crash node.
            if (g_hwbp_node_on && !g_hwbp_ring_dumped) {
                // At 0x1612209: r14 = the type DESCRIPTOR; [r14+0x30]=extra-field-count, [r14+0x20]=field
                // table ptr, [r14+0x18]=type-name ptr. Capture per struct read; dump on fault. The
                // MatrixParameter reads (near the crash) should show [r14+0x30]==0 (the bug).
                // At the FileCacher lookup 0xb1760d: rdi=cacher, rsi=requested block. Capture the requested
                // block vs cached slots [rdi+0x50]/[rdi+0x68] to see which wrongly matches for block 2.
                uint64_t cacher = (uint64_t)gr[REG_RDI], req = (uint64_t)gr[REG_RSI];
                unsigned long long s50 = probe_readable(cacher + 0x50)? *(const uint64_t*)(cacher + 0x50): 0xBADBADull;
                unsigned long long s68 = probe_readable(cacher + 0x68)? *(const uint64_t*)(cacher + 0x68): 0xBADBADull;
                unsigned long long f160= probe_readable(cacher + 0x160)?*(const uint64_t*)(cacher + 0x160):0xBADBADull;
                int p = g_hwbp_ring_pos % HWBP_RING;
                g_hwbp_ring[p] = { (unsigned long long)((uint64_t)gr[REG_RIP] - 0x400000000ull),
                                   req/*cur=requested block*/, (unsigned long long)cacher, 0,
                                   s50/*[+0x50]*/, s68/*[+0x68]*/, f160/*[+0x160]*/, 0, 0 };
                g_hwbp_ring_pos = g_hwbp_ring_pos + 1;
            }
            if (g_hwbp_strdump && cond_ok) {
                auto pstr = [&](const char* rn, uint64_t p) {
                    if (!probe_readable(p)) return;
                    char s[80]; size_t k = 0;
                    for (; k < 72 && probe_readable(p + k); ++k) {
                        char c = *(const char*)(p + k);
                        if (c == 0) break;
                        if (c < 0x20 || c > 0x7e) { s[k] = 0; if (k < 2) return; break; }
                        s[k] = c;
                    }
                    s[k] = 0;
                    if (k >= 2) { char b[128]; int n = snprintf(b, sizeof b, "[hwbp-str] %s=0x%llx -> \"%s\"\n", rn, (unsigned long long)p, s); syscall(SYS_write, 2, b, (size_t)n);   /* raw: glibc write() reads the TCB via %fs (guest-fs unsafe in this handler) */ }
                };
                char hdr[64]; int hn = snprintf(hdr, sizeof hdr, "[hwbp-str] hit @eboot+0x%llx:\n",
                    (unsigned long long)((uint64_t)gr[REG_RIP] - 0x400000000ull)); syscall(SYS_write, 2,hdr, hn);
                pstr("rdi", (uint64_t)gr[REG_RDI]); pstr("rsi", (uint64_t)gr[REG_RSI]);
                pstr("rdx", (uint64_t)gr[REG_RDX]); pstr("rcx", (uint64_t)gr[REG_RCX]);
                pstr("r8",  (uint64_t)gr[REG_R8]);  pstr("r9",  (uint64_t)gr[REG_R9]);
            }
            if (g_hwbp_klass && cond_ok) {
                auto pklass = [&](const char* rn, uint64_t obj) {
                    if (!probe_readable(obj) || !probe_readable(obj + 7)) return;
                    uint64_t klass = *(const uint64_t*)obj;                 // Il2CppObject.klass
                    if (!probe_readable(klass + 0x10 + 7)) return;
                    uint64_t namep = *(const uint64_t*)(klass + 0x10);      // Il2CppClass.name
                    if (!probe_readable(namep)) return;
                    char s[80]; size_t k = 0;
                    for (; k < 72 && probe_readable(namep + k); ++k) { char c = *(const char*)(namep + k);
                        if (c == 0) break; if (c < 0x20 || c > 0x7e) { if (k < 2) return; break; } s[k] = c; }
                    s[k] = 0;
                    if (k >= 2) { char b[160]; int n = snprintf(b, sizeof b,
                        "[hwbp-klass] %s obj=0x%llx klass=0x%llx name=\"%s\" [obj+0x40]=0x%llx\n",
                        rn, (unsigned long long)obj, (unsigned long long)klass, s,
                        (unsigned long long)(probe_readable(obj+0x40)?*(const uint64_t*)(obj+0x40):0)); syscall(SYS_write, 2, b, (size_t)n);   /* raw: glibc write() reads the TCB via %fs (guest-fs unsafe in this handler) */ }
                };
                // PROSPER_HWBP_FIELDS=1: dump rbx's object fields [0x10..0x60] + the class name of any
                // field that is itself an object — to tell "whole object uninitialized (all null)" from
                // "one specific field null".
                if (g_hwbp_fields) {
                    uint64_t o = (uint64_t)gr[REG_RBX];
                    if (probe_readable(o + 0x60)) {
                        char b[400]; int n = snprintf(b, sizeof b, "[hwbp-fields] rbx=0x%llx:", (unsigned long long)o);
                        for (uint64_t off = 0x10; off <= 0x60; off += 8) {
                            uint64_t v = *(const uint64_t*)(o + off);
                            n += snprintf(b+n, sizeof b-n, " +0x%llx=0x%llx", (unsigned long long)off, (unsigned long long)v);
                        }
                        n += snprintf(b+n, sizeof b-n, "\n"); syscall(SYS_write, 2, b, (size_t)n);   /* raw: glibc write() reads the TCB via %fs (guest-fs unsafe in this handler) */
                        // Resolve the class name of each object-typed field (its [obj]->klass->[+0x10]=name).
                        for (uint64_t off = 0x10; off <= 0x88; off += 8) {
                            if (!probe_readable(o + off + 7)) continue;
                            uint64_t v = *(const uint64_t*)(o + off);
                            if (v < 0x1000 || !probe_readable(v + 7)) continue;
                            uint64_t klass = *(const uint64_t*)v;
                            if (!probe_readable(klass + 0x17)) continue;
                            uint64_t namep = *(const uint64_t*)(klass + 0x10);
                            if (!probe_readable(namep)) continue;
                            char s[64]; size_t k = 0;
                            for (; k < 56 && probe_readable(namep + k); ++k) { char c = *(const char*)(namep + k);
                                if (c == 0) break; if (c < 0x20 || c > 0x7e) { k = 0; break; } s[k] = c; }
                            s[k] = 0;
                            if (k >= 2) { char fb[128]; int fn = snprintf(fb, sizeof fb,
                                "[hwbp-field] +0x%llx -> %s\n", (unsigned long long)off, s); syscall(SYS_write, 2,fb, fn); }
                        }
                    }
                }
                pklass("rbx", (uint64_t)gr[REG_RBX]); pklass("rdi", (uint64_t)gr[REG_RDI]);
                pklass("rax", (uint64_t)gr[REG_RAX]); pklass("r14", (uint64_t)gr[REG_R14]);
                // Also treat rdi as a raw Il2CppClass* (name @ +0x10) — for a bp inside a getter where a
                // register holds the declaring class ptr, this reveals the class of the (null) receiver.
                auto pcname = [&](const char* rn, uint64_t klass) {
                    if (!probe_readable(klass + 0x10 + 7)) return;
                    uint64_t namep = *(const uint64_t*)(klass + 0x10);
                    if (!probe_readable(namep)) return;
                    char s[80]; size_t k = 0;
                    for (; k < 72 && probe_readable(namep + k); ++k) { char c = *(const char*)(namep + k);
                        if (c == 0) break; if (c < 0x20 || c > 0x7e) { if (k < 2) return; break; } s[k] = c; }
                    s[k] = 0;
                    if (k >= 2) { char b[128]; int n = snprintf(b, sizeof b, "[hwbp-cname] %s klass=0x%llx name=\"%s\"\n",
                        rn, (unsigned long long)klass, s); syscall(SYS_write, 2, b, (size_t)n);   /* raw: glibc write() reads the TCB via %fs (guest-fs unsafe in this handler) */ }
                };
                pcname("rdi(class)", (uint64_t)gr[REG_RDI]);
                // PROSPER_HWBP_GLOBAL=0xADDR: also resolve the class name stored at a fixed guest global
                // (the getter loads its declaring class from [0x442302515]); guest mem is 1:1-mapped.
                if (const char* g = g_hwbp_global) {
                    uint64_t ga = strtoull(g, nullptr, 0);
                    if (probe_readable(ga + 7)) pcname("global", *(const uint64_t*)ga);
                }
            }
            // PROSPER_HWBP_ARGS=1: at a method-entry bp, print the SysV integer-arg registers
            // (rdi/rsi/rdx/rcx/r8/r9) + the return address, so a call's arguments (e.g. an enum/mode
            // in esi) are visible per hit without a gdb round-trip. Generic; composes with the count cap.
            if (g_hwbp_args && cond_ok) {
                char b[192]; int n = snprintf(b, sizeof b,
                    "[hwbp-args] rdi=0x%llx rsi=0x%llx rdx=0x%llx rcx=0x%llx r8=0x%llx r9=0x%llx ret=eboot+0x%llx\n",
                    (unsigned long long)gr[REG_RDI], (unsigned long long)gr[REG_RSI],
                    (unsigned long long)gr[REG_RDX], (unsigned long long)gr[REG_RCX],
                    (unsigned long long)gr[REG_R8], (unsigned long long)gr[REG_R9],
                    (unsigned long long)(probe_readable((uint64_t)gr[REG_RSP]) ? (*(const uint64_t*)(uint64_t)gr[REG_RSP] - 0x400000000ull) : 0));
                syscall(SYS_write, 2, b, (size_t)n);
            }
            // PROSPER_HWBP_OBJ=1: treat rdi as an il2cpp object (at a method-entry bp); print its class
            // name ([[rdi]+0x10]) and fields [rdi+0x00..+0x60], plus [rdi+0x40] chased as an object too.
            if (g_hwbp_obj && cond_ok) {
                auto rd = [](uint64_t a) -> uint64_t { return probe_readable(a) ? *(const uint64_t*)a : 0; };
                auto nm = [&](uint64_t klass, char* o, int cap) { o[0]=0; if(!probe_readable(klass+0x18)) return;
                    uint64_t p = rd(klass+0x10); int k=0; if(!probe_readable(p)){return;}
                    for(;k<cap-1&&probe_readable(p+k);k++){char c=*(const char*)(p+k); if(!c)break; if(c<0x20||c>0x7e){o[0]=0;return;} o[k]=c;} o[k]=0; };
                uint64_t rdi = (uint64_t)gr[REG_RDI];
                char cn[96], sn[96]; nm(rd(rdi), cn, sizeof cn);
                uint64_t sw = rd(rdi+0x40); nm(rd(sw), sn, sizeof sn);
                char b[400]; int n = snprintf(b, sizeof b,
                    "[hwbp-obj] rdi=0x%llx class=\"%s\" | +0x40(sw)=0x%llx swclass=\"%s\" swRunning=[+0x20]=%d | fields: +8=0x%llx +10=0x%llx +18=0x%llx +20=0x%llx +28=0x%llx +30=0x%llx +38=0x%llx +48=0x%llx\n",
                    (unsigned long long)rdi, cn, (unsigned long long)sw, sn,
                    sw&&probe_readable(sw+0x20)?*(const unsigned char*)(sw+0x20):-1,
                    (unsigned long long)rd(rdi+8),(unsigned long long)rd(rdi+0x10),(unsigned long long)rd(rdi+0x18),
                    (unsigned long long)rd(rdi+0x20),(unsigned long long)rd(rdi+0x28),(unsigned long long)rd(rdi+0x30),
                    (unsigned long long)rd(rdi+0x38),(unsigned long long)rd(rdi+0x48));
                syscall(SYS_write, 2,b, n);
            }
            if (cond_ok && g_hwbp_r15_on) {
                // The matching read: dump the window around rax (the source pointer) + classify its mapping,
                // so we can see whether the deserializer's cursor points into a file buffer / heap / garbage.
                uint64_t rbx = (uint64_t)gr[REG_RBX];
                auto rd = [](uint64_t a) -> unsigned long long {
                    return probe_readable(a) ? (unsigned long long)*(const uint64_t*)a : 0xBADBADull; };
                char b2[320];
                int n2 = snprintf(b2, sizeof b2,
                    "[hwbp] MATCH r15=0x%llx rax(src)=0x%llx rbx=0x%llx | [rax-16..+24]= %llx %llx %llx %llx %llx %llx\n",
                    (unsigned long long)r15, (unsigned long long)rax, (unsigned long long)rbx,
                    rd(rax-16), rd(rax-8), rd(rax), rd(rax+8), rd(rax+16), rd(rax+24));
                syscall(SYS_write, 2,b2, n2);
                classify_addr(rax);
            }
            if (cond_ok && g_hwbp_count < g_hwbp_max) {
                g_hwbp_count = g_hwbp_count + 1;
                auto rd = [](uint64_t a) -> unsigned long long {
                    return probe_readable(a) ? (unsigned long long)*(const uint64_t*)a : 0xBADBADull; };
                uint64_t rbp = (uint64_t)gr[REG_RBP], rsp = (uint64_t)gr[REG_RSP];
                auto off = [](unsigned long long v) -> unsigned long long {
                    return (v >= 0x400000000ull && v < 0x420000000ull) ? v - 0x400000000ull : v; };
                // Also surface rax + the u32 at [rax] — at the deserializer length-read site (0x7e40d9/e1)
                // rax is the stream cursor and [rax] the value being read, so a trace shows the parse walk.
                unsigned rax_u32 = probe_readable(rax) ? *(const uint32_t*)rax : 0xBADBADu;
                // Also surface a candidate stream-reader object in rbx: cursor [+0x38], base [+0x40],
                // end [+0x48] — at the 0x1612 deserializer driver rbx IS the reader, so this shows the
                // cursor's position/window per field (to see which read walks it into the shader blob).
                uint64_t rbx = (uint64_t)gr[REG_RBX];
                char b[512];
                int n = snprintf(b, sizeof b,
                    "[hwbp] #%d rip=eboot+0x%llx rax=0x%llx [rax]=0x%08x r14=0x%llx rbx=0x%llx cur=0x%llx base=0x%llx end=0x%llx ret=eboot+0x%llx caller_rbp=eboot+0x%llx tid=%ld\n",
                    (int)g_hwbp_count, (unsigned long long)((uint64_t)gr[REG_RIP] - 0x400000000ull),
                    (unsigned long long)rax, rax_u32, (unsigned long long)r14, (unsigned long long)rbx,
                    rd(rbx+0x38), rd(rbx+0x40), rd(rbx+0x48), off(rd(rsp)), off(rd(rbp + 8)), cur_tid());
                (void)r15; (void)rdi; (void)rsi;
                syscall(SYS_write, 2, b, (size_t)n);   /* raw: glibc write() reads the TCB via %fs (guest-fs unsafe in this handler) */
                // PROSPER_HWBP_DIVCAP: log the typetree-vector transfer's elemSize/byteSize/count from the
                // caller frame (rbp-relative), for the ~6 `call *0x88` array reads.
                if (g_hwbp_divcap) {
                    // Single compact write (minimal perturbation): the transfer count triplet from the caller
                    // frame (crbp-0xf0/-0xe8/-0xf8) + the typetree NODE saved at crbp-0x1c0 (its fields
                    // [n]/[n+8]=byteSize/[n+0x10]/[n+0x18]) — compare CubeBlur (#6) to the working #4/#5.
                    unsigned long long crbp = rd(rbp);
                    unsigned long long node = rd(crbp - 0x1c0);
                    // On hit #4 dump a large window around the Shader typetree node array (nodes + likely
                    // string buffer) so we can locate MatrixParameter's node and check its byteSize/children.
                    if ((int)g_hwbp_count == 4 && node > 0x10000) {
                        uint64_t lo = node - 0x6000, hi = node + 0x6000;
                        if (probe_readable(lo) && probe_readable(hi - 1)) {
                            int fd = open("/tmp/prosper_ttnodes.bin", O_WRONLY | O_CREAT | O_TRUNC, 0644);
                            if (fd >= 0) { (void)!syscall(SYS_write, fd, (const void*)lo, (size_t)(hi - lo)); close(fd);
                                char m2[96]; int mn = snprintf(m2, sizeof m2, "[ttnodes] dumped [0x%llx..0x%llx] base=0x%llx\n",
                                    (unsigned long long)lo, (unsigned long long)hi, (unsigned long long)node); syscall(SYS_write, 2,m2, mn); }
                        }
                    }
                    char db[300];
                    int dn = snprintf(db, sizeof db,
                        "[hwbp-divcap] #%d elemSz=0x%llx byteSz=0x%llx cnt=0x%llx | node=0x%llx [n]=0x%llx [n+8]=0x%llx [n+0x10]=0x%llx [n+0x18]=0x%llx [n+0x20]=0x%llx | cur=0x%llx\n",
                        (int)g_hwbp_count, rd(crbp - 0xf0), rd(crbp - 0xe8), rd(crbp - 0xf8),
                        node, rd(node), rd(node + 8), rd(node + 0x10), rd(node + 0x18), rd(node + 0x20),
                        rd((uint64_t)gr[REG_RBX] + 0x38));
                    syscall(SYS_write, 2,db, dn);
                }
                // PROSPER_HWBP_BUFDUMP: write the reader window [base..end] to a per-hit file.
                if (g_hwbp_bufdump) {
                    uint64_t base = rd(rbx + 0x40), end = rd(rbx + 0x48);
                    if (end > base && end - base < 0x400000 && probe_readable(base) && probe_readable(end - 1)) {
                        char fn[64]; snprintf(fn, sizeof fn, "/tmp/prosper_buf_%d.bin", (int)g_hwbp_count);
                        int fd = open(fn, O_WRONLY | O_CREAT | O_TRUNC, 0644);
                        if (fd >= 0) { (void)!syscall(SYS_write, fd, (const void*)base, (size_t)(end - base)); close(fd);
                            char m2[96]; int mn = snprintf(m2, sizeof m2, "[hwbp] bufdump #%d -> %s (%llu bytes)\n",
                                (int)g_hwbp_count, fn, (unsigned long long)(end - base)); syscall(SYS_write, 2,m2, mn); }
                    }
                }
            }
            // On the first exec hit, optionally arm a data write-watch on [<reg> + delta].
            if (g_hwwatch_req && g_hwwatch_fd < 0) {
                const char* r = g_hwwatch_reg; uint64_t base = rax;
                if      (!strcmp(r,"rbx")) base = (uint64_t)gr[REG_RBX];
                else if (!strcmp(r,"rcx")) base = (uint64_t)gr[REG_RCX];
                else if (!strcmp(r,"rdx")) base = (uint64_t)gr[REG_RDX];
                else if (!strcmp(r,"rdi")) base = rdi;
                else if (!strcmp(r,"rsi")) base = rsi;
                else if (!strcmp(r,"r14")) base = r14;
                else if (!strcmp(r,"r15")) base = r15;
                else if (!strcmp(r,"rbp")) base = (uint64_t)gr[REG_RBP];
                arm_hwwatch(base + (uint64_t)g_hwwatch_delta);
            }
            ioctl(t_hwbp_fd >= 0 ? t_hwbp_fd : g_hwbp_fd, PERF_EVENT_IOC_DISABLE, 0);   // disable so we can step off the bp address
            // Step-window ENTRY: on the Nth driver hit, begin continuous single-stepping (bp stays disabled;
            // the window ends when we single-step back to the driver = next hit). Buffer range is [base,
            // base+0x20000) — the reader's own end grows across the window, so use a generous span.
            if (g_stepwin_on && !g_stepwin_active && (int)g_hwbp_count == g_stepwin_after) {
                uint64_t rbx = (uint64_t)gr[REG_RBX];
                auto rdq = [](uint64_t a) -> uint64_t { return probe_readable(a) ? *(const uint64_t*)a : 0; };
                g_stepwin_base = rdq(rbx + 0x40);
                g_stepwin_end  = g_stepwin_base + 0x20000;
                g_stepwin_prevcur = rdq(rbx + 0x38);
                g_stepwin_steps = 0; g_stepwin_pos = 0; g_stepwin_active = true;
                char sb[160]; int sn = snprintf(sb, sizeof sb,
                    "[stepwin] ENTER at driver hit #%d: base=0x%llx cur=0x%llx (off 0x%llx)\n",
                    (int)g_hwbp_count, (unsigned long long)g_stepwin_base, (unsigned long long)g_stepwin_prevcur,
                    (unsigned long long)(g_stepwin_prevcur - g_stepwin_base));
                syscall(SYS_write, 2,sb, sn);
                uc->uc_mcontext.gregs[REG_EFL] |= 0x100ll;   // start single-stepping
                guest_fs_restore_scoped(saved_fs);
                return;
            }
            // Stay armed while logging OR while the anomaly-ring trace is still hunting (PROSPER_HWBP_MAX
            // can be 0 to suppress the per-hit log yet keep the bp live feeding the ring until it dumps).
            if (g_hwbp_count < g_hwbp_max || ((g_hwbp_anom_on || g_hwbp_node_on) && !g_hwbp_ring_dumped)) {
                uc->uc_mcontext.gregs[REG_EFL] |= 0x100ll;
                if (t_hwbp_fd >= 0) t_hwbp_stepping = true; else g_hwbp_stepping = true;
            }                                              // else: leave disabled (one-and-done at the cap)
            guest_fs_restore_scoped(saved_fs);             // back to guest %fs before returning to guest code
            return;
        }
        // PROSPER_BP int3 breakpoint-logger. Two SIGTRAP cases:
        //  (a) mid-step: we restored the orig byte + set TF; now re-insert 0xCC and clear TF.
        //  (b) hit: RIP is one past our int3 -> log r15's fields, then restore orig byte, back up RIP,
        //      set TF to execute the real instruction, and re-arm on the following (a) trap.
        if (sig == SIGTRAP && g_bp_on) {
            auto* uc = (ucontext_t*)uctx;
            if (g_bp_stepping) {
                bp_write_byte(g_bp_addr, 0xCC);
                uc->uc_mcontext.gregs[REG_EFL] &= ~0x100ll;   // clear TF
                g_bp_stepping = false;
                return;
            }
            uint64_t rip = (uint64_t)uc->uc_mcontext.gregs[REG_RIP];
            if (rip == g_bp_addr + 1) {
                uint64_t r15 = (uint64_t)uc->uc_mcontext.gregs[REG_R15];
                auto& gr0 = uc->uc_mcontext.gregs;
                // PROSPER_BP_SHIFT=1 (#312): only log when the store source rsi is a byte-shifted pool
                // pointer (the primary corruptor's value). Filters the freelist push/pop hot path down
                // to just the corrupting event, so timing is barely perturbed and the log is decisive.
                bool bp_pass = !g_bp_shift ||
                    lwatch_is_pool_shift((uint64_t)gr0[REG_RSI]) ||
                    lwatch_is_pool_shift((uint64_t)gr0[REG_R10]) ||
                    lwatch_is_pool_shift((uint64_t)gr0[REG_RAX]) ||
                    lwatch_is_pool_shift((uint64_t)gr0[REG_RDX]) ||
                    lwatch_is_pool_shift((uint64_t)gr0[REG_RCX]);
                if (bp_pass && g_bp_count < g_bp_max) {
                    g_bp_count = g_bp_count + 1;
                    auto rd = [](uint64_t a) -> unsigned long long {
                        return probe_readable(a) ? (unsigned long long)*(const uint64_t*)a : 0xBADBADull; };
                    auto& gr = uc->uc_mcontext.gregs;
                    uint64_t rdi = (uint64_t)gr[REG_RDI], rsi = (uint64_t)gr[REG_RSI];
                    uint64_t rax = (uint64_t)gr[REG_RAX], r14 = (uint64_t)gr[REG_R14];
                    uint64_t rcx = (uint64_t)gr[REG_RCX];
                    uint64_t r10 = (uint64_t)gr[REG_R10], rdx = (uint64_t)gr[REG_RDX];
                    uint64_t r8 = (uint64_t)gr[REG_R8];
                    char b[512];
                    int n = snprintf(b, sizeof b,
                        "[bp] #%d rsi=0x%llx r10=0x%llx rdx=0x%llx rcx=0x%llx r8=0x%llx rax=0x%llx rdi=0x%llx r14=0x%llx r15=0x%llx [rdi]=0x%llx [rdi+0x1e4c]=0x%llx tid=%ld",
                        (int)g_bp_count, (unsigned long long)rsi, (unsigned long long)r10,
                        (unsigned long long)rdx, (unsigned long long)rcx, (unsigned long long)r8,
                        (unsigned long long)rax, (unsigned long long)rdi, (unsigned long long)r14,
                        (unsigned long long)r15, rd(rdi),
                        (unsigned long long)(probe_readable(rdi + 0x1e4c) ? *(const uint32_t*)(rdi + 0x1e4c) : 0xBADBAD),
                        cur_tid());
                    // Caller stack: first guest-text return addresses (who called free with this ptr).
                    uint64_t rsp = (uint64_t)gr[REG_RSP];
                    int found = 0;
                    for (uint64_t o = 0; o < 0x200 && found < 8 && n < (int)sizeof b - 24; o += 8) {
                        if (!probe_readable(rsp + o)) break;
                        uint64_t v = *(const uint64_t*)(rsp + o);
                        if (v >= 0x400000000ull && v < 0x40a000000ull) {
                            n += snprintf(b + n, sizeof b - (size_t)n, " s:%llx",
                                          (unsigned long long)(v - 0x400000000ull));
                            found++;
                        }
                    }
                    if (n < (int)sizeof b - 1) b[n++] = '\n';
                    syscall(SYS_write, 2, b, (size_t)n);   /* raw: glibc write() reads the TCB via %fs (guest-fs unsafe in this handler) */
                }
                bp_write_byte(g_bp_addr, g_bp_orig);              // restore real instruction
                uc->uc_mcontext.gregs[REG_RIP] = (greg_t)g_bp_addr;  // re-execute it
                uc->uc_mcontext.gregs[REG_EFL] |= 0x100ll;        // single-step (TF)
                g_bp_stepping = true;
                return;
            }
        }
        // #312 label-slot watch: SIGTRAP after single-stepping a write to the watched label page —
        // log the slot's post-write value, re-protect, keep watching.
        if (sig == SIGTRAP && g_lwatch_stepping) {
            auto* uc = (ucontext_t*)uctx;
            uc->uc_mcontext.gregs[REG_EFL] &= ~0x100ll;   // clear TF
            if (g_lwatch_shift) {
                // Value-triggered: only report if the just-completed store left a byte-shifted pool
                // pointer at the faulting address (the primary #312 corruptor). Silent otherwise.
                uint64_t v = probe_readable(g_lwatch_fa) ? *(const uint64_t*)g_lwatch_fa : 0;
                if (lwatch_is_pool_shift(v)) {
                    g_lwatch_hits = g_lwatch_hits + 1;
                    char b[512];
                    bool guest = g_lwatch_step_rip >= 0x400000000ull && g_lwatch_step_rip < 0x4c0000000ull;
                    int n = snprintf(b, sizeof b,
                        "[lwatch] SHIFT-STOMP fa=0x%llx val=0x%llx (<<8=0x%llx) rip=%s0x%llx tid=%ld",
                        (unsigned long long)g_lwatch_fa, (unsigned long long)v,
                        (unsigned long long)(v << 8), guest ? "eboot+" : "host:",
                        (unsigned long long)(guest ? g_lwatch_step_rip - 0x400000000ull : g_lwatch_step_rip),
                        cur_tid());
                    for (int i = 0; i < g_lwatch_stkn && n < (int)sizeof b - 24; i++)
                        n += snprintf(b + n, sizeof b - (size_t)n, " s:%llx",
                                      (unsigned long long)(g_lwatch_stk[i] - 0x400000000ull));
                    if (n < (int)sizeof b - 1) b[n++] = '\n';
                    syscall(SYS_write, 2, b, (size_t)n);
                    if (g_lwatch_hits >= g_lwatch_max) {   // caught enough — disarm, leave page RW
                        mprotect((void*)g_lwatch_page, 0x1000, PROT_READ | PROT_WRITE);
                        g_lwatch_armed = 0; g_lwatch_stepping = false; g_lwatch_step_rip = 0;
                        return;
                    }
                }
                g_lwatch_step_rip = 0;
                if (g_lwatch_armed) mprotect((void*)g_lwatch_page, 0x1000, PROT_READ);
                g_lwatch_stepping = false;
                return;
            }
            if (g_lwatch_step_rip) {   // the stepped write was inside the watched block: log post-value
                char b[160];
                int n = snprintf(b, sizeof b, "[lwatch]   -> slot now [0]=0x%llx [8]=0x%llx\n",
                                 (unsigned long long)*(const uint64_t*)g_lwatch_slot,
                                 (unsigned long long)*(const uint64_t*)(g_lwatch_slot + 8));
                syscall(SYS_write, 2, b, (size_t)n);
                g_lwatch_step_rip = 0;
            }
            if (g_lwatch_armed) mprotect((void*)g_lwatch_page, 0x1000, PROT_READ);
            g_lwatch_stepping = false;
            return;
        }
        // A write into the watched label page while armed: log writes that land in the label's
        // 0x20-byte block (with pre-content + writer rip), then single-step past the write.
        if (g_lwatch_armed && sig == SIGSEGV && si->si_addr) {
            uint64_t fa = (uint64_t)si->si_addr;
            auto* uc = (ucontext_t*)uctx;
            if ((fa & ~(uint64_t)0xfff) == g_lwatch_page && (uc->uc_mcontext.gregs[REG_ERR] & 2)) {
                uint64_t rip = (uint64_t)uc->uc_mcontext.gregs[REG_RIP];
                g_lwatch_step_rip = 0;
                if (g_lwatch_shift) {
                    // Value-triggered: capture writer + stack silently now, decide at SIGTRAP whether
                    // the store produced the byte-shifted pool pointer. Watch the whole page (any
                    // write) so the corruptor is caught wherever in the FPoolInfo page it lands.
                    g_lwatch_fa = fa;
                    g_lwatch_step_rip = rip;
                    uint64_t rsp = (uint64_t)uc->uc_mcontext.gregs[REG_RSP];
                    g_lwatch_stkn = 0;
                    for (uint64_t o = 0; o < 0x400 && g_lwatch_stkn < 8; o += 8) {
                        if (!probe_readable(rsp + o)) break;
                        uint64_t v = *(const uint64_t*)(rsp + o);
                        if (v >= 0x400000000ull && v < 0x40a000000ull) g_lwatch_stk[g_lwatch_stkn++] = v;
                    }
                    mprotect((void*)g_lwatch_page, 0x1000, PROT_READ | PROT_WRITE);
                    uc->uc_mcontext.gregs[REG_EFL] |= 0x100ll;   // TF -> single-step the write
                    g_lwatch_stepping = true;
                    return;
                }
                if (fa >= g_lwatch_slot - 0x18 && fa < g_lwatch_slot + 0x20) {
                    g_lwatch_hits = g_lwatch_hits + 1;
                    char b[512];
                    bool guest = rip >= 0x400000000ull && rip < 0x4c0000000ull;
                    int n = snprintf(b, sizeof b,
                        "[lwatch] #%d write fa=0x%llx rip=%s0x%llx tid=%ld pre[0]=0x%llx pre[8]=0x%llx",
                        (int)g_lwatch_hits, (unsigned long long)fa,
                        guest ? "eboot+" : "host:",
                        (unsigned long long)(guest ? rip - 0x400000000ull : rip), cur_tid(),
                        (unsigned long long)*(const uint64_t*)g_lwatch_slot,
                        (unsigned long long)*(const uint64_t*)(g_lwatch_slot + 8));
                    // #312: call-stack capture — scan the writer's stack for eboot-text return
                    // addresses (up to 8), so the guest free/alloc path above the raw write is
                    // identifiable (async-signal-safe: bounded reads of the faulting thread's
                    // own stack).
                    uint64_t rsp = (uint64_t)uc->uc_mcontext.gregs[REG_RSP];
                    int found = 0;
                    for (uint64_t o = 0; o < 0x400 && found < 8 && n < (int)sizeof b - 24; o += 8) {
                        if (!probe_readable(rsp + o)) break;
                        uint64_t v = *(const uint64_t*)(rsp + o);
                        if (v >= 0x400000000ull && v < 0x40a000000ull) {
                            n += snprintf(b + n, sizeof b - (size_t)n, " s:%llx",
                                          (unsigned long long)(v - 0x400000000ull));
                            found++;
                        }
                    }
                    if (n < (int)sizeof b - 1) b[n++] = '\n';
                    syscall(SYS_write, 2, b, (size_t)n);
                    g_lwatch_step_rip = rip;
                    if (g_lwatch_hits >= g_lwatch_max) {   // bounded: disarm after enough evidence
                        mprotect((void*)g_lwatch_page, 0x1000, PROT_READ | PROT_WRITE);
                        g_lwatch_armed = 0;
                        return;
                    }
                }
                mprotect((void*)g_lwatch_page, 0x1000, PROT_READ | PROT_WRITE);
                uc->uc_mcontext.gregs[REG_EFL] |= 0x100ll;   // TF -> single-step the write
                g_lwatch_stepping = true;
                return;
            }
        }
        // Companion-slot write-watchpoint: SIGTRAP after single-stepping a write to the watched page —
        // re-arm (re-protect read-only) and clear the trap flag so we keep watching.
        if (sig == SIGTRAP && g_watch_stepping) {
            auto* uc = (ucontext_t*)uctx;
            uc->uc_mcontext.gregs[REG_EFL] &= ~0x100ll;   // clear TF
            // Post-write: log the slot's new value + the object's type tag [r15+0] (0x2b if it's still
            // the same live object; changed => the memory was freed and reused, so the write is churn).
            unsigned long long slot = *(const uint64_t*)g_watch_addr;
            unsigned long long tag  = *(const uint64_t*)(g_watch_addr - 0x140);
            char b[128];
            int n = snprintf(b, sizeof b, "[watch]   -> slot now=0x%llx  obj[+0]=0x%llx\n", slot, tag);
            syscall(SYS_write, 2, b, (size_t)n);   /* raw: glibc write() reads the TCB via %fs (guest-fs unsafe in this handler) */
            mprotect((void*)g_watch_page, 0x1000, PROT_READ);
            g_watch_stepping = false;
            return;
        }
        // Arm the watchpoint on the first companion read (rip == the reader deref, slot still null).
        if (g_watch_companion && sig == SIGSEGV && !g_watch_armed) {
            auto* uc = (ucontext_t*)uctx;
            if ((uint64_t)uc->uc_mcontext.gregs[REG_RIP] == g_skip_rip) {
                uint64_t r15 = (uint64_t)uc->uc_mcontext.gregs[REG_R15];
                g_watch_addr = r15 + 0x140;
                g_watch_page = g_watch_addr & ~(uint64_t)0xfff;
                if (mprotect((void*)g_watch_page, 0x1000, PROT_READ) == 0) {
                    g_watch_armed = true;
                    char b[128];
                    int n = snprintf(b, sizeof b, "[watch] armed on companion slot 0x%llx (obj r15=0x%llx) reader-tid=%ld\n",
                                     (unsigned long long)g_watch_addr, (unsigned long long)r15, cur_tid());
                    syscall(SYS_write, 2, b, (size_t)n);   /* raw: glibc write() reads the TCB via %fs (guest-fs unsafe in this handler) */
                }
                // Skip this (null) read so the boot proceeds; the slot's page is now watched.
                uc->uc_mcontext.gregs[REG_RIP] = (greg_t)g_skip_target;
                return;
            }
        }
        // A write to the watched page (x86 page-fault error code bit 1 = write) while armed: log if it
        // lands in the 8-byte companion slot, then single-step past it (unprotect + set TF).
        if (g_watch_armed && sig == SIGSEGV && si->si_addr) {
            uint64_t fa = (uint64_t)si->si_addr;
            auto* uc = (ucontext_t*)uctx;
            if ((fa & ~(uint64_t)0xfff) == g_watch_page && (uc->uc_mcontext.gregs[REG_ERR] & 2)) {
                if (fa >= g_watch_addr && fa < g_watch_addr + 8) {
                    g_watch_hits = g_watch_hits + 1;
                    char b[160];
                    int n = snprintf(b, sizeof b, "[watch] WRITE to companion slot 0x%llx from rip=0x%llx writer-tid=%ld (hit #%d)\n",
                                     (unsigned long long)fa,
                                     (unsigned long long)uc->uc_mcontext.gregs[REG_RIP], cur_tid(),
                                     (int)g_watch_hits);
                    syscall(SYS_write, 2, b, (size_t)n);   /* raw: glibc write() reads the TCB via %fs (guest-fs unsafe in this handler) */
                }
                mprotect((void*)g_watch_page, 0x1000, PROT_READ | PROT_WRITE);
                uc->uc_mcontext.gregs[REG_EFL] |= 0x100ll;   // set TF -> single-step the write
                g_watch_stepping = true;
                return;
            }
        }
        // Repeated companion reads (later reader passes): skip them too so the boot keeps running.
        if (g_watch_armed && sig == SIGSEGV) {
            auto* uc = (ucontext_t*)uctx;
            if ((uint64_t)uc->uc_mcontext.gregs[REG_RIP] == g_skip_rip) {
                uc->uc_mcontext.gregs[REG_RIP] = (greg_t)g_skip_target;
                return;
            }
        }
        // Lazy commit inside a guest-RESERVED range: the guest reserved this VA (tracked by the
        // memory HLE) and touches a page it believes committed. Observed with UE4 (PPSA17942): its
        // binned allocator touches one pool page per size-class bucket whose BatchMap commit it
        // skips (bookkeeping real HW satisfies via semantics we don't fully replicate yet). Back
        // the 64KB page on first touch — the same faithful unified-memory model as the GPU-VA
        // window below — and log every page so a systematic commit-protocol gap stays visible.
        // CONFIDENCE: MED (unblocks boot; the committed-page protocol deserves a real RE pass).
        if (sig == SIGSEGV && si->si_addr) {
            uint64_t a = (uint64_t)si->si_addr;
            if (a >= 0x1000000000ull && prosper_reserved_range_state(a) == 1) {
                void* page = (void*)(a & ~(uint64_t)0xffff);
                bool ok = mmap(page, 0x10000, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == page;
                char b[128]; auto* uc2 = (ucontext_t*)uctx;
                int n = snprintf(b, sizeof b, "[lazy-commit] %s page=0x%llx rip=0x%llx\n",
                                 ok ? "mapped" : "MMAP-FAILED", (unsigned long long)(uint64_t)(uintptr_t)page,
                                 (unsigned long long)uc2->uc_mcontext.gregs[REG_RIP]);
                // RAW syscall, NOT glibc write(): this handler can run on a thread whose %fs is the
                // GUEST TCB (PROSPER_GUEST_FS), and glibc write()'s cancellation prologue reads the
                // TCB through %fs — a nested SIGSEGV inside the handler killed the process (the UE4
                // boot's silent exit-139).
                syscall(SYS_write, 2, b, (size_t)n);
                if (ok) return;   // re-execute against the now-backed page
            }
        }
        // Lazy unified-memory backing: back an unmapped GPU-VA page on demand and retry.
        // Gates (both required — everything interesting lives inside the 4-64 GiB window,
        // including guest module images and dmem mappings):
        //  - si_code == SEGV_MAPERR: only back genuinely UNMAPPED pages. A protection fault
        //    (SEGV_ACCERR, e.g. a guest write to a CPU_READ-only dmem mapping) must NOT be
        //    "handled" by mmap'ing a detached zero page over the live mapping — that silently
        //    destroys the original contents and breaks memfd CPU/GPU aliasing.
        //  - fault addr != RIP: an instruction fetch through a garbage-but-in-window pointer
        //    would get a non-executable RW page mapped at RIP, refault with SEGV_ACCERR at the
        //    same address forever (infinite fault loop) instead of a clean fault report.
        if (sig == SIGSEGV && si->si_addr && si->si_code == SEGV_MAPERR) {
            uint64_t a = (uint64_t)si->si_addr;
            uint64_t rip = (uint64_t)((ucontext_t*)uctx)->uc_mcontext.gregs[REG_RIP];
            if (a >= GPU_VA_LO && a < GPU_VA_HI && a != rip) {
                void* page = (void*)(a & ~(uint64_t)0xfff);
                bool ok = mmap(page, 0x1000, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == page;
                if (g_faultlog) {
                    char b[128];
                    int n = snprintf(b, sizeof b, "[fault] GPU-VA %s addr=0x%llx rip=0x%llx\n",
                                     ok ? "mapped" : "MMAP-FAILED", (unsigned long long)a,
                                     (unsigned long long)rip);
                    syscall(SYS_write, 2, b, (size_t)n);   /* raw: glibc write() reads the TCB via %fs (guest-fs unsafe in this handler) */
                }
                if (ok) { g_lazy_pages = g_lazy_pages + 1; return; }  // re-execute against the now-mapped page
            }
        }
        // NULL-CHAIN probe (PROSPER_NULL_PAGE, default off): back a low-address read fault with a
        // read-only zero page so null-field *reads* return 0 and the chain of null derefs proceeds —
        // revealing how far it cascades and where it truly stops. WRITE-to-null and CALL-through-null
        // (execute at 0) still fault (page is read-only, non-exec), which bounds it. Distinct fault
        // RIPs are logged (capped) so we see the cascade shape: bounded → narrow validation path;
        // endless → the resource subsystem is systematically dead. Diagnostic, NOT a fix.
        if (g_null_page && sig == SIGSEGV) {
            uint64_t a = (uint64_t)si->si_addr;
            auto* uc = (ucontext_t*)uctx;
            uint64_t rip = (uint64_t)uc->uc_mcontext.gregs[REG_RIP];
            // Only a DATA READ from low memory is backable. An instruction fetch at null (a==rip, i.e.
            // a call/jmp through a null pointer) or a fault on an already-backed page (a null WRITE)
            // must NOT be re-backed — those are the real chain endpoints; let them terminate + report.
            unsigned pg = (unsigned)(a >> 12);
            if (a < 0x10000ull && a != rip && pg < 16 && !(g_null_page_mask & (1u << pg))) {
                void* page = (void*)(a & ~(uint64_t)0xfff);
                if (mmap(page, 0x1000, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == page) {
                    g_null_page_mask |= (1u << pg);
                    g_null_page_count = g_null_page_count + 1;
                    char b[128];
                    int n = snprintf(b, sizeof b, "[nullpage] #%d addr=0x%llx rip=eboot+0x%llx\n",
                                     (int)g_null_page_count, (unsigned long long)a,
                                     (unsigned long long)(rip - 0x400000000ull));
                    syscall(SYS_write, 2, b, (size_t)n);   /* raw: glibc write() reads the TCB via %fs (guest-fs unsafe in this handler) */
                    nullpage_deep_dump(uc, rip);   // issue-#312 attribution dump (registers + heap + GPU ring)
                    return;   // re-execute; the null read now sees zero
                }
            }
        }
        // Null-companion skip probe (diagnostic; see g_skip_null_companion). Redirect the reader past
        // the null [obj+0x140] deref to its own skip label, logging each object's state.
        if (g_skip_null_companion && sig == SIGSEGV) {
            auto* uc = (ucontext_t*)uctx;
            uint64_t rip = (uint64_t)uc->uc_mcontext.gregs[REG_RIP];
            if (rip == g_skip_rip) {
                uint64_t r15 = (uint64_t)uc->uc_mcontext.gregs[REG_R15];
                g_skip_count = g_skip_count + 1;
                auto rd = [](uint64_t a) -> unsigned long long {
                    return probe_readable(a) ? (unsigned long long)*(const uint64_t*)a : 0xBADBADull;
                };
                char b[256];
                int n = snprintf(b, sizeof b,
                    "[skip] #%d r15=0x%llx [+0xc0]=0x%llx [+0xe0]=0x%llx [+0x138]=0x%llx [+0x140]=0x%llx "
                    "[+0x1a0]=0x%llx -> skip to 0x%llx\n",
                    (int)g_skip_count, (unsigned long long)r15, rd(r15+0xc0), rd(r15+0xe0),
                    rd(r15+0x138), rd(r15+0x140), rd(r15+0x1a0), (unsigned long long)g_skip_target);
                syscall(SYS_write, 2, b, (size_t)n);   /* raw: glibc write() reads the TCB via %fs (guest-fs unsafe in this handler) */
                uc->uc_mcontext.gregs[REG_RIP] = (greg_t)g_skip_target;
                return;   // re-execute from the reader's skip label
            }
        }
        // Fatal crash path (all diagnostic/stepping cases above already returned). If this thread was
        // running on OUR guest %fs, restore the host %fs NOW so the host-libc reporting below (snprintf/
        // write) + the siglongjmp-return into host C++ don't double-fault reading guest TLS as glibc's TCB.
        // No-op when guest-fs is off. (The GC RT-signal handler is separate and keeps the guest %fs.)
        guest_fs_enter_host_for_signal();
        if (g_faultlog) {
            char b[128]; auto* uc = (ucontext_t*)uctx;
            int n = snprintf(b, sizeof b, "[fault] sig=%d addr=%p rip=0x%llx armed=%d tid=%ld\n",
                             sig, si->si_addr, (unsigned long long)uc->uc_mcontext.gregs[REG_RIP],
                             (int)(g_armed_tid && cur_tid() == g_armed_tid), cur_tid());
            syscall(SYS_write, 2, b, (size_t)n);   /* raw: glibc write() reads the TCB via %fs (guest-fs unsafe in this handler) */
        }
        g_fault_addr = si->si_addr;
        auto* uc = (ucontext_t*)uctx;
        auto& g = uc->uc_mcontext.gregs;
        g_fault_rip = (uint64_t)g[REG_RIP];
        g_rbp = (uint64_t)g[REG_RBP]; g_rsp = (uint64_t)g[REG_RSP];
        g_rax = (uint64_t)g[REG_RAX]; g_rdi = (uint64_t)g[REG_RDI];
        g_rsi = (uint64_t)g[REG_RSI]; g_rdx = (uint64_t)g[REG_RDX];
        g_rbx = (uint64_t)g[REG_RBX]; g_rcx = (uint64_t)g[REG_RCX];
        g_r8  = (uint64_t)g[REG_R8];  g_r9  = (uint64_t)g[REG_R9];
        g_r10 = (uint64_t)g[REG_R10]; g_r11 = (uint64_t)g[REG_R11];
        g_r12 = (uint64_t)g[REG_R12]; g_r13 = (uint64_t)g[REG_R13];
        g_r14 = (uint64_t)g[REG_R14]; g_r15 = (uint64_t)g[REG_R15];
        g_trap_sig = sig;
        g_trap_kind = (sig == SIGILL) ? 3 : 2;
        // PROSPER_FAULT_SKIP="<fault-off>[:<resume-off>]" DIAGNOSTIC (bring-up, NOT a fix): when a fault
        // hits at eboot+<fault-off>, set rax=0 and resume the guest at eboot+<resume-off> (default
        // fault-off+4, i.e. skip one short instruction). Lets the guest survive ONE specific crash so
        // you can see whether it gates later progress. Found PPSA02664's boot gate (#238): skipping the
        // engine state-machine null-handler deref chain (`PROSPER_FAULT_SKIP=0x410b3:0x410bb`) makes the
        // title load its scene, instantiate MonoBehaviours, and run 76k GPU submits crash-free.
        if (const char* fsk = getenv("PROSPER_FAULT_SKIP")) {
            uint64_t foff = strtoull(fsk, nullptr, 0);
            const char* colon = strchr(fsk, ':');
            uint64_t roff = colon ? strtoull(colon + 1, nullptr, 0) : foff + 4;
            if (g_base && g_fault_rip == g_base + foff) { g[REG_RAX] = 0; g[REG_RIP] = g_base + roff; return; }
        }
        dump_fault_mem();   // no-op unless PROSPER_FAULTMEM is set
        // Dump the HWBP ring on the recoverable (armed/main-thread) crash too — the deser fault is kind=2.
        if (g_hwbp_node_on && !g_hwbp_ring_dumped) { uint64_t sfs = guest_fs_to_host_scoped();
            hwbp_dump_ring("recover"); guest_fs_restore_scoped(sfs); }
        if (g_armed_tid && cur_tid() == g_armed_tid) siglongjmp(g_jb, 1);
        // Fault on a thread with no recovery point (a guest worker thread). Report where
        // (async-signal-safe write) then terminate cleanly instead of a cross-thread longjmp.
        {
            char b[200];
            int n = snprintf(b, sizeof b, "[prosper] WORKER-THREAD FAULT: sig=%d addr=%p rip=0x%llx (image+0x%llx) rbp=0x%llx\n",
                             sig, g_fault_addr, (unsigned long long)g_fault_rip,
                             (unsigned long long)(g_base && g_fault_rip >= g_base ? g_fault_rip - g_base : g_fault_rip),
                             (unsigned long long)g_rbp);
            syscall(SYS_write, 2, b, (size_t)n);   /* raw: glibc write() reads the TCB via %fs (guest-fs unsafe in this handler) */
            // Classify the fault rip + fault-addr regions (which mapping / module) and dump the instruction
            // bytes at rip — turns the ASLR-relocated "rip=0x...48b" into an identifiable location.
            classify_addr(g_fault_rip);
            classify_addr((uint64_t)g_fault_addr);
            auto rdb = [](uint64_t a) -> unsigned { return probe_readable(a) ? *(const uint8_t*)a : 0x100u; };
            char ib[160]; int m = snprintf(ib, sizeof ib,
                "[prosper]   insn bytes @rip: %02x %02x %02x %02x %02x %02x %02x %02x  ret@[rsp]=0x%llx\n",
                rdb(g_fault_rip), rdb(g_fault_rip+1), rdb(g_fault_rip+2), rdb(g_fault_rip+3),
                rdb(g_fault_rip+4), rdb(g_fault_rip+5), rdb(g_fault_rip+6), rdb(g_fault_rip+7),
                (unsigned long long)(probe_readable(g_rsp) ? *(const uint64_t*)g_rsp : 0));
            syscall(SYS_write, 2,ib, m);
            // #312: full register + heap-window + GPU-write-ring dump at the worker fault too (the
            // MallocBinned3 freelist-pop faults — e.g. rcx=0x20015f00 at eboot+0x2316acf — land here,
            // not on the nullpage path). Reuses the async-signal-safe nullpage attribution dump.
            nullpage_deep_dump(uc, g_fault_rip);
            // If PROSPER_HWBP_NODE ring-capture is active, the crash node's typetree metadata is the tail.
            if (g_hwbp_node_on) hwbp_dump_ring("worker-fault");
        }
        // PROSPER_WORKER_PARK=1 (diagnostic): instead of terminating the whole process on a guest
        // worker-thread fault, park the faulting thread so the main thread can proceed to its own
        // recoverable crash (lets CRASHPEEK/PEEK_CLASS run). May deadlock if the worker held a lock.
        if (getenv("PROSPER_WORKER_PARK")) { for (;;) pause(); }
        _exit(90);
    }

    std::string trap_detail() {
        char buf[256];
        const char* sn = g_trap_sig == SIGILL ? "SIGILL" : g_trap_sig == SIGBUS ? "SIGBUS" : "SIGSEGV";
        uint64_t off = (g_base && g_fault_rip >= g_base) ? g_fault_rip - g_base : 0;
        snprintf(buf, sizeof buf, "%s at addr=%p  rip=0x%llx (image+0x%llx)",
                 sn, g_fault_addr, (unsigned long long)g_fault_rip, (unsigned long long)off);
        return buf;
    }

    uint64_t page_up(uint64_t v) { return (v + 0xfff) & ~((uint64_t)0xfff); }

    // Emit machine code into a stub slot.
    void emit_impl(uint8_t* p, uint64_t fn) {          // movabs rax,fn ; jmp rax
        p[0] = 0x48; p[1] = 0xB8; memcpy(p + 2, &fn, 8); p[10] = 0xFF; p[11] = 0xE0;
    }
    void emit_unimpl(uint8_t* p, uint32_t idx, uint64_t fn) { // mov edi,idx ; movabs rax,fn ; jmp rax
        p[0] = 0xBF; memcpy(p + 1, &idx, 4);
        p[5] = 0x48; p[6] = 0xB8; memcpy(p + 7, &fn, 8); p[15] = 0xFF; p[16] = 0xE0;
    }
    // --- Guest-%fs swap stubs (only when PROSPER_GUEST_FS is enabled). Guest code runs with %fs = guest
    // TP; HLE handlers need the host %fs (host libc TLS). The stub is ROBUST to the calling thread's fs:
    // it checks a magic at [fs+0x108] that marks OUR guest TCB (guest_tls.cpp). If present (guest thread),
    // it swaps %fs to the stashed host TCB [fs+0x100] for the handler call, then restores the guest %fs.
    // If absent (a host-context thread, or the main thread during pre-entry init), it just tail-calls the
    // handler on the current fs — exactly the old behavior. Uses FSGSBASE. Clobbers only rax/r11 (never
    // the arg regs rdi..r9).
    //
    // STACK-ARG FORWARDING: the guest path interposes `push r11` + `call` between the guest caller and
    // the handler, which shifts the caller's STACK args (args 7+) by two qwords — a handler reading its
    // 7th arg at the SysV frame slot fp[2] silently got the saved guest-fs base instead (ReleaseMem read
    // a TCB pointer as data_sel and the guest return address as the 64-bit fence value). The stub now
    // re-pushes the first TWO stack args before the call, so handlers see the documented SysV layout
    // (fp[2]=arg7, fp[3]=arg8) on BOTH paths. Alignment: entry rsp≡8 (caller's call), +3 pushes ≡0,
    // call ≡8 at handler entry — the SysV contract. Handlers needing >2 stack args would need this
    // widened. Keep 0x108/0x100/magic in sync with guest_tls.cpp (MAGIC_OFF/HOSTFS_OFF/TCB_MAGIC).
    void emit_swap_stub(uint8_t* p, uint32_t idx, uint64_t fn, bool unimpl) {
        uint8_t* s = p;
        auto mid = [&](uint8_t*& q) {   // per-variant: unimpl loads its slot index into edi first
            if (unimpl) { *q++ = 0xBF; memcpy(q, &idx, 4); q += 4; }         // mov edi, idx
            *q++ = 0x48; *q++ = 0xB8; memcpy(q, &fn, 8); q += 8;            // movabs rax, fn
        };
        // rdfsbase r11 ; cmp dword [r11+0x108], MAGIC ; jne .host
        *p++=0xF3; *p++=0x49; *p++=0x0F; *p++=0xAE; *p++=0xC3;
        *p++=0x41; *p++=0x81; *p++=0xBB; uint32_t mo=0x108; memcpy(p,&mo,4); p+=4;
        uint32_t magic=0x50524F53u; memcpy(p,&magic,4); p+=4;
        *p++=0x75; uint8_t* jne_rel = p++;                                  // jne rel8 (patched below)
        // .guest: push r11 ; push [rsp+0x18] (arg8) ; push [rsp+0x18] (arg7) ; mov rax,[r11+0x100] ;
        //         wrfsbase rax ; <mid> ; call rax ; add rsp,0x10 ; pop r11 ; wrfsbase r11 ; ret
        *p++=0x41; *p++=0x53;
        *p++=0xFF; *p++=0x74; *p++=0x24; *p++=0x18;                         // push qword [rsp+0x18] = arg8
        *p++=0xFF; *p++=0x74; *p++=0x24; *p++=0x18;                         // push qword [rsp+0x18] = arg7
        *p++=0x49; *p++=0x8B; *p++=0x83; uint32_t ho=0x100; memcpy(p,&ho,4); p+=4;
        *p++=0xF3; *p++=0x48; *p++=0x0F; *p++=0xAE; *p++=0xD0;
        mid(p);
        *p++=0xFF; *p++=0xD0;                                               // call rax
        *p++=0x48; *p++=0x83; *p++=0xC4; *p++=0x10;                         // add rsp, 0x10
        *p++=0x41; *p++=0x5B;                                               // pop r11
        *p++=0xF3; *p++=0x49; *p++=0x0F; *p++=0xAE; *p++=0xD3;              // wrfsbase r11
        *p++=0xC3;                                                          // ret
        // .host: <mid> ; jmp rax   (host-context: no swap, tail-call as before)
        *jne_rel = (uint8_t)(p - (jne_rel + 1));
        mid(p);
        *p++=0xFF; *p++=0xE0;                                               // jmp rax
        (void)s;
    }
    void emit_impl_swap(uint8_t* p, uint64_t fn)              { emit_swap_stub(p, 0,   fn, false); }
    void emit_unimpl_swap(uint8_t* p, uint32_t idx, uint64_t fn) { emit_swap_stub(p, idx, fn, true); }
}

bool map_image(const LoadedImage& img, std::string* err) {
    auto fail = [&](const char* s){ if (err) *err = s; return false; };
    void* want = (void*)(img.base + img.min_vaddr);
    size_t sz  = img.mem.size();
    // RWX for bring-up; per-segment W^X is a later refinement (shared LOAD pages).
    void* got = mmap(want, sz, PROT_READ | PROT_WRITE | PROT_EXEC,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (got == MAP_FAILED || got != want) return fail("mmap image at guest base failed");
    memcpy(got, img.mem.data(), sz);
    if (!g_base) g_base = img.base;   // main image, for fault-offset reporting
    return true;
}

bool install_stubs(const std::vector<ImportSlot>& slots, uint64_t stub_base,
                   uint64_t stub_size, std::string* err) {
    auto fail = [&](const char* s){ if (err) *err = s; return false; };
    if (stub_size < 24) return fail("stub_size too small (need >= 24)");
    if (!g_nid_db) g_nid_db = new NidDb();
    dispatch_init(&slots, g_nid_db);

    uint64_t n = slots.size();
    // Zero unresolved imports (e.g. a title whose every import resolved cross-module, or a dump whose
    // dynamic section yields none): nothing to emit — a 0-byte mmap would fail with EINVAL, so record
    // the empty table and succeed.
    if (n == 0) { g_stub_base = stub_base; g_stub_size = stub_size; g_nstubs = 0; return true; }
    uint64_t region = page_up(n * stub_size);
    void* want = (void*)stub_base;
    void* got = mmap(want, region, PROT_READ | PROT_WRITE | PROT_EXEC,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (got == MAP_FAILED || got != want) return fail("mmap stub region failed");

    bool swap = guest_tls_enabled();   // gated: emit the %fs swap stubs so HLE handlers run on the host TCB
    if (swap && stub_size < 96) return fail("stub_size too small for guest-%fs swap stub (need >= 96)");
    uint8_t* base = (uint8_t*)got;
    for (uint64_t i = 0; i < n; i++) {
        uint8_t* slot = base + i * stub_size;
        HleFn fn = Hle::lookup(slots[i].nid);
        if (fn) { if (swap) emit_impl_swap(slot, (uint64_t)fn);       else emit_impl(slot, (uint64_t)fn); }
        else    { if (swap) emit_unimpl_swap(slot, (uint32_t)i, (uint64_t)&prosper_on_unimpl);
                  else      emit_unimpl(slot, (uint32_t)i, (uint64_t)&prosper_on_unimpl); }
    }
    g_stub_base = stub_base; g_stub_size = stub_size; g_nstubs = n;
    // PROSPER_STUBDUMP: dump the stub table (index, guest offset from stub_base, lib::nid + resolved name).
    // Used to map a stub address seen on a stack (e.g. 0x600000000+off) back to the import it calls.
    if (getenv("PROSPER_STUBDUMP")) {
        for (uint64_t i = 0; i < n; i++) {
            const std::string& nm = g_nid_db ? g_nid_db->resolve(slots[i].nid) : std::string();
            fprintf(stderr, "[stub] #%llu off=0x%llx %s::%s %s\n", (unsigned long long)i,
                    (unsigned long long)(i * stub_size), slots[i].lib.c_str(), slots[i].nid.c_str(), nm.c_str());
        }
    }
    return true;
}

// #312 label-slot write watch (PROSPER_WATCH_LABEL=1): called by the AGC fence builder with the
// first heap-resident value-1 fence label address; protects the label's page and logs every write
// into the label's block with the writer's rip (guest allocator free-path vs prosper fence write),
// catching a write-after-free in the act. Diagnostic, one page, bounded to 400 logged hits.
// Arm the watch on `addr` unconditionally (shared body). Used by the env-gated wrapper below and
// by the PROSPER_WATCH_HOT trigger (hle_agc), which picks its own slot.
extern "C" void prosper_arm_label_watch_force(uint64_t addr) {
    if (g_lwatch_armed || !addr) return;
    if (const char* mx = getenv("PROSPER_WATCH_MAX")) {
        long v = atol(mx); if (v > 0) g_lwatch_max = (int)v;
    }
    g_lwatch_shift = getenv("PROSPER_WATCH_SHIFT") != nullptr;
    g_lwatch_slot = addr;
    g_lwatch_page = addr & ~(uint64_t)0xfff;
    if (mprotect((void*)g_lwatch_page, 0x1000, PROT_READ) == 0) {
        g_lwatch_armed = 1;
        fprintf(stderr, "[lwatch] armed slot=0x%llx page=0x%llx max=%d shift=%d\n",
                (unsigned long long)addr, (unsigned long long)g_lwatch_page, g_lwatch_max,
                g_lwatch_shift ? 1 : 0);
    }
}
extern "C" void prosper_arm_label_watch(uint64_t addr) {
    static const bool on = getenv("PROSPER_WATCH_LABEL") != nullptr;
    // PROSPER_WATCH_ABS=0xADDR (#312): watch a FIXED slot (e.g. the MallocBinned3 pool free-list
    // head found by the fault deep-dump) instead of the first fence label. The arm is retried on
    // every builder call until the slot's page is actually mapped (the pool region appears later
    // in boot than the first fence). Catches EVERY writer process-wide (guest or host) — the
    // definitive attribution for the 0x20015f00 head stomp.
    static const uint64_t abs_slot = [] { const char* e = getenv("PROSPER_WATCH_ABS");
                                          return e ? strtoull(e, nullptr, 0) : 0ull; }();
    if (g_lwatch_armed) return;
    if (abs_slot) addr = abs_slot;
    else if (!on || !addr) return;
    prosper_arm_label_watch_force(addr);
}

// Install a per-thread alternate signal stack so the fault handler can run even when the faulting
// thread's own stack is exhausted (a guest stack overflow otherwise delivers SIGSEGV with no usable
// stack -> the handler can't run -> the process cores uncatchably). Idempotent per thread. Worker
// threads call this from their trampoline; the main thread from install_trap_handler.
void install_sigaltstack() {
    static thread_local uint8_t* alt = nullptr;
    if (alt) return;
    const size_t sz = 256 * 1024;              // generous: deep real-libc call chains
    alt = (uint8_t*)malloc(sz);
    if (!alt) return;
    stack_t ss{}; ss.ss_sp = alt; ss.ss_size = sz; ss.ss_flags = 0;
    sigaltstack(&ss, nullptr);
}

void install_trap_handler() {
    g_faultmem = getenv("PROSPER_FAULTMEM") != nullptr;   // read once (getenv is not signal-safe)
    // Expose the perf HW write-watch to HLE code (dispatch.hpp g_hwwatch_hook): lets an HLE-side
    // diagnostic arm a watch on a runtime-discovered guest slot (one watch; extra calls ignored).
    // Installs the SIGTRAP handler on demand (the watch may be armed without PROSPER_HWBP).
    g_hwwatch_hook = [](uint64_t addr) {
        if (g_hwwatch_fd >= 0) return;
        struct sigaction ta{}; ta.sa_sigaction = fault_handler; ta.sa_flags = SA_SIGINFO;
        sigemptyset(&ta.sa_mask); sigaction(SIGTRAP, &ta, nullptr);
        arm_hwwatch(addr);
    };
    g_faultlog = getenv("PROSPER_FAULTLOG") != nullptr;
    g_skip_null_companion = getenv("PROSPER_SKIP_NULL_COMPANION") != nullptr;
    g_null_page = getenv("PROSPER_NULL_PAGE") != nullptr;
    g_watch_companion = getenv("PROSPER_WATCH_COMPANION") != nullptr;
    if (const char* r = getenv("PROSPER_SKIP_RIP"))    g_skip_rip    = strtoull(r, nullptr, 0);
    if (const char* t = getenv("PROSPER_SKIP_TARGET")) g_skip_target = strtoull(t, nullptr, 0);
    g_bp_shift = getenv("PROSPER_BP_SHIFT") != nullptr;
    // #312 per-thread MB3 head watch: expose an HLE-side arm hook. k_getspecific/k_setspecific call
    // through it with the per-thread cache base the instant it is handed to the guest; we arm a HW
    // write-watch on base+0x20 on the OWNING thread, before any store. Region-filtered to the MB3
    // FPoolInfo/cache range [0x2000000000,0x2100000000) so only real MB3 caches are watched.
    g_mb3w_on = getenv("PROSPER_MB3WATCH") != nullptr;
    if (const char* o = getenv("PROSPER_MB3WATCH_OFF")) g_mb3w_off = (int)strtol(o, nullptr, 0);
    if (const char* n = getenv("PROSPER_MB3WATCH_N"))   g_mb3w_nslots = (int)strtol(n, nullptr, 0);
    if (g_mb3w_on) {
        g_mb3_arm_hook = [](uint64_t base) {
            if (base < 0x2000000000ull || base >= 0x2100000000ull) return;   // MB3 pool-region only
            mb3w_arm_current_thread(base);
        };
    }
    if (g_faultmem || g_skip_null_companion || g_null_page) ensure_probe_pipe();
    // PROSPER_PEEK="r15:0x140,0x1a0;rbx:0x78,0x88;r15:*0x18+0x0" — N specs (';'), each reg:off[,off];
    // a '*pre+off' offset chases one pointer level ([[reg+pre]+off]). Parsed once (getenv unsafe at fault).
    if (const char* pk = getenv("PROSPER_PEEK")) {
        const char* spec = pk;
        while (*spec && g_peek_specs < 6) {
            const char* semi = strchr(spec, ';');
            const char* end  = semi ? semi : spec + strlen(spec);
            const char* colon = strchr(spec, ':');
            if (colon && colon < end && colon - spec <= 3) {
                PeekSpec& ps = g_peek[g_peek_specs++];
                for (const char* c = spec; c < colon; c++) ps.reg[c - spec] = *c;
                const char* s = colon + 1;
                while (s < end && ps.n < 12) {
                    if (*s == '*') { ps.deref[ps.n] = true; s++;
                        ps.pre[ps.n] = strtoull(s, nullptr, 0);
                        const char* plus = strchr(s, '+');
                        ps.off[ps.n] = (plus && plus < end) ? strtoull(plus + 1, nullptr, 0) : 0;
                    } else ps.off[ps.n] = strtoull(s, nullptr, 0);
                    ps.n++;
                    const char* comma = strchr(s, ','); if (!comma || comma >= end) break; s = comma + 1;
                }
            }
            if (!semi) break; spec = semi + 1;
        }
    }
    // PROSPER_HWWATCH_ABS=0xADDR — arm a hardware WRITE-watch on a FIXED absolute guest VA right now
    // (owner = this = the main guest thread). Unlike PROSPER_HWWATCH (register-relative, armed on the
    // first exec-bp hit), this catches writes to a known slot with no exec breakpoint. Each write logs
    // the writer RIP + the value stored (see the g_hwwatch handler), so a corrupt value (e.g. a magic
    // constant landing where a pointer belongs) is traced to its writer.
    if (const char* wa = getenv("PROSPER_HWWATCH_ABS")) {
        uint64_t addr = strtoull(wa, nullptr, 0);
        struct sigaction ta{}; ta.sa_sigaction = fault_handler; ta.sa_flags = SA_SIGINFO;
        sigemptyset(&ta.sa_mask); sigaction(SIGTRAP, &ta, nullptr);
        arm_hwwatch(addr);
    }
    // PROSPER_DUMPAT="0xADDR[,0xADDR...]" — absolute guest addresses to hex-dump at fault time.
    if (const char* da = getenv("PROSPER_DUMPAT")) {
        const char* s = da;
        while (*s && g_dumpat_n < 6) {
            g_dumpat[g_dumpat_n++] = strtoull(s, nullptr, 0);
            const char* comma = strchr(s, ','); if (!comma) break; s = comma + 1;
        }
        ensure_probe_pipe();   // DUMPAT uses probe_readable(); the readability probe now backs on a pipe (master #61)
    }
    // PROSPER_BP=0xOFFSET installs an int3 code-breakpoint-logger at guest VA 0x400000000+offset.
    if (const char* bp = getenv("PROSPER_BP")) {
        g_bp_addr = 0x400000000ull + strtoull(bp, nullptr, 0);
        if (const char* m = getenv("PROSPER_BP_MAX")) g_bp_max = (int)strtoul(m, nullptr, 0);
        ensure_probe_pipe();
        g_bp_on = true;   // the actual 0xCC is written after the image is mapped (arm_bp below)
    }
    // PROSPER_HWBP=0xOFFSET installs a race-free hardware execute breakpoint at guest VA 0x400000000+off.
    if (const char* hb = getenv("PROSPER_HWBP")) {
        g_hwbp_addr = 0x400000000ull + strtoull(hb, nullptr, 0);
        if (const char* m = getenv("PROSPER_HWBP_MAX")) g_hwbp_max = (int)strtoul(m, nullptr, 0);
        if (const char* c = getenv("PROSPER_HWBP_R15")) { g_hwbp_r15 = strtoull(c, nullptr, 0); g_hwbp_r15_on = true; }
        if (const char* c = getenv("PROSPER_HWBP_RAXMIN")) g_hwbp_raxmin = strtoull(c, nullptr, 0);
        if (const char* c = getenv("PROSPER_HWBP_ANOM")) { g_hwbp_anom = strtoull(c, nullptr, 0); g_hwbp_anom_on = true; }
        if (getenv("PROSPER_HWBP_NODE")) g_hwbp_node_on = true;
        if (getenv("PROSPER_STEPWIN")) { g_stepwin_on = true;
            if (const char* a = getenv("PROSPER_STEPWIN_AFTER")) g_stepwin_after = (int)strtol(a, nullptr, 0); }
        if (getenv("PROSPER_HWBP_BUFDUMP")) g_hwbp_bufdump = true;
        if (getenv("PROSPER_HWBP_DIVCAP")) g_hwbp_divcap = true;
        if (getenv("PROSPER_HWBP_STRDUMP")) g_hwbp_strdump = true;
        if (getenv("PROSPER_HWBP_KLASS")) g_hwbp_klass = true;
        if (getenv("PROSPER_HWBP_FIELDS")) g_hwbp_fields = true;   // latch here (#159): handler must not getenv()
        if (getenv("PROSPER_HWBP_OBJ")) g_hwbp_obj = true;
        if (getenv("PROSPER_HWBP_ARGS")) g_hwbp_args = true;
        g_hwbp_global = getenv("PROSPER_HWBP_GLOBAL");             // may be null; handler null-checks
        if (getenv("PROSPER_HWBP_ALLTHREADS")) g_hwbp_allthreads = true;
        ensure_probe_pipe();
        g_hwbp_on = true;   // perf_event_open done in arm_hwbp() after the image is mapped
        // PROSPER_HWWATCH=<signed delta>: on the first exec-bp hit, arm a data write-watch on [rax+delta]
        // (default -0x80). Reveals what writes the thread-local device slot the ctor reads.
        if (const char* w = getenv("PROSPER_HWWATCH")) {
            g_hwwatch_req = true;
            if (*w) g_hwwatch_delta = (int64_t)strtoll(w, nullptr, 0);
            if (const char* rg = getenv("PROSPER_HWWATCH_REG")) { strncpy(g_hwwatch_reg, rg, sizeof(g_hwwatch_reg)-1); g_hwwatch_reg[sizeof(g_hwwatch_reg)-1]=0; }
        }
    }
    install_sigaltstack();
    struct sigaction sa{};
    sa.sa_sigaction = fault_handler;
    sa.sa_flags = SA_SIGINFO;   // (SA_ONSTACK disabled by default: siglongjmp from the alt stack tripped
                                // glibc's %fs-guarded ____longjmp_chk -> jump-to-garbage fault storm)
    // PROSPER_FAULT_ONSTACK: run the handler on the per-thread sigaltstack. Needed to diagnose a
    // guest-thread STACK OVERFLOW: the fault destroys the thread's own stack, so a handler without an
    // alt stack cannot even enter (nested #PF -> forced-default SIGSEGV, killed with no report — the
    // exact "fatal signal 11, no output" we see mid-load). On the alt stack the worker-thread fault
    // path (which does NOT siglongjmp) can print the faulting RIP. Gated so the default boot's
    // main-thread siglongjmp recovery is unchanged. CONFIDENCE: HIGH (mechanism).
    if (getenv("PROSPER_FAULT_ONSTACK")) sa.sa_flags |= SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS,  &sa, nullptr);
    sigaction(SIGILL,  &sa, nullptr);
    if (g_watch_companion || g_bp_on || g_hwbp_on
        || getenv("PROSPER_WATCH_LABEL")
        || getenv("PROSPER_WATCH_ABS")
        || getenv("PROSPER_MB3WATCH")
        || getenv("PROSPER_WATCH_HOT")) sigaction(SIGTRAP, &sa, nullptr);   // single-step / breakpoint
}

// Write the 0xCC breakpoint into the (now-mapped) guest image. Call after the image load.
void arm_bp() {
    if (!g_bp_on || !g_bp_addr) return;
    g_bp_orig = *(volatile uint8_t*)g_bp_addr;
    bp_write_byte(g_bp_addr, 0xCC);
    char b[96];
    int n = snprintf(b, sizeof b, "[bp] armed int3 at eboot+0x%llx (orig=0x%02x)\n",
                     (unsigned long long)(g_bp_addr - 0x400000000ull), g_bp_orig);
    syscall(SYS_write, 2, b, (size_t)n);   /* raw: glibc write() reads the TCB via %fs (guest-fs unsafe in this handler) */
}

// Open + enable the PROSPER_HWBP hardware breakpoint (must run on the main/guest thread — the perf
// event monitors the calling thread). Call after the image is mapped, before jumping to the entry.
void arm_hwbp() {
    if (!g_hwbp_on || !g_hwbp_addr) return;
    long fd = perf_bp_open(g_hwbp_addr, HW_BREAKPOINT_X);
    char b[160];
    if (fd < 0) {
        int n = snprintf(b, sizeof b, "[hwbp] perf_event_open FAILED for eboot+0x%llx (errno=%d) — HW bp disabled\n",
                         (unsigned long long)(g_hwbp_addr - 0x400000000ull), errno);
        syscall(SYS_write, 2, b, (size_t)n);   /* raw: glibc write() reads the TCB via %fs (guest-fs unsafe in this handler) */ g_hwbp_on = false; return;
    }
    g_hwbp_fd = (int)fd;
    fcntl(g_hwbp_fd, F_SETFL, O_ASYNC);
    fcntl(g_hwbp_fd, F_SETSIG, SIGTRAP);
    struct f_owner_ex ow; ow.type = F_OWNER_TID; ow.pid = (pid_t)syscall(SYS_gettid);
    fcntl(g_hwbp_fd, F_SETOWN_EX, &ow);
    ioctl(g_hwbp_fd, PERF_EVENT_IOC_ENABLE, 0);
    t_hwbp_fd = g_hwbp_fd;   // main thread uses the same fd for its per-thread stepping state
    int n = snprintf(b, sizeof b, "[hwbp] armed HW execute bp at eboot+0x%llx (fd=%d tid=%ld)\n",
                     (unsigned long long)(g_hwbp_addr - 0x400000000ull), g_hwbp_fd, (long)syscall(SYS_gettid));
    syscall(SYS_write, 2, b, (size_t)n);   /* raw: glibc write() reads the TCB via %fs (guest-fs unsafe in this handler) */
}

// Arm the same execute bp on the CURRENT (worker) thread, gated by PROSPER_HWBP_ALLTHREADS. Each thread
// owns its own perf fd + SIGTRAP delivery, so off-main-thread execution of the target is observed too.
void arm_hwbp_this_thread() {
    if (getenv("PROSPER_HWBP_ARMLOG")) { char b[128]; int n = snprintf(b, sizeof b,
        "[hwbp] arm_this_thread tid=%ld on=%d all=%d addr=0x%llx tfd=%d\n", (long)syscall(SYS_gettid),
        g_hwbp_on, g_hwbp_allthreads, (unsigned long long)g_hwbp_addr, t_hwbp_fd); syscall(SYS_write, 2,b, n); }
    if (!g_hwbp_on || !g_hwbp_allthreads || !g_hwbp_addr || t_hwbp_fd >= 0) return;
    long fd = perf_bp_open(g_hwbp_addr, HW_BREAKPOINT_X);
    if (fd < 0) { char b[96]; int n = snprintf(b, sizeof b, "[hwbp] worker-arm FAILED tid=%ld errno=%d\n",
        (long)syscall(SYS_gettid), errno); syscall(SYS_write, 2,b, n); return; }
    t_hwbp_fd = (int)fd;
    fcntl(t_hwbp_fd, F_SETFL, O_ASYNC);
    fcntl(t_hwbp_fd, F_SETSIG, SIGTRAP);
    struct f_owner_ex ow; ow.type = F_OWNER_TID; ow.pid = (pid_t)syscall(SYS_gettid);
    fcntl(t_hwbp_fd, F_SETOWN_EX, &ow);
    ioctl(t_hwbp_fd, PERF_EVENT_IOC_ENABLE, 0);
    char b[128]; int n = snprintf(b, sizeof b, "[hwbp] armed on worker tid=%ld (fd=%d) for eboot+0x%llx\n",
        (long)syscall(SYS_gettid), t_hwbp_fd, (unsigned long long)(g_hwbp_addr - 0x400000000ull));
    syscall(SYS_write, 2,b, n);
}

uint64_t stub_addr(uint64_t idx) { return g_stub_base + idx * g_stub_size; }

uint64_t invoke_stub(uint64_t idx) {
    auto fn = (uint64_t(*)())(stub_addr(idx));
    return fn();
}

void register_thread_stack(uint64_t tid, void* base, uint64_t size) {
    std::lock_guard<std::mutex> lk(g_smx);
    g_stacks[tid] = { (uint64_t)base, size };
}
void unregister_thread_stack(uint64_t tid) {
    std::lock_guard<std::mutex> lk(g_smx);
    g_stacks.erase(tid);
}
bool guest_stack_for_thread(uint64_t tid, void** base, size_t* size) {
    std::lock_guard<std::mutex> lk(g_smx);
    auto it = g_stacks.find(tid);
    if (it == g_stacks.end()) return false;
    *base = (void*)it->second.first; *size = (size_t)it->second.second;
    return true;
}
bool guest_stack_for_current_thread(void** base, size_t* size) {
    return guest_stack_for_thread((uint64_t)pthread_self(), base, size);
}

// Guest-address ranges whose module_start needs a real SCE module-param descriptor (see header).
std::vector<std::pair<uint64_t, uint64_t>> g_modstart_param_ranges;
// The descriptor Sony's PSN/SaveData plugin module_start validates: [+0]=0x10 (size), [+4]=0x200
// (interface version the native plugin requires), [+8]=0 (optional callback-registration fn ptr;
// NULL is honored — PSN_PrxInitialize skips the optional callbacks and still reports its real
// native version). Static so it has a stable, guest-readable address for the whole run.
static const struct __attribute__((packed)) { uint32_t size; uint32_t version; uint64_t cb; }
    g_modstart_desc = { 0x10, 0x200, 0 };

void set_module_start_param_ranges(const std::vector<std::pair<uint64_t, uint64_t>>& ranges) {
    g_modstart_param_ranges = ranges;
}

// Find the /proc/self/maps region containing `addr`: [s,e) and its "rwxp" perms. False if unmapped.
static bool region_of(uint64_t addr, uint64_t& s, uint64_t& e, char perms[5]) {
    FILE* mf = fopen("/proc/self/maps", "re");
    if (!mf) return false;
    char line[512];
    while (fgets(line, sizeof line, mf)) {
        unsigned long long ls = 0, le = 0; char pr[8] = {0};
        if (sscanf(line, "%llx-%llx %4s", &ls, &le, pr) != 3) continue;
        if (addr >= ls && addr < le) { s = ls; e = le; memcpy(perms, pr, 4); perms[4] = 0; fclose(mf); return true; }
    }
    fclose(mf);
    return false;
}

// Print the `n` bytes at [start, start+n) for the init-fault report WITHOUT risking a second,
// fatal fault (#128): the faulting rip can be a wild/null jump — the exact case this diagnostic
// exists for — and by the time the report runs the sigsetjmp guard is disarmed, so an unguarded
// deref (or an mprotect that silently failed) turns a tolerated, logged failure into process
// death. Each region the range touches is checked in /proc/self/maps; unmapped stretches print a
// marker. Execute-only pages get PROT_READ temporarily and their ORIGINAL protection restored
// (the old code force-set R|X, stripping W from an RW span and leaving the page readable forever).
static void dump_fault_bytes(uint64_t start, int n) {
    uint64_t cur = start;
    int remaining = n;
    while (remaining > 0) {
        uint64_t s = 0, e = 0; char perms[5] = {0};
        if (!region_of(cur, s, e, perms)) { fprintf(stderr, " (unmapped@0x%llx)", (unsigned long long)cur); return; }
        int chunk = (int)((e - cur) < (uint64_t)remaining ? (e - cur) : (uint64_t)remaining);
        bool readable = perms[0] == 'r';
        int prot_orig = (perms[0] == 'r' ? PROT_READ : 0) | (perms[1] == 'w' ? PROT_WRITE : 0)
                      | (perms[2] == 'x' ? PROT_EXEC : 0);
        uint64_t pg_lo = cur & ~0xfffull;
        size_t span = (size_t)(((cur + chunk - 1) & ~0xfffull) - pg_lo + 0x1000);
        if (!readable && mprotect((void*)pg_lo, span, prot_orig | PROT_READ) != 0) {
            fprintf(stderr, " (unreadable@0x%llx)", (unsigned long long)cur); return;
        }
        const uint8_t* p = (const uint8_t*)cur;
        for (int i = 0; i < chunk; i++) fprintf(stderr, " %02x", p[i]);
        if (!readable) mprotect((void*)pg_lo, span, prot_orig);   // restore execute-only
        cur += chunk; remaining -= chunk;
    }
}

size_t run_guest_inits(const std::vector<uint64_t>& fns) {
    size_t ok = 0;
    for (uint64_t f : fns) {
        g_trap_kind = 0; g_armed_tid = cur_tid();
        // Call with the PS5 module-entry ABI module_start(size_t argc, const void* argp): pass argc=0,
        // argp=NULL. Plain init_array ctors take no args and harmlessly ignore rdi/rsi (SysV), but a real
        // module_start (e.g. the PSN.prx / SaveData.prx plugins) READS these — calling f() left garbage in
        // rdi/rsi, so it dereferenced a garbage argp and faulted at boot (SIGSEGV addr=0x1/0x5). CONFIDENCE: HIGH.
        // A plugin module_start registered in g_modstart_param_ranges further requires a valid descriptor:
        // its user module_start validates argp={0x10,0x200,...} and null-faults on (0,NULL). CONFIDENCE: HIGH.
        uint64_t argc = 0, argp = 0;
        for (auto& r : g_modstart_param_ranges)
            if (f >= r.first && f < r.second) { argc = 0x10; argp = (uint64_t)&g_modstart_desc; break; }
        if (sigsetjmp(g_jb, 1) == 0) { ((void (*)(uint64_t, uint64_t))(uintptr_t)f)(argc, argp); ok++; }
        g_armed_tid = 0;
        if (g_trap_kind) {
            fprintf(stderr, "[prosper] init fn 0x%llx faulted (%s); continuing\n",
                    (unsigned long long)f, trap_detail().c_str());
            // Full register capture at the init fault — diagnoses the PSN.prx module_start fault
            // (null global / TLS / unresolved import). g_r* were latched by the fault handler.
            fprintf(stderr, "[prosper]   rax=%016llx rbx=%016llx rcx=%016llx rdx=%016llx\n"
                            "[prosper]   rsi=%016llx rdi=%016llx rbp=%016llx rsp=%016llx\n"
                            "[prosper]   r8 =%016llx r9 =%016llx r10=%016llx r11=%016llx\n"
                            "[prosper]   r12=%016llx r13=%016llx r14=%016llx r15=%016llx\n",
                    (unsigned long long)g_rax, (unsigned long long)g_rbx, (unsigned long long)g_rcx,
                    (unsigned long long)g_rdx, (unsigned long long)g_rsi, (unsigned long long)g_rdi,
                    (unsigned long long)g_rbp, (unsigned long long)g_rsp, (unsigned long long)g_r8,
                    (unsigned long long)g_r9, (unsigned long long)g_r10, (unsigned long long)g_r11,
                    (unsigned long long)g_r12, (unsigned long long)g_r13, (unsigned long long)g_r14,
                    (unsigned long long)g_r15);
            // Dump the 24 bytes around the faulting rip — resolves the exact faulting instruction.
            // The exec segment is often mapped execute-only (PS5 PF_X with no PF_R), so
            // dump_fault_bytes temporarily adds PROT_READ (and restores the original protection);
            // a wild/null rip prints an unmapped marker instead of faulting the reporter (#128).
            fprintf(stderr, "[prosper]   bytes@rip-8:");
            dump_fault_bytes(g_fault_rip - 8, 24);
            fprintf(stderr, "  (rip=0x%llx)\n", (unsigned long long)g_fault_rip);
            fprintf(stderr, "[prosper]   bytes@entry:");
            dump_fault_bytes(f, 24);
            fprintf(stderr, "  (entry=0x%llx)\n", (unsigned long long)f);
        }
    }
    return ok;
}

BootResult run_entry(const LoadedImage& img) {
    const size_t STK = 16 * 1024 * 1024;
    void* stk = mmap(nullptr, STK, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    BootResult r;
    if (stk == MAP_FAILED) { r.kind = 2; r.detail = "guest stack mmap failed"; return r; }

    register_thread_stack((uint64_t)pthread_self(), stk, STK);   // guest main thread

    uint64_t top = ((uint64_t)stk + STK) & ~(uint64_t)0xf;
    // Guest argv[0] is always the eboot path. Optionally inject extra args via PROSPER_GUEST_ARGS
    // (space-separated) — e.g. Unity's own switches like "-force-gfx-direct" to force single-threaded
    // rendering. This is a legitimate compat-layer configuration (the game parses these switches), off by
    // default so the normal boot is unchanged. CONFIDENCE: HIGH on the argv/crt0 stack layout.
    std::vector<std::string> args = { "/app0/eboot.bin" };
    if (const char* extra = getenv("PROSPER_GUEST_ARGS")) {
        const char* p = extra;
        while (*p) {
            while (*p == ' ') p++;
            if (!*p) break;
            const char* s = p;
            while (*p && *p != ' ') p++;
            args.emplace_back(s, (size_t)(p - s));
        }
    }
    // Copy each arg string onto the stack (high→low), collecting guest pointers.
    std::vector<uint64_t> argptrs;
    for (const auto& a : args) {
        size_t n = a.size() + 1;
        top -= n; memcpy((void*)top, a.c_str(), n);
        argptrs.push_back(top);
    }
    // PROSPER_GUEST_ENV="KEY=VAL;KEY=VAL": inject environment strings the guest's libc getenv() will see
    // (the crt0 envp is otherwise empty). Used to feed the guest's bundled Boehm GC its env knobs
    // (e.g. GC_DONT_GC=1 to disable collection, or GC_INITIAL_HEAP_SIZE) when investigating GC-driven
    // corruption. ';'-separated so values may contain '='. CONFIDENCE: HIGH on the SysV envp layout.
    std::vector<uint64_t> envptrs;
    if (const char* ge = getenv("PROSPER_GUEST_ENV")) {
        std::string s(ge); size_t i = 0;
        while (i < s.size()) {
            size_t j = s.find(';', i);
            std::string kv = s.substr(i, j == std::string::npos ? std::string::npos : j - i);
            if (!kv.empty()) { size_t n = kv.size() + 1; top -= n; memcpy((void*)top, kv.c_str(), n); envptrs.push_back(top); }
            if (j == std::string::npos) break; i = j + 1;
        }
        top &= ~(uint64_t)0xf;
    }
    top &= ~(uint64_t)0xf;
    // crt0 vector: argc, argv[0..N-1], NULL, envp[0..M-1], NULL, auxv AT_NULL(0,0).
    std::vector<uint64_t> vecv;
    vecv.push_back(args.size());
    for (uint64_t pp : argptrs) vecv.push_back(pp);
    vecv.push_back(0);   // argv terminator
    for (uint64_t pp : envptrs) vecv.push_back(pp);
    vecv.push_back(0);   // envp terminator
    vecv.push_back(0); vecv.push_back(0);   // auxv AT_NULL
    if (vecv.size() & 1) vecv.push_back(0);   // keep an even count so the ≡8(mod16) placement below holds
    const uint64_t* vec = vecv.data(); size_t vecsz = vecv.size() * sizeof(uint64_t);
    top -= vecsz; top &= ~(uint64_t)0xf;   // 16-aligned base for the vector
    // The Sony crt _start pushes an odd number of words before its first call, so it
    // expects entry rsp ≡ 8 (mod 16) (like a normal callee), NOT 16-aligned. Placing
    // the vector 8 below a 16-boundary makes every downstream call correctly aligned,
    // so alignment-sensitive SIMD (vmovaps) in callees doesn't #GP.
    top -= 8;
    memcpy((void*)top, vec, vecsz);
    uint64_t sp = top, rdi = sp, rsi = 0;

    g_trap_kind = 0; g_fault_addr = nullptr; g_fault_rip = 0; g_armed_tid = cur_tid();
    arm_bp();     // write the PROSPER_BP int3 now that the guest image is fully mapped
    arm_hwbp();   // open the PROSPER_HWBP hardware breakpoint on this (guest main) thread
    if (sigsetjmp(g_jb, 1) == 0) {
        // gated (PROSPER_GUEST_FS): switch %fs to a guest TCB as the LAST host action before entering the
        // guest — any host C++ after this point would run on the guest TCB. Import stubs swap back per-call.
        guest_tls_activate_thread();
        register uint64_t e  asm("rax") = img.entry;
        register uint64_t s  asm("r8")  = sp;
        register uint64_t d  asm("r9")  = rdi;
        register uint64_t si asm("r10") = rsi;
        __asm__ volatile(
            "mov %%r8, %%rsp\n\t" "mov %%r9, %%rdi\n\t" "mov %%r10, %%rsi\n\t"
            "xor %%rbp, %%rbp\n\t" "jmp *%%rax\n\t"
            : : "r"(e), "r"(s), "r"(d), "r"(si) : "memory");
        r.kind = 0; r.detail = "entry returned";
    } else {
        r.kind = (int)g_trap_kind;
        r.detail = trap_detail();
        r.fault_addr = (uint64_t)g_fault_addr;
        r.fault_rip = g_fault_rip;
        r.rbp = g_rbp; r.rsp = g_rsp; r.rax = g_rax;
        r.rdi = g_rdi; r.rsi = g_rsi; r.rdx = g_rdx; r.rbx = g_rbx;
        // Walk the rbp chain for a backtrace, guarded so a bad frame can't crash us.
        g_armed_tid = cur_tid();
        if (sigsetjmp(g_jb, 1) == 0) {
            uint64_t bp = g_rbp;
            for (int i = 0; i < 24 && bp > 0x10000; i++) {
                uint64_t ret = *(uint64_t*)(bp + 8);
                if (ret) r.backtrace.push_back(ret);
                uint64_t nbp = *(uint64_t*)bp;
                if (nbp <= bp) break;
                bp = nbp;
            }
        }
        g_armed_tid = 0;
    }
    return r;
}

} // namespace prosper
#endif // __linux__
