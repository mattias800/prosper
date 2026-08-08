// exec_image_linux.cpp — Linux host backing + HLE stubs (M2/M3). Compiles to nothing
// on non-Linux so the shared (mingw) build is unaffected.
#include "exec_image.hpp"
#include "sse4a.hpp"
#include "x86_read_decode.hpp"
#include "guest_write_watch.hpp"
#include "fs_emu.hpp"
#include "raw_syscall.hpp"
#include "boot_program.hpp"   // #1659: shared guest-module labelling (BOOT_* bases)
#include "../hle/nid.hpp"
#include "../hle/dispatch.hpp"

#if defined(__linux__) || defined(__APPLE__)
#include "posix_shim.hpp"
#include "fault_context.hpp"   // #2018: one fault's own registers, snapshotted from ITS ucontext
#include <sys/mman.h>
#include <signal.h>
#include <setjmp.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#ifdef __linux__
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#endif
#include <cstdio>
#include <cstring>
#include <dlfcn.h>   // dladdr: name the host object behind a backtrace frame (#2194)
#include <cstdlib>
#include <cstdint>
#include <cerrno>
#include <algorithm>
#include <map>
#include <mutex>
#include <atomic>

// Darwin mcontext/perf compatibility (PROSPER_GREGS / PROSPER_REG_ERR / PROSPER_SI_FD, the
// HW_BREAKPOINT_*/F_SETSIG constants, and mach headers) lives in posix_shim.hpp — shared with the
// guest-exception delivery handler in hle_kernel.cpp.
#ifdef __APPLE__
// Hardware break/watchpoints are a Linux perf_event capability with no Darwin userland
// equivalent (and Rosetta exposes no x86 debug registers anyway). perf_bp_open below returns
// -1/ENOTSUP, so every PROSPER_HWBP/HWWATCH arm attempt takes its existing "arm FAILED" path and
// the software-watch (mprotect) + int3 diagnostics remain the macOS toolset. These constants only
// keep the never-reached call sites compiling; the fds involved are always -1 here.
enum { HW_BREAKPOINT_X = 4, HW_BREAKPOINT_W = 2, HW_BREAKPOINT_LEN_8 = 8 };
enum { PERF_EVENT_IOC_ENABLE = 0x2400, PERF_EVENT_IOC_DISABLE = 0x2401 };
enum { F_OWNER_TID = 0 };
struct f_owner_ex { int type; pid_t pid; };
#ifndef F_SETSIG
#define F_SETSIG    -1000   // unknown fcntl cmd -> EINVAL; only ever issued on fd=-1 (EBADF)
#endif
#ifndef F_SETOWN_EX
#define F_SETOWN_EX -1001
#endif
#endif

namespace prosper {

namespace {
    uint64_t g_base = 0, g_stub_base = 0, g_stub_size = 0, g_nstubs = 0;
    // #1659: label guest addresses through the shared, module-aware helpers instead of subtracting a
    // literal base. These sites hard-coded 0x400000000 — the eboot's address BEFORE #825 relocated it
    // to 0x410000000 — so every printed offset was 0x10000000 too high and did not round-trip through
    // PROSPER_BP, which adds the real mapped base. Async-signal-safe: pure comparisons and arithmetic.
    inline const char* gmod(uint64_t a) { return prosper::guest_module_name(a); }
    inline uint64_t    goff(uint64_t a) { return prosper::guest_module_offset(a); }
    inline bool        gin(uint64_t a)  { return prosper::guest_va_in_module(a); }
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
    inline long cur_tid() { return (long)prosper_gettid(); }
    volatile sig_atomic_t g_trap_kind = 0;   // 0 none, 2 fatal fault, 3 ILL
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
    // #1944: PROSPER_LAZY_COMMIT_STRICT=1 declines to back a reserved-but-uncommitted page, so the
    // SIGSEGV reports at the LOADING instruction with the real faulting address instead of handing
    // the guest a zero and reporting the dereference of that zero one instruction later. Read once
    // at install time — getenv() is not async-signal-safe. Default OFF: the normal boot keeps the
    // lazy commit, which several titles depend on to get past their allocator bring-up.
    bool g_lazy_commit_strict = false;
    // Whole-run census (#1944). ATOMIC, not `volatile sig_atomic_t`: this handler runs on every
    // guest thread, and the ordinal now backs the claim "exactly one lazy-commit event per affected
    // run" — a non-atomic read-modify-write could lose an increment and turn two events into one,
    // which is precisely the statement being made. Relaxed is enough (a counter, not a fence) and
    // `fetch_add` on a lock-free atomic is async-signal-safe.
    std::atomic<unsigned> g_lazy_commit_events{0};
    bool g_int41_skip = true;   // guest int $0x41 (RAGE debugbreak) -> skip; PROSPER_NO_INT41_SKIP disables (#1138)
    // Probe pipe for probe_readable() below — a pipe write imports the source pages (EFAULT on
    // unmapped memory) where a /dev/null write does not. O_NONBLOCK so a full pipe can never
    // block the fault handler; drained after every probe.
    int  g_probe_pipe[2] = {-1, -1};
    // One-time init via a C++11 magic static (thread-safe): the unguarded `if (fd < 0) pipe2`
    // pattern could tear the fd pair under two concurrent first-calls (PR #61 review). Called
    // only from install-time paths (never from the signal handler itself), so the guard's
    // internal locking is safe here.
    bool make_probe_pipe() {
#ifdef __linux__
        return pipe2(g_probe_pipe, O_CLOEXEC | O_NONBLOCK) == 0;
#else   // Darwin: no pipe2; pipe + fcntl inside the same once-only guard, so no torn-pair race
        if (pipe(g_probe_pipe) != 0) return false;
        for (int fd : g_probe_pipe)
            if (fcntl(fd, F_SETFD, FD_CLOEXEC) != 0 || fcntl(fd, F_SETFL, O_NONBLOCK) != 0) return false;
        return true;
#endif
    }
    void ensure_probe_pipe() {
        static const bool ok = make_probe_pipe();
        if (!ok) g_probe_pipe[0] = g_probe_pipe[1] = -1;
    }
    // DIAGNOSTIC (PROSPER_SKIP_NULL_COMPANION, default off): at the Unity GfxDevice pipeline reader
    // eboot+0xba6e08 — which derefs a null GPU-companion pointer [obj+0x140] for pipelines that were
    // never processed — log the object state and redirect RIP to the reader's own skip label
    // (eboot+0xba6e40, where its type/flag-check branches already land) so processing continues as if
    // the companion weren't needed. This is a *probe* to reveal whether the null companion is the sole
    // blocker or one of a cascade — NOT a fix (the companions are still not real). Overridable via
    // PROSPER_SKIP_RIP / PROSPER_SKIP_TARGET. NOTE: the two defaults below are absolute guest VAs
    // computed against the PRE-#825 eboot base (0x400000000) and are stale for that reason (#1659);
    // they are Messenger-specific probe addresses and are only meaningful when explicitly overridden.
    bool     g_skip_null_companion = false;
    uint64_t g_skip_rip    = 0x400ba6e08ull;   // reader companion deref (mov 0x8(%rsi),%rax; rsi=null)
    uint64_t g_skip_target = 0x400ba6e40ull;   // reader skip label (continues via [obj+0x520/0x530])
    volatile sig_atomic_t g_skip_count = 0;
    bool     g_null_page = false;              // PROSPER_NULL_PAGE: back low null reads with zero page
    volatile sig_atomic_t g_null_page_count = 0;
    unsigned g_null_page_mask = 0;             // which of the 16 low pages [0,0x10000) are backed
    void nullpage_deep_dump(ucontext_t* uc, uint64_t rip);
    // #1226: the command processor's per-label protocol ring, already exported for exactly this
    // kind of out-of-band question. Writes "events(total=0, no-history)" for an untracked address.
    extern "C" void prosper_label_hist_dump(uint64_t addr, char* out, unsigned cap);
#ifdef __linux__
    // Linux's gregset order is not the architectural x86 register encoding used by
    // decode_low_read_dest(): map rax..r15 explicitly rather than treating the two layouts as alike.
    constexpr int kX86ToLinuxGreg[16] = { REG_RAX, REG_RCX, REG_RDX, REG_RBX,
                                           REG_RSP, REG_RBP, REG_RSI, REG_RDI,
                                           REG_R8,  REG_R9,  REG_R10, REG_R11,
                                           REG_R12, REG_R13, REG_R14, REG_R15 };

    // When the host forbids mapping below vm.mmap_min_addr, preserve the NULL_PAGE probe's narrow
    // read-as-zero behavior only for an instruction form the shared decoder proves safe. This is a
    // fallback for the same low-page mapping the probe would otherwise install, not a general null
    // dereference recovery: writes, instruction fetches, protection faults, map failures other than
    // the host-policy denials, and every unknown instruction all remain on the normal fatal path.
    bool try_emulate_denied_null_read(ucontext_t* uc, uint64_t addr, uint64_t rip, long map_result) {
        constexpr uint64_t kPageFaultWrite = 1ull << 1;
        constexpr uint64_t kPageFaultInstructionFetch = 1ull << 4;
        if (map_result != -EPERM && map_result != -EACCES) return false;
        if (addr >= 0x10000ull || addr == rip || !uc ||
            (PROSPER_REG_ERR(uc) & (kPageFaultWrite | kPageFaultInstructionFetch))) return false;

        // The current instruction page was executable (or the CPU could not have reached this data
        // fault). Do not read beyond that page in the handler: a cross-page instruction is declined
        // rather than risking a nested fault while decoding it.
        size_t avail = (size_t)(0x1000ull - (rip & 0xfffull));
        if (avail > 15) avail = 15; // x86-64's architectural maximum instruction length
        int dest = 0, insn_len = 0;
        if (!decode_low_read_dest((const uint8_t*)rip, avail, &dest, &insn_len)) return false;

        auto regs = PROSPER_GREGS(uc);
        regs[kX86ToLinuxGreg[dest]] = 0;
        regs[REG_RIP] = (greg_t)(rip + (uint64_t)insn_len);

        g_null_page_count = g_null_page_count + 1;
        const int count = (int)g_null_page_count;
        // Keep the fallback observable without allowing a malformed/null-chain loop to flood a log.
        // The printed monotonic count makes the cap explicit rather than disguising it as frequency.
        if (count <= 64 || (count & 1023) == 0) {
            char b[192];
            int n = snprintf(b, sizeof b,
                             "[nullpage] #%d EMULATED-READ addr=0x%llx rip=%s+0x%llx maperr=%ld\n",
                             count, (unsigned long long)addr, gmod(rip),
                             (unsigned long long)goff(rip), map_result);
            raw_write_fmt(2, b, sizeof b, n);
        }
        nullpage_deep_dump(uc, rip);
        return true;
    }
#endif
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
        // #1226: widened from the DOLL-era [0x20..0x21) to [0x20..0x40) — the current faults on both
        // DOLL and ArcRunner store 0x30015f00 (<<8 = 0x30015f0000), which the old window could not
        // see (dmem layout moved). Matches is_byteshift_poolptr in command_processor.cpp.
        return (v >> 32) == 0 && ((v << 8) >= 0x2000000000ull && (v << 8) < 0x4000000000ull);
    }
    // The value a MallocBinned3 free-list head slot may legally hold: either 0 (empty list) or the
    // address of an FBundleNode, which is a naturally 0x20-aligned block inside the guest's mapped
    // heap/dmem window. ANY other value in that slot is corruption by construction, whatever shape it
    // has — so this, not a value-shape guess, is what the head watch must trigger on. Without it the
    // watch reports only `lwatch_is_pool_shift` values and stays SILENT for every other corrupt head
    // (Crisis Core PPSA07809 observed 0xff000000ff000000 and 0x0002400100024001 in the same slot),
    // which reads exactly like "armed and saw nothing wrong".
    static inline bool lwatch_is_plausible_bundle_node(uint64_t v) {
        if (v == 0) return true;
        if (v & 0x1full) return false;
        return v >= 0x2000000000ull && v < 0x4000000000ull;
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
    uint64_t g_bp_addr = 0;                    // guest VA of the breakpoint (mapped base + offset)
    uint8_t  g_bp_orig = 0;                    // original byte replaced by 0xCC
    bool     g_bp_stepping = false;            // mid single-step (orig byte restored, TF set)
    volatile sig_atomic_t g_bp_count = 0;
    int      g_bp_max = 400;                   // cap log volume
    // PROSPER_BP_KLASS=1: at each int3 hit, print the IL2CPP class name of the object in each SysV-arg
    // register (rdi/rsi/rdx/rcx/r8/r9) plus rbx/rax/r14/r15 — reliably names the managed receiver/args
    // WITHOUT gdb (the int3 is armed by prosper after it maps the module, so it is race-free w.r.t. the
    // demand-map timing that makes gdb managed breakpoints nondeterministic). Reuses the HWBP obj->klass->
    // name layout. For an IL2CPP-module method use PROSPER_BP=0x30000000+RVA (module base 0x440000000 =
    // eboot base 0x410000000 + 0x30000000).
    bool     g_bp_klass = false;
    // PROSPER_BP_PROBE="<chain>[;<chain>...]": at each hit, evaluate pointer chains and print each result.
    // A chain is "<start>[+0xN|*]...:<t>": start is a register name (rdi,r14,...) or a 0xADDR; ops apply
    // left-to-right — "+N" adds an offset, "*" dereferences a qword; the trailing ":t" reads the final
    // address as q(word,default)/d(word)/b(yte)/f(loat-bits). Lets a single breakpoint read a managed
    // field chain live, e.g. a mode flag "0x441f32088*+0xb8*:b" or a curtain color.a "rdi+0x20*+...:f".
    const char* g_bp_probe = nullptr;
    // PROSPER_KSCAN=0x<Il2CppClass> : on the FIRST PROSPER_BP hit, scan the guest GC heap for every live
    // object whose [obj]==this class and dump each address + PROSPER_KSCAN_FIELDS. Get the class address from a
    // [bp-klass] line (which prints klass=0x..). This enumerates all instances of a class WITHOUT gdb/GC
    // internals — pure guest-memory scan. PROSPER_KSCAN_FIELDS uses the probe grammar with "@" = the found
    // object (e.g. "@+0x18:b;@+0x40*+0x10:s"). PROSPER_KSCAN_MAX caps dumped matches.
    uint64_t    g_kscan_klass = 0;
    const char* g_kscan_fields = nullptr;
    int         g_kscan_max = 64;
    bool        g_kscan_done = false;
    void il2cpp_kscan(void* uctx);   // defined after bp_eval_probes
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
    std::atomic<GuestExecutionThreadEnterTestHook> g_guest_execution_enter_test_hook{nullptr};
    std::atomic<void*> g_guest_execution_enter_test_opaque{nullptr};
    // Signal handlers can run while the guest owns %fs, so C++ thread_local state is not usable there.
    // Publish perf fds by kernel tid before enabling them; the handler then finds its own step state
    // without touching host TLS. Fixed storage also keeps lookup allocation/lock-free in the handler.
    static constexpr int HWBP_THREAD_MAX = 512;
    struct HwbpThreadState {
        volatile long tid;
        volatile sig_atomic_t fd;
        volatile sig_atomic_t stepping;
    };
    HwbpThreadState g_hwbp_threads[HWBP_THREAD_MAX] = {};
    volatile int g_hwbp_thread_count = 0;
    HwbpThreadState* hwbp_thread_state(long tid) {
        const int count = __atomic_load_n(&g_hwbp_thread_count, __ATOMIC_ACQUIRE);
        for (int i = 0; i < count && i < HWBP_THREAD_MAX; ++i) {
            if (__atomic_load_n(&g_hwbp_threads[i].tid, __ATOMIC_ACQUIRE) == tid)
                return &g_hwbp_threads[i];
        }
        return nullptr;
    }
    bool hwbp_register_thread(long tid, int fd) {
        const int slot = __atomic_fetch_add(&g_hwbp_thread_count, 1, __ATOMIC_ACQ_REL);
        if (slot >= HWBP_THREAD_MAX) return false;
        g_hwbp_threads[slot].fd = fd;
        g_hwbp_threads[slot].stepping = 0;
        __atomic_store_n(&g_hwbp_threads[slot].tid, tid, __ATOMIC_RELEASE);
        return true;
    }
    // PROSPER_HWBP_R15=<hex>: only LOG when r15 == this value (a condition, evaluated in-process = no gdb
    // round-trip). When matched, also dump a window of memory around rax + classify rax's mapping. Built to
    // catch the one deserializer read that produces a garbage std::string length without the gdb-bp overhead
    // that times out on hot addresses. r15 is used because it carries the read length at eboot+0x7e40e1.
    uint64_t g_hwbp_r15 = 0; bool g_hwbp_r15_on = false;
    // PROSPER_HWBP_R14=<hex>: the same exact-value log gate for r14. In particular, value 0 lets a
    // breakpoint stay armed through valid resource descriptors and report only the first missing one.
    uint64_t g_hwbp_r14 = 0; bool g_hwbp_r14_on = false;
    // PROSPER_HWBP_RET=<absolute VA>: only log calls whose stack return address matches. This isolates
    // one caller of a hot shared function while still stepping/rearming silently for all other hits.
    uint64_t g_hwbp_ret = 0; bool g_hwbp_ret_on = false;
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
    struct HwbpRingEnt { const char* rip_mod; unsigned long long rip_off, cur, rax; unsigned val;
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
    struct StepEnt { const char* rip_mod; unsigned long long rip_off, cur; };
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
    const char* g_hwbp_probe = nullptr;  // PROSPER_HWBP_PROBE: generic pointer chains on gated hits
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
#ifdef __linux__
        struct perf_event_attr pe; memset(&pe, 0, sizeof pe);
        pe.type = PERF_TYPE_BREAKPOINT; pe.size = sizeof pe;
        pe.bp_type = bp_type; pe.bp_addr = addr;
        pe.bp_len = (bp_type == HW_BREAKPOINT_X) ? sizeof(long) : (uint64_t)HW_BREAKPOINT_LEN_8;
        pe.sample_period = 1; pe.disabled = 1; pe.exclude_kernel = 1; pe.exclude_hv = 1;
        return syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0UL);
#else
        (void)addr; (void)bp_type; errno = ENOTSUP; return -1;   // see the Darwin compat note above
#endif
    }
    // Async-safe-ish: write the /proc/self/maps line containing `addr` to stderr (identifies the module
    // that a writer RIP belongs to). Fixed buffers, no malloc; open/read/write are signal-safe.
    void classify_addr(uint64_t addr) {
#ifdef __APPLE__
        // No /proc on Darwin: report the containing mach VM region (bounds + prot). mach_vm_region
        // is a plain trap — no malloc, async-safe enough for this diagnostic path.
        mach_vm_address_t ra = addr; mach_vm_size_t rs = 0;
        vm_region_basic_info_data_64_t info; mach_msg_type_number_t cnt = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t obj = MACH_PORT_NULL;
        char b[160]; int n;
        if (mach_vm_region(mach_task_self(), &ra, &rs, VM_REGION_BASIC_INFO_64,
                           (vm_region_info_t)&info, &cnt, &obj) == KERN_SUCCESS && addr >= ra) {
            if (obj != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), obj);
            n = snprintf(b, sizeof b, "[hwwatch]   writer 0x%llx is in: 0x%llx-0x%llx %c%c%c\n",
                         (unsigned long long)addr, (unsigned long long)ra, (unsigned long long)(ra + rs),
                         (info.protection & VM_PROT_READ) ? 'r' : '-',
                         (info.protection & VM_PROT_WRITE) ? 'w' : '-',
                         (info.protection & VM_PROT_EXECUTE) ? 'x' : '-');
        } else {
            n = snprintf(b, sizeof b, "[hwwatch]   writer 0x%llx: no containing region\n",
                         (unsigned long long)addr);
        }
        // `n > 0` only rejects the encoding error; a truncating format still returns MORE than the
        // buffer holds, so the length needs the same clamp as every other report site (#2050).
        raw_write_fmt(2, b, sizeof b, n);
        return;
#else
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
                if (ok && addr >= s && addr < e) { raw_write_fmt(2, pend, sizeof pend, pn); raw_write(2,line, (size_t)li);
                    raw_write(2,"\n", 1); close(fd); return; }
                li = 0;
            }
        }
        close(fd);
#endif
    }
    // Dump the HWBP ring (oldest->newest) once. Prints reader fields (cur/[cur]/rax) for anom traces and
    // typetree-node fields (r8/[r8+8]/[r8+0x10]/r14) for node traces. Signal-safe (fixed buffers, write()).
    void hwbp_dump_ring(const char* why) {
        if (g_hwbp_ring_dumped) return;
        g_hwbp_ring_dumped = true;
        int total = g_hwbp_ring_pos < HWBP_RING ? g_hwbp_ring_pos : HWBP_RING;
        int start = g_hwbp_ring_pos < HWBP_RING ? 0 : (g_hwbp_ring_pos % HWBP_RING);
        char hdr[96]; int hn = snprintf(hdr, sizeof hdr, "[hwbp-ring] dump (%s): last %d hits:\n", why, total);
        raw_write_fmt(2, hdr, sizeof hdr, hn);
        unsigned long long prev_cur = 0;
        for (int i = 0; i < total; ++i) {
            const HwbpRingEnt& e = g_hwbp_ring[(start + i) % HWBP_RING];
            long long dcur = prev_cur ? (long long)(e.cur - prev_cur) : 0;
            char lb[220];
            int ln = snprintf(lb, sizeof lb,
                "  [%3d] rip=%s+0x%llx req/cur=0x%llx cacher/rax=0x%llx | s50/r8=0x%llx s68/f8=0x%llx f10=0x%llx r14=0x%llx  %s\n",
                i, e.rip_mod ? e.rip_mod : "?", e.rip_off, e.cur, e.rax, e.r8, e.f8, e.f10, e.r14,
                (e.cur == e.r8 || e.cur == e.f8) ? "<<< MATCH (skip fetch)" : "");
            (void)dcur;
            raw_write_fmt(2, lb, sizeof lb, ln);
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
        raw_write_fmt(2, hdr, sizeof hdr, hn);
        unsigned long long prev = 0;
        for (int i = 0; i < total; ++i) {
            const StepEnt& e = g_stepwin_ring[(start + i) % STEPWIN_RING];
            long long d = prev ? (long long)(e.cur - prev) : 0;
            char lb[160];
            int ln = snprintf(lb, sizeof lb, "  [%3d] rip=%s+0x%llx cur=0x%llx (off 0x%llx) delta=%+lld\n",
                i, e.rip_mod ? e.rip_mod : "?", e.rip_off, e.cur, (unsigned long long)(e.cur - g_stepwin_base), d);
            raw_write_fmt(2, lb, sizeof lb, ln);
            prev = e.cur;
        }
    }
    void arm_hwwatch(uint64_t addr) {
        long fd = perf_bp_open(addr, HW_BREAKPOINT_W);
        char b[160];
        if (fd < 0) { int n = snprintf(b, sizeof b, "[hwwatch] perf W-watch FAILED addr=0x%llx errno=%d\n",
                          (unsigned long long)addr, errno); raw_write_fmt(2, b, sizeof b, n);   /* raw syscall: no libc TLS access */ return; }
        g_hwwatch_fd = (int)fd; g_hwwatch_addr = addr;
        fcntl(g_hwwatch_fd, F_SETFL, O_ASYNC);
        fcntl(g_hwwatch_fd, F_SETSIG, SIGTRAP);
        struct f_owner_ex ow; ow.type = F_OWNER_TID; ow.pid = (pid_t)prosper_gettid();
        fcntl(g_hwwatch_fd, F_SETOWN_EX, &ow);
        ioctl(g_hwwatch_fd, PERF_EVENT_IOC_ENABLE, 0);
        int n = snprintf(b, sizeof b, "[hwwatch] armed W-watch at 0x%llx (fd=%d)\n",
                         (unsigned long long)addr, g_hwwatch_fd); raw_write_fmt(2, b, sizeof b, n);   /* raw syscall: no libc TLS access */
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
    unsigned         g_mb3w_loud[64] = {0};   // per-slot ORDINAL for the structural corrupt-head arm
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
                raw_write_fmt(2, b, sizeof b, k); continue; }
            fcntl((int)fd, F_SETFL, O_ASYNC);
            fcntl((int)fd, F_SETSIG, SIGTRAP);
            struct f_owner_ex ow; ow.type = F_OWNER_TID; ow.pid = (pid_t)prosper_gettid();
            fcntl((int)fd, F_SETOWN_EX, &ow);
            ioctl((int)fd, PERF_EVENT_IOC_ENABLE, 0);
            int slot = g_mb3w_cnt.fetch_add(1, std::memory_order_acq_rel);
            if (slot < 64) { g_mb3w[slot].addr = addr; g_mb3w[slot].fd = (int)fd; }
            char b[176]; int k = snprintf(b, sizeof b,
                "[mb3watch] ARMED head watch @0x%llx (base=0x%llx off=0x%x) fd=%ld tid=%ld slot=%d\n",
                (unsigned long long)addr, (unsigned long long)base, g_mb3w_off,
                fd, cur_tid(), slot);
            raw_write_fmt(2, b, sizeof b, k);
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
        long w = raw_write(g_probe_pipe[1], (const void*)a, 8);
        if (w > 0) { char b[8]; syscall(SYS_read, g_probe_pipe[0], b, (size_t)w); }
        return w == 8;
    }
    // Resolve an Il2CppObject's class name: obj -> [obj]=Il2CppClass* -> [klass+0x10]=name (C string).
    // Returns bytes written (0 if unresolvable / not a plausible object). Async-signal-safe: probe_readable
    // + reads only. Used by PROSPER_BP_KLASS (and reusable by any handler that has a candidate object ptr).
    size_t bp_il2cpp_cname(uint64_t obj, char* s, size_t cap) {
        if (cap == 0) return 0; s[0] = 0;
        // probe_readable(X) verifies [X, X+8) — so guard each read with its BASE address (not base+size-1,
        // which would leave the leading bytes of an unaligned read unverified and could fault in-handler).
        if (obj < 0x1000 || !probe_readable(obj)) return 0;
        uint64_t klass = *(const uint64_t*)obj;
        if (klass < 0x1000 || !probe_readable(klass + 0x10)) return 0;
        uint64_t namep = *(const uint64_t*)(klass + 0x10);
        if (!probe_readable(namep)) return 0;
        size_t k = 0;
        for (; k + 1 < cap && k < 96 && probe_readable(namep + k); ++k) {
            char c = *(const char*)(namep + k);
            if (c == 0) break;
            if (c < 0x20 || c > 0x7e) { if (k < 2) { s[0] = 0; return 0; } break; }
            s[k] = c;
        }
        s[k] = 0;
        return k >= 2 ? k : 0;
    }
    // Evaluate PROSPER_BP_PROBE chains against a saved register file and write the results to stderr.
    // Grammar per chain: <start>[+0xN|*]...:<t>  where start = a register name or 0xADDR, "+N" adds an
    // offset, "*" dereferences a qword, and the trailing ":t" reads q/d/b/f. Chains are separated by ';'
    // or ','. Async-signal-safe: hand-rolled hex parse (no strtoull/locale), probe_readable + reads only.
    // base != 0 makes the "@" start-token resolve to that object (used by PROSPER_KSCAN to dump each scanned
    // instance's fields); uctx may be null (then register start-tokens simply fail to resolve). tag is the log
    // line prefix.
    void bp_eval_probes(const char* spec, void* uctx, uint64_t base, const char* tag) {
        // PROSPER_GREGS resolves to a raw greg array on Linux but a view object on Darwin, so construct it with
        // auto INSIDE reg_val (and only when uctx is provided — KSCAN passes uctx, so register tokens still work
        // there, but a hypothetical null-uctx caller degrades cleanly to "@"/0xADDR only).
        auto reg_val = [&](const char* p, size_t len, uint64_t* out) -> bool {
            if (!uctx) return false;
            auto gr = PROSPER_GREGS((ucontext_t*)uctx);
            struct { const char* n; int idx; } R[] = {
                {"rax",REG_RAX},{"rbx",REG_RBX},{"rcx",REG_RCX},{"rdx",REG_RDX},
                {"rsi",REG_RSI},{"rdi",REG_RDI},{"rbp",REG_RBP},{"rsp",REG_RSP},
                {"r10",REG_R10},{"r11",REG_R11},{"r12",REG_R12},{"r13",REG_R13},
                {"r14",REG_R14},{"r15",REG_R15},{"r8",REG_R8},{"r9",REG_R9} };
            for (auto& r : R) { size_t rn = 0; while (r.n[rn]) rn++;
                if (rn == len) { bool eq = true; for (size_t i=0;i<len && eq;i++) if (r.n[i]!=p[i]) eq=false;
                    if (eq) { *out = (uint64_t)gr[r.idx]; return true; } } }
            return false;
        };
        auto hexval = [](const char** pp) -> uint64_t {
            const char* p = *pp; uint64_t v = 0;
            if (p[0]=='0' && (p[1]=='x'||p[1]=='X')) p += 2;
            for (;; ++p) { char c=*p; uint64_t d;
                if (c>='0'&&c<='9') d=c-'0'; else if (c>='a'&&c<='f') d=c-'a'+10;
                else if (c>='A'&&c<='F') d=c-'A'+10; else break; v = v*16 + d; }
            *pp = p; return v;
        };
        // Every cursor advance in this function goes through raw_fmt_advance (#2161): snprintf
        // returns the WOULD-BE length, so an unclamped `on` walks past `out` and the next
        // `sizeof out - on` underflows to ~2^64. Saturated at `sizeof out`, so `out + on` stays
        // in range, later appends see size 0 and write nothing, and the terminator/flush below
        // still read `on >= sizeof out` as "this line truncated".
        char out[400]; int on = raw_fmt_advance(0, snprintf(out, sizeof out, "%s", tag), sizeof out);
        const char* p = spec;
        // #2192: the reserve was 64 while one append can reach 81 (" %.*s=%s" with slen capped at
        // 40 and a 40-byte value buffer), so an admitted iteration could truncate mid-field.
        //
        // This site differs from the other three the issue lists, and the difference is worth
        // stating because it changes what the defect IS: every cursor advance here goes through
        // raw_fmt_advance, which saturates at `sizeof out`, and the flush below emits
        // raw_write_trunc_mark() when it did. So truncation here was never SILENT -- the marker
        // has always fired. The defect is only that the reserve did not match the format, so the
        // loop admitted an entry it could not hold.
        //
        // Corrected to the true maximum rather than removed: the marker is the backstop, not the
        // primary mechanism, and keeping the reserve means whole entries in the ordinary case
        // instead of a marked partial one. If the format ever grows past this, the marker still
        // catches it -- which is why 96 is safe to state as a number rather than fragile.
        while (*p && on < (int)sizeof out - 96) {
            while (*p==';'||*p==','||*p==' ') p++;
            if (!*p) break;
            const char* start = p; uint64_t v = 0; bool have = false;
            if (p[0]=='@') { v = base; have = true; p++; }
            else if (p[0]=='0' && (p[1]=='x'||p[1]=='X')) { v = hexval(&p); have = true; }
            else { const char* q = p; while ((*q>='a'&&*q<='z')||(*q>='0'&&*q<='9')) q++;
                   if (reg_val(p, (size_t)(q-p), &v)) { have = true; p = q; } }
            if (!have) break;
            char type = 'q'; bool bad = false;
            while (*p && *p!=';' && *p!=',') {
                if (*p=='+') { p++; v += hexval(&p); }
                else if (*p=='*') { p++; if (!probe_readable(v)) { bad = true; break; } v = *(const uint64_t*)v; }   // probe_readable(v) covers [v,v+8)
                else if (*p==':') { p++; if (*p) { type = *p; p++; } break; }
                else p++;
            }
            int slen = (int)(p - start); if (slen > 40) slen = 40;
            if (bad) { on = raw_fmt_advance(on, snprintf(out+on, sizeof out-on, " %.*s=<unmapped>", slen, start), sizeof out); continue; }
            // probe_readable(v) verifies [v,v+8), covering any 1..8-byte read at v (guard the read's base).
            if (type=='s') {   // Il2CppString at v: length@0x10 (i32), UTF-16 chars@0x14 — print ASCII, capped.
                on = raw_fmt_advance(on, snprintf(out+on, sizeof out-on, " %.*s=", slen, start), sizeof out);
                if (probe_readable(v + 0x10)) {
                    int len = *(const int32_t*)(v + 0x10);
                    if (len < 0) len = 0; if (len > 48) len = 48;
                    if (on < (int)sizeof out - 2) out[on++] = '"';
                    for (int i = 0; i < len && on < (int)sizeof out - 3; ++i) {
                        uint64_t ca = v + 0x14 + (uint64_t)i * 2;
                        if (!probe_readable(ca)) break;
                        uint16_t c = *(const uint16_t*)ca;
                        out[on++] = (c >= 0x20 && c < 0x7f) ? (char)c : '.';
                    }
                    if (on < (int)sizeof out - 2) out[on++] = '"';
                } else if (on < (int)sizeof out - 1) out[on++] = '?';
                continue;
            }
            char vb[40];
            if (type=='b')      snprintf(vb, sizeof vb, probe_readable(v) ? "%u"     : "?", probe_readable(v) ? *(const uint8_t*)v  : 0);
            else if (type=='d') snprintf(vb, sizeof vb, probe_readable(v) ? "0x%x"   : "?", probe_readable(v) ? *(const uint32_t*)v : 0);
            else if (type=='f') snprintf(vb, sizeof vb, probe_readable(v) ? "f:0x%x" : "?", probe_readable(v) ? *(const uint32_t*)v : 0);
            else                snprintf(vb, sizeof vb, probe_readable(v) ? "0x%llx" : "?", (unsigned long long)(probe_readable(v) ? *(const uint64_t*)v : 0));
            on = raw_fmt_advance(on, snprintf(out+on, sizeof out-on, " %.*s=%s", slen, start, vb), sizeof out);
        }
        // Since #2161 every append above saturates `on` at sizeof out, so the only value the
        // clamp still has to absorb here is that saturation itself — the newline store needs one
        // byte more room than the appends do. `cut` is exactly `on == sizeof out`, which is how
        // #2050's contract reads "this line truncated": the write covers the bytes that really
        // landed and the marker tells the reader the tail is missing.
        const bool cut = on < 0 || (size_t)on >= sizeof out;
        on = (int)raw_fmt_len(on, sizeof out);
        out[on] = '\n';
        raw_write(2, out, (size_t)on + 1);   /* raw syscall: no libc TLS access */
        if (cut) raw_write_trunc_mark(2);
    }
    // PROSPER_KSCAN one-shot enumerator: scan the guest GC-heap mappings (from /proc/self/maps, in the
    // [0x20_00000000,0x28_00000000) object band) for every 8-aligned qword equal to g_kscan_klass — i.e. every
    // live Il2CppObject of that class ([obj][0]=klass) — and dump each + PROSPER_KSCAN_FIELDS. Reads are on
    // mapped, readable, private-anonymous ranges (untouched pages fault to the shared zero page: no RAM commit,
    // no SIGSEGV), so no probe_readable is needed for the qword compare. Bounded by KSCAN_MAX and a byte cap.
    void il2cpp_kscan(void* uctx) {
        static char maps[256 * 1024];
        long fd = syscall(SYS_open, "/proc/self/maps", O_RDONLY, 0);
        if (fd < 0) { const char* e = "[kscan] open /proc/self/maps failed\n"; raw_write(2, e, 36); return; }
        long total = 0, r;
        while (total < (long)sizeof(maps) - 1 &&
               (r = syscall(SYS_read, fd, maps + total, sizeof(maps) - 1 - (size_t)total)) > 0) total += r;
        syscall(SYS_close, fd);
        maps[total < 0 ? 0 : total] = 0;
        char hdr[96]; int hn = snprintf(hdr, sizeof hdr, "[kscan] klass=0x%llx max=%d\n",
                                        (unsigned long long)g_kscan_klass, g_kscan_max);
        raw_write_fmt(2, hdr, sizeof hdr, hn);
        auto rdhex = [](const char** pp) -> uint64_t { uint64_t v = 0; const char* p = *pp;
            for (;; ++p) { char c=*p; uint64_t d; if(c>='0'&&c<='9')d=c-'0'; else if(c>='a'&&c<='f')d=c-'a'+10;
                else if(c>='A'&&c<='F')d=c-'A'+10; else break; v=v*16+d; } *pp=p; return v; };
        int found = 0; uint64_t scanned = 0; const uint64_t CAP = 6ull << 30;
        const char* p = maps;
        while (*p && found < g_kscan_max && scanned < CAP) {
            while (*p==' ') p++;
            uint64_t s = rdhex(&p);
            if (*p != '-') { while (*p && *p!='\n') p++; if (*p) p++; continue; }
            p++; uint64_t e = rdhex(&p);
            while (*p==' ') p++;
            char perm0 = *p;                       // 'r' iff readable
            while (*p && *p!='\n') p++; if (*p) p++;
            // Scan only readable chunks in the guest object band. On this port that band is prosper's own device
            // memory (a /memfd:prosper-dmem mapping split into readable chunks with PROT_NONE `---p` guard gaps
            // between them); the guards are separate, non-readable map lines that this `'r'` check already
            // excludes, so each scanned chunk is contiguously readable. That memfd is prosper-owned and never
            // externally truncated, so the read cannot SIGBUS. (Best-effort snapshot: another guest thread could
            // remap an in-band chunk between the /proc/self/maps read and a qword read — inherent to a one-shot
            // live-heap scan; acceptable for an off-by-default diagnostic.)
            if (perm0 != 'r' || e <= s) continue;
            if (s < 0x2000000000ull || s >= 0x2800000000ull) continue;
            uint64_t lim = e; if (lim - s > CAP - scanned) lim = s + (CAP - scanned);
            for (uint64_t pos = (s + 7) & ~7ull; pos + 16 <= lim && found < g_kscan_max; pos += 8) {
                // A genuine Il2CppObject header is {klass@0, monitor@8}; the monitor is null for a typical
                // instance. Requiring [pos+8]==0 rejects the false positives that are just klass-pointer FIELDS
                // inside Il2CppClass metadata (whose next qword is another non-null metadata pointer).
                if (*(const uint64_t*)pos != g_kscan_klass) continue;
                if (pos + 0x400 > g_kscan_klass && pos < g_kscan_klass + 0x400) continue;  // skip the Il2CppClass struct's own self-referencing fields
                if (*(const uint64_t*)(pos + 8) != 0) continue;
                found++;
                char tag[80]; snprintf(tag, sizeof tag, "[kscan] #%d obj=0x%llx", found, (unsigned long long)pos);
                bp_eval_probes(g_kscan_fields ? g_kscan_fields : "", uctx, pos, tag);
            }
            scanned += (lim - s);
        }
        char sum[96]; int sn = snprintf(sum, sizeof sum, "[kscan] done: %d instance(s), %llu MiB scanned\n",
                                        found, (unsigned long long)(scanned >> 20));
        raw_write_fmt(2, sum, sizeof sum, sn);
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
            raw_write_fmt(2, b, sizeof b, n);
            for (int r = 0; r < 8; r++) {
                uint64_t addr = a + (uint64_t)r * 8;
                if (probe_readable(addr))
                    n = snprintf(b, sizeof b, "  +0x%02x = 0x%016llx\n", r * 8, (unsigned long long)*(const uint64_t*)addr);
                else
                    n = snprintf(b, sizeof b, "  +0x%02x = <unmapped>\n", r * 8);
                raw_write_fmt(2, b, sizeof b, n);
            }
        }
    }
    // Takes this fault's own snapshot rather than reading the shared globals (#2018): it runs on the
    // worker path too, where a second faulting thread can rewrite them mid-dump.
    void dump_fault_mem(const prosper::host::FaultContext& fc) {
        if (g_dumpat_n) dump_at_addrs();
        if (!g_faultmem) return;
        const struct { const char* n; uint64_t v; } regs[] = {
            {"rdi", fc.rdi}, {"rsi", fc.rsi}, {"rdx", fc.rdx}, {"rcx", fc.rcx},
            {"rax", fc.rax}, {"rbx", fc.rbx}, {"rbp", fc.rbp}, {"rsp", fc.rsp},
            {"r8 ", fc.r8},  {"r9 ", fc.r9},  {"r12", fc.r12}, {"r13", fc.r13},
            {"r14", fc.r14}, {"r15", fc.r15},
        };
        char b[256];
        int n = snprintf(b, sizeof b, "[prosper] FAULTMEM dump (regs -> guest memory):\n");
        raw_write_fmt(2, b, sizeof b, n);   /* raw syscall: no libc TLS access */
        for (auto& r : regs) {
            if (!probe_readable(r.v)) {
                n = snprintf(b, sizeof b, "  %s=0x%llx  (unmapped/immediate)\n",
                             r.n, (unsigned long long)r.v);
                raw_write_fmt(2, b, sizeof b, n);   /* raw syscall: no libc TLS access */
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
            raw_write_fmt(2, b, sizeof b, n);   /* raw syscall: no libc TLS access */
        }
        // Deep field peek (PROSPER_PEEK, parsed once at arm time — getenv is not signal-safe): read
        // specific offsets off one or more registers to classify a large object beyond the 0x20-byte
        // window above. See g_peek definition for the syntax.
        for (int sp = 0; sp < g_peek_specs; sp++) {
            const PeekSpec& ps = g_peek[sp];
            uint64_t base = 0;
            for (auto& r : regs) { bool m = true; for (int i=0;i<3;i++) if (r.n[i]!=ps.reg[i]&&!(r.n[i]==' '&&ps.reg[i]==0)) { m=false; break; } if (m) { base=r.v; break; } }
            n = snprintf(b, sizeof b, "  PEEK %s=0x%llx:\n", ps.reg, (unsigned long long)base);
            raw_write_fmt(2, b, sizeof b, n);   /* raw syscall: no libc TLS access */
            for (int k = 0; k < ps.n; k++) {
                uint64_t obj = base;
                if (ps.deref[k]) {   // chase one pointer level: obj = [base + pre]
                    uint64_t pa = base + ps.pre[k];
                    // raw_write_fmt, like every other write in this dump -- including the one
                    // eleven lines below. This site had NO guard at all: snprintf's would-be length
                    // went straight to write(), so a truncating format over-read `b` (#2180, the
                    // #2050 class). It also used glibc write() inside a fault handler, where the
                    // rest of this function deliberately uses raw syscalls to avoid libc TLS on a
                    // guest-%fs thread -- so it was two defects wearing one line.
                    if (!probe_readable(pa)) {
                        n = snprintf(b, sizeof b, "    [+0x%llx]-><unmapped>\n",
                                     (unsigned long long)ps.pre[k]);
                        raw_write_fmt(2, b, sizeof b, n);
                        continue;
                    }
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
                raw_write_fmt(2, b, sizeof b, n);   /* raw syscall: no libc TLS access */
            }
        }
    }

    // GPU write-attribution ring scanner (src/gpu/command_processor.cpp). Weak: tools that link
    // the host exec image without the gpu lib get a null and skip the scan.
    extern "C" int prosper_gpu_write_ring_scan(uint64_t lo, uint64_t hi, char* out, size_t cap)
        __attribute__((weak));
    // #1226: persistent clock-fence provenance scanner (same TU as the ring). Unlike the ring, its
    // records never age out, so a stomp seconds before the fault is still attributable.
    extern "C" int prosper_gpu_clockfence_scan(uint64_t lo, uint64_t hi, char* out, size_t cap)
        __attribute__((weak));
    // #1226: reverse lookup — which fence target ever wrote a clock whose low32 == the corrupted
    // value's high dword (the {orig_low32, clock_low32} A-4 read-back signature; see gpu TU).
    extern "C" int prosper_gpu_clockfence_find_low32(uint32_t low32, char* out, size_t cap)
        __attribute__((weak));
    // #1226: APR-destination provenance (hle_file.cpp) — the bulk guest-writer no ring tracks.
    extern "C" int prosper_apr_dest_scan(uint64_t lo, uint64_t hi, char* out, size_t cap)
        __attribute__((weak));
    // #1226: pool-candidate membership (mb3_freelist.cpp) — async-signal-safe atomic reads.
    extern "C" int prosper_mb3_is_pool_candidate(uint64_t base) __attribute__((weak));

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
            raw_write_fmt(2, b, sizeof b, n);
        }
        auto g = PROSPER_GREGS(uc);
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
            n = raw_fmt_advance(n, snprintf(b + n, sizeof b - (size_t)n, "%s%s=0x%llx", (i % 4) ? " " : "[nullpage]   ",
                          regs[i].nm, (unsigned long long)regs[i].v), sizeof b);
            if (i % 4 == 3) {
                // Same shape as the [il2cpp-probe] line above: the append saturates `n` at
                // sizeof b (#2161), and the newline store needs one byte more room than that,
                // so `cut` is exactly the saturated case and the clamp lands it in range (#2050).
                const bool cut = n < 0 || (size_t)n >= sizeof b;
                n = (int)raw_fmt_len(n, sizeof b);
                b[n++] = '\n';
                raw_write(2, b, (size_t)n);
                if (cut) raw_write_trunc_mark(2);
                n = 0;
            }
        }
        // #1226 — is a faulting register a GPU LABEL SLOT? On ArcRunner every terminal fault carries
        // the shape {small nonzero high dword, low dword 1}, which is byte-identical to a 0x20-byte
        // label whose 4-byte fence value 1 sits under stale malloc residue in the high half (the
        // #1226 "residue 0x2020e31680 -> 0x2000000000" mechanism, one generation later). That reading
        // and "prosper stomped a live pointer" predict the same VALUE and are told apart only by
        // whether the address has a label protocol history — which the command processor already
        // records per label and exports here. Printed only for registers that HAVE one, so a silent
        // run means no register named a tracked label rather than a disabled probe.
        for (int i = 0; i < 16; i++) {
            const uint64_t v = regs[i].v;
            if (v < 0x10000) continue;
            char hist[512];
            prosper_label_hist_dump(v, hist, (unsigned)sizeof hist);
            if (!hist[0] || strstr(hist, "no-history")) continue;
            snprintf(b, sizeof b, "[labelhist] %s=0x%llx %s\n", regs[i].nm,
                     (unsigned long long)v, hist);
            // strlen, NOT snprintf's return: snprintf returns the WOULD-BE length, and `hist` is the
            // same 512 bytes as `b`. A full 16-event ring formats to ~450-500 characters, so the
            // prefix pushes the would-be length past `sizeof b` on exactly the richest labels this
            // instrument exists to print — handing that value to raw_write() reads past the end of a
            // stack buffer. This was the first site whose format embeds a %s of comparable size to
            // the buffer, so it is where the class was caught (review of #2077, instrument trap 109);
            // the accompanying claim that every other raw_write here was already length-bounded was
            // wrong — the sweep in #2050 found 88 report sites handing the raw return to raw_write,
            // and they now go through raw_write_fmt(). strlen is kept here because it is exact:
            // snprintf NUL-terminates, so it needs no clamp and reports no truncation it did not see.
            raw_write(2, b, strlen(b));
        }
        for (int i = 0; i < 16; i++) {
            uint64_t v = regs[i].v;
            if (v < 0x10000 || !probe_readable(v)) continue;
            // rbp: the canary-walk keeps its locals at rbp-0x78..-0x48 (pool-table ptr, chain
            // heads) — start the window at rbp-0x80 so they're captured.
            uint64_t base = (v & ~0xfull) - (i == 6 /*rbp*/ ? 0x80 : 0x20);
            if (!probe_readable(base)) base = v & ~0xfull;
            int nw = (i == 6) ? 20 : 12;
            n = raw_fmt_advance(0, snprintf(b, sizeof b, "[nullpage]   %s=0x%llx mem@0x%llx:", regs[i].nm,
                         (unsigned long long)v, (unsigned long long)base), sizeof b);
            for (int w = 0; w < nw; w++) {
                uint64_t addr = base + (uint64_t)w * 8;
                if (probe_readable(addr))
                    n = raw_fmt_advance(n, snprintf(b + n, sizeof b - (size_t)n, " %016llx",
                                  (unsigned long long)*(const uint64_t*)addr), sizeof b);
                else
                    n = raw_fmt_advance(n, snprintf(b + n, sizeof b - (size_t)n, " ????????????????"), sizeof b);
            }
            n = raw_fmt_advance(n, snprintf(b + n, sizeof b - (size_t)n, "\n"), sizeof b);
            raw_write_fmt(2, b, sizeof b, n);
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
                        raw_write(2, slab_out, strlen(slab_out));
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
                    raw_write_fmt(2, b, sizeof b, n);
                    if (prosper_gpu_write_ring_scan) {
                        static char hit_out[1024];
                        if (prosper_gpu_write_ring_scan(addr - 8, addr + 16, hit_out, sizeof hit_out) > 0)
                            raw_write(2, hit_out, strlen(hit_out));
                    }
                }
            } else if (prosper_gpu_write_ring_scan) {
                // recent GPU writes near this register's address (attribution ring, gpu lib; weak —
                // tools that link exec_image without the gpu lib simply skip this)
                static char ring_out[4096];
                if (prosper_gpu_write_ring_scan(v - 0x100, v + 0x100, ring_out, sizeof ring_out) > 0)
                    raw_write(2, ring_out, strlen(ring_out));
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
    // source xmm (EXTRQ: rm[13:0]; INSERTQ: rm[77:64]). len==0 means 64. CONFIDENCE: HIGH
    // (AMD64 Architecture Programmer's Manual, Volume 4, publication 26568).
    volatile unsigned long g_sse4a_emulated = 0;
    const bool g_sse4a_stat = getenv("PROSPER_SSE4A_STAT") != nullptr;
#ifdef __APPLE__
    // macOS/Rosetta guest %fs TLS emulation. Rosetta keeps the CPU fs base at 0 and rejects wrfsbase,
    // so every guest `%fs:disp` access faults at linear address == the raw offset. With the per-thread
    // guest TCB (guest_tls trap mode), the real target is guest_TP + fault_addr. Decode the fs-prefixed
    // instruction, perform the access against the TCB, advance RIP, and resume. Handles the mov family
    // (the overwhelming majority of TLS touches: self-ptr read, IE-var load/store, imm store, and
    // movzx/movsx); logs+bails on any other form so it can be extended from real boot evidence.
    volatile unsigned long g_fs_emulated = 0;
    const bool g_fslog = getenv("PROSPER_FSLOG") != nullptr;
    // x86 register number (rax=0..r15=15) -> Darwin greg view index.
    static const int kX86ToReg[16] = { REG_RAX,REG_RCX,REG_RDX,REG_RBX,REG_RSP,REG_RBP,REG_RSI,REG_RDI,
                                       REG_R8,REG_R9,REG_R10,REG_R11,REG_R12,REG_R13,REG_R14,REG_R15 };
    bool try_emulate_fs_access(ucontext_t* uc, uint64_t fault_addr) {
        uint64_t tp = guest_tls_tp();
        if (!tp) return false;
        auto g = PROSPER_GREGS(uc);
        const uint8_t* p = (const uint8_t*)(uintptr_t)g[REG_RIP];
        // Safety: only touch the guest TCB window [TP - total_below, TP + 0x200). A mis-decode with a
        // wild offset is treated as a real fault (bail) rather than corrupting/reading random memory.
        int64_t off = (int64_t)fault_addr;
        if (off < -(int64_t)guest_tls_total_below() || off >= 0x200) return false;
        volatile uint8_t* mem = (volatile uint8_t*)(uintptr_t)(tp + fault_addr);

        // Decode + execute via the platform-neutral core (fs_emu.hpp — unit-tested on Linux,
        // tests/test_fs_emu.cpp, incl. the #727 AH/CH/DH/BH byte-register rules).
        uint64_t regs[16];
        for (int k = 0; k < 16; k++) regs[k] = (uint64_t)g[kX86ToReg[k]];
        FsEmuResult res = fs_emulate_access(p, mem, regs);
        if (res.status == FsEmuStatus::NotFs) return false;   // genuine fault, not guest TLS
        if (res.status == FsEmuStatus::Unhandled) {
            char b[128]; int n = snprintf(b, sizeof b,
                "[fs-emu] UNHANDLED fs insn rip=0x%llx off=%lld bytes: %02x %02x %02x %02x %02x %02x %02x %02x\n",
                (unsigned long long)g[REG_RIP], (long long)off,
                p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]);
            // `n > 0` rejects only the ENCODING error; a truncating format returns MORE than the
            // buffer holds and the write over-reads it. #2171's own comment at the raw_write sites
            // says exactly that, and these two glibc sites were missed because that sweep's
            // enumeration was scoped to `raw_write` (#2180). raw_write_fmt applies the full clamp
            // and keeps the truncation marker.
            raw_write_fmt(2, b, sizeof b, n);
            return false;   // fall through to the normal (fatal) fault path so the form is visible
        }
        for (int k = 0; k < 16; k++) g[kX86ToReg[k]] = (greg_t)regs[k];
        g[REG_RIP] = (greg_t)(uintptr_t)(p + res.insn_len);   // advance past the emulated instruction
        g_fs_emulated++;
        if (g_fslog && (g_fs_emulated & 0x3FFF) == 0) {
            char b[64]; int n = snprintf(b, sizeof b, "[fs-emu] %lu accesses\n", g_fs_emulated);
            raw_write_fmt(2, b, sizeof b, n);
        }
        return true;
    }
#endif  // __APPLE__

    bool try_emulate_sse4a(ucontext_t* uc) {
        auto g = PROSPER_GREGS(uc);
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
#ifdef __APPLE__
        // Darwin keeps the xmm registers as 16 contiguous _STRUCT_XMM_REG fields in the mcontext's
        // float state; index off __fpu_xmm0.
        if (!uc->uc_mcontext) return false;
        auto xmm = [&](int n) { return (uint32_t*)(&uc->uc_mcontext->__fs.__fpu_xmm0 + n); };
#else
        auto* fp = uc->uc_mcontext.fpregs;
        if (!fp) return false;
        auto xmm = [&](int n) { return (uint32_t*)fp->_xmm[n].element; };
#endif
        auto lo  = [&](int n) { const uint32_t* e = xmm(n); return (uint64_t)e[0] | ((uint64_t)e[1] << 32); };
        auto hi  = [&](int n) { const uint32_t* e = xmm(n); return (uint64_t)e[2] | ((uint64_t)e[3] << 32); };
        auto set = [&](int n, uint64_t v) { uint32_t* e = xmm(n); e[0] = (uint32_t)v; e[1] = (uint32_t)(v >> 32); };
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
            // AMD documents the upper qword as undefined. Preserve it to match the AMD CPU model
            // and prosper's established guest-visible trap-emulation behavior.
        }
        g[REG_RIP] = (greg_t)(rip + i);
        g_sse4a_emulated++;
        // PROSPER_SSE4A_STAT: async-safe rate probe — every 2^20 emulations, write the count so a
        // timed run reveals whether the SIGILL round-trip is the throughput wall. Diagnostic only.
        if (g_sse4a_stat && (g_sse4a_emulated & 0xFFFFF) == 0) {
            char b[64]; int n = snprintf(b, sizeof b, "[sse4a] %lu emulated\n", g_sse4a_emulated);
            raw_write_fmt(2, b, sizeof b, n);   // the old `n > 0` guard is raw_write_fmt's job now
        }
        return true;
    }

    // Guest write-watch handling enters std::mutex and glibc mprotect. Both use host TLS through %fs
    // (`pthread_mutex_trylock` reads the host TID and a failing mprotect stores errno), while a guest
    // worker reaches this signal handler with %fs pointing at its guest TCB. Scope only that host work
    // onto the saved host TCB, then restore the original guest %fs before either resuming the store or
    // continuing through the ordinary fault path. Keep this helper stack-protector-free because it
    // deliberately changes %fs within its own frame.
    __attribute__((no_stack_protector))
    prosper::host::GuestWriteWatchFaultAction handle_guest_write_watch_fault(
            uint64_t addr, uint64_t rip, int64_t tid, bool trap_flag_already_owned) {
        const uint64_t saved_guest_fs = guest_fs_to_host_scoped();
        const auto handled = prosper::host::guest_write_watch_handle_fault_ex(
            addr, rip, tid, trap_flag_already_owned);
        guest_fs_restore_scoped(saved_guest_fs);
        return handled;
    }

    __attribute__((no_stack_protector))
    prosper::host::GuestDmemWriteTraceStepAction complete_guest_dmem_write_trace_step(
            int64_t tid, uint64_t next_rip, prosper::host::GuestDmemWriteTraceEvent& event) {
        const uint64_t saved_guest_fs = guest_fs_to_host_scoped();
        const auto result = prosper::host::guest_dmem_write_trace_complete_step(tid, next_rip, event);
        guest_fs_restore_scoped(saved_guest_fs);
        return result;
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
#ifdef __APPLE__
        // macOS/Rosetta: a guest `%fs:`-relative TLS access faults (fs base is 0). Redirect it to the
        // guest TCB (guest_TP + offset) and resume. Only fires when trap-mode TLS is active and the
        // faulting instruction actually carries an fs prefix; otherwise falls through to the real path.
        if ((sig == SIGSEGV || sig == SIGBUS) && try_emulate_fs_access((ucontext_t*)uctx, (uint64_t)si->si_addr))
            return;
#endif
        // The dmem writer overlay opens its watched page for exactly one instruction. Complete that
        // TF step before any unrelated breakpoint machinery sees the trace trap, retain post-store
        // bytes, and re-arm. A contended state lock is acquired without returning to guest execution;
        // exhaustion is fatal and fail-visible rather than silently stepping additional instructions.
        if (sig == SIGTRAP && si->si_code == TRAP_TRACE) {
            auto* uc = (ucontext_t*)uctx;
            prosper::host::GuestDmemWriteTraceEvent event{};
            const auto step = complete_guest_dmem_write_trace_step(
                cur_tid(), static_cast<uint64_t>(PROSPER_GREGS(uc)[REG_RIP]), event);
            if (step == prosper::host::GuestDmemWriteTraceStepAction::LockTimeout) {
                static constexpr char message[] =
                    "[dmem-write-trace] invalid reason=step-lock-timeout\n";
                raw_write(2, message, sizeof(message) - 1);
                _exit(190);
            }
            if (step == prosper::host::GuestDmemWriteTraceStepAction::Complete ||
                step == prosper::host::GuestDmemWriteTraceStepAction::CompleteInvalid) {
                PROSPER_GREGS(uc)[REG_EFL] &= ~0x100ll;
                const uint64_t saved_guest_fs = guest_fs_to_host_scoped();
                constexpr char hex[] = "0123456789abcdef";
                char line[1024] = {};
                const bool guest = gin(event.writer_rip);
                const char* selected = event.selected ? "yes"
                                       : event.selection_uncertain ? "unknown" : "no";
                int written = snprintf(
                    line, sizeof line,
                    "[dmem-write-trace] event=%llu selected=%s completion=%s"
                    " coverage-valid-before=%s"
                    " changed-during-window=%s"
                    " fault=0x%llx phys=0x%llx decoded-write-bytes=%u"
                    " writer=%s%s0x%llx next=0x%llx tid=%lld"
                    " rearmed=%s before=",
                    static_cast<unsigned long long>(event.ordinal), selected,
                    step == prosper::host::GuestDmemWriteTraceStepAction::CompleteInvalid
                        ? "invalid" : "valid",
                    event.coverage_valid_before ? "yes" : "no",
                    event.changed_during_window ? "yes" : "no",
                    static_cast<unsigned long long>(event.fault_addr),
                    static_cast<unsigned long long>(event.fault_phys),
                    event.decoded_write_size,
                    guest ? gmod(event.writer_rip) : "host", guest ? "+" : ":",
                    static_cast<unsigned long long>(guest ? goff(event.writer_rip)
                                                          : event.writer_rip),
                    static_cast<unsigned long long>(event.next_rip),
                    static_cast<long long>(event.tid), event.rearmed ? "yes" : "no");
                size_t used = written > 0
                                  ? std::min<size_t>(static_cast<size_t>(written), sizeof(line) - 1)
                                  : 0;
                auto append_bytes = [&](const auto& bytes) {
                    for (uint32_t i = 0; i < event.size && used + 2 < sizeof(line); ++i) {
                        line[used++] = hex[bytes[i] >> 4];
                        line[used++] = hex[bytes[i] & 0xf];
                    }
                };
                append_bytes(event.before);
                const char middle[] = " post=";
                if (used + sizeof(middle) < sizeof(line)) {
                    std::memcpy(line + used, middle, sizeof(middle) - 1);
                    used += sizeof(middle) - 1;
                }
                used += prosper::host::guest_dmem_write_trace_format_post(
                    event, line + used, sizeof(line) - used);
                if (used < sizeof(line)) line[used++] = '\n';
                raw_write(2, line, used);
                guest_fs_restore_scoped(saved_guest_fs);
                return;
            }
        }
        // Texture write-watch (#1144): a guest store into a page we armed read-only for cross-submit
        // dirty-tracking. Restore the page (all its aliases) to writable, mark it dirty, and resume so
        // the store re-executes. Returns false for any address we did not arm, so genuine guest faults
        // fall through untouched. Only ever true when this handler is on the sigaltstack (red-zone-safe).
        if (sig == SIGSEGV && si->si_addr) {
            auto* uc = (ucontext_t*)uctx;
            if (PROSPER_REG_ERR(uc) & 2) {
                const auto action = handle_guest_write_watch_fault(
                    reinterpret_cast<uint64_t>(si->si_addr),
                    static_cast<uint64_t>(PROSPER_GREGS(uc)[REG_RIP]), cur_tid(),
                    (PROSPER_GREGS(uc)[REG_EFL] & 0x100ll) != 0);
                if (action == prosper::host::GuestWriteWatchFaultAction::SingleStep)
                    PROSPER_GREGS(uc)[REG_EFL] |= 0x100ll;
                if (action != prosper::host::GuestWriteWatchFaultAction::NotHandled) return;
            }
        }
        // A SIGILL we did NOT emulate: log the faulting instruction bytes so the offending opcode can be
        // identified (another AMD-only ISA extension, or a decode miss in try_emulate_sse4a). #163-progress.
        if (sig == SIGILL) {
            uint64_t rip = (uint64_t)PROSPER_GREGS((ucontext_t*)uctx)[REG_RIP];
            const uint8_t* p = (const uint8_t*)rip;
            char b[96]; int n = raw_fmt_advance(0, snprintf(b, sizeof b, "[sigill] rip=0x%llx bytes:", (unsigned long long)rip), sizeof b);
            for (int k = 0; k < 16 && n < (int)sizeof b - 4; k++) n = raw_fmt_advance(n, snprintf(b + n, sizeof b - n, " %02x", p[k]), sizeof b);
            if (n < (int)sizeof b - 1) b[n++] = '\n';
            ssize_t w = raw_write_fmt(2, b, sizeof b, n); (void)w;
        }
        // #312 per-thread MB3 head watch: a hardware WRITE-watch on a poolArray head slot fired.
        // Write-watchpoints trap AFTER the store, so RIP already points past it — just log the
        // writer RIP + the value now in the slot and return (the watch stays armed for the next
        // write). A byte-shifted pool pointer (0x20015f0000 stored as 0x20015f00) IS the corruptor;
        // dump its full register + guest-stack context. Raw syscalls only (may run on a guest-%fs
        // worker); globals + fd table, no thread_local.
        if (sig == SIGTRAP && g_mb3w_cnt.load(std::memory_order_acquire) > 0) {
            int mi = mb3w_match(PROSPER_SI_FD(si));
            if (mi >= 0) {
                auto gr2 = PROSPER_GREGS((ucontext_t*)uctx);
                uint64_t addr = g_mb3w[mi].addr;
                unsigned long long v = *(volatile uint64_t*)addr;
                uint64_t wr = (uint64_t)gr2[REG_RIP];
                bool in_eboot = (gin(wr));
                // A byte-shifted pool pointer is one corrupt shape; a head that is not a plausible
                // FBundleNode at all is the general case, and the one this watch must never miss.
                //
                // BUDGETED, unlike the shifted case it joins. The loud arm writes ~1 KB (regs plus a
                // guest-stack scan) from inside a SIGTRAP handler on the owning thread, and this
                // PR's own result is that the duration of work on that thread decides whether the
                // title survives — an unbudgeted loud path would be a timing confound in the
                // instrument built to study a timing bug. The first hits are what matter; a slot
                // that keeps re-reporting adds nothing.
                bool shifted = lwatch_is_pool_shift(v);
                unsigned loud_ord = 0;
                if (!shifted && !lwatch_is_plausible_bundle_node(v)) {
                    // Trap 51: a bare cap makes the ceiling the finding. Count every corrupt head,
                    // print the first 8 and then every 256th, and carry the ordinal on the line —
                    // otherwise "8" reads as the population, and a head that keeps being re-poisoned
                    // would be silenced for good by the benign-write budget below.
                    loud_ord = ++g_mb3w_loud[mi];
                    if (loud_ord <= 8 || (loud_ord & 0xff) == 0) shifted = true;
                    else return;
                }
                // base+0x20 is a HOT free-list head (every idx-1 malloc/free touches it), so logging
                // every benign write floods I/O and stalls the run before the t~60s corruption burst.
                // Log only the corruptor (a byte-shifted pool pointer) + the first couple of hits per
                // slot (confirmation that the watch is live). g_mb3w_hits packs a per-slot small count.
                if (!shifted && g_mb3w[mi].fd >= 0) {
                    if (g_mb3w_seen[mi] >= 3) return;   // silent steady-state
                    g_mb3w_seen[mi]++;
                }
                // The ordinal is folded into ONE conditional string: a bare `%u` beside a
                // conditional separator prints `0` on every benign write and fuses it onto the tid
                // field (`tid=12340`), which is exactly the kind of misreading this log exists to
                // prevent.
                char ord[24] = "";
                if (loud_ord) snprintf(ord, sizeof ord, "  #%u", loud_ord);
                char b[256]; int n = snprintf(b, sizeof b,
                    "[mb3watch] WRITE [0x%llx]=0x%llx by rip=%s%s0x%llx tid=%ld%s%s\n",
                    (unsigned long long)addr, v, in_eboot ? gmod(wr) : "host:", in_eboot ? "+" : "",
                    (unsigned long long)(in_eboot ? goff(wr) : wr), cur_tid(),
                    shifted ? "  <<<<< CORRUPT HEAD" : "", ord);
                raw_write_fmt(2, b, sizeof b, n);
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
                    raw_write_fmt(2, rb, sizeof rb, rn);
                    // Guest call stack: scan [rsp, rsp+0x200) for eboot-range return addresses.
                    uint64_t rsp = (uint64_t)gr2[REG_RSP];
                    // The one chain in this file with NO cursor guard at all AND a variable-length
                    // %s in its append (`gmod` returns module names up to 25 characters), so ten
                    // captured frames can format to ~470 characters into 320 bytes. Before #2161
                    // the append after the first truncating one computed `sizeof sb - sn` with
                    // sn > 320 and wrote at `sb + sn` with a ~2^64 size: a real out-of-bounds
                    // write, off the end of the signal stack, on realistic guest stacks.
                    char sb[320]; int sn = raw_fmt_advance(0, snprintf(sb, sizeof sb, "[mb3watch]   guest-stack:"), sizeof sb);
                    int found = 0;
                    for (uint64_t p = rsp; p < rsp + 0x200 && found < 10; p += 8) {
                        if (!probe_readable(p)) break;
                        uint64_t ra = *(const uint64_t*)p;
                        if (gin(ra)) {
                            sn = raw_fmt_advance(sn, snprintf(sb + sn, sizeof sb - sn, " %s+0x%llx",
                                           gmod(ra), (unsigned long long)goff(ra)), sizeof sb);
                            found++;
                        }
                    }
                    sn = raw_fmt_advance(sn, snprintf(sb + sn, sizeof sb - sn, "\n"), sizeof sb);
                    raw_write_fmt(2, sb, sizeof sb, sn);
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
            HwbpThreadState* hwbp_state = hwbp_thread_state(cur_tid());
            // Step-window: continuous single-step from driver hit N to the next driver hit. On each step,
            // find the live read cursor (a GP reg pointing inside [base,end]) and ring-log advances.
            if (g_stepwin_active && si->si_code == TRAP_TRACE) {
                auto gs = PROSPER_GREGS(uc);
                uint64_t rip = (uint64_t)gs[REG_RIP];
                g_stepwin_steps++;
                static const int idx[] = { REG_RAX,REG_RBX,REG_RCX,REG_RDX,REG_RSI,REG_RDI,REG_R8,
                                           REG_R9,REG_R10,REG_R11,REG_R12,REG_R13,REG_R14,REG_R15 };
                unsigned long long cur = 0;
                for (int i = 0; i < 14; i++) { uint64_t v = (uint64_t)gs[idx[i]];
                    if (v >= g_stepwin_base && v < g_stepwin_end) { cur = v; break; } }
                if (cur && cur != g_stepwin_prevcur) {
                    g_stepwin_ring[g_stepwin_pos % STEPWIN_RING] =
                        { gmod(rip), (unsigned long long)goff(rip), cur };
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
            // perf's asynchronous SIGTRAP delivery does not reliably preserve TRAP_TRACE for the
            // TF completion on every host/kernel combination. The perf event is disabled while this
            // state is set, so the next trap is the single-step completion by construction.
            if ((hwbp_state && hwbp_state->stepping) || g_hwbp_stepping) {
                const int fd = hwbp_state ? hwbp_state->fd : g_hwbp_fd;
                ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
                PROSPER_GREGS(uc)[REG_EFL] &= ~0x100ll;
                if (hwbp_state) hwbp_state->stepping = 0;
                g_hwbp_stepping = false;
                return;
            }
            // Chained DATA write-watchpoint hit (a write to the watched slot completed): log the writer
            // RIP + the value just stored. Write-watchpoints trap AFTER the store, so no step is needed.
            if (g_hwwatch_fd >= 0 && PROSPER_SI_FD(si) == g_hwwatch_fd) {
                auto gr2 = PROSPER_GREGS(uc);
                unsigned long long v = 0; { auto p=(volatile uint64_t*)g_hwwatch_addr; v=*p; }
                // Always log an ANOMALOUS store (not a plausible guest heap pointer 0x1000000000..
                // 0x1720000000, and nonzero) even past the count cap — this is how a corrupt value
                // (e.g. a magic constant landing where a pointer belongs) is caught among the churn.
                bool anomalous = v && (v < 0x1000000000ull || v >= 0x1730000000ull);
                if (g_hwwatch_count < 60 || anomalous) {
                    g_hwwatch_count = g_hwwatch_count + 1;
                    char b[200];
                    uint64_t wr = (uint64_t)gr2[REG_RIP];
                    bool in_eboot = (gin(wr));
                    int n = snprintf(b, sizeof b, "[hwwatch] #%d WRITE [0x%llx]=0x%llx by rip=%s%s0x%llx tid=%ld\n",
                        (int)g_hwwatch_count, (unsigned long long)g_hwwatch_addr, v,
                        in_eboot ? gmod(wr) : "", in_eboot ? "+" : "", (unsigned long long)(in_eboot ? goff(wr) : wr), cur_tid());
                    raw_write_fmt(2, b, sizeof b, n);   /* raw syscall: no libc TLS access */
                    if (!in_eboot) classify_addr(wr);
                }
                return;
            }
            auto gr = PROSPER_GREGS(uc);
            uint64_t rdi = (uint64_t)gr[REG_RDI], rsi = (uint64_t)gr[REG_RSI];
            uint64_t rax = (uint64_t)gr[REG_RAX], r14 = (uint64_t)gr[REG_R14], r15 = (uint64_t)gr[REG_R15];
            // This handler returns to the guest, which may be on the guest %fs — swap to host %fs for the
            // host-libc logging below, restore before returning. No-op off the guest-fs path.
            uint64_t saved_fs = guest_fs_to_host_scoped();
            const uint64_t rsp = (uint64_t)gr[REG_RSP];
            const uint64_t ret = probe_readable(rsp + 7) ? *(const uint64_t*)rsp : 0;
            bool cond_ok = (!g_hwbp_r15_on || (r15 == g_hwbp_r15)) &&
                           (!g_hwbp_r14_on || (r14 == g_hwbp_r14)) &&
                           (!g_hwbp_ret_on || (ret == g_hwbp_ret)) && (rax >= g_hwbp_raxmin);
            // PROSPER_HWBP_ANOM ring trace: push every gated hit; when a read yields a value >= the
            // anomaly threshold, dump the ring (the cursor walk into the over-read) exactly once.
            if (g_hwbp_anom_on && cond_ok && !g_hwbp_ring_dumped) {
                uint64_t rbxr = (uint64_t)gr[REG_RBX];
                unsigned val = probe_readable(rax) ? *(const uint32_t*)rax : 0xBADBADu;
                unsigned long long cur = probe_readable(rbxr + 0x38) ? *(const uint64_t*)(rbxr + 0x38) : 0;
                int p = g_hwbp_ring_pos % HWBP_RING;
                g_hwbp_ring[p] = { gmod((uint64_t)gr[REG_RIP]), (unsigned long long)goff((uint64_t)gr[REG_RIP]),
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
                g_hwbp_ring[p] = { gmod((uint64_t)gr[REG_RIP]), (unsigned long long)goff((uint64_t)gr[REG_RIP]),
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
                    if (k >= 2) { char b[128]; int n = snprintf(b, sizeof b, "[hwbp-str] %s=0x%llx -> \"%s\"\n", rn, (unsigned long long)p, s); raw_write_fmt(2, b, sizeof b, n);   /* raw syscall: no libc TLS access */ }
                };
                char hdr[80]; int hn = snprintf(hdr, sizeof hdr, "[hwbp-str] hit @%s+0x%llx:\n",
                    gmod((uint64_t)gr[REG_RIP]),
                    (unsigned long long)goff((uint64_t)gr[REG_RIP])); raw_write_fmt(2, hdr, sizeof hdr, hn);
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
                        (unsigned long long)(probe_readable(obj+0x40)?*(const uint64_t*)(obj+0x40):0)); raw_write_fmt(2, b, sizeof b, n);   /* raw syscall: no libc TLS access */ }
                };
                // PROSPER_HWBP_FIELDS=1: dump rbx's object fields [0x10..0x60] + the class name of any
                // field that is itself an object — to tell "whole object uninitialized (all null)" from
                // "one specific field null".
                if (g_hwbp_fields) {
                    uint64_t o = (uint64_t)gr[REG_RBX];
                    if (probe_readable(o + 0x60)) {
                        char b[400]; int n = raw_fmt_advance(0, snprintf(b, sizeof b, "[hwbp-fields] rbx=0x%llx:", (unsigned long long)o), sizeof b);
                        for (uint64_t off = 0x10; off <= 0x60; off += 8) {
                            uint64_t v = *(const uint64_t*)(o + off);
                            n = raw_fmt_advance(n, snprintf(b+n, sizeof b-n, " +0x%llx=0x%llx", (unsigned long long)off, (unsigned long long)v), sizeof b);
                        }
                        n = raw_fmt_advance(n, snprintf(b+n, sizeof b-n, "\n"), sizeof b); raw_write_fmt(2, b, sizeof b, n);   /* raw syscall: no libc TLS access */
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
                                "[hwbp-field] +0x%llx -> %s\n", (unsigned long long)off, s); raw_write_fmt(2, fb, sizeof fb, fn); }
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
                        rn, (unsigned long long)klass, s); raw_write_fmt(2, b, sizeof b, n);   /* raw syscall: no libc TLS access */ }
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
            // (rdi/rsi/rdx/rcx/r8/r9), guarded [rdi+0x10]/[rdx+0x10], and the return address, so a
            // call's arguments (and UnityEngine.Object m_CachedPtr values) are visible per hit without
            // a gdb round-trip. Generic; composes with the count cap.
            if (g_hwbp_args && cond_ok) {
                auto arg_field = [](uint64_t p) -> uint64_t {
                    return probe_readable(p + 0x17) ? *(const uint64_t*)(p + 0x10) : 0;
                };
                char b[256]; int n = snprintf(b, sizeof b,
                    "[hwbp-args] rdi=0x%llx [+10]=0x%llx rsi=0x%llx rdx=0x%llx [+10]=0x%llx rcx=0x%llx r8=0x%llx r9=0x%llx ret=%s+0x%llx\n",
                    (unsigned long long)gr[REG_RDI],
                    (unsigned long long)arg_field((uint64_t)gr[REG_RDI]),
                    (unsigned long long)gr[REG_RSI],
                    (unsigned long long)gr[REG_RDX],
                    (unsigned long long)arg_field((uint64_t)gr[REG_RDX]),
                    (unsigned long long)gr[REG_RCX],
                    (unsigned long long)gr[REG_R8], (unsigned long long)gr[REG_R9],
                    probe_readable((uint64_t)gr[REG_RSP])
                        ? gmod(*(const uint64_t*)(uint64_t)gr[REG_RSP]) : "?",
                    (unsigned long long)(probe_readable((uint64_t)gr[REG_RSP]) ? goff(*(const uint64_t*)(uint64_t)gr[REG_RSP]) : 0));
                raw_write_fmt(2, b, sizeof b, n);
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
                raw_write_fmt(2, b, sizeof b, n);
            }
            // Reuse the software breakpoint's generic pointer-chain evaluator for race-free hardware
            // breakpoints. Evaluate only hits that pass every register/return gate and remain under the
            // ordinary output cap; unmatched hot-path hits neither print nor consume that cap.
            if (g_hwbp_probe && cond_ok && g_hwbp_count < g_hwbp_max)
                bp_eval_probes(g_hwbp_probe, uc, 0, "[hwbp-probe]");
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
                raw_write_fmt(2, b2, sizeof b2, n2);
                classify_addr(rax);
            }
            if (cond_ok && g_hwbp_count < g_hwbp_max) {
                g_hwbp_count = g_hwbp_count + 1;
                auto rd = [](uint64_t a) -> unsigned long long {
                    return probe_readable(a) ? (unsigned long long)*(const uint64_t*)a : 0xBADBADull; };
                uint64_t rbp = (uint64_t)gr[REG_RBP], rsp = (uint64_t)gr[REG_RSP];
                auto off = [](unsigned long long v) -> unsigned long long {
                    return (gin(v)) ? goff(v) : v; };
                // Companion namer: a widened filter must not keep a hard-coded "eboot+" label, or an
                // Il2Cpp frame prints a small in-range offset under the wrong module name — worse than
                // the old out-of-range value, which at least announced itself (#1659 review).
                auto nm = [](unsigned long long v) -> const char* { return gin(v) ? gmod(v) : "host"; };
                // Also surface rax + the u32 at [rax] — at the deserializer length-read site (0x7e40d9/e1)
                // rax is the stream cursor and [rax] the value being read, so a trace shows the parse walk.
                unsigned rax_u32 = probe_readable(rax) ? *(const uint32_t*)rax : 0xBADBADu;
                // Also surface a candidate stream-reader object in rbx: cursor [+0x38], base [+0x40],
                // end [+0x48] — at the 0x1612 deserializer driver rbx IS the reader, so this shows the
                // cursor's position/window per field (to see which read walks it into the shader blob).
                uint64_t rbx = (uint64_t)gr[REG_RBX];
                char b[512];
                int n = snprintf(b, sizeof b,
                    "[hwbp] #%d rip=%s+0x%llx rax=0x%llx [rax]=0x%08x r14=0x%llx rbx=0x%llx cur=0x%llx base=0x%llx end=0x%llx ret=%s+0x%llx caller_rbp=%s+0x%llx tid=%ld\n",
                    (int)g_hwbp_count, gmod((uint64_t)gr[REG_RIP]),
                    (unsigned long long)goff((uint64_t)gr[REG_RIP]),
                    (unsigned long long)rax, rax_u32, (unsigned long long)r14, (unsigned long long)rbx,
                    rd(rbx+0x38), rd(rbx+0x40), rd(rbx+0x48), nm(rd(rsp)), off(rd(rsp)),
                    nm(rd(rbp + 8)), off(rd(rbp + 8)), cur_tid());
                (void)r15; (void)rdi; (void)rsi;
                raw_write_fmt(2, b, sizeof b, n);   /* raw syscall: no libc TLS access */
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
                            if (fd >= 0) { (void)!raw_write(fd, (const void*)lo, (size_t)(hi - lo)); close(fd);
                                char m2[96]; int mn = snprintf(m2, sizeof m2, "[ttnodes] dumped [0x%llx..0x%llx] base=0x%llx\n",
                                    (unsigned long long)lo, (unsigned long long)hi, (unsigned long long)node); raw_write_fmt(2, m2, sizeof m2, mn); }
                        }
                    }
                    char db[300];
                    int dn = snprintf(db, sizeof db,
                        "[hwbp-divcap] #%d elemSz=0x%llx byteSz=0x%llx cnt=0x%llx | node=0x%llx [n]=0x%llx [n+8]=0x%llx [n+0x10]=0x%llx [n+0x18]=0x%llx [n+0x20]=0x%llx | cur=0x%llx\n",
                        (int)g_hwbp_count, rd(crbp - 0xf0), rd(crbp - 0xe8), rd(crbp - 0xf8),
                        node, rd(node), rd(node + 8), rd(node + 0x10), rd(node + 0x18), rd(node + 0x20),
                        rd((uint64_t)gr[REG_RBX] + 0x38));
                    raw_write_fmt(2, db, sizeof db, dn);
                }
                // PROSPER_HWBP_BUFDUMP: write the reader window [base..end] to a per-hit file.
                if (g_hwbp_bufdump) {
                    uint64_t base = rd(rbx + 0x40), end = rd(rbx + 0x48);
                    if (end > base && end - base < 0x400000 && probe_readable(base) && probe_readable(end - 1)) {
                        char fn[64]; snprintf(fn, sizeof fn, "/tmp/prosper_buf_%d.bin", (int)g_hwbp_count);
                        int fd = open(fn, O_WRONLY | O_CREAT | O_TRUNC, 0644);
                        if (fd >= 0) { (void)!raw_write(fd, (const void*)base, (size_t)(end - base)); close(fd);
                            char m2[96]; int mn = snprintf(m2, sizeof m2, "[hwbp] bufdump #%d -> %s (%llu bytes)\n",
                                (int)g_hwbp_count, fn, (unsigned long long)(end - base)); raw_write_fmt(2, m2, sizeof m2, mn); }
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
                else if (!strcmp(r,"r12")) base = (uint64_t)gr[REG_R12];
                else if (!strcmp(r,"r14")) base = r14;
                else if (!strcmp(r,"r15")) base = r15;
                else if (!strcmp(r,"rbp")) base = (uint64_t)gr[REG_RBP];
                arm_hwwatch(base + (uint64_t)g_hwwatch_delta);
            }
            const int hit_fd = hwbp_state ? hwbp_state->fd :
                               (PROSPER_SI_FD(si) >= 0 ? PROSPER_SI_FD(si) : g_hwbp_fd);
            ioctl(hit_fd, PERF_EVENT_IOC_DISABLE, 0);   // disable so we can step off the bp address
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
                raw_write_fmt(2, sb, sizeof sb, sn);
                PROSPER_GREGS(uc)[REG_EFL] |= 0x100ll;   // start single-stepping
                guest_fs_restore_scoped(saved_fs);
                return;
            }
            // Stay armed while logging OR while the anomaly-ring trace is still hunting (PROSPER_HWBP_MAX
            // can be 0 to suppress the per-hit log yet keep the bp live feeding the ring until it dumps).
            if (g_hwbp_count < g_hwbp_max || ((g_hwbp_anom_on || g_hwbp_node_on) && !g_hwbp_ring_dumped)) {
                PROSPER_GREGS(uc)[REG_EFL] |= 0x100ll;
                if (hwbp_state) hwbp_state->stepping = 1; else g_hwbp_stepping = true;
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
                PROSPER_GREGS(uc)[REG_EFL] &= ~0x100ll;   // clear TF
                g_bp_stepping = false;
                return;
            }
            uint64_t rip = (uint64_t)PROSPER_GREGS(uc)[REG_RIP];
            if (rip == g_bp_addr + 1) {
                uint64_t r15 = (uint64_t)PROSPER_GREGS(uc)[REG_R15];
                auto gr0 = PROSPER_GREGS(uc);
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
                    auto gr = PROSPER_GREGS(uc);
                    uint64_t rdi = (uint64_t)gr[REG_RDI], rsi = (uint64_t)gr[REG_RSI];
                    uint64_t rax = (uint64_t)gr[REG_RAX], r14 = (uint64_t)gr[REG_R14];
                    uint64_t rcx = (uint64_t)gr[REG_RCX];
                    uint64_t r10 = (uint64_t)gr[REG_R10], rdx = (uint64_t)gr[REG_RDX];
                    uint64_t r8 = (uint64_t)gr[REG_R8];
                    char b[512];
                    int n = raw_fmt_advance(0, snprintf(b, sizeof b,
                        "[bp] #%d rsi=0x%llx r10=0x%llx rdx=0x%llx rcx=0x%llx r8=0x%llx rax=0x%llx rdi=0x%llx r14=0x%llx r15=0x%llx [rdi]=0x%llx [rdi+0x1e4c]=0x%llx tid=%ld",
                        (int)g_bp_count, (unsigned long long)rsi, (unsigned long long)r10,
                        (unsigned long long)rdx, (unsigned long long)rcx, (unsigned long long)r8,
                        (unsigned long long)rax, (unsigned long long)rdi, (unsigned long long)r14,
                        (unsigned long long)r15, rd(rdi),
                        (unsigned long long)(probe_readable(rdi + 0x1e4c) ? *(const uint32_t*)(rdi + 0x1e4c) : 0xBADBAD),
                        cur_tid()), sizeof b);
                    // Caller stack: first guest-text return addresses (who called free with this ptr).
                    uint64_t rsp = (uint64_t)gr[REG_RSP];
                    int found = 0;
                    for (uint64_t o = 0; o < 0x200 && found < 8 && n < (int)sizeof b - 24; o += 8) {
                        if (!probe_readable(rsp + o)) break;
                        uint64_t v = *(const uint64_t*)(rsp + o);
                        if (gin(v)) {
                            n = raw_fmt_advance(n, snprintf(b + n, sizeof b - (size_t)n, " s:%llx",
                                          (unsigned long long)goff(v)), sizeof b);
                            found++;
                        }
                    }
                    if (n < (int)sizeof b - 1) b[n++] = '\n';
                    raw_write_fmt(2, b, sizeof b, n);   /* raw syscall: no libc TLS access */
                    // PROSPER_BP_PROBE: evaluate the configured pointer/field chains at this hit (first,
                    // so it is never skipped by a KLASS-candidate issue below).
                    if (g_bp_probe) bp_eval_probes(g_bp_probe, uc, 0, "[bp-probe]");
                    // PROSPER_KSCAN: on the first hit, scan the guest heap for every live instance of the
                    // target class and dump each + its fields (one-shot; see il2cpp_kscan).
                    if (g_kscan_klass && !g_kscan_done) { g_kscan_done = true; il2cpp_kscan(uc); }
                    // PROSPER_BP_KLASS: name the IL2CPP object in each candidate register. Only probe values
                    // in the guest GC-heap band [0x20_00000000, 0x28_00000000) — host stack pointers (0x7f..),
                    // module/code pointers (0x4..) and small values are not objects and needlessly walk the
                    // obj->klass->name chain over unrelated memory.
                    if (g_bp_klass) {
                        const struct { const char* rn; uint64_t v; } cand[] = {
                            {"rdi",rdi},{"rsi",rsi},{"rdx",rdx},{"rcx",rcx},{"r8",r8},{"r9",(uint64_t)gr[REG_R9]},
                            {"rbx",(uint64_t)gr[REG_RBX]},{"rax",rax},{"r14",r14},{"r15",r15} };
                        for (auto& c : cand) {
                            if (c.v < 0x2000000000ull || c.v >= 0x2800000000ull) continue;
                            char nm[112];
                            if (bp_il2cpp_cname(c.v, nm, sizeof nm)) {
                                char kb[176]; int kn = snprintf(kb, sizeof kb, "[bp-klass] %s=0x%llx %s\n",
                                    c.rn, (unsigned long long)c.v, nm);
                                raw_write_fmt(2, kb, sizeof kb, kn);
                            }
                        }
                    }
                }
                bp_write_byte(g_bp_addr, g_bp_orig);              // restore real instruction
                PROSPER_GREGS(uc)[REG_RIP] = (greg_t)g_bp_addr;  // re-execute it
                PROSPER_GREGS(uc)[REG_EFL] |= 0x100ll;        // single-step (TF)
                g_bp_stepping = true;
                return;
            }
        }
        // #312 label-slot watch: SIGTRAP after single-stepping a write to the watched label page —
        // log the slot's post-write value, re-protect, keep watching.
        if (sig == SIGTRAP && g_lwatch_stepping) {
            auto* uc = (ucontext_t*)uctx;
            PROSPER_GREGS(uc)[REG_EFL] &= ~0x100ll;   // clear TF
            if (g_lwatch_shift) {
                // Value-triggered: only report if the just-completed store left a byte-shifted pool
                // pointer at the faulting address (the primary #312 corruptor). Silent otherwise.
                uint64_t v = probe_readable(g_lwatch_fa) ? *(const uint64_t*)g_lwatch_fa : 0;
                if (lwatch_is_pool_shift(v)) {
                    g_lwatch_hits = g_lwatch_hits + 1;
                    char b[512];
                    bool guest = gin(g_lwatch_step_rip);
                    int n = raw_fmt_advance(0, snprintf(b, sizeof b,
                        "[lwatch] SHIFT-STOMP fa=0x%llx val=0x%llx (<<8=0x%llx) rip=%s0x%llx tid=%ld",
                        (unsigned long long)g_lwatch_fa, (unsigned long long)v,
                        (unsigned long long)(v << 8), guest ? "eboot+" : "host:",
                        (unsigned long long)(guest ? goff(g_lwatch_step_rip) : g_lwatch_step_rip),
                        cur_tid()), sizeof b);
                    for (int i = 0; i < g_lwatch_stkn && n < (int)sizeof b - 24; i++)
                        n = raw_fmt_advance(n, snprintf(b + n, sizeof b - (size_t)n, " s:%llx",
                                      (unsigned long long)goff(g_lwatch_stk[i])), sizeof b);
                    if (n < (int)sizeof b - 1) b[n++] = '\n';
                    raw_write_fmt(2, b, sizeof b, n);
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
                raw_write_fmt(2, b, sizeof b, n);
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
            if ((fa & ~(uint64_t)0xfff) == g_lwatch_page && (PROSPER_REG_ERR(uc) & 2)) {
                uint64_t rip = (uint64_t)PROSPER_GREGS(uc)[REG_RIP];
                g_lwatch_step_rip = 0;
                if (g_lwatch_shift) {
                    // Value-triggered: capture writer + stack silently now, decide at SIGTRAP whether
                    // the store produced the byte-shifted pool pointer. Watch the whole page (any
                    // write) so the corruptor is caught wherever in the FPoolInfo page it lands.
                    g_lwatch_fa = fa;
                    g_lwatch_step_rip = rip;
                    uint64_t rsp = (uint64_t)PROSPER_GREGS(uc)[REG_RSP];
                    g_lwatch_stkn = 0;
                    for (uint64_t o = 0; o < 0x400 && g_lwatch_stkn < 8; o += 8) {
                        if (!probe_readable(rsp + o)) break;
                        uint64_t v = *(const uint64_t*)(rsp + o);
                        if (gin(v)) g_lwatch_stk[g_lwatch_stkn++] = v;
                    }
                    mprotect((void*)g_lwatch_page, 0x1000, PROT_READ | PROT_WRITE);
                    PROSPER_GREGS(uc)[REG_EFL] |= 0x100ll;   // TF -> single-step the write
                    g_lwatch_stepping = true;
                    return;
                }
                if (fa >= g_lwatch_slot - 0x18 && fa < g_lwatch_slot + 0x20) {
                    g_lwatch_hits = g_lwatch_hits + 1;
                    char b[512];
                    bool guest = gin(rip);
                    int n = raw_fmt_advance(0, snprintf(b, sizeof b,
                        "[lwatch] #%d write fa=0x%llx rip=%s0x%llx tid=%ld pre[0]=0x%llx pre[8]=0x%llx",
                        (int)g_lwatch_hits, (unsigned long long)fa,
                        guest ? "eboot+" : "host:",
                        (unsigned long long)(guest ? goff(rip) : rip), cur_tid(),
                        (unsigned long long)*(const uint64_t*)g_lwatch_slot,
                        (unsigned long long)*(const uint64_t*)(g_lwatch_slot + 8)), sizeof b);
                    // #312: call-stack capture — scan the writer's stack for eboot-text return
                    // addresses (up to 8), so the guest free/alloc path above the raw write is
                    // identifiable (async-signal-safe: bounded reads of the faulting thread's
                    // own stack).
                    uint64_t rsp = (uint64_t)PROSPER_GREGS(uc)[REG_RSP];
                    int found = 0;
                    for (uint64_t o = 0; o < 0x400 && found < 8 && n < (int)sizeof b - 24; o += 8) {
                        if (!probe_readable(rsp + o)) break;
                        uint64_t v = *(const uint64_t*)(rsp + o);
                        if (gin(v)) {
                            n = raw_fmt_advance(n, snprintf(b + n, sizeof b - (size_t)n, " s:%llx",
                                          (unsigned long long)goff(v)), sizeof b);
                            found++;
                        }
                    }
                    if (n < (int)sizeof b - 1) b[n++] = '\n';
                    raw_write_fmt(2, b, sizeof b, n);
                    g_lwatch_step_rip = rip;
                    if (g_lwatch_hits >= g_lwatch_max) {   // bounded: disarm after enough evidence
                        mprotect((void*)g_lwatch_page, 0x1000, PROT_READ | PROT_WRITE);
                        g_lwatch_armed = 0;
                        return;
                    }
                }
                mprotect((void*)g_lwatch_page, 0x1000, PROT_READ | PROT_WRITE);
                PROSPER_GREGS(uc)[REG_EFL] |= 0x100ll;   // TF -> single-step the write
                g_lwatch_stepping = true;
                return;
            }
        }
        // Companion-slot write-watchpoint: SIGTRAP after single-stepping a write to the watched page —
        // re-arm (re-protect read-only) and clear the trap flag so we keep watching.
        if (sig == SIGTRAP && g_watch_stepping) {
            auto* uc = (ucontext_t*)uctx;
            PROSPER_GREGS(uc)[REG_EFL] &= ~0x100ll;   // clear TF
            // Post-write: log the slot's new value + the object's type tag [r15+0] (0x2b if it's still
            // the same live object; changed => the memory was freed and reused, so the write is churn).
            unsigned long long slot = *(const uint64_t*)g_watch_addr;
            unsigned long long tag  = *(const uint64_t*)(g_watch_addr - 0x140);
            char b[128];
            int n = snprintf(b, sizeof b, "[watch]   -> slot now=0x%llx  obj[+0]=0x%llx\n", slot, tag);
            raw_write_fmt(2, b, sizeof b, n);   /* raw syscall: no libc TLS access */
            mprotect((void*)g_watch_page, 0x1000, PROT_READ);
            g_watch_stepping = false;
            return;
        }
        // Arm the watchpoint on the first companion read (rip == the reader deref, slot still null).
        if (g_watch_companion && sig == SIGSEGV && !g_watch_armed) {
            auto* uc = (ucontext_t*)uctx;
            if ((uint64_t)PROSPER_GREGS(uc)[REG_RIP] == g_skip_rip) {
                uint64_t r15 = (uint64_t)PROSPER_GREGS(uc)[REG_R15];
                g_watch_addr = r15 + 0x140;
                g_watch_page = g_watch_addr & ~(uint64_t)0xfff;
                if (mprotect((void*)g_watch_page, 0x1000, PROT_READ) == 0) {
                    g_watch_armed = true;
                    char b[128];
                    int n = snprintf(b, sizeof b, "[watch] armed on companion slot 0x%llx (obj r15=0x%llx) reader-tid=%ld\n",
                                     (unsigned long long)g_watch_addr, (unsigned long long)r15, cur_tid());
                    raw_write_fmt(2, b, sizeof b, n);   /* raw syscall: no libc TLS access */
                }
                // Skip this (null) read so the boot proceeds; the slot's page is now watched.
                PROSPER_GREGS(uc)[REG_RIP] = (greg_t)g_skip_target;
                return;
            }
        }
        // A write to the watched page (x86 page-fault error code bit 1 = write) while armed: log if it
        // lands in the 8-byte companion slot, then single-step past it (unprotect + set TF).
        if (g_watch_armed && sig == SIGSEGV && si->si_addr) {
            uint64_t fa = (uint64_t)si->si_addr;
            auto* uc = (ucontext_t*)uctx;
            if ((fa & ~(uint64_t)0xfff) == g_watch_page && (PROSPER_REG_ERR(uc) & 2)) {
                if (fa >= g_watch_addr && fa < g_watch_addr + 8) {
                    g_watch_hits = g_watch_hits + 1;
                    char b[160];
                    int n = snprintf(b, sizeof b, "[watch] WRITE to companion slot 0x%llx from rip=0x%llx writer-tid=%ld (hit #%d)\n",
                                     (unsigned long long)fa,
                                     (unsigned long long)PROSPER_GREGS(uc)[REG_RIP], cur_tid(),
                                     (int)g_watch_hits);
                    raw_write_fmt(2, b, sizeof b, n);   /* raw syscall: no libc TLS access */
                }
                mprotect((void*)g_watch_page, 0x1000, PROT_READ | PROT_WRITE);
                PROSPER_GREGS(uc)[REG_EFL] |= 0x100ll;   // set TF -> single-step the write
                g_watch_stepping = true;
                return;
            }
        }
        // Repeated companion reads (later reader passes): skip them too so the boot keeps running.
        if (g_watch_armed && sig == SIGSEGV) {
            auto* uc = (ucontext_t*)uctx;
            if ((uint64_t)PROSPER_GREGS(uc)[REG_RIP] == g_skip_rip) {
                PROSPER_GREGS(uc)[REG_RIP] = (greg_t)g_skip_target;
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
                auto* uc2 = (ucontext_t*)uctx;
                // #1944: the line used to carry only the 64 KiB page and the rip, which is why a
                // masked wild READ was indistinguishable from a lost commit. Report the exact
                // faulting address, whether the access was a read or a write (x86 page-fault error
                // code bit 1), and a running ordinal so the whole-run population is visible without
                // an atexit summary (the worker-fault path calls _exit()).
                const bool is_write = (PROSPER_REG_ERR(uc2) & 2) != 0;
                const unsigned ord =
                    g_lazy_commit_events.fetch_add(1, std::memory_order_relaxed) + 1u;
                // #1226 tell: a faulting address that lands in the FIRST 64 KiB PAGE of a heap
                // pointer's high half is not a page the guest ever populated — it is a pointer that
                // lost its low dword (a 4-byte immediate-zero init, or any other sub-qword write)
                // and then took a small structure offset. ArcRunner's terminal `addr=(nil)` fault
                // dereferences `rdi=0x2100000001` and faults at `rdi+0x40` = `0x2100000041`.
                //
                // Test the PAGE, not the address. An earlier form tested `(uint32_t)a <= 1` and was
                // therefore INERT on the exact case it was written for — `0x41` is not `<= 1` — a
                // marker that could never fire on its own founding evidence. Every `0x21000000xx`
                // value lands in page `0x2100000000`, which is the same argument the doc makes for
                // why two different guest sites appear to first-touch one page, so the page is the
                // right granularity. The enclosing block already requires `a >= 0x1000000000`, so
                // no separate high-half test is needed.
                const bool forged_shape = ((uint32_t)a) <= 0xffffu;
                bool ok = false;
                if (!g_lazy_commit_strict)
                    ok = mmap(page, 0x10000, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == page;
                // 320, not 256: the worst-case line is ~228 bytes (max ordinal + two 16-digit
                // addresses + rip + the FORGED-PTR-SHAPE suffix). The write below is clamped, so a
                // suffix edit that pushed past the buffer now costs a visibly truncated line rather
                // than an over-read — keep the headroom anyway so it stays a complete line.
                char b[320];
                int n = snprintf(b, sizeof b,
                                 "[lazy-commit] #%u %s page=0x%llx addr=0x%llx access=%s rip=0x%llx%s\n",
                                 ord,
                                 g_lazy_commit_strict ? "DECLINED(strict)"
                                                      : (ok ? "mapped" : "MMAP-FAILED"),
                                 (unsigned long long)(uint64_t)(uintptr_t)page,
                                 (unsigned long long)a, is_write ? "write" : "read",
                                 (unsigned long long)PROSPER_GREGS(uc2)[REG_RIP],
                                 forged_shape ? "  FORGED-PTR-SHAPE(low-dword<=0xffff: a truncated pointer, "
                                                "not a page the guest populated; see #1226)" : "");
                // raw_write (raw_syscall.hpp), NOT glibc write() OR syscall(): this handler can run on a
                // thread whose %fs is the GUEST TCB (PROSPER_GUEST_FS). glibc write()'s cancellation
                // prologue reads the TCB through %fs, and even glibc's syscall() wrapper stores errno
                // via %fs on write failure — either is a nested SIGSEGV inside the handler that killed
                // the process (the UE4 boot's silent exit-139; #1071/#1075).
                // raw_write_fmt clamps: snprintf returns the length it WOULD have written, so an
                // unclamped `n` after truncation reads past the buffer. This site had the clamp
                // open-coded; it now shares the one helper every report site uses (#2050).
                raw_write_fmt(2, b, sizeof b, n);
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
            uint64_t rip = (uint64_t)PROSPER_GREGS((ucontext_t*)uctx)[REG_RIP];
            if (a >= GPU_VA_LO && a < GPU_VA_HI && a != rip) {
                void* page = (void*)(a & ~(uint64_t)0xfff);
                bool ok = mmap(page, 0x1000, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == page;
                if (g_faultlog) {
                    char b[128];
                    int n = snprintf(b, sizeof b, "[fault] GPU-VA %s addr=0x%llx rip=0x%llx\n",
                                     ok ? "mapped" : "MMAP-FAILED", (unsigned long long)a,
                                     (unsigned long long)rip);
                    raw_write_fmt(2, b, sizeof b, n);   /* raw syscall: no libc TLS access */
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
            uint64_t rip = (uint64_t)PROSPER_GREGS(uc)[REG_RIP];
            // Only a DATA READ from low memory is backable. An instruction fetch at null (a==rip, i.e.
            // a call/jmp through a null pointer) or a fault on an already-backed page (a null WRITE)
            // must NOT be re-backed — those are the real chain endpoints; let them terminate + report.
            unsigned pg = (unsigned)(a >> 12);
            if (a < 0x10000ull && a != rip && pg < 16 && !(g_null_page_mask & (1u << pg))) {
                uint64_t page = a & ~(uint64_t)0xfff;
                // Raw syscall, NOT glibc mmap()/syscall(): both store errno through %fs on
                // failure, and %fs is guest TLS on this thread — the store itself faults inside
                // the handler and kills the process before any report (issue #1071).
                long mr = raw_mmap_fixed_ro(page, 0x1000);
                if (mr == 0) {
                    g_null_page_mask |= (1u << pg);
                    g_null_page_count = g_null_page_count + 1;
                    char b[128];
                    int n = snprintf(b, sizeof b, "[nullpage] #%d addr=0x%llx rip=eboot+0x%llx\n",
                                     (int)g_null_page_count, (unsigned long long)a,
                                     (unsigned long long)(rip - g_base));
                    raw_write_fmt(2, b, sizeof b, n);   /* raw_syscall: no errno/TLS access, %fs-safe in the handler (#1075) */
                    nullpage_deep_dump(uc, rip);   // issue-#312 attribution dump (registers + heap + GPU ring)
                    return;   // re-execute; the null read now sees zero
                }
#ifdef __linux__
                // Rootless/containerized Linux commonly rejects only this diagnostic mapping with
                // EPERM/EACCES. Emulate a strictly decoder-proven *read* instead of letting host
                // policy decide whether that diagnostic can reach the guest's next real endpoint.
                // Any unsupported access remains fail-visible below.
                if (si->si_code == SEGV_MAPERR && try_emulate_denied_null_read(uc, a, rip, mr)) return;
#endif
                // Fail-visible: low mappings need CAP_SYS_RAWIO once addr < vm.mmap_min_addr
                // (default 65536). If the conservative fallback above cannot prove this is an
                // emulatable read, fall through to the normal fatal report instead of guessing.
                char b[160];
                int n = snprintf(b, sizeof b,
                                 "[nullpage] BACKING DENIED addr=0x%llx rip=eboot+0x%llx err=%ld"
                                 " (vm.mmap_min_addr/CAP_SYS_RAWIO)\n",
                                 (unsigned long long)a,
                                 (unsigned long long)(rip - g_base), mr);
                raw_write_fmt(2, b, sizeof b, n);   /* not even glibc syscall(): it stores errno via %fs on failure */
            }
        }
        // Null-companion skip probe (diagnostic; see g_skip_null_companion). Redirect the reader past
        // the null [obj+0x140] deref to its own skip label, logging each object's state.
        if (g_skip_null_companion && sig == SIGSEGV) {
            auto* uc = (ucontext_t*)uctx;
            uint64_t rip = (uint64_t)PROSPER_GREGS(uc)[REG_RIP];
            if (rip == g_skip_rip) {
                uint64_t r15 = (uint64_t)PROSPER_GREGS(uc)[REG_R15];
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
                raw_write_fmt(2, b, sizeof b, n);   /* raw syscall: no libc TLS access */
                PROSPER_GREGS(uc)[REG_RIP] = (greg_t)g_skip_target;
                return;   // re-execute from the reader's skip label
            }
        }
        // Fault-report path (env-gated diagnostic/stepping cases above already returned; the int-0x41
        // skip and PROSPER_FAULT_SKIP return-to-guest cases are handled a few lines below). If this
        // thread was running on OUR guest %fs, switch to the host %fs NOW so the host-libc reporting
        // below (snprintf/write) + the siglongjmp-return into host C++ don't double-fault reading guest
        // TLS as glibc's TCB. No-op when guest-fs is off. (The GC RT-signal handler keeps the guest %fs.)
        // Capture the guest %fs (scoped) rather than dropping it: the paths below that RETURN to guest
        // code (int $0x41 skip, PROSPER_FAULT_SKIP) must restore it first — sigreturn does NOT restore
        // fs_base on x86-64, so resuming the guest after a host-%fs swap would run guest code (incl. its
        // %fs-relative TLS reads) on the host glibc TCB. That leaked host %fs after every int-0x41 skip
        // was the RAGE `[RAGE] Main Thr` TLS crash at eboot+0x2b1f0e5 (#1155: TLS[-0x10] read host
        // garbage 0x3d and dereferenced it). CONFIDENCE: HIGH (live probe: thread on host TCB at fault).
        const uint64_t saved_guest_fs = guest_fs_to_host_scoped();
        if (g_faultlog) {
            char b[128]; auto* uc = (ucontext_t*)uctx;
            int n = snprintf(b, sizeof b, "[fault] sig=%d addr=%p rip=0x%llx armed=%d tid=%ld\n",
                             sig, si->si_addr, (unsigned long long)PROSPER_GREGS(uc)[REG_RIP],
                             (int)(g_armed_tid && cur_tid() == g_armed_tid), cur_tid());
            raw_write_fmt(2, b, sizeof b, n);   /* raw syscall: no libc TLS access */
        }
        g_fault_addr = si->si_addr;
        auto* uc = (ucontext_t*)uctx;
        auto g = PROSPER_GREGS(uc);
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
        // Guest `int $0x41` (opcode CD 41) is RAGE's software breakpoint / assert trap (GTA V,
        // PPSA04263, #1138). On PS5 an int with a userspace-illegal vector raises #GP; the kernel's
        // trap handler, with no debugger attached, simply returns past it — so the game continues to
        // its OWN error-reporting/recovery code after the trap. prosper must emulate the same "no
        // debugger → skip the breakpoint" behavior, or the trap SIGSEGVs and kills the thread (the
        // reason GTA V needed the PROSPER_FAULT_SKIP=0x2b2c463:0x2b2c465 diagnostic crutch just to
        // render its intro). Only fires for a guest-module rip whose bytes really are CD 41; every
        // other title is unaffected (none emit int 0x41). The rip page is mapped+exec (the trap was
        // fetched from it), so a direct 2-byte read guarded by the guest band is safe in-handler.
        // Disable with PROSPER_NO_INT41_SKIP to fall back to the fatal path for debugging.
        if (g_int41_skip && sig == SIGSEGV && g_base && g_fault_rip >= g_base &&
            g_fault_rip < g_base + 0x100000000ull) {
            const auto* ins = (const uint8_t*)g_fault_rip;
            if (ins[0] == 0xCDu && ins[1] == 0x41u) {
                static std::atomic<int> logged{0};
                if (logged.fetch_add(1) < 8) {
                    char b[96];
                    int n = snprintf(b, sizeof b,
                        "[int41] guest int $0x41 (debugbreak) at eboot+0x%llx -> skipped\n",
                        (unsigned long long)(g_fault_rip - g_base));
                    raw_write_fmt(2, b, sizeof b, n);
                }
                g[REG_RIP] = g_fault_rip + 2;   // advance past the 2-byte int instruction
                guest_fs_restore_scoped(saved_guest_fs);   // resume guest on its guest %fs, not host (#1155)
                return;
            }
        }
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
            if (g_base && g_fault_rip == g_base + foff) { g[REG_RAX] = 0; g[REG_RIP] = g_base + roff;
                guest_fs_restore_scoped(saved_guest_fs); return; }   // resume guest on guest %fs (#1155)
        }
        // #2018: this fault's own registers, captured once and used by everything below. Taken HERE
        // rather than beside the global stores at the top: every path above this point either
        // returns to the guest (the int-0x41 skip, PROSPER_FAULT_SKIP, the sse4a and read-emulation
        // paths) or has already returned, and RAGE's debugbreak path in particular is hot. Nothing
        // above needs a snapshot, so it should not pay for one.
        const prosper::host::FaultContext fc =
            prosper::host::capture_fault_context(sig, cur_tid(), si->si_addr, uc);
        dump_fault_mem(fc);   // no-op unless PROSPER_FAULTMEM is set
        // Dump the HWBP ring on the recoverable (armed/main-thread) crash too — the deser fault is kind=2.
        if (g_hwbp_node_on && !g_hwbp_ring_dumped) { uint64_t sfs = guest_fs_to_host_scoped();
            hwbp_dump_ring("recover"); guest_fs_restore_scoped(sfs); }
        if (g_armed_tid && cur_tid() == g_armed_tid) siglongjmp(g_jb, 1);
        // Fault on a thread with no recovery point (a guest worker thread). Report where
        // (async-signal-safe write) then terminate cleanly instead of a cross-thread longjmp.
        {
            // #2018: the globals above are shared by every thread, and the guest's faults are
            // correlated (one heap corruption takes several worker threads within milliseconds). A
            // second thread faulting while the first is printing rewrites all of them underneath
            // it, and the first report then continues with the second thread's registers while
            // still reading as one complete fault. Keep the globals — the armed/main-thread
            // recovery path above and trap_detail()/record_fault() are single-threaded by
            // construction and use them — but drive this FATAL report from `fc`, the stack-local
            // snapshot taken above, which no other thread can touch. Every value it prints then
            // comes from one fault.
            //
            // #2018: one report at a time. Two worker threads faulting milliseconds apart used to
            // interleave their lines into a single apparent report; with the snapshot above each
            // line is now self-consistent, but two coherent reports written concurrently would still
            // be spliced together on stderr. First arrival owns the full dump; a thread that arrives
            // while one is in flight prints one clearly-labelled line for its own fault, so the
            // second fault is named rather than silently lost.
            //
            // Ownership is RELEASED when the dump finishes, not held for the life of the process.
            // Holding it would be sound only if a fatal fault always ended the process, and under
            // PROSPER_WORKER_PARK it deliberately does not: faulting threads park and the run
            // continues, so later faults arrive seconds apart with no concurrency at all and must
            // get their own full reports rather than a "someone else is reporting" line about a
            // report that finished long ago.
            static std::atomic<long> dump_owner{0};
            long expected = 0;
            if (!prosper::host::claim_fault_report(dump_owner, fc.tid, &expected)) {
                char cb2[224];
                int cn2 = snprintf(cb2, sizeof cb2,
                    "[prosper] CONCURRENT WORKER-THREAD FAULT: tid=%ld sig=%d addr=%p rip=0x%llx "
                    "(image+0x%llx) — full dump suppressed, tid=%ld is already reporting\n",
                    fc.tid, fc.sig, (void*)(uintptr_t)fc.addr, (unsigned long long)fc.rip,
                    (unsigned long long)(g_base && fc.rip >= g_base ? fc.rip - g_base : fc.rip),
                    expected);
                raw_write_fmt(2, cb2, sizeof cb2, cn2);
                // Wait for the owner to finish before ending the process — `_exit` here would
                // truncate the very report this gate exists to keep whole (measured: the header
                // line and nothing after it). BOUNDED, because the owner may itself die inside its
                // dump and a signal handler that waited unconditionally on another faulting thread
                // would hang a run that used to exit 90. `nanosleep` is async-signal-safe.
                int waited = 0;
                for (; waited < 400 &&
                       dump_owner.load(std::memory_order_acquire) == expected; waited++) {
                    struct timespec ts; ts.tv_sec = 0; ts.tv_nsec = 5 * 1000 * 1000;   // 5 ms
                    nanosleep(&ts, nullptr);
                }
                // Say so when the bound is what ended the wait. A silent timeout truncates the
                // owner's report exactly as the pre-fix `_exit` did, and a reader who is not told
                // cannot tell a complete report from a cut-off one.
                if (waited >= 400) {
                    static const char to[] =
                        "[prosper]   (report may be TRUNCATED: waited 2s for the owning thread's "
                        "dump and it had not finished)\n";
                    raw_write(2, to, sizeof to - 1);
                }
                if (getenv("PROSPER_WORKER_PARK")) { for (;;) pause(); }
                _exit(90);
            }
            char b[200];
            int n = snprintf(b, sizeof b, "[prosper] WORKER-THREAD FAULT: sig=%d addr=%p rip=0x%llx (image+0x%llx) rbp=0x%llx tid=%ld\n",
                             fc.sig, (void*)(uintptr_t)fc.addr, (unsigned long long)fc.rip,
                             (unsigned long long)(g_base && fc.rip >= g_base ? fc.rip - g_base : fc.rip),
                             (unsigned long long)fc.rbp, fc.tid);
            raw_write_fmt(2, b, sizeof b, n);   /* raw syscall: no libc TLS access */
            // Classify the fault rip + fault-addr regions (which mapping / module) and dump the instruction
            // bytes at rip — turns the ASLR-relocated "rip=0x...48b" into an identifiable location.
            classify_addr(fc.rip);
            classify_addr(fc.addr);
            auto rdb = [](uint64_t a) -> unsigned { return probe_readable(a) ? *(const uint8_t*)a : 0x100u; };
            char ib[160]; int m = snprintf(ib, sizeof ib,
                "[prosper]   insn bytes @rip: %02x %02x %02x %02x %02x %02x %02x %02x  ret@[rsp]=0x%llx\n",
                rdb(fc.rip), rdb(fc.rip+1), rdb(fc.rip+2), rdb(fc.rip+3),
                rdb(fc.rip+4), rdb(fc.rip+5), rdb(fc.rip+6), rdb(fc.rip+7),
                (unsigned long long)(probe_readable(fc.rsp) ? *(const uint64_t*)fc.rsp : 0));
            raw_write_fmt(2, ib, sizeof ib, m);
            // Guest-%fs attribution (#1155): is the faulting thread running guest code on OUR guest TCB
            // or leaked onto the host glibc TCB? The guest loads its TCB self-pointer with `mov %fs:0x0`;
            // at a fault right after, that value is still in rax (the common landing reg) and equals the
            // fs base. Our guest TCBs carry the "PROS" magic at +0x108; a mismatch means guest code ran
            // on the host TCB, so any %fs-relative TLS read returned host garbage (the #1155 fault class,
            // now fixed by restoring guest %fs on signal-handler return-to-guest paths above).
            //
            // The rax premise is a HEURISTIC and it is not always met — rax holds the TCB
            // self-pointer only when the fault lands shortly after that load, and when it holds
            // something else (a corrupt free-list head, a count) the magic check reads an unrelated
            // address and asserted `NO(host-%fs leak?)` for a thread with no %fs problem at all
            // (#2018: rax = 0x30016000). It does not need to guess for the positive answer:
            // `saved_guest_fs` above is a fact. `guest_fs_to_host_scoped()` read this thread's real
            // %fs base and returned it only after checking TCB_MAGIC there
            // (guest_tls.cpp:161-171), so non-zero IS "this thread was on our guest TCB". rax stays
            // on the line as raw evidence, without a claim attached to it.
            //
            // ZERO is NOT the mirror image of that, and the line must not print as though it were.
            // Zero has three causes and only one of them is a defect:
            //   (1) the thread really did run guest code on the host glibc TCB — the #1155 class,
            //       and the only reading worth alarming about;
            //   (2) guest %fs is not enabled for this process. On Linux/Windows it is enabled by
            //       DEFAULT — `g_enabled = getenv("PROSPER_NO_GUEST_FS") == nullptr`
            //       (guest_tls.cpp:58), an opt-OUT kept for compatibility bisection — so this arm
            //       means either that opt-out is set, or the fault landed before
            //       guest_tls_set_templates() ran (boot_program.cpp:252) and g_configured is still
            //       false. Both are RARE on an ordinary boot, so do NOT reach for this cause first.
            //       (guest_tls.cpp:46's PROSPER_GUEST_FS opt-IN is the macOS/Rosetta arm only, and
            //       this whole three-state line is compiled out there — see the Darwin note below.)
            //   (3) the faulting thread is one of prosper's OWN host threads, which never had a
            //       guest TCB to leak off. This report is reached by any non-armed thread, host or
            //       guest, so that is not a rare case.
            // `guest_tls_enabled()` (guest_tls.cpp:79) separates (2) from (1)/(3) as fact; nothing
            // available in a signal handler separates (1) from (3), so the line says so rather than
            // asserting a leak. It is two static bool loads: no allocation, no lock, handler-safe.
            // Practical consequence on Linux, which is where these reports are read: because guest
            // %fs is on by default, `n/a` is the rare arm and a zero here is almost always (1) or
            // (3) — the two the handler cannot tell apart. That is the honest state of knowledge,
            // and it is why the string hedges instead of naming a leak.
            //
            // Darwin: `guest_fs_to_host_scoped()` returns 0 unconditionally there — the
            // `#ifdef __APPLE__` early-out inside the shared POSIX definition (guest_tls.cpp:162-164),
            // NOT the separate Windows stub at guest_tls.cpp:288 — so zero carries no information on
            // that platform at all, and the Darwin arm below prints neither answer.
            {
                char nm[16] = {0}; pthread_getname_np(pthread_self(), nm, sizeof nm);   // portable (Linux+macOS)
#ifdef __APPLE__
                const char* tcb = "unknown (not determined on this platform)";
#else
                // Async-signal-safe: guest_tls_enabled() is two static bool loads, no lock and no
                // allocation, and this TU already calls it unqualified.
                const char* tcb = !guest_tls_enabled() ? "n/a (guest %fs not enabled)"
                                : saved_guest_fs       ? "yes"
                                : "no (host TCB — a leak only if this thread should have had a guest TCB)";
#endif
                char tb[224];
                int t = snprintf(tb, sizeof tb,
                                 "[fault] thread='%s' guest-fs=0x%llx rax=0x%llx on-guest-TCB=%s\n",
                                 nm, (unsigned long long)saved_guest_fs,
                                 (unsigned long long)fc.rax, tcb);
                raw_write_fmt(2, tb, sizeof tb, t);
            }
            // PROSPER_FAULTOBJ (#1226): dump the leading 0xC0 bytes of the objects in rax/r15/rbx via a
            // fault-safe self-read (process_vm_readv, EFAULT-safe on unmapped guest VAs). For ArcRunner's
            // RHIThread null-deref (r14=[rax+0x98]==0) this reveals the object type (leading vtable/tag)
            // and the null field at +0x98, so the missing RHI dependency is identifiable offline. Gated
            // so other titles' worker faults stay quiet; snprintf+SYS_write matches this handler's style.
            // Linux-only: the fault-safe self-read below uses SYS_process_vm_readv (no macOS syscall
            // number); this is a dev diagnostic and the exercised worker-fault RE runs on Linux.
#ifndef __APPLE__
            if (getenv("PROSPER_FAULTOBJ")) {
                // #1945: rdi/r12/r13/r14 as well as the original rax/r15/rbx. The corrupted object
                // is whichever register the faulting instruction dereferenced, and on PPSA21406's
                // eboot+0x117e21d that is rdi (the forged pointer) and r12 (the pointer array it
                // was loaded from) — neither of which the original three-register list dumped.
                const struct { const char* nm; uint64_t addr; } objs[] = {
                    {"rax", fc.rax}, {"r15", fc.r15}, {"rbx", fc.rbx}, {"rdi", fc.rdi},
                    {"r12", fc.r12}, {"r13", fc.r13}, {"r14", fc.r14},
                };
                for (const auto& o : objs) {
                    uint64_t buf[24];
                    struct iovec liov; liov.iov_base = buf; liov.iov_len = sizeof buf;
                    struct iovec riov; riov.iov_base = (void*)(uintptr_t)o.addr; riov.iov_len = sizeof buf;
                    ssize_t got = syscall(SYS_process_vm_readv, getpid(), &liov, 1UL, &riov, 1UL, 0UL);
                    char ob[700];
                    int on = raw_fmt_advance(0, snprintf(ob, sizeof ob, "[faultobj] %s=0x%llx got=%zd:",
                                      o.nm, (unsigned long long)o.addr, (ssize_t)got), sizeof ob);
                    for (int w = 0; got > 0 && w < (int)((size_t)got / 8) && on < (int)sizeof ob - 24; w++)
                        on = raw_fmt_advance(on, snprintf(ob + on, sizeof ob - (size_t)on, " +%02x=%016llx",
                                       w * 8, (unsigned long long)buf[w]), sizeof ob);
                    on = raw_fmt_advance(on, snprintf(ob + on, sizeof ob - (size_t)on, "\n"), sizeof ob);
                    raw_write_fmt(2, ob, sizeof ob, on);
                }
                // Corruption-source probe: dump rcx/rdx and scan the GPU-write ring around every
                // guest-pointer-shaped register. If a prosper ReleaseMem/fence write landed on a
                // guest allocator header or a live guest pointer array, it shows here (DOLL
                // #232/#241 stomp class).
                //
                // #1945: the probe set used to be exactly { rdx, rax } — chosen for DOLL's
                // `mov [rdx]`-style free-list pop. That silently answers nothing whenever the
                // corrupted pointer lives in a different register. Measured on PPSA21406 at
                // eboot+0x117e21d (`mov (%r12,%rcx,8),%rdi` / `mov 0x40(%rdi),%rcx` /
                // `vucomisd (%rcx),%xmm0`): the forged pointer is rdi = 0x2100000001 and the array
                // base is r12, while rax = 0 and rdx = 1 — both below 0x1000, so BOTH probes hit
                // the `continue` and the scans printed not one line. A ring scan that skipped the
                // only page worth reading is indistinguishable from a ring that held no hit, i.e.
                // void, not negative. Probe the fault address and every register whose value could
                // be a guest address; de-duplicate pages and cap the list so a 16-register dump
                // cannot flood the terminal fault report.
                uint64_t probe_pages[12];
                unsigned probe_page_n = 0, probe_page_considered = 0;
                {
                    const uint64_t cand[] = {
                        (uint64_t)(uintptr_t)fc.addr, fc.rax, fc.rbx, fc.rcx, fc.rdx, fc.rsi, fc.rdi,
                        fc.r8, fc.r9, fc.r10, fc.r11, fc.r12, fc.r13, fc.r14, fc.r15,
                    };
                    for (uint64_t v : cand) {
                        // Skip small integers (a count/flag, and the corrupt `0x1` head itself has
                        // no page worth scanning) and host addresses (stack/libc live at 0x7f…).
                        if (v < 0x10000 || v >= 0x7f0000000000ull) continue;
                        const uint64_t pg = v & ~0xfffull;
                        bool dup = false;
                        for (unsigned i = 0; i < probe_page_n; i++) dup = dup || probe_pages[i] == pg;
                        if (dup) continue;
                        // Count every distinct eligible page, then keep what fits. Reporting only
                        // the kept count would make a truncated probe read as an exhaustive one —
                        // the very failure this widening exists to remove — so print kept/considered.
                        probe_page_considered++;
                        if (probe_page_n < (unsigned)(sizeof probe_pages / sizeof probe_pages[0]))
                            probe_pages[probe_page_n++] = pg;
                    }
                }
                {
                    char cb[176];
                    int cn = snprintf(cb, sizeof cb,
                                      "[faultobj] rcx=0x%llx rdx=0x%llx probe-pages=%u/%u%s\n",
                                      (unsigned long long)fc.rcx, (unsigned long long)fc.rdx,
                                      probe_page_n, probe_page_considered,
                                      probe_page_n < probe_page_considered ? " (CAPPED)" : "");
                    raw_write_fmt(2, cb, sizeof cb, cn);
                    if (prosper_gpu_write_ring_scan) {
                        static char ring[8192];
                        for (unsigned i = 0; i < probe_page_n; i++) {
                            const uint64_t pg = probe_pages[i];
                            int rn = prosper_gpu_write_ring_scan(pg, pg + 0x1000, ring, sizeof ring);
                            if (rn > 0) {
                                char hb[96]; int hn = snprintf(hb, sizeof hb,
                                    "[faultobj] GPU-write-ring hits in page 0x%llx:\n",
                                    (unsigned long long)pg);
                                raw_write_fmt(2, hb, sizeof hb, hn);
                                raw_write(2, ring, strlen(ring));
                            }
                        }
                    }
                    // #1226: cross-reference the same pages against the PERSISTENT clock-fence
                    // table (RELEASE_MEM data_sel==3 / EVENT_WRITE timestamps). The ring wraps in
                    // well under a second on a busy title, so "no ring hit" says nothing about a
                    // stomp seconds earlier; this table retains the last clock write per address
                    // for the whole run. A zero-hit result is itself evidence (the corrupted field
                    // was NEVER a clock-fence target), so report both outcomes.
                    if (prosper_gpu_clockfence_scan) {
                        static char cfb[8192];
                        for (unsigned i = 0; i < probe_page_n; i++) {
                            const uint64_t pg = probe_pages[i];
                            int fn2 = prosper_gpu_clockfence_scan(pg, pg + 0x1000, cfb, sizeof cfb);
                            char hb[112]; int hn = snprintf(hb, sizeof hb,
                                "[faultobj] clock-fence targets in page 0x%llx: %d\n",
                                (unsigned long long)pg, fn2);
                            raw_write_fmt(2, hb, sizeof hb, hn);
                            if (fn2 > 0) raw_write(2, cfb, strlen(cfb));
                        }
                    }
                    // #1226 reverse lookup: the corrupted value's shape is {orig_low32, clock_low32}
                    // (an 8-byte clock write at A read back at A-4, then guest-copied into the bin
                    // head). Ask the table which fence target ever wrote a clock with that low32 —
                    // probing the high dword of rax and of the qword at [rdx] (fault-safe read).
                    if (prosper_gpu_clockfence_find_low32) {
                        static char fb[8192];
                        uint64_t rdx_val = 0;
                        {
                            struct iovec liov; liov.iov_base = &rdx_val; liov.iov_len = 8;
                            struct iovec riov; riov.iov_base = (void*)(uintptr_t)fc.rdx; riov.iov_len = 8;
                            if (syscall(SYS_process_vm_readv, getpid(), &liov, 1UL, &riov, 1UL, 0UL) != 8)
                                rdx_val = 0;
                        }
                        const uint64_t vals[2] = { fc.rax, rdx_val };
                        for (uint64_t v : vals) {
                            uint32_t hi32 = (uint32_t)(v >> 32);
                            // Shape gate (run-10 lesson): a plain guest ADDRESS has a tiny high
                            // dword (0x21..0x41), and probing it floods NEAR noise near every
                            // 2^32 clock wrap. Only clock-plausible high dwords are worth asking
                            // about (>= ~16.7M ns == 16.7ms into a wrap).
                            if (hi32 < 0x01000000u || hi32 == 0xffffffffu) continue;
                            int fn3 = prosper_gpu_clockfence_find_low32(hi32, fb, sizeof fb);
                            char hb[128]; int hn = snprintf(hb, sizeof hb,
                                "[faultobj] clock-fence low32 matches for value 0x%llx (hi32=0x%x): %d\n",
                                (unsigned long long)v, hi32, fn3);
                            raw_write_fmt(2, hb, sizeof hb, hn);
                            if (fn3 > 0) raw_write(2, fb, strlen(fb));
                        }
                    }
                    // #1226 POOLSHIFT probe. Both residual fault classes dereference a value whose
                    // <<8 lands in the allocator-metadata region (0x30015f00 -> pool table
                    // 0x30015f0000 on current dmem layouts, both UE4 titles). Two questions a
                    // fault can answer directly:
                    // (1) mechanism discriminator — a BYTE dump around the source slot [rdx]: if
                    //     the UNSHIFTED pointer sits at rdx-1 (bytes align one low), the pointer
                    //     was STORED at an odd address; if the slot holds exactly the shifted
                    //     value with clean zero high bytes, it was COPIED/computed already-shifted.
                    // (2) writer cross-reference — scan the GPU-write ring and clock-fence table
                    //     around (v<<8), the REAL pool table, not just the corrupted slot.
                    {
                        uint64_t rdx_val2 = 0;
                        struct iovec l2; l2.iov_base = &rdx_val2; l2.iov_len = 8;
                        struct iovec r2; r2.iov_base = (void*)(uintptr_t)fc.rdx; r2.iov_len = 8;
                        if (syscall(SYS_process_vm_readv, getpid(), &l2, 1UL, &r2, 1UL, 0UL) != 8)
                            rdx_val2 = 0;
                        // Byte dump around the free-list SLOT — the mechanism discriminator. The
                        // two observed pop compilations keep the slot pointer in rdx (ArcRunner,
                        // image+0x127e751) or r8 (DOLL, image+0x2316acf); dump around both when
                        // guest-pointer-shaped. Mixed 4-byte compressed entries vs full 8-byte
                        // heads in the NEIGHBORING bins are visible directly in the byte stream.
                        // #1252 review: print the ABSOLUTE base address of the dump, never a
                        // relative label — byte-exact forensics must not require the reader to
                        // reconstruct the base from call-site arithmetic.
                        auto byte_dump = [&](const char* nm, uint64_t center) {
                            if (center < 0x10000) return;
                            uint8_t bytes[0x60];
                            struct iovec lb; lb.iov_base = bytes; lb.iov_len = sizeof bytes;
                            struct iovec rb; rb.iov_base = (void*)(uintptr_t)(center - 0x20); rb.iov_len = sizeof bytes;
                            ssize_t got = syscall(SYS_process_vm_readv, getpid(), &lb, 1UL, &rb, 1UL, 0UL);
                            if (got <= 0) return;
                            char bb[640];
                            int bn = raw_fmt_advance(0, snprintf(bb, sizeof bb, "[faultobj] bytes (%s) @0x%llx:", nm,
                                              (unsigned long long)(center - 0x20)), sizeof bb);
                            for (ssize_t k = 0; k < got && bn < (int)sizeof bb - 4; k++)
                                bn = raw_fmt_advance(bn, snprintf(bb + bn, sizeof bb - (size_t)bn, "%s%02x",
                                               (k % 8 == 0) ? " " : "", bytes[k]), sizeof bb);
                            bn = raw_fmt_advance(bn, snprintf(bb + bn, sizeof bb - (size_t)bn, "\n"), sizeof bb);
                            raw_write_fmt(2, bb, sizeof bb, bn);
                        };
                        byte_dump("rdx", fc.rdx);
                        byte_dump("r8", fc.r8);
                        // #1226: was the faulting pool array even visible to the window probe?
                        if (prosper_mb3_is_pool_candidate) {
                            for (uint64_t pb : { fc.rdx & ~0xffffull, fc.r8 & ~0xffffull }) {
                                if (pb < 0x10000) continue;
                                char cb2[96]; int cn2 = snprintf(cb2, sizeof cb2,
                                    "[faultobj] pool 0x%llx candidate=%d\n",
                                    (unsigned long long)pb, prosper_mb3_is_pool_candidate(pb));
                                raw_write_fmt(2, cb2, sizeof cb2, cn2);
                            }
                        }
                        // #1226: cross-reference the slot pages against APR write destinations —
                        // the bulk guest-writer outside every GPU provenance ring (#88 precedent).
                        if (prosper_apr_dest_scan) {
                            static char ab[4096];
                            const uint64_t aprobes[2] = { fc.rdx, fc.r8 };
                            for (uint64_t probe : aprobes) {
                                uint64_t pg = probe & ~0xfffull;
                                if (pg < 0x1000) continue;
                                int an = prosper_apr_dest_scan(pg, pg + 0x1000, ab, sizeof ab);
                                char h4[112]; int h4n = snprintf(h4, sizeof h4,
                                    "[faultobj] APR dests overlapping page 0x%llx: %d\n",
                                    (unsigned long long)pg, an);
                                raw_write_fmt(2, h4, sizeof h4, h4n);
                                raw_write(2, ab, strlen(ab));   // unconditional: the stores/evictions header IS the zero-case confidence signal
                            }
                        }
                        const uint64_t cands[3] = { fc.rax, fc.rcx, rdx_val2 };
                        for (uint64_t v : cands) {
                            if (v >> 32 || v < 0x100000) continue;           // must look byte-shifted
                            uint64_t real = v << 8;
                            if (real < 0x2000000000ull || real >= 0x4000000000ull) continue;
                            char pb[128]; int pn = snprintf(pb, sizeof pb,
                                "[faultobj] POOLSHIFT candidate 0x%llx -> real ptr 0x%llx\n",
                                (unsigned long long)v, (unsigned long long)real);
                            raw_write_fmt(2, pb, sizeof pb, pn);
                            // What actually lives at the reconstructed pointer? A live free block
                            // (next-pointer residue), a pool-info record, or unrelated data —
                            // discriminates "compressed pointer misread" from coincidence.
                            byte_dump("real ptr", real + 0x20);
                            // Was the reconstructed buffer itself ever an APR destination?
                            if (prosper_apr_dest_scan) {
                                static char ar[4096];
                                uint64_t rp = real & ~0xfffull;
                                int arn = prosper_apr_dest_scan(rp, rp + 0x1000, ar, sizeof ar);
                                char h5[112]; int h5n = snprintf(h5, sizeof h5,
                                    "[faultobj] APR dests overlapping real-ptr page 0x%llx: %d\n",
                                    (unsigned long long)rp, arn);
                                raw_write_fmt(2, h5, sizeof h5, h5n);
                                raw_write(2, ar, strlen(ar));   // unconditional (see above)
                            }
                            uint64_t pg = real & ~0xfffull;
                            if (prosper_gpu_write_ring_scan) {
                                static char rs[4096];
                                int rn2 = prosper_gpu_write_ring_scan(pg, pg + 0x1000, rs, sizeof rs);
                                char h2[112]; int h2n = snprintf(h2, sizeof h2,
                                    "[faultobj] GPU-write-ring hits at real-ptr page 0x%llx: %d\n",
                                    (unsigned long long)pg, rn2);
                                raw_write_fmt(2, h2, sizeof h2, h2n);
                                if (rn2 > 0) raw_write(2, rs, strlen(rs));
                            }
                            if (prosper_gpu_clockfence_scan) {
                                static char cs[4096];
                                int cn2 = prosper_gpu_clockfence_scan(pg, pg + 0x1000, cs, sizeof cs);
                                char h3[112]; int h3n = snprintf(h3, sizeof h3,
                                    "[faultobj] clock-fence targets at real-ptr page 0x%llx: %d\n",
                                    (unsigned long long)pg, cn2);
                                raw_write_fmt(2, h3, sizeof h3, h3n);
                                if (cn2 > 0) raw_write(2, cs, strlen(cs));
                            }
                        }
                    }
                }
            }
#endif  // !__APPLE__ (PROSPER_FAULTOBJ)
            // #694: guest control-flow backtrace at the worker fault. When the fault is an intermittent
            // bad-pointer jump (e.g. the Blasphemous 2 asset-load worker reaching non-exec Rosetta memory
            // with ret@[rsp]=0x0), rip is uninformative garbage; the only way to attribute it to a guest
            // call site is the return-address chain. Two async-signal-safe passes (probe_readable + raw
            // SYS_write, no malloc): a frame-pointer chain walk from rbp, and a bounded stack scan flagging
            // qwords that fall in the main guest module band [g_base, g_base+4 GiB) (rsp may be a host
            // pthread stack, so non-guest words are filtered out).
            {
                char lb[160]; int ln;
#ifndef __APPLE__
                // The fault can occur before a pointer-dump mode has initialized the pipe-based probe,
                // and worker frames may live on either a host pthread stack or a guest mapping.
                // process_vm_readv is EFAULT-safe across both kinds of mapping; use it here so the most
                // important word -- the return address above a real-PRX call -- is not silently lost.
                auto safe_qword = [](uint64_t addr, uint64_t* value) -> bool {
                    struct iovec local = { value, sizeof *value };
                    struct iovec remote = { (void*)(uintptr_t)addr, sizeof *value };
                    return syscall(SYS_process_vm_readv, getpid(), &local, 1UL, &remote, 1UL, 0UL) ==
                           (ssize_t)sizeof *value;
                };
#else
                auto safe_qword = [](uint64_t addr, uint64_t* value) -> bool {
                    if (!probe_readable(addr)) return false;
                    *value = *(const uint64_t*)(uintptr_t)addr;
                    return true;
                };
#endif
                ln = snprintf(lb, sizeof lb, "[prosper]   guest fp-chain (rbp=0x%llx):\n", (unsigned long long)fc.rbp);
                raw_write_fmt(2, lb, sizeof lb, ln);
                uint64_t fp = fc.rbp;
                for (int depth = 0; depth < 16; depth++) {
                    uint64_t ra = 0, nfp = 0;
                    if (!safe_qword(fp, &nfp) || !safe_qword(fp + 8, &ra)) break;
                    if (g_base && ra >= g_base && ra < g_base + 0x100000000ull)
                        ln = snprintf(lb, sizeof lb, "[prosper]     #%d ra=0x%llx (eboot+0x%llx)\n",
                                      depth, (unsigned long long)ra, (unsigned long long)(ra - g_base));
                    else
                        ln = snprintf(lb, sizeof lb, "[prosper]     #%d ra=0x%llx\n", depth, (unsigned long long)ra);
                    raw_write_fmt(2, lb, sizeof lb, ln);
                    if (nfp <= fp || nfp - fp > 0x200000ull) break;   // non-increasing / implausible frame: stop
                    fp = nfp;
                }
                ln = snprintf(lb, sizeof lb, "[prosper]   guest ret-addr scan (rsp=0x%llx, 64 qwords):\n",
                              (unsigned long long)fc.rsp);
                raw_write_fmt(2, lb, sizeof lb, ln);
                for (int i = 0, shown = 0; i < 64 && shown < 16; i++) {
                    uint64_t sa = fc.rsp + (uint64_t)i * 8;
                    uint64_t q = 0;
                    if (!safe_qword(sa, &q)) continue;
                    if (g_base && q >= g_base && q < g_base + 0x100000000ull) {
                        ln = snprintf(lb, sizeof lb, "[prosper]     rsp+0x%03x = 0x%llx (eboot+0x%llx)\n",
                                      i * 8, (unsigned long long)q, (unsigned long long)(q - g_base));
                        raw_write_fmt(2, lb, sizeof lb, ln);
                        shown++;
                    }
                }
                // Register-band scan: when the stack is empty (a thread that faulted at/near its entry,
                // e.g. #694's asset-load worker: rbp unreadable, rsp at the guard page), the only guest
                // code lead is whichever GPR still holds a guest-module pointer. Flag them eboot-relative.
                const struct { const char* nm; uint64_t v; } gregs[] = {
                    {"rax",fc.rax},{"rbx",fc.rbx},{"rcx",fc.rcx},{"rdx",fc.rdx},{"rsi",fc.rsi},{"rdi",fc.rdi},
                    {"r8",fc.r8},{"r9",fc.r9},{"r10",fc.r10},{"r11",fc.r11},{"r12",fc.r12},{"r13",fc.r13},
                    {"r14",fc.r14},{"r15",fc.r15},{"rbp",fc.rbp},
                };
                for (const auto& r : gregs) {
                    if (g_base && r.v >= g_base && r.v < g_base + 0x100000000ull) {
                        ln = snprintf(lb, sizeof lb, "[prosper]     %-3s = 0x%llx (eboot+0x%llx)\n",
                                      r.nm, (unsigned long long)r.v, (unsigned long long)(r.v - g_base));
                        raw_write_fmt(2, lb, sizeof lb, ln);
                    }
                }
            }
            // #312: full register + heap-window + GPU-write-ring dump at the worker fault too (the
            // MallocBinned3 freelist-pop faults — e.g. rcx=0x20015f00 at eboot+0x2316acf — land here,
            // not on the nullpage path). Reuses the async-signal-safe nullpage attribution dump.
            // fc.rip, not g_fault_rip (#2018): this call pairs THIS fault's ucontext registers with a
            // rip, and reading the global there produced the report's most misleading line — an
            // `insn @rip` decoding to an instruction that could not have faulted at the printed
            // address, because the bytes came from another thread's fault.
            nullpage_deep_dump(uc, fc.rip);
            // If PROSPER_HWBP_NODE ring-capture is active, the crash node's typetree metadata is the tail.
            if (g_hwbp_node_on) hwbp_dump_ring("worker-fault");
            // The report is complete (#2018): a concurrently-faulting thread waiting above may now
            // end the process, and under PROSPER_WORKER_PARK a LATER fault can take the gate and
            // get a full report of its own.
            prosper::host::release_fault_report(dump_owner, fc.tid);
            // These two are INSIDE the block on purpose (#2163). fault_context.hpp explains that
            // release_fault_report is safe without a nesting count only because "the fatal report
            // block never returns -- its every exit is _exit or an unbounded park", and that property
            // used to be enforced by these statements sitting AFTER the block, which control flow
            // fell off the end of. Anyone adding an early `return` above broke the invariant with no
            // failing check and no test; the consequence is the report gate released while a report
            // is still printing, i.e. two re-entrant reports interleaved on one stderr.
            //
            // Inside the brace, the block demonstrably has no fall-off-the-end exit, so the comment
            // in fault_context.hpp is enforced by structure rather than by everyone remembering to
            // read it. Moving them is behaviour-preserving because the block at :2311 is an
            // UNCONDITIONAL bare block, not an `if` -- "inside, at the end" and "immediately after"
            // run in exactly the same cases.
            //
            // PROSPER_WORKER_PARK=1 (diagnostic): instead of terminating the whole process on a guest
            // worker-thread fault, park the faulting thread so the main thread can proceed to its own
            // recoverable crash (lets CRASHPEEK/PEEK_CLASS run). May deadlock if the worker held a lock.
            if (getenv("PROSPER_WORKER_PARK")) { for (;;) pause(); }
            _exit(90);
        }
    }

    std::string trap_detail() {
        char buf[256];
        const char* sn = g_trap_sig == SIGILL ? "SIGILL"
                       : g_trap_sig == SIGBUS ? "SIGBUS"
                       : g_trap_sig == SIGFPE ? "SIGFPE" : "SIGSEGV";
        uint64_t off = (g_base && g_fault_rip >= g_base) ? g_fault_rip - g_base : 0;
        snprintf(buf, sizeof buf, "%s at addr=%p  rip=0x%llx (image+0x%llx)",
                 sn, g_fault_addr, (unsigned long long)g_fault_rip, (unsigned long long)off);
        return buf;
    }

    uint64_t page_up(uint64_t v) { return (v + 0xfff) & ~((uint64_t)0xfff); }

    // Return-hook imports jump through one shared trampoline so the fixed 96-byte slot does not
    // duplicate argument forwarding, optional guest-%fs switching, result preservation and the
    // checkpoint call. r10=handler, r11=hook. Seven qwords make both paths share one handler-frame
    // contract: the original guest return is at handler-entry RSP+0x40.
    extern "C" __attribute__((naked)) void prosper_hle_hook_host_trampoline() {
        __asm__ volatile(
            "pushq %rax\n"                         // saved dummy / alignment
            "pushq %r11\n"                         // return hook
            "pushq %rax\n"                         // result spill / alignment
            "pushq 0x38(%rsp)\n"                   // original arg10
            "pushq 0x38(%rsp)\n"                   // original arg9
            "pushq 0x38(%rsp)\n"                   // original arg8
            "pushq 0x38(%rsp)\n"                   // original arg7
            "callq *%r10\n"
            "movq %rax, 0x20(%rsp)\n"
            "movq 0x28(%rsp), %r10\n"
            "callq *%r10\n"                        // true post-handler return checkpoint
            "movq 0x20(%rsp), %rax\n"
            "addq $0x38, %rsp\n"
            "retq\n");
    }
#ifdef __linux__
    extern "C" __attribute__((naked)) void prosper_hle_hook_swap_trampoline() {
        __asm__ volatile(
            "rdfsbase %rax\n"
            "cmpl $0x50524f53, 0x108(%rax)\n"
            "jne 1f\n"
            // Guest-FS path: save the guest base as the outer slot, switch to the host TCB, then
            // checkpoint while host TLS is still active and restore guest FS immediately before ret.
            "pushq %rax\n"
            "pushq %r11\n"
            "pushq %rax\n"
            "pushq 0x38(%rsp)\n"
            "pushq 0x38(%rsp)\n"
            "pushq 0x38(%rsp)\n"
            "pushq 0x38(%rsp)\n"
            "movq 0x100(%rax), %rax\n"
            "wrfsbase %rax\n"
            "callq *%r10\n"
            "movq %rax, 0x20(%rsp)\n"
            "movq 0x28(%rsp), %r10\n"
            "callq *%r10\n"
            "movq 0x20(%rsp), %rax\n"
            "movq 0x30(%rsp), %r11\n"
            "addq $0x38, %rsp\n"
            "wrfsbase %r11\n"
            "retq\n"
            // Host-FS path: same seven-qword frame, without an FS transition.
            "1:\n"
            "pushq %rax\n"
            "pushq %r11\n"
            "pushq %rax\n"
            "pushq 0x38(%rsp)\n"
            "pushq 0x38(%rsp)\n"
            "pushq 0x38(%rsp)\n"
            "pushq 0x38(%rsp)\n"
            "callq *%r10\n"
            "movq %rax, 0x20(%rsp)\n"
            "movq 0x28(%rsp), %r10\n"
            "callq *%r10\n"
            "movq 0x20(%rsp), %rax\n"
            "addq $0x38, %rsp\n"
            "retq\n");
    }
#endif

    // Emit machine code into a stub slot.
    size_t emit_impl(uint8_t* p, uint64_t fn) {        // movabs rax,fn ; jmp rax
        p[0] = 0x48; p[1] = 0xB8; memcpy(p + 2, &fn, 8); p[10] = 0xFF; p[11] = 0xE0;
        return 12;
    }
    size_t emit_unimpl(uint8_t* p, uint32_t idx, uint64_t fn) { // mov edi,idx ; movabs rax,fn ; jmp rax
        p[0] = 0xBF; memcpy(p + 1, &idx, 4);
        p[5] = 0x48; p[6] = 0xB8; memcpy(p + 7, &fn, 8); p[15] = 0xFF; p[16] = 0xE0;
        return 17;
    }
    size_t emit_impl_hook(uint8_t* p, uint64_t fn, uint64_t hook, bool swap) {
        p[0] = 0x49; p[1] = 0xBA; memcpy(p + 2, &fn, 8);       // movabs r10,handler
        p[10] = 0x49; p[11] = 0xBB; memcpy(p + 12, &hook, 8); // movabs r11,hook
        uint64_t trampoline = (uint64_t)(uintptr_t)&prosper_hle_hook_host_trampoline;
#ifdef __linux__
        if (swap) trampoline = (uint64_t)(uintptr_t)&prosper_hle_hook_swap_trampoline;
#else
        (void)swap;
#endif
        p[20] = 0x48; p[21] = 0xB8; memcpy(p + 22, &trampoline, 8); // movabs rax,trampoline
        p[30] = 0xFF; p[31] = 0xE0;                               // jmp rax
        return 32;
    }
    // --- Guest-%fs swap stubs (only when PROSPER_GUEST_FS is enabled). Guest code runs with %fs = guest
    // TP; HLE handlers need the host %fs (host libc TLS). The stub is ROBUST to the calling thread's fs:
    // it checks a magic at [fs+0x108] that marks OUR guest TCB (guest_tls.cpp). If present (guest thread),
    // it swaps %fs to the stashed host TCB [fs+0x100] for the handler call, then restores the guest %fs.
    // If absent (a host-context thread, or the main thread during pre-entry init), it just tail-calls the
    // handler on the current fs — exactly the old behavior. Uses FSGSBASE. Clobbers only
    // caller-saved rax/r10/r11 (never the argument registers rdi..r9).
    //
    // STACK-ARG FORWARDING: the guest path interposes a call between the guest caller and handler.
    // Re-push args 10,9,8,7 so fixed-arity handlers receive the normal SysV layout ([rsp+8]=arg7
    // through [rsp+32]=arg10). Saved r11 remains behind those arguments. Alignment: entry rsp is
    // 8 mod 16; five qwords of storage make the call site 0 mod 16 and the handler entry 8 mod 16.
    // Keep offsets/magic in sync with guest_tls.cpp.
    size_t emit_swap_stub(uint8_t* p, uint32_t idx, uint64_t fn, bool unimpl) {
        uint8_t* s = p;
        // Keep the target in caller-saved r10 across the FS probe/swap. Loading it once avoids
        // duplicating a movabs in both branches and leaves room for the fourth guest stack arg.
        if (unimpl) { *p++ = 0xBF; memcpy(p, &idx, 4); p += 4; }             // mov edi, idx
        *p++ = 0x49; *p++ = 0xBA; memcpy(p, &fn, 8); p += 8;                // movabs r10, fn
        // rdfsbase r11 ; cmp dword [r11+0x108], MAGIC ; jne .host
        *p++=0xF3; *p++=0x49; *p++=0x0F; *p++=0xAE; *p++=0xC3;
        *p++=0x41; *p++=0x81; *p++=0xBB; uint32_t mo=0x108; memcpy(p,&mo,4); p+=4;
        uint32_t magic=0x50524F53u; memcpy(p,&magic,4); p+=4;
        *p++=0x75; uint8_t* jne_rel = p++;                                  // jne rel8 (patched below)
        // .guest: save r11, then re-push original args 10/9/8/7. Each source remains at rsp+0x28
        // as the stack moves. Five qwords put the handler call site at the required alignment.
        *p++=0x41; *p++=0x53;                                               // push r11
        *p++=0xFF; *p++=0x74; *p++=0x24; *p++=0x28;                         // push original arg10
        *p++=0xFF; *p++=0x74; *p++=0x24; *p++=0x28;                         // push original arg9
        *p++=0xFF; *p++=0x74; *p++=0x24; *p++=0x28;                         // push original arg8
        *p++=0xFF; *p++=0x74; *p++=0x24; *p++=0x28;                         // push original arg7
        *p++=0x49; *p++=0x8B; *p++=0x83; uint32_t ho=0x100; memcpy(p,&ho,4); p+=4;
        *p++=0xF3; *p++=0x48; *p++=0x0F; *p++=0xAE; *p++=0xD0;              // wrfsbase host FS
        *p++=0x41; *p++=0xFF; *p++=0xD2;                                    // call r10
        *p++=0x48; *p++=0x83; *p++=0xC4; *p++=0x20;                         // discard arg copies
        *p++=0x41; *p++=0x5B;                                               // pop r11
        *p++=0xF3; *p++=0x49; *p++=0x0F; *p++=0xAE; *p++=0xD3;              // restore guest FS
        *p++=0xC3;                                                          // ret
        // .host: no FS swap is needed; tail-call the already loaded target.
        *jne_rel = (uint8_t)(p - (jne_rel + 1));
        *p++=0x41; *p++=0xFF; *p++=0xE2;                                    // jmp r10
        return static_cast<size_t>(p - s);
    }
    size_t emit_impl_swap(uint8_t* p, uint64_t fn)                 { return emit_swap_stub(p, 0,   fn, false); }
    size_t emit_unimpl_swap(uint8_t* p, uint32_t idx, uint64_t fn) { return emit_swap_stub(p, idx, fn, true); }
}

bool map_image(const LoadedImage& img, std::string* err) {
    auto fail = [&](const char* s){ if (err) *err = s; return false; };
    void* want = (void*)(img.base + img.min_vaddr);
    size_t sz  = img.mem.size();
    // RWX for bring-up; per-segment W^X is a later refinement (shared LOAD pages).
    void* got = prosper_mmap_noreplace(want, sz, PROT_READ | PROT_WRITE | PROT_EXEC,
                                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (got == MAP_FAILED || got != want) return fail("mmap image at guest base failed");
    memcpy(got, img.mem.data(), sz);
    if (!g_base) g_base = img.base;   // main image, for fault-offset reporting
    return true;
}

// The one place a slot's stub bytes are chosen. install_stubs and append_stubs (#639) must emit
// byte-identical stubs for the same slot, or a runtime-loaded module's imports would take a
// different path into the HLE than the pre-linked ones.
static size_t emit_one_stub(uint8_t* slot, const ImportSlot& s, uint32_t idx, bool swap) {
    HleFn fn = Hle::lookup(s.nid);
    HleReturnHook return_hook = Hle::return_hook_of(s.nid);
    if (fn) {
        if (return_hook) return emit_impl_hook(slot, (uint64_t)fn, (uint64_t)return_hook, swap);
        return swap ? emit_impl_swap(slot, (uint64_t)fn) : emit_impl(slot, (uint64_t)fn);
    }
    return swap ? emit_unimpl_swap(slot, idx, (uint64_t)&prosper_on_unimpl)
                : emit_unimpl(slot, idx, (uint64_t)&prosper_on_unimpl);
}

// Whether the emitted stubs must swap %fs (see install_stubs' original comment).
static bool stub_swap_mode() {
#ifdef __APPLE__
    // macOS trap mode NEVER swaps %fs: the CPU fs base stays 0 and HLE handlers run on the host's own
    // (%gs) TLS, so the plain tail-jump stub is correct. (The swap stub uses rdfsbase/wrfsbase, which
    // SIGILL on Rosetta — emitting it here made every guest import call trap.)
    return false;
#else
    return guest_tls_enabled();   // gated: emit the %fs swap stubs so HLE handlers run on the host TCB
#endif
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
    if (region > kStubApertureBytes) return fail("import stub table exceeds the stub aperture");
    void* want = (void*)stub_base;
    void* got = prosper_mmap_noreplace(want, region, PROT_READ | PROT_WRITE | PROT_EXEC,
                                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (got == MAP_FAILED || got != want) return fail("mmap stub region failed");

    const bool swap = stub_swap_mode();
    if (swap && stub_size < 96) return fail("stub_size too small for guest-%fs swap stub (need >= 96)");
    uint8_t* base = (uint8_t*)got;
    for (uint64_t i = 0; i < n; i++) {
        const size_t emitted = emit_one_stub(base + i * stub_size, slots[i], (uint32_t)i, swap);
        if (emitted > stub_size) return fail("generated import stub exceeds stub_size");
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

bool append_stubs(const std::vector<ImportSlot>& slots, size_t first_new, std::string* err) {
    auto fail = [&](const char* s){ if (err) *err = s; return false; };
    if (!g_stub_size) return fail("append_stubs before install_stubs");
    const uint64_t n = slots.size();
    if (first_new > n || first_new != g_nstubs) return fail("append_stubs: slot table is not an extension");
    if (first_new == n) return true;                    // nothing new to emit
    const bool swap = stub_swap_mode();
    if (swap && g_stub_size < 96) return fail("stub_size too small for guest-%fs swap stub (need >= 96)");
    // Grow the region only by the pages the new slots need. The already-mapped pages are NEVER
    // remapped: relocated guest code already holds addresses inside them, and a fresh MAP_FIXED
    // would tear a stub out from under a thread executing it.
    const uint64_t mapped_end = page_up(g_nstubs * g_stub_size);
    const uint64_t need_end   = page_up(n * g_stub_size);
    if (need_end > kStubApertureBytes) return fail("import stub table exceeds the stub aperture");
    if (need_end > mapped_end) {
        void* want = (void*)(g_stub_base + mapped_end);
        void* got = prosper_mmap_noreplace(want, need_end - mapped_end,
                                           PROT_READ | PROT_WRITE | PROT_EXEC,
                                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (got == MAP_FAILED || got != want) return fail("mmap stub region extension failed");
    }
    for (uint64_t i = first_new; i < n; i++) {
        const size_t emitted =
            emit_one_stub((uint8_t*)(uintptr_t)(g_stub_base + i * g_stub_size), slots[i],
                          (uint32_t)i, swap);
        if (emitted > g_stub_size) return fail("generated import stub exceeds stub_size");
    }
    g_nstubs = n;
    // Publish the grown table only after every new stub is written: prosper_on_unimpl indexes it.
    dispatch_grow_slots(&slots);
    if (getenv("PROSPER_STUBDUMP"))
        for (uint64_t i = first_new; i < n; i++) {
            const std::string& nm = g_nid_db ? g_nid_db->resolve(slots[i].nid) : std::string();
            fprintf(stderr, "[stub] +#%llu off=0x%llx %s::%s %s\n", (unsigned long long)i,
                    (unsigned long long)(i * g_stub_size), slots[i].lib.c_str(),
                    slots[i].nid.c_str(), nm.c_str());
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
    g_int41_skip = getenv("PROSPER_NO_INT41_SKIP") == nullptr;   // read once (getenv is not signal-safe) — #1138
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
    // #1944 discriminator, default OFF (see the declaration): decline the lazy commit so the fault
    // reports at the loading instruction with the real address. A malformed value is refused LOUDLY
    // — `PROSPER_LAZY_COMMIT_STRICT=yes` silently parsing to OFF would make an unarmed run
    // indistinguishable from one where the commit legitimately never fired.
    if (const char* s = getenv("PROSPER_LAZY_COMMIT_STRICT")) {
        char* end = nullptr;
        const long parsed = strtol(s, &end, 0);
        if (end == s || (end && *end))
            fprintf(stderr, "[lazy-commit] NOT ARMED: PROSPER_LAZY_COMMIT_STRICT='%s' is not a "
                            "number — the lazy commit stays enabled\n", s);
        else
            g_lazy_commit_strict = parsed != 0;
    }
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
            // #1945: the window was the DOLL-era [0x20..0x21), the same stale filter #1998 found on
            // PROSPER_WATCH_LABEL/PROSPER_WATCH_HOT. prosper's dmem layout moved: Crisis Core
            // (PPSA07809) hands the guest its per-thread cache at 0x301ac50000 and ArcRunner at
            // 0x3152350000, so on every current title this hook armed NOTHING and the instrument
            // printed NOTHING — indistinguishable from "armed and saw no write". [0x20..0x40) is the
            // window command_processor.cpp already uses post-#1226 (is_byteshift_poolptr).
            if (base < 0x2000000000ull || base >= 0x4000000000ull) return;   // MB3 pool-region only
            mb3w_arm_current_thread(base);
        };
    }
    // UNCONDITIONAL (#2078). This used to be gated on
    // `g_faultmem || g_skip_null_companion || g_null_page`, and the gate was wrong because it does
    // not cover every consumer: `nullpage_deep_dump` also runs from the WORKER-FAULT path (:2782),
    // which no flag guards -- that call site exists precisely because those faults "land here, not
    // on the nullpage path".
    //
    // Without the pipe, `probe_readable` returns false for EVERY address (:612 short-circuits on
    // `g_probe_pipe[1] < 0`), so an ordinary worker-fault report loses its `insn @rip` line and all
    // sixteen register memory windows -- including `rsp`, which is the faulting thread's own stack
    // and cannot be unreadable. The report still prints four bare register lines, so it looks like a
    // report that ran and found nothing rather than one whose probe was disabled.
    //
    // It must be created HERE and not from the handler: `ensure_probe_pipe` guards a magic static,
    // whose initialisation lock is not async-signal-safe. Cost when unused is one pipe pair.
    ensure_probe_pipe();
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
    // PROSPER_BP=0xOFFSET installs an int3 code-breakpoint logger relative to the mapped eboot.
    if (const char* bp = getenv("PROSPER_BP")) {
        g_bp_addr = g_base + strtoull(bp, nullptr, 0);
        if (const char* m = getenv("PROSPER_BP_MAX")) g_bp_max = (int)strtoul(m, nullptr, 0);
        g_bp_klass = getenv("PROSPER_BP_KLASS") != nullptr;
        g_bp_probe = getenv("PROSPER_BP_PROBE");   // stable environ string; read (not copied) in the handler
        if (const char* k = getenv("PROSPER_KSCAN")) g_kscan_klass = strtoull(k, nullptr, 0);
        g_kscan_fields = getenv("PROSPER_KSCAN_FIELDS");
        if (const char* m = getenv("PROSPER_KSCAN_MAX")) g_kscan_max = (int)strtoul(m, nullptr, 0);
        ensure_probe_pipe();
        g_bp_on = true;   // the actual 0xCC is written after the image is mapped (arm_bp below)
    }
    // PROSPER_HWBP=0xOFFSET installs a race-free hardware execute breakpoint relative to the eboot.
    if (const char* hb = getenv("PROSPER_HWBP")) {
        g_hwbp_addr = g_base + strtoull(hb, nullptr, 0);
        if (const char* m = getenv("PROSPER_HWBP_MAX")) g_hwbp_max = (int)strtoul(m, nullptr, 0);
        if (const char* c = getenv("PROSPER_HWBP_R15")) { g_hwbp_r15 = strtoull(c, nullptr, 0); g_hwbp_r15_on = true; }
        if (const char* c = getenv("PROSPER_HWBP_R14")) { g_hwbp_r14 = strtoull(c, nullptr, 0); g_hwbp_r14_on = true; }
        if (const char* c = getenv("PROSPER_HWBP_RET")) { g_hwbp_ret = strtoull(c, nullptr, 0); g_hwbp_ret_on = true; }
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
        g_hwbp_probe = getenv("PROSPER_HWBP_PROBE");               // stable environ string; handler only reads it
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
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    // Run the handler on the per-thread sigaltstack. Besides making stack-overflow diagnostics
    // possible, this keeps page-protection write-watch faults out of a guest function's live SysV red
    // zone. The old Linux opt-in predated the host-%fs restoration in fault_handler: siglongjmp once
    // entered glibc with the guest TCB and its bogus stack guard, causing a jump-to-garbage storm.
    // Recovery from wild/null/SIGFPE init faults now runs on the alt stack in
    // test_initfault_dump_survives, proving that the repaired host-%fs path is safe by default.
    // PROSPER_FAULT_NO_ONSTACK remains a diagnostic escape hatch and deliberately disables write
    // watches through guest_write_watch_set_fault_onstack below.
    // On macOS the alt stack is also mandatory: a guest fault whose access is on the
    // faulting thread's own stack (a bad indirect branch into the stack, a stack overflow) cannot
    // push a signal frame onto that same stack, so without SA_ONSTACK the kernel force-kills the
    // process with no handler entry (the "SIGSEGV, no output" we first saw). The Darwin recovery
    // path likewise siglongjmps out of the handler safely.
    if (getenv("PROSPER_FAULT_NO_ONSTACK")) sa.sa_flags &= ~SA_ONSTACK;
    // The texture write-watch (guest_write_watch.cpp) resolves its faults by return, not siglongjmp, so
    // it is red-zone-safe ONLY when this handler runs on the sigaltstack. Tell it whether that holds;
    // otherwise create() refuses to arm and callers keep the exact byte-comparison fallback (#1144).
    prosper::host::guest_write_watch_set_fault_onstack((sa.sa_flags & SA_ONSTACK) != 0);
    // Parse the dynamic caller-chain/allocation/offset selector only after the red-zone-safety gate is
    // known. No guest mapping exists yet; the matching allocation and map notifications arm it later.
    prosper::host::guest_dmem_write_trace_init_from_environment();
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS,  &sa, nullptr);
    sigaction(SIGILL,  &sa, nullptr);
    sigaction(SIGFPE,  &sa, nullptr);
    if (g_watch_companion || g_bp_on || g_hwbp_on
        || getenv("PROSPER_WATCH_LABEL")
        || getenv("PROSPER_WATCH_ABS")
        || getenv("PROSPER_MB3WATCH")
        || getenv("PROSPER_WATCH_HOT")
        || prosper::host::guest_dmem_write_trace_enabled())
        sigaction(SIGTRAP, &sa, nullptr);   // single-step / breakpoint
}

// Write the 0xCC breakpoint into the (now-mapped) guest image. Call after the image load.
void arm_bp() {
    if (!g_bp_on || !g_bp_addr) return;
    g_bp_orig = *(volatile uint8_t*)g_bp_addr;
    bp_write_byte(g_bp_addr, 0xCC);
    char b[96];
    int n = snprintf(b, sizeof b, "[bp] armed int3 at eboot+0x%llx (orig=0x%02x)\n",
                     (unsigned long long)(g_bp_addr - g_base), g_bp_orig);
    raw_write_fmt(2, b, sizeof b, n);   /* raw syscall: no libc TLS access */
}

// Open + enable the PROSPER_HWBP hardware breakpoint on a primary guest-execution thread — the perf
// event monitors the calling thread. Call after the image is mapped, before its first guest code.
void arm_hwbp() {
    // run_guest_inits and run_entry commonly execute on the same host thread. The init boundary
    // must arm before the first guest instruction, while the later entry boundary must not consume
    // a second perf slot or split that thread's single-step ownership across two fds.
    if (!g_hwbp_on || !g_hwbp_addr || t_hwbp_fd >= 0) return;
    const bool already_anchored = g_hwbp_fd >= 0;
    long fd = perf_bp_open(g_hwbp_addr, HW_BREAKPOINT_X);
    char b[160];
    if (fd < 0) {
        int n = snprintf(b, sizeof b,
                         "[hwbp] perf_event_open FAILED for eboot+0x%llx (errno=%d) — %s\n",
                         (unsigned long long)(g_hwbp_addr - g_base), errno,
                         already_anchored ? "this primary thread not armed"
                                          : "HW bp disabled");
        raw_write_fmt(2, b, sizeof b, n);   /* raw syscall: no libc TLS access */
        // A frontend may run module init on the process thread, then run_entry on a distinct
        // std::thread. A failure on that second primary must not disable the first valid per-TID fd.
        if (!already_anchored) g_hwbp_on = false;
        return;
    }
    const int thread_fd = (int)fd;
    fcntl(thread_fd, F_SETFL, O_ASYNC);
    fcntl(thread_fd, F_SETSIG, SIGTRAP);
    struct f_owner_ex ow; ow.type = F_OWNER_TID; ow.pid = (pid_t)prosper_gettid();
    fcntl(thread_fd, F_SETOWN_EX, &ow);
    if (!hwbp_register_thread(ow.pid, thread_fd)) {
        int n = snprintf(b, sizeof b, "[hwbp] thread-state table full for tid=%ld - %s\n",
                         (long)ow.pid, already_anchored ? "this primary thread not armed"
                                                        : "HW bp disabled");
        raw_write_fmt(2, b, sizeof b, n);
        close(thread_fd);
        if (!already_anchored) g_hwbp_on = false;
        return;
    }
    // g_hwbp_fd is an always-live anchor/fallback for the signal handler, not "the latest primary".
    // Every hit normally resolves its exact fd/step state through the per-TID table above. Retaining
    // the first anchor makes init-thread + later frontend-thread ownership safe and keeps a failed
    // second arm from invalidating the first.
    if (g_hwbp_fd < 0) g_hwbp_fd = thread_fd;
    t_hwbp_fd = thread_fd;
    ioctl(thread_fd, PERF_EVENT_IOC_ENABLE, 0);
    int n = snprintf(b, sizeof b, "[hwbp] armed HW execute bp at eboot+0x%llx (fd=%d tid=%ld)\n",
                     (unsigned long long)(g_hwbp_addr - g_base), thread_fd, (long)prosper_gettid());
    raw_write_fmt(2, b, sizeof b, n);   /* raw syscall: no libc TLS access */
}

// Arm the same execute bp on the CURRENT (worker) thread, gated by PROSPER_HWBP_ALLTHREADS. Each thread
// owns its own perf fd + SIGTRAP delivery, so off-main-thread execution of the target is observed too.
void arm_hwbp_this_thread() {
    if (getenv("PROSPER_HWBP_ARMLOG")) { char b[128]; int n = snprintf(b, sizeof b,
        "[hwbp] arm_this_thread tid=%ld on=%d all=%d addr=0x%llx tfd=%d\n", (long)prosper_gettid(),
        g_hwbp_on, g_hwbp_allthreads, (unsigned long long)g_hwbp_addr, t_hwbp_fd); raw_write_fmt(2, b, sizeof b, n); }
    if (!g_hwbp_on || !g_hwbp_allthreads || !g_hwbp_addr || t_hwbp_fd >= 0) return;
    long fd = perf_bp_open(g_hwbp_addr, HW_BREAKPOINT_X);
    if (fd < 0) { char b[96]; int n = snprintf(b, sizeof b, "[hwbp] worker-arm FAILED tid=%ld errno=%d\n",
        (long)prosper_gettid(), errno); raw_write_fmt(2, b, sizeof b, n); return; }
    t_hwbp_fd = (int)fd;
    fcntl(t_hwbp_fd, F_SETFL, O_ASYNC);
    fcntl(t_hwbp_fd, F_SETSIG, SIGTRAP);
    struct f_owner_ex ow; ow.type = F_OWNER_TID; ow.pid = (pid_t)prosper_gettid();
    fcntl(t_hwbp_fd, F_SETOWN_EX, &ow);
    if (!hwbp_register_thread(ow.pid, t_hwbp_fd)) {
        char b[112]; int n = snprintf(b, sizeof b,
            "[hwbp] thread-state table full for worker tid=%ld - not armed\n", (long)ow.pid);
        raw_write_fmt(2, b, sizeof b, n);
        close(t_hwbp_fd); t_hwbp_fd = -1; return;
    }
    ioctl(t_hwbp_fd, PERF_EVENT_IOC_ENABLE, 0);
    char b[128]; int n = snprintf(b, sizeof b, "[hwbp] armed on worker tid=%ld (fd=%d) for eboot+0x%llx\n",
        (long)prosper_gettid(), t_hwbp_fd, (unsigned long long)goff(g_hwbp_addr));
    raw_write_fmt(2, b, sizeof b, n);
}

void guest_execution_thread_enter(bool primary) {
    if (primary) arm_hwbp();
    else arm_hwbp_this_thread();
    // Observe completion of the real arm boundary, not merely a helper called by the test. This
    // stays useful on machines where perf_event is unavailable: the seam proves ordering and the
    // production arm path remains responsible for its existing fail-visible error.
    if (auto hook = g_guest_execution_enter_test_hook.load(std::memory_order_acquire))
        hook(primary, g_guest_execution_enter_test_opaque.load(std::memory_order_acquire));
}

void set_guest_execution_thread_enter_test_hook(GuestExecutionThreadEnterTestHook hook,
                                                void* opaque) {
    if (!hook) {
        g_guest_execution_enter_test_hook.store(nullptr, std::memory_order_release);
        g_guest_execution_enter_test_opaque.store(nullptr, std::memory_order_release);
        return;
    }
    g_guest_execution_enter_test_opaque.store(opaque, std::memory_order_release);
    g_guest_execution_enter_test_hook.store(hook, std::memory_order_release);
}

uint64_t stub_addr(uint64_t idx) { return g_stub_base + idx * g_stub_size; }

uint64_t hle_guest_return_address(uint64_t entry_rsp) {
    if (!entry_rsp) return 0;
    const uint64_t immediate = *(const uint64_t*)(uintptr_t)entry_rsp;
    // Return-hook imports CALL the handler from a shared trampoline. Both its host-FS and guest-FS
    // paths use the same seven-qword forwarding frame, so the original guest return is +0x40 from
    // the handler entry (including the handler call's own return slot).
    const uint64_t host_hook_begin =
        (uint64_t)(uintptr_t)&prosper_hle_hook_host_trampoline;
    bool from_hook_trampoline = immediate >= host_hook_begin && immediate < host_hook_begin + 256;
#ifdef __linux__
    const uint64_t swap_hook_begin =
        (uint64_t)(uintptr_t)&prosper_hle_hook_swap_trampoline;
    from_hook_trampoline |= immediate >= swap_hook_begin && immediate < swap_hook_begin + 256;
#endif
    if (from_hook_trampoline)
        return *(const uint64_t*)(uintptr_t)(entry_rsp + 0x40);
#if defined(__linux__)
    // Only the Linux guest-FS path CALLS the HLE handler. At handler entry its frame is:
    //   +0x00 return-to-stub, +0x08..0x18 forwarded args 7..9, +0x20 alignment pad,
    //   +0x28 saved guest FS, +0x30 original guest return address.
    // The host-context and macOS paths tail-jump, so their immediate return is not in the table.
    const bool from_stub = g_stub_size && immediate >= g_stub_base &&
                           (immediate - g_stub_base) / g_stub_size < g_nstubs;
    if (from_stub) return *(const uint64_t*)(uintptr_t)(entry_rsp + 0x30);
#endif
    return immediate;
}

uint64_t invoke_stub(uint64_t idx) {
    auto fn = (uint64_t(*)())(stub_addr(idx));
    return fn();
}

std::string describe_code_address(uint64_t address) {
    char text[160];
    const char* module = prosper::guest_module_name(address);
    if (std::strcmp(module, "mapped/host") != 0) {
        std::snprintf(text, sizeof text, "%s+0x%llx", module,
                      (unsigned long long)prosper::guest_module_offset(address));
        return text;
    }
    // Host frame. dladdr resolves the containing object and, when the symbol is exported, its name.
    Dl_info info{};
    if (dladdr((void*)(uintptr_t)address, &info) && info.dli_fname) {
        const char* leaf = std::strrchr(info.dli_fname, '/');
        leaf = leaf ? leaf + 1 : info.dli_fname;
        const auto base = (uint64_t)(uintptr_t)info.dli_fbase;
        if (info.dli_sname)
            std::snprintf(text, sizeof text, "%s!%s+0x%llx", leaf, info.dli_sname,
                          (unsigned long long)(address - (uint64_t)(uintptr_t)info.dli_saddr));
        else
            std::snprintf(text, sizeof text, "%s+0x%llx", leaf,
                          (unsigned long long)(address - base));
        return text;
    }
    std::snprintf(text, sizeof text, "host:0x%llx", (unsigned long long)address);
    return text;
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
#ifdef __APPLE__
    // mach_vm_region: first region at or after `addr` (containing it when mapped).
    mach_vm_address_t ra = addr; mach_vm_size_t rs = 0;
    vm_region_basic_info_data_64_t info; mach_msg_type_number_t cnt = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t obj = MACH_PORT_NULL;
    if (mach_vm_region(mach_task_self(), &ra, &rs, VM_REGION_BASIC_INFO_64,
                       (vm_region_info_t)&info, &cnt, &obj) != KERN_SUCCESS) return false;
    if (obj != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), obj);
    if (addr < ra || addr >= ra + rs) return false;   // gap: addr itself is unmapped
    s = ra; e = ra + rs;
    perms[0] = (info.protection & VM_PROT_READ)    ? 'r' : '-';
    perms[1] = (info.protection & VM_PROT_WRITE)   ? 'w' : '-';
    perms[2] = (info.protection & VM_PROT_EXECUTE) ? 'x' : '-';
    perms[3] = 'p'; perms[4] = 0;
    return true;
#else
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
#endif
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
    // boot_program runs module init code before run_entry. This is already guest execution and can
    // create more guest threads, so waiting for run_entry to arm leaves the earliest code invisible.
    if (!fns.empty()) guest_execution_thread_enter(/*primary=*/true);
#ifdef __APPLE__
    // macOS only: the module .init_array ctors touch guest %fs TLS and run on this (main) thread BEFORE
    // run_entry, so activate the guest TCB now to give the %fs emulator its per-thread guest_TP
    // (idempotent — run_entry's later call reuses the same TP). On Linux the inits run under the host
    // glibc %fs (a valid TCB) and activation stays in run_entry, so this is deliberately NOT done there
    // to avoid perturbing the tuned Messenger/PROSPER_GUEST_FS boot ordering.
    guest_tls_activate_thread();
#endif
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
    guest_execution_thread_enter(/*primary=*/true);
    if (sigsetjmp(g_jb, 1) == 0) {
        // Switch %fs to a guest TCB as the LAST host action before entering the
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
