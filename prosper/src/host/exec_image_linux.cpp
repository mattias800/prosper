// exec_image_linux.cpp — Linux host backing + HLE stubs (M2/M3). Compiles to nothing
// on non-Linux so the shared (mingw) build is unaffected.
#include "exec_image.hpp"
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
    int  g_devnull_fd = -1;
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
                if (ok && addr >= s && addr < e) { write(2, pend, pn); write(2, line, (size_t)li);
                    write(2, "\n", 1); close(fd); return; }
                li = 0;
            }
        }
        close(fd);
    }
    void arm_hwwatch(uint64_t addr) {
        long fd = perf_bp_open(addr, HW_BREAKPOINT_W);
        char b[160];
        if (fd < 0) { int n = snprintf(b, sizeof b, "[hwwatch] perf W-watch FAILED addr=0x%llx errno=%d\n",
                          (unsigned long long)addr, errno); write(2, b, n); return; }
        g_hwwatch_fd = (int)fd; g_hwwatch_addr = addr;
        fcntl(g_hwwatch_fd, F_SETFL, O_ASYNC);
        fcntl(g_hwwatch_fd, F_SETSIG, SIGTRAP);
        struct f_owner_ex ow; ow.type = F_OWNER_TID; ow.pid = (pid_t)syscall(SYS_gettid);
        fcntl(g_hwwatch_fd, F_SETOWN_EX, &ow);
        ioctl(g_hwwatch_fd, PERF_EVENT_IOC_ENABLE, 0);
        int n = snprintf(b, sizeof b, "[hwwatch] armed W-watch at 0x%llx (fd=%d)\n",
                         (unsigned long long)addr, g_hwwatch_fd); write(2, b, n);
    }
    // PROSPER_PEEK dumps offsets off registers at fault time. Supports N specs separated by ';',
    // each "reg:off,off,..."; an offset may be prefixed '*' to chase one pointer level first
    // (e.g. "r15:*0x18+0x0" = read [[r15+0x18]+0x0]).
    struct PeekSpec { char reg[4]; uint64_t off[12]; bool deref[12]; uint64_t pre[12]; int n; };
    PeekSpec g_peek[6] = {};
    int      g_peek_specs = 0;

    // Async-signal-safe readability probe: write() to /dev/null returns EFAULT (not a fault) for
    // an unmapped source, so we can test guest addresses without risking a nested SIGSEGV.
    bool probe_readable(uint64_t a) {
        if (a < 0x1000 || g_devnull_fd < 0) return false;
        return write(g_devnull_fd, (const void*)a, 8) == 8;
    }
    void dump_fault_mem() {
        if (!g_faultmem) return;
        const struct { const char* n; uint64_t v; } regs[] = {
            {"rdi", g_rdi}, {"rsi", g_rsi}, {"rdx", g_rdx}, {"rcx", g_rcx},
            {"rax", g_rax}, {"rbx", g_rbx}, {"rbp", g_rbp}, {"rsp", g_rsp},
            {"r8 ", g_r8},  {"r9 ", g_r9},  {"r12", g_r12}, {"r13", g_r13},
            {"r14", g_r14}, {"r15", g_r15},
        };
        char b[256];
        int n = snprintf(b, sizeof b, "[prosper] FAULTMEM dump (regs -> guest memory):\n");
        write(2, b, n);
        for (auto& r : regs) {
            if (!probe_readable(r.v)) {
                n = snprintf(b, sizeof b, "  %s=0x%llx  (unmapped/immediate)\n",
                             r.n, (unsigned long long)r.v);
                write(2, b, n);
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
            write(2, b, n);
        }
        // Deep field peek (PROSPER_PEEK, parsed once at arm time — getenv is not signal-safe): read
        // specific offsets off one or more registers to classify a large object beyond the 0x20-byte
        // window above. See g_peek definition for the syntax.
        for (int sp = 0; sp < g_peek_specs; sp++) {
            const PeekSpec& ps = g_peek[sp];
            uint64_t base = 0;
            for (auto& r : regs) { bool m = true; for (int i=0;i<3;i++) if (r.n[i]!=ps.reg[i]&&!(r.n[i]==' '&&ps.reg[i]==0)) { m=false; break; } if (m) { base=r.v; break; } }
            n = snprintf(b, sizeof b, "  PEEK %s=0x%llx:\n", ps.reg, (unsigned long long)base);
            write(2, b, n);
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
                write(2, b, n);
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

    void fault_handler(int sig, siginfo_t* si, void* uctx) {
        // PROSPER_HWBP race-free hardware breakpoint. The perf event is disabled while single-stepping,
        // so a SIGTRAP while g_hwbp_stepping is the step-completion (TF) -> re-enable + clear TF. Any
        // other SIGTRAP while armed is a HW-breakpoint hit -> log registers, then disable + single-step
        // over it (TF), re-enabling on the step trap. No code bytes are touched (unlike int3), so this is
        // safe on hot/multi-threaded functions.
        if (sig == SIGTRAP && g_hwbp_on && g_hwbp_fd >= 0) {
            auto* uc = (ucontext_t*)uctx;
            if (g_hwbp_stepping && si->si_code == TRAP_TRACE) {
                ioctl(g_hwbp_fd, PERF_EVENT_IOC_ENABLE, 0);
                uc->uc_mcontext.gregs[REG_EFL] &= ~0x100ll;
                g_hwbp_stepping = false;
                return;
            }
            // Chained DATA write-watchpoint hit (a write to the watched slot completed): log the writer
            // RIP + the value just stored. Write-watchpoints trap AFTER the store, so no step is needed.
            if (g_hwwatch_fd >= 0 && si->si_fd == g_hwwatch_fd) {
                auto& gr2 = uc->uc_mcontext.gregs;
                unsigned long long v = 0; { auto p=(volatile uint64_t*)g_hwwatch_addr; v=*p; }
                if (g_hwwatch_count < 60) {
                    g_hwwatch_count = g_hwwatch_count + 1;
                    char b[200];
                    uint64_t wr = (uint64_t)gr2[REG_RIP];
                    bool in_eboot = (wr >= 0x400000000ull && wr < 0x420000000ull);
                    int n = snprintf(b, sizeof b, "[hwwatch] #%d WRITE [0x%llx]=0x%llx by rip=%s0x%llx tid=%ld\n",
                        (int)g_hwwatch_count, (unsigned long long)g_hwwatch_addr, v,
                        in_eboot ? "eboot+" : "", (unsigned long long)(in_eboot ? wr - 0x400000000ull : wr), cur_tid());
                    write(2, b, n);
                    if (!in_eboot) classify_addr(wr);
                }
                return;
            }
            auto& gr = uc->uc_mcontext.gregs;
            uint64_t rdi = (uint64_t)gr[REG_RDI], rsi = (uint64_t)gr[REG_RSI];
            uint64_t rax = (uint64_t)gr[REG_RAX], r14 = (uint64_t)gr[REG_R14], r15 = (uint64_t)gr[REG_R15];
            if (g_hwbp_count < g_hwbp_max) {
                g_hwbp_count = g_hwbp_count + 1;
                auto rd = [](uint64_t a) -> unsigned long long {
                    return probe_readable(a) ? (unsigned long long)*(const uint64_t*)a : 0xBADBADull; };
                uint64_t rbp = (uint64_t)gr[REG_RBP], rsp = (uint64_t)gr[REG_RSP];
                auto off = [](unsigned long long v) -> unsigned long long {
                    return (v >= 0x400000000ull && v < 0x420000000ull) ? v - 0x400000000ull : v; };
                char b[380];
                int n = snprintf(b, sizeof b,
                    "[hwbp] #%d rip=eboot+0x%llx rdi=0x%llx rsi=0x%llx rdx=0x%llx r14=0x%llx ret=eboot+0x%llx caller_rbp=eboot+0x%llx tid=%ld\n",
                    (int)g_hwbp_count, (unsigned long long)((uint64_t)gr[REG_RIP] - 0x400000000ull),
                    (unsigned long long)rdi, (unsigned long long)rsi, (unsigned long long)gr[REG_RDX],
                    (unsigned long long)r14, off(rd(rsp)), off(rd(rbp + 8)), cur_tid());
                (void)r15; (void)rax;
                write(2, b, n);
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
            ioctl(g_hwbp_fd, PERF_EVENT_IOC_DISABLE, 0);   // disable so we can step off the bp address
            if (g_hwbp_count < g_hwbp_max) {               // step past, then re-enable for the next hit
                uc->uc_mcontext.gregs[REG_EFL] |= 0x100ll; g_hwbp_stepping = true;
            }                                              // else: leave disabled (one-and-done at the cap)
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
                if (g_bp_count < g_bp_max) {
                    g_bp_count = g_bp_count + 1;
                    auto rd = [](uint64_t a) -> unsigned long long {
                        return probe_readable(a) ? (unsigned long long)*(const uint64_t*)a : 0xBADBADull; };
                    auto& gr = uc->uc_mcontext.gregs;
                    uint64_t rdi = (uint64_t)gr[REG_RDI], rsi = (uint64_t)gr[REG_RSI];
                    uint64_t rax = (uint64_t)gr[REG_RAX], r14 = (uint64_t)gr[REG_R14];
                    char b[320];
                    int n = snprintf(b, sizeof b,
                        "[bp] #%d rdi=0x%llx rsi=0x%llx rax=0x%llx r14=0x%llx r15=0x%llx [rdi]=0x%llx [rdi+0x1e4c]=0x%llx tid=%ld\n",
                        (int)g_bp_count, (unsigned long long)rdi, (unsigned long long)rsi,
                        (unsigned long long)rax, (unsigned long long)r14, (unsigned long long)r15,
                        rd(rdi), (unsigned long long)(probe_readable(rdi+0x1e4c) ? *(const uint32_t*)(rdi+0x1e4c) : 0xBADBAD),
                        cur_tid());
                    write(2, b, n);
                }
                bp_write_byte(g_bp_addr, g_bp_orig);              // restore real instruction
                uc->uc_mcontext.gregs[REG_RIP] = (greg_t)g_bp_addr;  // re-execute it
                uc->uc_mcontext.gregs[REG_EFL] |= 0x100ll;        // single-step (TF)
                g_bp_stepping = true;
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
            write(2, b, n);
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
                    write(2, b, n);
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
                    write(2, b, n);
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
        // Lazy unified-memory backing: back an unmapped GPU-VA page on demand and retry.
        if (sig == SIGSEGV && si->si_addr) {
            uint64_t a = (uint64_t)si->si_addr;
            if (a >= GPU_VA_LO && a < GPU_VA_HI) {
                void* page = (void*)(a & ~(uint64_t)0xfff);
                bool ok = mmap(page, 0x1000, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == page;
                if (g_faultlog) {
                    char b[128]; auto* uc = (ucontext_t*)uctx;
                    int n = snprintf(b, sizeof b, "[fault] GPU-VA %s addr=0x%llx rip=0x%llx\n",
                                     ok ? "mapped" : "MMAP-FAILED", (unsigned long long)a,
                                     (unsigned long long)uc->uc_mcontext.gregs[REG_RIP]);
                    write(2, b, n);
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
                    write(2, b, n);
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
                write(2, b, n);
                uc->uc_mcontext.gregs[REG_RIP] = (greg_t)g_skip_target;
                return;   // re-execute from the reader's skip label
            }
        }
        if (g_faultlog) {
            char b[128]; auto* uc = (ucontext_t*)uctx;
            int n = snprintf(b, sizeof b, "[fault] sig=%d addr=%p rip=0x%llx armed=%d tid=%ld\n",
                             sig, si->si_addr, (unsigned long long)uc->uc_mcontext.gregs[REG_RIP],
                             (int)(g_armed_tid && cur_tid() == g_armed_tid), cur_tid());
            write(2, b, n);
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
        dump_fault_mem();   // no-op unless PROSPER_FAULTMEM is set
        if (g_armed_tid && cur_tid() == g_armed_tid) siglongjmp(g_jb, 1);
        // Fault on a thread with no recovery point (a guest worker thread). Report where
        // (async-signal-safe write) then terminate cleanly instead of a cross-thread longjmp.
        {
            char b[160];
            int n = snprintf(b, sizeof b, "[prosper] WORKER-THREAD FAULT: sig=%d addr=%p rip=0x%llx (image+0x%llx)\n",
                             sig, g_fault_addr, (unsigned long long)g_fault_rip,
                             (unsigned long long)(g_base && g_fault_rip >= g_base ? g_fault_rip - g_base : g_fault_rip));
            write(2, b, n);
        }
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
    // handler on the current fs — exactly the old behavior. Uses FSGSBASE. `push r11` also re-aligns rsp to
    // the SysV rsp≡8(mod16) contract. Clobbers only rax/r11 (never the arg regs rdi..r9).
    // Keep 0x108/0x100/magic in sync with guest_tls.cpp (MAGIC_OFF/HOSTFS_OFF/TCB_MAGIC).
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
        // .guest: push r11 ; mov rax,[r11+0x100] ; wrfsbase rax ; <mid> ; call rax ; pop r11 ; wrfsbase r11 ; ret
        *p++=0x41; *p++=0x53;
        *p++=0x49; *p++=0x8B; *p++=0x83; uint32_t ho=0x100; memcpy(p,&ho,4); p+=4;
        *p++=0xF3; *p++=0x48; *p++=0x0F; *p++=0xAE; *p++=0xD0;
        mid(p);
        *p++=0xFF; *p++=0xD0;                                               // call rax
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
    uint64_t region = page_up(n * stub_size);
    void* want = (void*)stub_base;
    void* got = mmap(want, region, PROT_READ | PROT_WRITE | PROT_EXEC,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (got == MAP_FAILED || got != want) return fail("mmap stub region failed");

    bool swap = guest_tls_enabled();   // gated: emit the %fs swap stubs so HLE handlers run on the host TCB
    if (swap && stub_size < 80) return fail("stub_size too small for guest-%fs swap stub (need >= 80)");
    uint8_t* base = (uint8_t*)got;
    for (uint64_t i = 0; i < n; i++) {
        uint8_t* slot = base + i * stub_size;
        HleFn fn = Hle::lookup(slots[i].nid);
        if (fn) { if (swap) emit_impl_swap(slot, (uint64_t)fn);       else emit_impl(slot, (uint64_t)fn); }
        else    { if (swap) emit_unimpl_swap(slot, (uint32_t)i, (uint64_t)&prosper_on_unimpl);
                  else      emit_unimpl(slot, (uint32_t)i, (uint64_t)&prosper_on_unimpl); }
    }
    g_stub_base = stub_base; g_stub_size = stub_size; g_nstubs = n;
    return true;
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
    g_faultlog = getenv("PROSPER_FAULTLOG") != nullptr;
    g_skip_null_companion = getenv("PROSPER_SKIP_NULL_COMPANION") != nullptr;
    g_null_page = getenv("PROSPER_NULL_PAGE") != nullptr;
    g_watch_companion = getenv("PROSPER_WATCH_COMPANION") != nullptr;
    if (const char* r = getenv("PROSPER_SKIP_RIP"))    g_skip_rip    = strtoull(r, nullptr, 0);
    if (const char* t = getenv("PROSPER_SKIP_TARGET")) g_skip_target = strtoull(t, nullptr, 0);
    if ((g_faultmem || g_skip_null_companion) && g_devnull_fd < 0)
        g_devnull_fd = open("/dev/null", O_WRONLY | O_CLOEXEC);
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
    // PROSPER_BP=0xOFFSET installs an int3 code-breakpoint-logger at guest VA 0x400000000+offset.
    if (const char* bp = getenv("PROSPER_BP")) {
        g_bp_addr = 0x400000000ull + strtoull(bp, nullptr, 0);
        if (const char* m = getenv("PROSPER_BP_MAX")) g_bp_max = (int)strtoul(m, nullptr, 0);
        if (g_devnull_fd < 0) g_devnull_fd = open("/dev/null", O_WRONLY | O_CLOEXEC);
        g_bp_on = true;   // the actual 0xCC is written after the image is mapped (arm_bp below)
    }
    // PROSPER_HWBP=0xOFFSET installs a race-free hardware execute breakpoint at guest VA 0x400000000+off.
    if (const char* hb = getenv("PROSPER_HWBP")) {
        g_hwbp_addr = 0x400000000ull + strtoull(hb, nullptr, 0);
        if (const char* m = getenv("PROSPER_HWBP_MAX")) g_hwbp_max = (int)strtoul(m, nullptr, 0);
        if (g_devnull_fd < 0) g_devnull_fd = open("/dev/null", O_WRONLY | O_CLOEXEC);
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
    sa.sa_flags = SA_SIGINFO;   // (SA_ONSTACK disabled: siglongjmp from the alt stack tripped glibc's
                                // %fs-guarded ____longjmp_chk -> jump-to-garbage fault storm)
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS,  &sa, nullptr);
    sigaction(SIGILL,  &sa, nullptr);
    if (g_watch_companion || g_bp_on || g_hwbp_on) sigaction(SIGTRAP, &sa, nullptr);   // single-step / breakpoint
}

// Write the 0xCC breakpoint into the (now-mapped) guest image. Call after the image load.
void arm_bp() {
    if (!g_bp_on || !g_bp_addr) return;
    g_bp_orig = *(volatile uint8_t*)g_bp_addr;
    bp_write_byte(g_bp_addr, 0xCC);
    char b[96];
    int n = snprintf(b, sizeof b, "[bp] armed int3 at eboot+0x%llx (orig=0x%02x)\n",
                     (unsigned long long)(g_bp_addr - 0x400000000ull), g_bp_orig);
    write(2, b, n);
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
        write(2, b, n); g_hwbp_on = false; return;
    }
    g_hwbp_fd = (int)fd;
    fcntl(g_hwbp_fd, F_SETFL, O_ASYNC);
    fcntl(g_hwbp_fd, F_SETSIG, SIGTRAP);
    struct f_owner_ex ow; ow.type = F_OWNER_TID; ow.pid = (pid_t)syscall(SYS_gettid);
    fcntl(g_hwbp_fd, F_SETOWN_EX, &ow);
    ioctl(g_hwbp_fd, PERF_EVENT_IOC_ENABLE, 0);
    int n = snprintf(b, sizeof b, "[hwbp] armed HW execute bp at eboot+0x%llx (fd=%d tid=%ld)\n",
                     (unsigned long long)(g_hwbp_addr - 0x400000000ull), g_hwbp_fd, (long)syscall(SYS_gettid));
    write(2, b, n);
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

size_t run_guest_inits(const std::vector<uint64_t>& fns) {
    size_t ok = 0;
    for (uint64_t f : fns) {
        g_trap_kind = 0; g_armed_tid = cur_tid();
        if (sigsetjmp(g_jb, 1) == 0) { ((void (*)())(uintptr_t)f)(); ok++; }
        g_armed_tid = 0;
        if (g_trap_kind) fprintf(stderr, "[prosper] init fn 0x%llx faulted (%s); continuing\n",
                                 (unsigned long long)f, trap_detail().c_str());
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
    top &= ~(uint64_t)0xf;
    // crt0 vector: argc, argv[0..N-1], NULL, envp NULL, auxv AT_NULL(0,0).
    std::vector<uint64_t> vecv;
    vecv.push_back(args.size());
    for (uint64_t pp : argptrs) vecv.push_back(pp);
    vecv.push_back(0);   // argv terminator
    vecv.push_back(0);   // envp NULL
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
        r.rdi = g_rdi; r.rsi = g_rsi; r.rdx = g_rdx;
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
