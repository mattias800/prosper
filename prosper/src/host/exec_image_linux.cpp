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
#include <fcntl.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
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
                if (ok) { g_lazy_pages++; return; }  // re-execute against the now-mapped page
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

    uint8_t* base = (uint8_t*)got;
    for (uint64_t i = 0; i < n; i++) {
        uint8_t* slot = base + i * stub_size;
        HleFn fn = Hle::lookup(slots[i].nid);
        if (fn) emit_impl(slot, (uint64_t)fn);
        else    emit_unimpl(slot, (uint32_t)i, (uint64_t)&prosper_on_unimpl);
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
    if (g_faultmem && g_devnull_fd < 0)
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
    install_sigaltstack();
    struct sigaction sa{};
    sa.sa_sigaction = fault_handler;
    sa.sa_flags = SA_SIGINFO;   // (SA_ONSTACK disabled: siglongjmp from the alt stack tripped glibc's
                                // %fs-guarded ____longjmp_chk -> jump-to-garbage fault storm)
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS,  &sa, nullptr);
    sigaction(SIGILL,  &sa, nullptr);
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
    static const char argstr[] = "/app0/eboot.bin";
    top -= sizeof(argstr); uint64_t arg0 = top;
    memcpy((void*)arg0, argstr, sizeof(argstr));
    top &= ~(uint64_t)0xf;
    uint64_t vec[] = { 1, arg0, 0, 0, 0, 0 };   // argc, argv0, NULL, envp NULL, auxv AT_NULL
    top -= sizeof(vec); top &= ~(uint64_t)0xf;   // 16-aligned base for the vector
    // The Sony crt _start pushes an odd number of words before its first call, so it
    // expects entry rsp ≡ 8 (mod 16) (like a normal callee), NOT 16-aligned. Placing
    // the vector 8 below a 16-boundary makes every downstream call correctly aligned,
    // so alignment-sensitive SIMD (vmovaps) in callees doesn't #GP.
    top -= 8;
    memcpy((void*)top, vec, sizeof(vec));
    uint64_t sp = top, rdi = sp, rsi = 0;

    g_trap_kind = 0; g_fault_addr = nullptr; g_fault_rip = 0; g_armed_tid = cur_tid();
    if (sigsetjmp(g_jb, 1) == 0) {
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
