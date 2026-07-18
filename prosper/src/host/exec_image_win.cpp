// exec_image_win.cpp — Windows host backing for the guest-execution substrate.
//
// The Windows sibling of exec_image_linux.cpp. It implements the same exec_image.hpp contract with
// Win32 primitives instead of POSIX: VirtualAlloc for fixed-address guest mappings, a Vectored
// Exception Handler (VEH) instead of a SIGSEGV/SIGILL handler, and CONTEXT register access instead
// of ucontext. The guest is System V AMD64; the host is Microsoft x64. Emitted import stubs marshal
// guest calls into the host ABI, and prosper_call_guest_sysv marshals host calls into the guest ABI.
//
// STATUS (2026-07-14): runtime-verified through repeated GC cycles and the native live Vulkan
// renderer. Remaining work is tracked in docs/PORTING.md "Windows", including physical-memory
// alias fidelity, float/XMM import arguments, and deeper frontend/gameplay validation. Diagnostics
// that depend on Linux perf_event / ptrace (PROSPER_HWBP/HWWATCH/PEEK/DUMPAT) remain absent. A small
// one-shot int3 logger (`PROSPER_WIN_BP=off[,off...]`) is available for native branch-order probes.
#ifdef _WIN32

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00   // GetCurrentThreadStackLimits (Windows 8+)
#endif

#include "exec_image.hpp"
#include "guest_write_watch.hpp"
#include "sse4a.hpp"
#include "../hle/nid.hpp"
#include "../hle/dispatch.hpp"

#include <windows.h>
#include <pthread.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <csetjmp>
#include <map>
#include <mutex>
#include <limits>
#include <string>
#include <vector>

// Host→guest SysV call trampoline. The guest is System V AMD64; the host is Microsoft x64. On
// Linux/macOS a GuestInitFn is tagged __attribute__((sysv_abi)) so the compiler marshals the call,
// but on Windows that attribute breaks MinGW SEH unwinding and PROSPER_SYSV_ABI is empty — so a
// plain C call `((GuestInitFn)f)(argc, argp)` passes args in the MS x64 registers (rcx/rdx) while
// the guest reads the SysV registers (rdi/rsi). The guest then acts on garbage argc/argp and jumps
// wild (observed: a module_start crash landing in host data). This shim moves argc→rdi, argp→rsi
// (SysV) and calls the guest, saving the MS x64 callee-saved registers the SysV callee is free to
// clobber (rsi, rdi, and xmm6–xmm15); rbx/rbp/r12–r15 are callee-saved in both ABIs. Returns the
// guest's rax. run_entry does the equivalent inline (it jmp's and never returns); this is the
// call-and-return path used for the init/module_start functions.
extern "C" uint64_t prosper_call_guest_sysv(uint64_t fn, uint64_t argc, uint64_t argp);
extern "C" uint64_t prosper_call_guest_sysv4(uint64_t fn, uint64_t a0, uint64_t a1,
                                               uint64_t a2, uint64_t a3);
__asm__(
    ".text\n"
    ".p2align 4\n"
    ".globl prosper_call_guest_sysv\n"
    "prosper_call_guest_sysv:\n"
    "    pushq %rbp\n"
    "    movq  %rsp, %rbp\n"
    "    pushq %rsi\n"
    "    pushq %rdi\n"
    "    subq  $176, %rsp\n"           // 10*16 for xmm6-15 (+16 slack); keeps rsp 16-aligned at the call
    "    movaps %xmm6,    0(%rsp)\n"
    "    movaps %xmm7,   16(%rsp)\n"
    "    movaps %xmm8,   32(%rsp)\n"
    "    movaps %xmm9,   48(%rsp)\n"
    "    movaps %xmm10,  64(%rsp)\n"
    "    movaps %xmm11,  80(%rsp)\n"
    "    movaps %xmm12,  96(%rsp)\n"
    "    movaps %xmm13, 112(%rsp)\n"
    "    movaps %xmm14, 128(%rsp)\n"
    "    movaps %xmm15, 144(%rsp)\n"
    "    movq  %rcx, %rax\n"           // MS x64: rcx=fn, rdx=argc, r8=argp
    "    movq  %rdx, %rdi\n"           // argc -> SysV arg0
    "    movq  %r8,  %rsi\n"           // argp -> SysV arg1
    "    callq *%rax\n"
    "    movaps    0(%rsp), %xmm6\n"
    "    movaps   16(%rsp), %xmm7\n"
    "    movaps   32(%rsp), %xmm8\n"
    "    movaps   48(%rsp), %xmm9\n"
    "    movaps   64(%rsp), %xmm10\n"
    "    movaps   80(%rsp), %xmm11\n"
    "    movaps   96(%rsp), %xmm12\n"
    "    movaps  112(%rsp), %xmm13\n"
    "    movaps  128(%rsp), %xmm14\n"
    "    movaps  144(%rsp), %xmm15\n"
    "    addq  $176, %rsp\n"
    "    popq  %rdi\n"
    "    popq  %rsi\n"
    "    popq  %rbp\n"
    "    retq\n"
);

// Four-argument host->guest bridge used by registered Sony callbacks.  Microsoft x64 passes
// fn/a0/a1/a2 in rcx/rdx/r8/r9 and a3 at [entry-rsp+0x28]; the PS5 callback reads the four payload
// arguments from rdi/rsi/rdx/rcx.  Preserve every register that is nonvolatile to the Windows caller
// but volatile to a SysV callee, exactly as the two-argument module-init bridge above does.
__asm__(
    ".text\n"
    ".p2align 4\n"
    ".globl prosper_call_guest_sysv4\n"
    "prosper_call_guest_sysv4:\n"
    "    pushq %rbp\n"
    "    movq  %rsp, %rbp\n"
    "    pushq %rsi\n"
    "    pushq %rdi\n"
    "    subq  $176, %rsp\n"
    "    movaps %xmm6,    0(%rsp)\n"
    "    movaps %xmm7,   16(%rsp)\n"
    "    movaps %xmm8,   32(%rsp)\n"
    "    movaps %xmm9,   48(%rsp)\n"
    "    movaps %xmm10,  64(%rsp)\n"
    "    movaps %xmm11,  80(%rsp)\n"
    "    movaps %xmm12,  96(%rsp)\n"
    "    movaps %xmm13, 112(%rsp)\n"
    "    movaps %xmm14, 128(%rsp)\n"
    "    movaps %xmm15, 144(%rsp)\n"
    "    movq  %rcx, %rax\n"           // callback address
    "    movq  %r9,  %r10\n"           // save a2 before r9 is reused
    "    movq  %rdx, %rdi\n"           // a0 -> SysV rdi
    "    movq  %r8,  %rsi\n"           // a1 -> SysV rsi
    "    movq  %r10, %rdx\n"           // a2 -> SysV rdx
    "    movq  48(%rbp), %rcx\n"       // a3 -> SysV rcx (5th MS argument)
    "    callq *%rax\n"
    "    movaps    0(%rsp), %xmm6\n"
    "    movaps   16(%rsp), %xmm7\n"
    "    movaps   32(%rsp), %xmm8\n"
    "    movaps   48(%rsp), %xmm9\n"
    "    movaps   64(%rsp), %xmm10\n"
    "    movaps   80(%rsp), %xmm11\n"
    "    movaps   96(%rsp), %xmm12\n"
    "    movaps  112(%rsp), %xmm13\n"
    "    movaps  128(%rsp), %xmm14\n"
    "    movaps  144(%rsp), %xmm15\n"
    "    addq  $176, %rsp\n"
    "    popq  %rdi\n"
    "    popq  %rsi\n"
    "    popq  %rbp\n"
    "    retq\n"
);

namespace prosper {

// A guest function pointer: called by host code (init arrays, entry) but obeys the guest's SysV ABI.
using GuestInitFn = PROSPER_SYSV_ABI void (*)(uint64_t argc, uint64_t argp);

namespace {
    uint64_t g_base = 0, g_stub_base = 0, g_stub_size = 0, g_nstubs = 0;
    NidDb*   g_nid_db = nullptr;

    // Per-thread recovery point. A guest fault on the armed thread is turned into a longjmp back to
    // run_entry / run_guest_inits (see the VEH). thread_local is correct: the VEH runs on the
    // faulting thread. Keyed implicitly by being thread_local (no tid needed, unlike the Linux path
    // which must dodge a guest %fs — Windows has no guest %fs swap).
    //
    // We use __builtin_setjmp/__builtin_longjmp, NOT the CRT setjmp/longjmp. On Windows x64 the CRT
    // longjmp performs a full SEH stack UNWIND (RtlUnwindEx) from the longjmp site back to the setjmp
    // frame. Our recovery longjmps ACROSS a guest frame and the hand-written prosper_call_guest_sysv
    // trampoline — neither has Windows .pdata/.xdata unwind info — so RtlUnwindEx walks into frames it
    // cannot describe and blows the stack (STATUS_STACK_OVERFLOW 0xC00000FD) instead of recovering.
    // The __builtin_* pair restores only rsp/rbp/rip (no unwind), which is exactly right for crossing
    // foreign frames — the same simple-register-restore behavior the Linux longjmp has. The buffer is
    // GCC's fixed 5-word layout.
    thread_local void*       t_jb[5];
    thread_local volatile int t_armed = 0;

    // Fault state latched by the VEH for the BootResult / init-fault report (single-threaded boot use).
    volatile int g_trap_kind = 0;   // 0 none, 2 access-violation, 3 illegal-instruction
    uint64_t g_fault_addr = 0, g_fault_rip = 0;
    uint64_t g_rax=0,g_rbx=0,g_rcx=0,g_rdx=0,g_rsi=0,g_rdi=0,g_rbp=0,g_rsp=0;
    uint64_t g_r8=0,g_r9=0,g_r10=0,g_r11=0,g_r12=0,g_r13=0,g_r14=0,g_r15=0;

    struct WinDiagBreakpoint {
        uint64_t addr = 0;
        uint8_t original = 0;
        volatile LONG armed = 0;
    };
    WinDiagBreakpoint g_diag_bps[16];
    size_t g_diag_bp_count = 0;
    bool g_diag_bp_repeat = false;
    thread_local WinDiagBreakpoint* t_diag_bp_rearm = nullptr;

    // Intel hosts raise #UD for every AMD SSE4a instruction. A dense animation-unpack loop in
    // Astro contains 48 EXTRQs per iteration. More importantly, Windows builds the #UD dispatch
    // frame below the interrupted RSP and can overwrite the guest's SysV red zone before VEH runs.
    // Translate register-form EXTRQs into a nearby code cache before guest entry. Five-byte forms
    // detour directly; four-byte forms consume only a fail-closed decoded successor or an adjacent
    // five-byte EXTRQ so every overwritten byte has an equivalent secondary entry.
    constexpr size_t kSse4aFastpathCacheSize = 4 * 1024 * 1024;
    constexpr size_t kSse4aFastpathSlotSize = 256;
    constexpr size_t kSse4aSecondaryOffset = 160;
    constexpr size_t kSse4aFastpathMaxPatches = 8192;
    uint8_t* g_sse4a_fastpath_cache = nullptr;
    size_t g_sse4a_fastpath_used = 0;
    uint8_t* g_sse4a_successor_cache = nullptr;
    size_t g_sse4a_successor_used = 0;
    uint8_t* g_sse4a_chain_cache = nullptr;
    size_t g_sse4a_chain_used = 0;
    uint64_t g_sse4a_fastpath_sites[kSse4aFastpathMaxPatches]{};
    volatile LONG g_sse4a_fastpath_count = 0;
    struct Sse4aSecondaryRedirect { uint64_t marker = 0, target = 0; };
    Sse4aSecondaryRedirect g_sse4a_secondary_redirects[kSse4aFastpathMaxPatches]{};
    volatile LONG g_sse4a_secondary_count = 0;
    volatile LONG g_sse4a_fastpath_lock = 0;

    // Thread-stack registry (portable; mirrors the Linux one) so GC/thread code gets real bounds.
    std::map<uint64_t, std::pair<uint64_t, uint64_t>> g_stacks;
    std::mutex g_smx;

    // Module initialization may attach a frontend-owned thread to IL2CPP before run_entry switches
    // to its dedicated guest stack. Keep the init thread's native stack registered for that
    // thread's lifetime, then remove it before Windows can recycle the native TID.
    struct InitThreadStackRegistration {
        uint64_t native_tid = 0;
        ~InitThreadStackRegistration() {
            if (!native_tid) return;
            std::lock_guard<std::mutex> lk(g_smx);
            g_stacks.erase(native_tid);
        }
    };
    thread_local InitThreadStackRegistration t_init_stack_registration;

    std::vector<std::pair<uint64_t, uint64_t>> g_modstart_param_ranges;
    struct ModStartDesc { uint64_t a, b, c; } g_modstart_desc = { 0x10, 0x200, 0 };

    inline uint64_t cur_tid() { return (uint64_t)GetCurrentThreadId(); }
    inline uint64_t page_up(uint64_t v) { return (v + 0xfffull) & ~0xfffull; }

    // Import-stub machine code. Unlike Linux (where host ABI == guest SysV, so the stub is a bare
    // tail-jump with args intact), Windows handlers are Microsoft x64, so the stub is a TRAMPOLINE
    // that converts the guest's SysV integer arguments to MS x64 before calling the handler:
    //   SysV in: rdi rsi rdx rcx r8 r9 [guest rsp+8] [guest rsp+16] [guest rsp+24] (a0..a8)
    //   MS out:  rcx rdx r8 r9 [rsp+20]...[rsp+40] (+32B shadow, stack args a4..a8)
    // Return value is rax in both ABIs. This handles integer/pointer args (the whole HleFn surface);
    // XMM/float args (e.g. some libc formatters) are NOT converted yet — see docs/PORTING.md.
    // The 0x48-byte outgoing area is 8 mod 16, so a normally-entered stub (RSP%16==8) has the required
    // RSP%16==0 at its handler call site. The ABI conformance test verifies args 1-9 and alignment.
    size_t emit_sysv_to_ms_prologue(uint8_t* p) {
        static const uint8_t seq[] = {
            0x48,0x83,0xEC,0x48,              // sub rsp,0x48   (shadow + a4..a8; call rsp%16==0)
            0x48,0x8B,0x44,0x24,0x50,         // mov rax,[rsp+0x50]   ; guest a6 at original rsp+8
            0x48,0x89,0x44,0x24,0x30,         // mov [rsp+0x30],rax  ; MS 7th arg = a6
            0x48,0x8B,0x44,0x24,0x58,         // mov rax,[rsp+0x58]   ; guest a7 at original rsp+16
            0x48,0x89,0x44,0x24,0x38,         // mov [rsp+0x38],rax  ; MS 8th arg = a7
            0x48,0x8B,0x44,0x24,0x60,         // mov rax,[rsp+0x60]   ; guest a8 at original rsp+24
            0x48,0x89,0x44,0x24,0x40,         // mov [rsp+0x40],rax  ; MS 9th arg = a8
            0x4C,0x89,0x44,0x24,0x20,         // mov [rsp+0x20],r8    ; MS 5th arg = a4
            0x4C,0x89,0x4C,0x24,0x28,         // mov [rsp+0x28],r9    ; MS 6th arg = a5
            0x49,0x89,0xC9,                   // mov r9,rcx           ; MS 4th = a3  (save before rcx clobbered)
            0x49,0x89,0xD0,                   // mov r8,rdx           ; MS 3rd = a2  (save before rdx clobbered)
            0x48,0x89,0xF2,                   // mov rdx,rsi          ; MS 2nd = a1
            0x48,0x89,0xF9,                   // mov rcx,rdi          ; MS 1st = a0
        };
        memcpy(p, seq, sizeof seq); return sizeof seq;
    }
    size_t emit_hle_return_checkpoint(uint8_t* p) {
        size_t o = 0;
        p[o++] = 0x48; p[o++] = 0x89; p[o++] = 0x44; p[o++] = 0x24; p[o++] = 0x20;
                                                                        // mov [rsp+0x20],rax
        const uint64_t checkpoint = (uint64_t)(uintptr_t)&dispatch_pending_guest_exception;
        p[o++] = 0x49; p[o++] = 0xBB; memcpy(p + o, &checkpoint, 8); o += 8;
                                                                        // movabs r11,checkpoint
        p[o++] = 0x41; p[o++] = 0xFF; p[o++] = 0xD3;                    // call r11
        p[o++] = 0x48; p[o++] = 0x8B; p[o++] = 0x44; p[o++] = 0x24; p[o++] = 0x20;
                                                                        // mov rax,[rsp+0x20]
        return o;
    }
    void emit_impl(uint8_t* p, uint64_t fn) {
        size_t o = emit_sysv_to_ms_prologue(p);
        p[o++] = 0x48; p[o++] = 0xB8; memcpy(p + o, &fn, 8); o += 8;   // movabs rax,fn
        p[o++] = 0xFF; p[o++] = 0xD0;                                  // call rax
        o += emit_hle_return_checkpoint(p + o);
        p[o++] = 0x48; p[o++] = 0x83; p[o++] = 0xC4; p[o++] = 0x48;    // add rsp,0x48
        p[o++] = 0xC3;                                                 // ret
    }
    void emit_unimpl(uint8_t* p, uint32_t idx, uint64_t fn) {
        size_t o = 0;
        p[o++] = 0x48; p[o++] = 0x83; p[o++] = 0xEC; p[o++] = 0x28;    // sub rsp,0x28 (shadow + align)
        p[o++] = 0x48; p[o++] = 0x8B; p[o++] = 0x54; p[o++] = 0x24; p[o++] = 0x28;
                                                                        // mov rdx,[rsp+0x28] (guest return)
        p[o++] = 0x4C; p[o++] = 0x8D; p[o++] = 0x44; p[o++] = 0x24; p[o++] = 0x28;
                                                                        // lea r8,[rsp+0x28] (guest rsp)
        p[o++] = 0xB9; memcpy(p + o, &idx, 4); o += 4;                 // mov ecx,idx  (MS 1st arg)
        p[o++] = 0x48; p[o++] = 0xB8; memcpy(p + o, &fn, 8); o += 8;   // movabs rax,fn
        p[o++] = 0xFF; p[o++] = 0xD0;                                  // call rax  (prosper_on_unimpl)
        o += emit_hle_return_checkpoint(p + o);
        p[o++] = 0x48; p[o++] = 0x83; p[o++] = 0xC4; p[o++] = 0x28;    // add rsp,0x28
        p[o++] = 0xC3;                                                 // ret
    }

    // Is [addr] readable right now? VirtualQuery-based, so the fault reporter never nests a fault.
    bool addr_readable(uint64_t addr) {
        MEMORY_BASIC_INFORMATION mbi;
        if (!VirtualQuery((LPCVOID)(uintptr_t)addr, &mbi, sizeof mbi)) return false;
        if (mbi.State != MEM_COMMIT) return false;
        DWORD ok = PAGE_READONLY | PAGE_READWRITE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE
                 | PAGE_WRITECOPY | PAGE_EXECUTE_WRITECOPY;
        return (mbi.Protect & ok) != 0 && !(mbi.Protect & PAGE_GUARD);
    }
    void arm_diag_breakpoints() {
        static bool initialized = false;
        if (initialized) return;
        initialized = true;
        const char* spec = getenv("PROSPER_WIN_BP");
        if (!spec || !*spec) return;
        g_diag_bp_repeat = getenv("PROSPER_WIN_BP_REPEAT") != nullptr;
        const char* p = spec;
        while (*p && g_diag_bp_count < sizeof(g_diag_bps) / sizeof(g_diag_bps[0])) {
            char* end = nullptr;
            uint64_t addr = strtoull(p, &end, 0);
            if (end == p) break;
            if (addr < g_base) addr += g_base;  // match Linux PROSPER_BP's eboot-offset convention
            if (addr_readable(addr)) {
                WinDiagBreakpoint& bp = g_diag_bps[g_diag_bp_count++];
                bp.addr = addr;
                bp.original = *(uint8_t*)(uintptr_t)addr;
                *(uint8_t*)(uintptr_t)addr = 0xcc;
                FlushInstructionCache(GetCurrentProcess(), (const void*)(uintptr_t)addr, 1);
                InterlockedExchange(&bp.armed, 1);
                fprintf(stderr, "[winbp] armed at eboot+0x%llx\n",
                        (unsigned long long)(addr - g_base));
            }
            p = end;
            while (*p == ',' || *p == ';' || *p == ' ') ++p;
        }
    }
    bool addr_executable(uint64_t addr) {
        MEMORY_BASIC_INFORMATION mbi;
        if (!VirtualQuery((LPCVOID)(uintptr_t)addr, &mbi, sizeof mbi)) return false;
        if (mbi.State != MEM_COMMIT || (mbi.Protect & PAGE_GUARD)) return false;
        const DWORD protection = mbi.Protect & 0xff;
        return protection == PAGE_EXECUTE || protection == PAGE_EXECUTE_READ ||
               protection == PAGE_EXECUTE_READWRITE || protection == PAGE_EXECUTE_WRITECOPY;
    }

    // Does the instruction at p carry an %fs segment-override prefix (0x64)? Scans the legacy
    // prefixes and REX; stops at the opcode. Used to tell a drifted-guest-%fs fault (Windows zeroed
    // the FS base at a kernel transition) apart from a genuine fault, so we only retry fs accesses.
    bool insn_is_fs_relative(const uint8_t* p) {
        for (int i = 0; i < 8; i++) {
            uint8_t b = p[i];
            if (b == 0x64) return true;                               // fs override
            if (b==0x66||b==0x67||b==0xF0||b==0xF2||b==0xF3||         // opnd/addr size, lock, rep
                b==0x2E||b==0x36||b==0x3E||b==0x26||b==0x65) continue; // other segment overrides
            if ((b & 0xF0) == 0x40) continue;                        // REX
            return false;                                            // opcode reached; no fs prefix
        }
        return false;
    }
    void dump_fault_bytes(uint64_t start, int n) {
        for (int i = 0; i < n; i++) {
            uint64_t a = start + i;
            if (addr_readable(a)) fprintf(stderr, " %02x", *(const uint8_t*)(uintptr_t)a);
            else { fprintf(stderr, " (unmapped@0x%llx)", (unsigned long long)a); return; }
        }
    }
    std::string trap_detail() {
        char b[160];
        snprintf(b, sizeof b, "%s at addr=0x%llx  rip=0x%llx (image+0x%llx)",
                 g_trap_kind == 3 ? "ILLEGAL-INSN" : "ACCESS-VIOLATION",
                 (unsigned long long)g_fault_addr, (unsigned long long)g_fault_rip,
                 (unsigned long long)(g_base && g_fault_rip >= g_base ? g_fault_rip - g_base : g_fault_rip));
        return b;
    }

    // AMD SSE4a INSERTQ/EXTRQ emulation (used by the PS5 guest; #UD on Intel hosts and the Intel ISA
    // exposed by Rosetta). Decode from Rip, mutate the xmm regs in CONTEXT, advance Rip. Mirrors the
    // Linux emulator but reads/writes CONTEXT.Xmm* instead of fpregs. The load-time EXTRQ translator
    // below avoids this live-exception path for primary-image sites on Windows.
    bool try_emulate_sse4a(CONTEXT* c, size_t* decoded_size = nullptr) {
        const uint8_t* p = (const uint8_t*)(uintptr_t)c->Rip;
        if (!addr_readable(c->Rip)) return false;
        size_t i = 0; uint8_t pfx = 0, rex = 0;
        if (p[i] == 0x66 || p[i] == 0xF2) pfx = p[i++];
        if ((p[i] & 0xF0) == 0x40) rex = p[i++];
        if (p[i++] != 0x0F) return false;
        uint8_t op = p[i++];
        if (op != 0x78 && op != 0x79) return false;
        uint8_t modrm = p[i++];
        if ((modrm >> 6) != 3) return false;
        int reg = ((modrm >> 3) & 7) | ((rex & 4) ? 8 : 0);
        int rm  = (modrm & 7)       | ((rex & 1) ? 8 : 0);
        M128A* xmm = &c->FltSave.XmmRegisters[0];   // 16 x {Low, High}
        auto lo  = [&](int n) { return (uint64_t)xmm[n].Low; };
        auto set = [&](int n, uint64_t v) { xmm[n].Low = (ULONGLONG)v; };
        bool insertq = (pfx == 0xF2);
        uint32_t len, idx;
        if (op == 0x78)   { len = p[i++]; idx = p[i++]; }
        else if (insertq) { uint64_t cc = (uint64_t)xmm[rm].High; len = (uint32_t)(cc & 0x3f); idx = (uint32_t)((cc >> 8) & 0x3f); }
        else              { uint64_t cc = (uint64_t)xmm[rm].Low;  len = (uint32_t)(cc & 0x3f); idx = (uint32_t)((cc >> 8) & 0x3f); }
        len &= 0x3f; idx &= 0x3f;
        if (insertq) set(reg, sse4a_insertq(lo(reg), lo(rm), len, idx));
        else {
            int dr = (op == 0x78) ? rm : reg;
            set(dr, sse4a_extrq(lo(dr), len, idx));
            // AMD documents EXTRQ's upper qword as undefined. Preserve it: that matches the AMD CPU
            // model used by QEMU and the behavior prosper's established trap emulators exposed to
            // guests. Astro's packed-vector kernels keep live data in the neighboring lane.
        }
        c->Rip += i;
        if (decoded_size) *decoded_size = i;
        return true;
    }

    bool sse4a_fastpath_site(uint64_t site) {
        const LONG count = InterlockedCompareExchange(&g_sse4a_fastpath_count, 0, 0);
        for (LONG i = 0; i < count; ++i)
            if (g_sse4a_fastpath_sites[i] == site) return true;
        return false;
    }

    // Return the exact register-form EXTRQ length that the code-cache expansion implements. The
    // instruction has a mandatory 66 prefix, an optional REX byte, 0F 79, and register-only ModRM.
    // Immediate forms and INSERTQ stay in the VEH emulator.
    size_t sse4a_fastpath_candidate_bytes(const uint8_t* p) {
        if (p[0] != 0x66) return 0;
        if ((p[1] & 0xf0) == 0x40)
            return p[2] == 0x0f && p[3] == 0x79 && (p[4] >> 6) == 3 ? 5 : 0;
        return p[1] == 0x0f && p[2] == 0x79 && (p[3] >> 6) == 3 ? 4 : 0;
    }

    size_t sse4a_fastpath_candidate(uint64_t site) {
        if (!addr_readable(site) || !addr_readable(site + 4)) return 0;
        return sse4a_fastpath_candidate_bytes((const uint8_t*)(uintptr_t)site);
    }

    // Minimal fail-closed decoder for the location-independent AVX instruction immediately after
    // a four-byte EXTRQ. It accepts only the register/stack-memory VEX forms exercised by the guest
    // unpack kernels; RIP-relative memory, branches, and unknown opcodes cannot be stolen.
    size_t patchable_vex_length(uint64_t site) {
        if (!addr_readable(site) || !addr_readable(site + 14)) return 0;
        const uint8_t* p = (const uint8_t*)(uintptr_t)site;
        size_t i = 0;
        if (p[0] == 0xc5) i = 2;
        else if (p[0] == 0xc4) i = 3;
        else return 0;
        const uint8_t opcode = p[i++];
        switch (opcode) {
            case 0x62: case 0x6a: case 0x6f: case 0x73: case 0x7f: case 0xf3: break;
            default: return 0;
        }
        const uint8_t modrm = p[i++];
        const unsigned mod = modrm >> 6;
        const unsigned rm = modrm & 7;
        if (mod != 3) {
            if (mod == 0 && rm == 5) return 0; // RIP-relative: copying would change its address
            if (rm == 4) {
                const uint8_t sib = p[i++];
                if (mod == 0 && (sib & 7) == 5) i += 4;
            }
            if (mod == 1) ++i;
            else if (mod == 2) i += 4;
        }
        if (opcode == 0x73) ++i;
        return i <= 15 ? i : 0;
    }

    bool emit_sse4a_fastpath(uint64_t site, size_t instruction_size,
                             const uint8_t* stolen = nullptr, size_t stolen_size = 0,
                             const uint8_t* second_sse4a = nullptr,
                             size_t second_sse4a_size = 0) {
        const bool chains_sse4a = instruction_size == 4 && second_sse4a_size;
        const bool consumes_vex = instruction_size == 4 && stolen_size;
        const bool consumes_successor = chains_sse4a || consumes_vex;
        uint8_t* selected_cache = chains_sse4a ? g_sse4a_chain_cache :
            (consumes_vex ? g_sse4a_successor_cache : g_sse4a_fastpath_cache);
        size_t& selected_used = chains_sse4a ? g_sse4a_chain_used :
            (consumes_vex ? g_sse4a_successor_used : g_sse4a_fastpath_used);
        const LONG records_needed = consumes_successor ? 2 : 1;
        if (!selected_cache ||
            selected_used + kSse4aFastpathSlotSize > kSse4aFastpathCacheSize ||
            InterlockedCompareExchange(&g_sse4a_fastpath_count, 0, 0) >
                (LONG)kSse4aFastpathMaxPatches - records_needed)
            return false;
        const uint8_t* instruction = (const uint8_t*)(uintptr_t)site;
        const uint8_t rex = instruction_size == 5 ? instruction[1] : 0;
        const uint8_t modrm = instruction[instruction_size - 1];
        const unsigned destination = ((modrm >> 3) & 7u) | ((rex & 4u) ? 8u : 0u);
        const unsigned control = (modrm & 7u) | ((rex & 1u) ? 8u : 0u);
        uint8_t* main_code = selected_cache + selected_used;
        const uintptr_t cache_page = (uintptr_t)main_code & ~(uintptr_t)0xfff;
        DWORD old_cache_protection = 0;
        if (!VirtualProtect((void*)cache_page, 0x1000, PAGE_EXECUTE_READWRITE,
                            &old_cache_protection))
            return false;
        selected_used += kSse4aFastpathSlotSize;
        uint8_t* code = main_code;
        size_t o = 0;
        auto byte = [&](uint8_t value) { code[o++] = value; };
        auto movq_xmm_to_gpr = [&](unsigned xmm, unsigned gpr) {
            byte(0x66); byte((uint8_t)(0x48 | (xmm >= 8 ? 4 : 0)));
            byte(0x0f); byte(0x7e); byte((uint8_t)(0xc0 | ((xmm & 7) << 3) | gpr));
        };
        auto pinsrq_gpr_to_xmm = [&](unsigned xmm, unsigned gpr) {
            // PINSRQ replaces only qword 0. Legacy MOVQ would clear qword 1, unlike the AMD CPU
            // model and prosper's established trap-emulation behavior.
            byte(0x66); byte((uint8_t)(0x48 | (xmm >= 8 ? 4 : 0) | (gpr >= 8 ? 1 : 0)));
            byte(0x0f); byte(0x3a); byte(0x22);
            byte((uint8_t)(0xc0 | ((xmm & 7) << 3) | (gpr & 7)));
            byte(0x00);
        };
        auto emit_extract = [&](unsigned value_xmm, unsigned control_xmm) {
            movq_xmm_to_gpr(value_xmm, 0);               // rax = value to extract from
            movq_xmm_to_gpr(control_xmm, 1);             // rcx = {index,length} control
            byte(0x89); byte(0xca);                      // mov edx,ecx (length)
            byte(0xc1); byte(0xe9); byte(0x08);          // shr ecx,8 (index)
            byte(0x83); byte(0xe1); byte(0x3f);          // and ecx,63
            byte(0x48); byte(0xd3); byte(0xe8);          // shr rax,cl
            byte(0x83); byte(0xe2); byte(0x3f);          // and edx,63
            byte(0x85); byte(0xd2);                      // test edx,edx (zero means 64)
            byte(0x74); const size_t no_mask_jump = o++; // jz .no_mask
            byte(0x89); byte(0xd1);                      // mov ecx,edx
            byte(0x48); byte(0xc7); byte(0xc2);          // mov rdx,-1
            byte(0xff); byte(0xff); byte(0xff); byte(0xff);
            byte(0x48); byte(0xd3); byte(0xe2);          // shl rdx,cl
            byte(0x48); byte(0xf7); byte(0xd2);          // not rdx
            byte(0x48); byte(0x21); byte(0xd0);          // and rax,rdx
            code[no_mask_jump] = (uint8_t)(o - (no_mask_jump + 1));
            pinsrq_gpr_to_xmm(value_xmm, 0);
        };
        auto emit_prologue = [&] {
            // The guest follows the SysV ABI and may keep live leaf-function data in the 128-byte
            // red zone below RSP. Move below it without changing flags before spilling scratch.
            byte(0x48); byte(0x8d); byte(0x64); byte(0x24); byte(0x80); // lea rsp,[rsp-128]
            byte(0x9c); byte(0x50); byte(0x51); byte(0x52); // pushfq; push rax/rcx/rdx
        };
        auto emit_epilogue = [&] {
            byte(0x5a); byte(0x59); byte(0x58); byte(0x9d); // pop rdx/rcx/rax; popfq
            byte(0x48); byte(0x8d); byte(0xa4); byte(0x24); // lea rsp,[rsp+128]
            byte(0x80); byte(0x00); byte(0x00); byte(0x00);
        };
        unsigned second_destination = 0, second_control = 0;
        if (second_sse4a_size) {
            const uint8_t second_rex = second_sse4a_size == 5 ? second_sse4a[1] : 0;
            const uint8_t second_modrm = second_sse4a[second_sse4a_size - 1];
            second_destination =
                ((second_modrm >> 3) & 7u) | ((second_rex & 4u) ? 8u : 0u);
            second_control =
                (second_modrm & 7u) | ((second_rex & 1u) ? 8u : 0u);
        }
        emit_prologue();
        emit_extract(destination, control);
        emit_epilogue();
        if (stolen && stolen_size) {
            memcpy(code + o, stolen, stolen_size);
            o += stolen_size;
        }
        byte(0xe9);
        const size_t replaced_size = instruction_size + stolen_size + second_sse4a_size;
        const uint64_t main_target = second_sse4a_size
            ? (uint64_t)(uintptr_t)(main_code + kSse4aSecondaryOffset)
            : site + replaced_size;
        const int64_t back_delta = (int64_t)main_target -
                                   (int64_t)((uint64_t)(uintptr_t)code + o + 4);
        if (back_delta < std::numeric_limits<int32_t>::min() ||
            back_delta > std::numeric_limits<int32_t>::max())
            return false;
        const int32_t back_rel = (int32_t)back_delta;
        memcpy(code + o, &back_rel, sizeof(back_rel)); o += sizeof(back_rel);
        const size_t main_size = o;
        uint8_t* secondary_code = nullptr;
        size_t secondary_size = 0;
        if (consumes_successor) {
            if (main_size > kSse4aSecondaryOffset) return false;
            secondary_code = main_code + kSse4aSecondaryOffset;
            code = secondary_code;
            o = 0;
            if (second_sse4a_size) {
                emit_prologue();
                emit_extract(second_destination, second_control);
                emit_epilogue();
            } else {
                memcpy(code + o, stolen, stolen_size);
                o += stolen_size;
            }
            byte(0xe9);
            const int64_t secondary_back_delta = (int64_t)(site + replaced_size) -
                (int64_t)((uint64_t)(uintptr_t)code + o + 4);
            if (secondary_back_delta < std::numeric_limits<int32_t>::min() ||
                secondary_back_delta > std::numeric_limits<int32_t>::max())
                return false;
            const int32_t secondary_back_rel = (int32_t)secondary_back_delta;
            memcpy(code + o, &secondary_back_rel, sizeof(secondary_back_rel));
            o += sizeof(secondary_back_rel);
            secondary_size = o;
            if (kSse4aSecondaryOffset + secondary_size > kSse4aFastpathSlotSize)
                return false;
        } else if (main_size > kSse4aFastpathSlotSize) {
            return false;
        }

        const int64_t detour_delta =
            (int64_t)(uint64_t)(uintptr_t)main_code - (int64_t)(site + 5);
        if (detour_delta < std::numeric_limits<int32_t>::min() ||
            detour_delta > std::numeric_limits<int32_t>::max())
            return false;
        const int32_t detour_rel = (int32_t)detour_delta;
        // The high displacement byte is also the instruction at site+4. Chained SSE uses E9 plus
        // the following four bytes as a second near jump. A copied VEX successor uses EB to reach
        // its permanent secondary-entry marker inside the shorter consumed span.
        const uint8_t overlap_opcode = chains_sse4a ? 0xe9 : 0xeb;
        if (consumes_successor && ((uint32_t)detour_rel >> 24) != overlap_opcode) return false;
        int32_t secondary_detour_rel = 0;
        if (chains_sse4a) {
            const int64_t secondary_detour_delta =
                (int64_t)(uint64_t)(uintptr_t)secondary_code - (int64_t)(site + 9);
            if (secondary_detour_delta < std::numeric_limits<int32_t>::min() ||
                secondary_detour_delta > std::numeric_limits<int32_t>::max())
                return false;
            secondary_detour_rel = (int32_t)secondary_detour_delta;
        }
        FlushInstructionCache(GetCurrentProcess(), main_code, main_size);
        if (secondary_code)
            FlushInstructionCache(GetCurrentProcess(), secondary_code, secondary_size);
        auto record_site = [&](uint64_t recorded_site) {
            const LONG record =
                InterlockedCompareExchange(&g_sse4a_fastpath_count, 0, 0);
            g_sse4a_fastpath_sites[record] = recorded_site;
            MemoryBarrier();
            InterlockedIncrement(&g_sse4a_fastpath_count);
        };
        record_site(site);
        if (consumes_successor) {
            record_site(site + instruction_size);
            if (consumes_vex) {
                const LONG redirect =
                    InterlockedCompareExchange(&g_sse4a_secondary_count, 0, 0);
                g_sse4a_secondary_redirects[redirect] =
                    {site + 6, (uint64_t)(uintptr_t)secondary_code};
                MemoryBarrier();
                InterlockedIncrement(&g_sse4a_secondary_count);
            }
        }

        // An INT3 gate prevents another core from observing a half-written near jump. A thread that
        // arrives during these stores waits in the VEH breakpoint path below, then replays the site.
        if (consumes_successor) {
            *(volatile uint8_t*)(uintptr_t)(site + instruction_size) = 0xcc;
            FlushInstructionCache(GetCurrentProcess(),
                                  (const void*)(uintptr_t)(site + instruction_size), 1);
        }
        *(volatile uint8_t*)(uintptr_t)site = 0xcc;
        FlushInstructionCache(GetCurrentProcess(), (const void*)(uintptr_t)site, 1);
        if (chains_sse4a) {
            // E9 <low 24 bits> E9 <secondary rel32>: both the original four-byte EXTRQ entry and a
            // direct branch to its five-byte successor reach the correct exception-free expansion.
            memcpy((void*)(uintptr_t)(site + 1), &detour_rel, 3);
            memcpy((void*)(uintptr_t)(site + 5), &secondary_detour_rel,
                   sizeof(secondary_detour_rel));
            *(volatile uint8_t*)(uintptr_t)(site + instruction_size) = 0xe9;
        } else if (consumes_vex) {
            // E9 <low 24 bits> EB is the main near jump. Entry at +4 executes `EB 00`, reaches the
            // permanent INT3 marker at +6, and VEH redirects to the second-only expansion.
            memcpy((void*)(uintptr_t)(site + 1), &detour_rel, 3);
            *(volatile uint8_t*)(uintptr_t)(site + 5) = 0x00;
            *(volatile uint8_t*)(uintptr_t)(site + 6) = 0xcc;
            for (size_t i = 7; i < replaced_size; ++i)
                *(volatile uint8_t*)(uintptr_t)(site + i) = 0x90;
            *(volatile uint8_t*)(uintptr_t)(site + instruction_size) = 0xeb;
        } else {
            memcpy((void*)(uintptr_t)(site + 1), &detour_rel, sizeof(detour_rel));
            for (size_t i = 5; i < replaced_size; ++i)
                *(volatile uint8_t*)(uintptr_t)(site + i) = 0x90;
        }
        *(volatile uint8_t*)(uintptr_t)site = 0xe9;
        FlushInstructionCache(GetCurrentProcess(), (const void*)(uintptr_t)site, replaced_size);
        DWORD ignored_cache_protection = 0;
        if (!VirtualProtect((void*)cache_page, 0x1000, PAGE_EXECUTE_READ,
                            &ignored_cache_protection))
            return false;
        if (getenv("PROSPER_SSE4A_LOG"))
            fprintf(stderr, "[sse4a] fastpath eboot+0x%llx xmm%u,xmm%u span=%zu cache=%p\n",
                    (unsigned long long)(site - g_base), destination, control,
                    replaced_size, main_code);
        return true;
    }

    bool try_handle_sse4a_fastpath_breakpoint(CONTEXT* c, uint64_t exception_address) {
        uint64_t breakpoint = exception_address;
        const LONG redirects = InterlockedCompareExchange(&g_sse4a_secondary_count, 0, 0);
        for (LONG i = 0; i < redirects; ++i) {
            const Sse4aSecondaryRedirect& redirect = g_sse4a_secondary_redirects[i];
            if (breakpoint != redirect.marker && (!c->Rip || c->Rip - 1 != redirect.marker))
                continue;
            c->Rip = redirect.target;
            return true;
        }
        uint64_t site = exception_address;
        if (!sse4a_fastpath_site(site) && c->Rip && sse4a_fastpath_site(c->Rip - 1))
            site = c->Rip - 1;
        if (!sse4a_fastpath_site(site)) return false;
        while (*(volatile uint8_t*)(uintptr_t)site == 0xcc) YieldProcessor();
        c->Rip = site;
        return true;
    }

    // Windows constructs an exception-dispatch frame below the interrupted RSP. PS5 code follows
    // the SysV ABI and may keep live values in the 128-byte red zone there, so discovering SSE4a
    // through a live #UD can corrupt guest locals before VEH is entered. Translate the narrowly
    // recognized register-form EXTRQ sequences while the primary image is still single-threaded.
    // Five-byte forms are self-contained. Four-byte forms are accepted only when the already
    // fail-closed successor decoder can preserve the instruction that supplies the fifth detour byte.
    size_t prepatch_sse4a(const LoadedImage& img) {
        size_t patched = 0;
        for (const LoadedImage::Prot& protection : img.prot) {
            if (!protection.x || !protection.size) continue;
            const uint64_t mapped_begin = img.base + img.min_vaddr;
            const uint64_t mapped_end = mapped_begin + img.mem.size();
            uint64_t begin = img.base + protection.vaddr;
            uint64_t end = begin + protection.size;
            if (begin < mapped_begin) begin = mapped_begin;
            if (end > mapped_end) end = mapped_end;
            if (end <= begin || end - begin < 5) continue;
            for (uint64_t site = begin; site + 5 <= end; ++site) {
                const size_t instruction_size = sse4a_fastpath_candidate_bytes(
                    (const uint8_t*)(uintptr_t)site);
                if (!instruction_size) continue;
                bool emitted = false;
                size_t consumed = instruction_size;
                if (instruction_size == 5) {
                    emitted = emit_sse4a_fastpath(site, instruction_size);
                } else {
                    const size_t stolen_size = patchable_vex_length(site + instruction_size);
                    const size_t chained_size = !stolen_size &&
                        site + instruction_size + 5 <= end &&
                        sse4a_fastpath_candidate_bytes(
                            (const uint8_t*)(uintptr_t)(site + instruction_size)) == 5 ? 5 : 0;
                    if (stolen_size) {
                        emitted = emit_sse4a_fastpath(
                            site, instruction_size,
                            (const uint8_t*)(uintptr_t)(site + instruction_size),
                            stolen_size);
                        consumed += stolen_size;
                    } else if (chained_size) {
                        emitted = emit_sse4a_fastpath(
                            site, instruction_size, nullptr, 0,
                            (const uint8_t*)(uintptr_t)(site + instruction_size),
                            chained_size);
                        consumed += chained_size;
                    }
                }
                if (!emitted) continue;
                ++patched;
                site += consumed - 1;
            }
        }
        if (patched || getenv("PROSPER_SSE4A_LOG"))
            fprintf(stderr, "[sse4a] prepatched %zu EXTRQ sequences before guest entry\n",
                    patched);
        return patched;
    }

    // The VEH resumes at this assembly thunk instead of entering a compiled function directly. Its
    // entry contract is deliberately not a normal call frame: RSP is 16-byte aligned and there is no
    // return address. The thunk allocates the Microsoft-x64 32-byte home area, then makes a normal,
    // aligned call into the compiled helper. __builtin_longjmp never returns across the foreign guest
    // frames, so the missing caller frame is immaterial. The recorded alignment makes the contract
    // regression-testable on Windows rather than relying on current compiler tolerance (#633).
    extern "C" {
        volatile uint64_t prosper_veh_recover_call_rsp_mod16 = ~uint64_t{0};
        void prosper_veh_recover_thunk();
    }
    extern "C" __attribute__((noinline, noreturn)) void prosper_veh_recover_longjmp() {
        __builtin_longjmp(t_jb, 1);
        __builtin_unreachable();
    }
    __asm__(
        ".text\n"
        ".p2align 4\n"
        ".globl prosper_veh_recover_thunk\n"
        "prosper_veh_recover_thunk:\n"
        "    subq $32, %rsp\n"       // MS-x64 shadow space; preserves 16-byte call-site alignment
        "    movq %rsp, %rax\n"
        "    andq $15, %rax\n"
        "    movq %rax, prosper_veh_recover_call_rsp_mod16(%rip)\n"
        "    callq prosper_veh_recover_longjmp\n"
        "    ud2\n"
    );

    // Tracked-mapping state probe for the lazy-commit fault path (defined in hle_kernel_mem.cpp):
    // 0 = untracked/gap, 1 = reserved, 2 = committed, 3 = sparse direct-memory page.
    extern "C" int prosper_reserved_range_state(uint64_t addr);
    extern "C" int prosper_try_commit_dmem(uint64_t addr, uint64_t len, int write);

    LONG CALLBACK veh(EXCEPTION_POINTERS* ep) {
        CONTEXT* c = ep->ContextRecord;
        DWORD code = ep->ExceptionRecord->ExceptionCode;
        if (code == EXCEPTION_SINGLE_STEP && t_diag_bp_rearm) {
            WinDiagBreakpoint* bp = t_diag_bp_rearm;
            *(uint8_t*)(uintptr_t)bp->addr = 0xcc;
            FlushInstructionCache(GetCurrentProcess(), (const void*)(uintptr_t)bp->addr, 1);
            InterlockedExchange(&bp->armed, 1);
            t_diag_bp_rearm = nullptr;
            c->EFlags &= ~0x100ull;
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if (code == EXCEPTION_BREAKPOINT) {
            if (try_handle_sse4a_fastpath_breakpoint(
                    c, (uint64_t)(uintptr_t)ep->ExceptionRecord->ExceptionAddress))
                return EXCEPTION_CONTINUE_EXECUTION;
            for (size_t i = 0; i < g_diag_bp_count; ++i) {
                WinDiagBreakpoint& bp = g_diag_bps[i];
                if ((c->Rip != bp.addr && c->Rip != bp.addr + 1) ||
                    InterlockedExchange(&bp.armed, 0) != 1) continue;
                const uint64_t addr = bp.addr;
                *(uint8_t*)(uintptr_t)addr = bp.original;
                FlushInstructionCache(GetCurrentProcess(), (const void*)(uintptr_t)addr, 1);
                fprintf(stderr, "[winbp] hit eboot+0x%llx tid=%lu rax=%llx rbx=%llx rcx=%llx "
                                "rdx=%llx rsi=%llx rdi=%llx r8=%llx r9=%llx r12=%llx r13=%llx "
                                "r14=%llx r15=%llx rbp=%llx rsp=%llx\n",
                        (unsigned long long)(addr - g_base), (unsigned long)cur_tid(),
                        (unsigned long long)c->Rax, (unsigned long long)c->Rbx,
                        (unsigned long long)c->Rcx, (unsigned long long)c->Rdx,
                        (unsigned long long)c->Rsi, (unsigned long long)c->Rdi,
                        (unsigned long long)c->R8, (unsigned long long)c->R9,
                        (unsigned long long)c->R12, (unsigned long long)c->R13,
                        (unsigned long long)c->R14, (unsigned long long)c->R15,
                        (unsigned long long)c->Rbp, (unsigned long long)c->Rsp);
                auto qword = [&](uint64_t p) -> uint64_t {
                    return p && addr_readable(p) && addr_readable(p + 7)
                        ? *(const uint64_t*)(uintptr_t)p : 0;
                };
                fprintf(stderr, "[winbp] mem [rdx]=%llx [rdx+10]=%llx [r8]=%llx [r9]=%llx "
                                "[r12]=%llx [r13]=%llx\n",
                        (unsigned long long)qword(c->Rdx), (unsigned long long)qword(c->Rdx + 0x10),
                        (unsigned long long)qword(c->R8), (unsigned long long)qword(c->R9),
                        (unsigned long long)qword(c->R12), (unsigned long long)qword(c->R13));
                auto ascii = [&](uint64_t p, char out[33]) {
                    size_t n = 0;
                    while (n < 32 && addr_readable(p + n)) {
                        unsigned char ch = *(const unsigned char*)(uintptr_t)(p + n);
                        if (!ch) break;
                        if (ch < 0x20 || ch > 0x7e) { n = 0; break; }
                        out[n++] = (char)ch;
                    }
                    out[n] = 0;
                };
                char s12[33], s15[33], s9[33], s_node[33];
                ascii(c->R12, s12); ascii(c->R15, s15); ascii(c->R9, s9);
                ascii(qword(c->Rdx + 0x10), s_node);
                fprintf(stderr,
                        "[winbp] str r12=\"%s\" r15=\"%s\" r9=\"%s\" [rdx+10]=\"%s\"\n",
                        s12, s15, s9, s_node);
                if (g_diag_bp_repeat) {
                    t_diag_bp_rearm = &bp;
                    c->EFlags |= 0x100ull;
                }
                c->Rip = addr;  // replay the restored instruction
                return EXCEPTION_CONTINUE_EXECUTION;
            }
        }
        if (code == EXCEPTION_ILLEGAL_INSTRUCTION) {
            const uint64_t site = c->Rip;
            size_t candidate_size = sse4a_fastpath_candidate(site);
            size_t stolen_size = 0;
            size_t chained_sse4a_size = 0;
            if (candidate_size == 4)
                stolen_size = patchable_vex_length(site + candidate_size);
            if (candidate_size == 4 && !stolen_size)
                chained_sse4a_size =
                    sse4a_fastpath_candidate(site + candidate_size) == 5 ? 5 : 0;
            if (candidate_size == 5 || stolen_size || chained_sse4a_size ||
                sse4a_fastpath_site(site)) {
                while (InterlockedCompareExchange(&g_sse4a_fastpath_lock, 1, 0) != 0)
                    YieldProcessor();
                // Another faulting thread may have installed this detour while we waited. Replay the
                // now-patched site with the original register context instead of emulating it twice.
                if (sse4a_fastpath_site(site)) {
                    InterlockedExchange(&g_sse4a_fastpath_lock, 0);
                    return EXCEPTION_CONTINUE_EXECUTION;
                }
                candidate_size = sse4a_fastpath_candidate(site);
                stolen_size = candidate_size == 4
                    ? patchable_vex_length(site + candidate_size) : 0;
                chained_sse4a_size = candidate_size == 4 && !stolen_size &&
                    sse4a_fastpath_candidate(site + candidate_size) == 5 ? 5 : 0;
                bool emulated = false;
                if (candidate_size == 4 && stolen_size &&
                    emit_sse4a_fastpath(site, candidate_size,
                                        (const uint8_t*)(uintptr_t)(site + candidate_size),
                                        stolen_size)) {
                    // The detour expansion performs both the EXTRQ and copied successor. Replay it
                    // from the untouched fault context; resuming at site+4 would land inside E9.
                    c->Rip = site;
                    emulated = true;
                } else if (candidate_size == 4 && chained_sse4a_size &&
                           emit_sse4a_fastpath(
                               site, candidate_size, nullptr, 0,
                               (const uint8_t*)(uintptr_t)(site + candidate_size),
                               chained_sse4a_size)) {
                    // Keep the two validated single-EXTRQ expansions separate. The first jumps to
                    // the second-only entry, which then returns after the consumed instruction.
                    c->Rip = site;
                    emulated = true;
                } else {
                    size_t decoded_size = 0;
                    emulated = try_emulate_sse4a(c, &decoded_size);
                    if (emulated && decoded_size == 5)
                        emit_sse4a_fastpath(site, decoded_size);
                }
                InterlockedExchange(&g_sse4a_fastpath_lock, 0);
                if (emulated) return EXCEPTION_CONTINUE_EXECUTION;
            } else if (try_emulate_sse4a(c)) {
                return EXCEPTION_CONTINUE_EXECUTION;
            }
        }
        // Lazy-commit inside a guest-RESERVED range — parity with the Linux SIGSEGV handler. The guest
        // reserves a virtual range then touches pages it believes committed (its binned allocator relies
        // on a commit protocol real HW satisfies but our HLE doesn't fully replicate). If the faulting
        // read/write address is one the memory HLE tracks as reserved-but-uncommitted, commit the 16 KiB
        // guest page and retry. Windows reservations can be only one guest page long; committing a
        // 64 KiB allocation-granularity span would cross that reservation and fail with ERROR_INVALID_ADDRESS.
        // Gated to tracked-reserved addresses (state 1) and non-execute accesses
        // (ExceptionInformation[0]!=8) so a genuine wild access / bad instruction fetch still faults.
        if (code == EXCEPTION_ACCESS_VIOLATION && ep->ExceptionRecord->NumberParameters >= 2 &&
            ep->ExceptionRecord->ExceptionInformation[0] != 8) {
            uint64_t a = (uint64_t)ep->ExceptionRecord->ExceptionInformation[1];
            if (ep->ExceptionRecord->ExceptionInformation[0] == 1 &&
                host::guest_write_watch_handle_fault(a))
                return EXCEPTION_CONTINUE_EXECUTION;
            // Direct-memory views are allowed below the historical 64 GiB GPU-VA window. Astro's
            // texture-streaming allocator, for example, receives a sparse view near 12 GiB and can
            // place a libc heap object across a 16 KiB page boundary there. The helper performs the
            // strict tracked-view/protection checks, so an address floor would only reject a valid
            // first touch and turn it into an unhandled access violation.
            if (prosper_try_commit_dmem(
                    a, 1, ep->ExceptionRecord->ExceptionInformation[0] == 1))
                return EXCEPTION_CONTINUE_EXECUTION;
            if (a >= 0x1000000000ull && prosper_reserved_range_state(a) == 1) {
                void* page = (void*)(uintptr_t)(a & ~(uint64_t)0x3fff);
                if (VirtualAlloc(page, 0x4000, MEM_COMMIT, PAGE_READWRITE))
                    return EXCEPTION_CONTINUE_EXECUTION;   // re-execute against the now-committed page
            }
        }
        // Guest %fs base drift: Windows zeroes the user FS base on every kernel transition, so a
        // guest fs-relative TLS access can fault at a low linear address after a context switch /
        // syscall even though the guest TCB is fine. If the faulting instruction is fs-relative and
        // our guest TP just needs re-applying, do so and retry — costing one fault per kernel-transition
        // boundary, not per access. Runs BEFORE the t_armed gate so guest worker threads get it too.
        if (code == EXCEPTION_ACCESS_VIOLATION && addr_readable(c->Rip) &&
            insn_is_fs_relative((const uint8_t*)(uintptr_t)c->Rip) && guest_fs_reapply())
            return EXCEPTION_CONTINUE_EXECUTION;
        if (code != EXCEPTION_ACCESS_VIOLATION && code != EXCEPTION_ILLEGAL_INSTRUCTION &&
            code != EXCEPTION_IN_PAGE_ERROR && code != EXCEPTION_PRIV_INSTRUCTION &&
            code != EXCEPTION_DATATYPE_MISALIGNMENT)
            return EXCEPTION_CONTINUE_SEARCH;   // not a guest fault we own
        // PROSPER_VEHLOG: surface a fault we're about to decline (unarmed thread → the process will die
        // without a RUN ENDED report). This is the only visibility into a guest worker-thread crash.
        if (getenv("PROSPER_VEHLOG")) {
            uint64_t fa = (ep->ExceptionRecord->NumberParameters >= 2)
                        ? (uint64_t)ep->ExceptionRecord->ExceptionInformation[1] : 0;
            fprintf(stderr, "[veh] tid=%lu code=0x%lx rip=0x%llx addr=0x%llx armed=%d rstate=%d\n",
                    (unsigned long)cur_tid(), (unsigned long)code, (unsigned long long)c->Rip,
                    (unsigned long long)fa, (int)t_armed, prosper_reserved_range_state(fa));
            if (!t_armed) {
                fprintf(stderr,
                        "[veh] regs rax=%016llx rbx=%016llx rcx=%016llx rdx=%016llx "
                        "rsi=%016llx rdi=%016llx\n",
                        (unsigned long long)c->Rax, (unsigned long long)c->Rbx,
                        (unsigned long long)c->Rcx, (unsigned long long)c->Rdx,
                        (unsigned long long)c->Rsi, (unsigned long long)c->Rdi);
                fprintf(stderr,
                        "[veh] regs rbp=%016llx rsp=%016llx r8=%016llx r9=%016llx "
                        "r10=%016llx r11=%016llx r12=%016llx r13=%016llx "
                        "r14=%016llx r15=%016llx\n",
                        (unsigned long long)c->Rbp, (unsigned long long)c->Rsp,
                        (unsigned long long)c->R8, (unsigned long long)c->R9,
                        (unsigned long long)c->R10, (unsigned long long)c->R11,
                        (unsigned long long)c->R12, (unsigned long long)c->R13,
                        (unsigned long long)c->R14, (unsigned long long)c->R15);
                fprintf(stderr, "[veh] bytes@rip:");
                dump_fault_bytes(c->Rip, 24);
                fprintf(stderr, "\n[veh] executable stack candidates:");
                unsigned candidates = 0;
                for (uint64_t offset = 0; offset < 0x400; offset += sizeof(uint64_t)) {
                    const uint64_t slot = c->Rsp + offset;
                    if (!addr_readable(slot)) break;
                    const uint64_t value = *(const uint64_t*)(uintptr_t)slot;
                    if (!addr_executable(value)) continue;
                    fprintf(stderr, " +0x%llx=0x%llx", (unsigned long long)offset,
                            (unsigned long long)value);
                    if (++candidates == 24) break;
                }
                fprintf(stderr, "\n");
                dump_guest_exception_trace();
            }
        }
        if (!t_armed) return EXCEPTION_CONTINUE_SEARCH;   // worker thread with no recovery point

        g_trap_kind = (code == EXCEPTION_ILLEGAL_INSTRUCTION) ? 3 : 2;
        g_fault_rip = c->Rip;
        g_fault_addr = (ep->ExceptionRecord->NumberParameters >= 2)
                     ? (uint64_t)ep->ExceptionRecord->ExceptionInformation[1] : 0;
        g_rax=c->Rax; g_rbx=c->Rbx; g_rcx=c->Rcx; g_rdx=c->Rdx; g_rsi=c->Rsi; g_rdi=c->Rdi;
        g_rbp=c->Rbp; g_rsp=c->Rsp; g_r8=c->R8; g_r9=c->R9; g_r10=c->R10; g_r11=c->R11;
        g_r12=c->R12; g_r13=c->R13; g_r14=c->R14; g_r15=c->R15;
        // Resume at the explicit recovery thunk. Its entry contract is RSP%16==0; it creates the
        // MS-x64 shadow space and a normal call frame before entering compiled code.
        c->Rsp &= ~(uint64_t)0xf;
        c->Rip = (DWORD64)(uintptr_t)&prosper_veh_recover_thunk;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
} // namespace

bool map_image(const LoadedImage& img, std::string* err) {
    auto fail = [&](const char* s){ if (err) *err = s; return false; };
    void* want = (void*)(uintptr_t)(img.base + img.min_vaddr);
    size_t sz  = img.mem.size();
    void* got = VirtualAlloc(want, sz, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
    if (!got || got != want) return fail("VirtualAlloc image at guest base failed");
    memcpy(got, img.mem.data(), sz);
    if (!g_base) {
        g_base = img.base;
        const uint64_t cache_address =
            (img.base + img.max_vaddr + 0xffffull) & ~0xffffull;
        g_sse4a_fastpath_cache = (uint8_t*)VirtualAlloc(
            (void*)(uintptr_t)cache_address, kSse4aFastpathCacheSize,
            MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
        if (g_sse4a_fastpath_cache != (void*)(uintptr_t)cache_address) {
            if (g_sse4a_fastpath_cache)
                VirtualFree(g_sse4a_fastpath_cache, 0, MEM_RELEASE);
            g_sse4a_fastpath_cache = nullptr;
            fprintf(stderr, "[sse4a] trap-free cache unavailable; using VEH emulation\n");
        }
        // Four-byte EXTRQ needs to consume its successor to make room for a five-byte detour.
        // Put successor-consuming expansions where the detour's high displacement byte is 0xEB.
        // That byte doubles as a short-jump entry for a direct branch to the consumed successor;
        // see the secondary-entry marker installed by emit_sse4a_fastpath.
        if (img.base > 0x14800000ull) {
            const uint64_t successor_cache_address =
                (img.base - 0x14800000ull) & ~0xffffull;
            g_sse4a_successor_cache = (uint8_t*)VirtualAlloc(
                (void*)(uintptr_t)successor_cache_address, kSse4aFastpathCacheSize,
                MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
            if (g_sse4a_successor_cache != (void*)(uintptr_t)successor_cache_address) {
                if (g_sse4a_successor_cache)
                    VirtualFree(g_sse4a_successor_cache, 0, MEM_RELEASE);
                g_sse4a_successor_cache = nullptr;
            }
        }
        if (img.base > 0x16800000ull) {
            const uint64_t chain_cache_address =
                (img.base - 0x16800000ull) & ~0xffffull;
            g_sse4a_chain_cache = (uint8_t*)VirtualAlloc(
                (void*)(uintptr_t)chain_cache_address, kSse4aFastpathCacheSize,
                MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
            if (g_sse4a_chain_cache != (void*)(uintptr_t)chain_cache_address) {
                if (g_sse4a_chain_cache)
                    VirtualFree(g_sse4a_chain_cache, 0, MEM_RELEASE);
                g_sse4a_chain_cache = nullptr;
            }
        }
        prepatch_sse4a(img);
    }
    return true;
}

bool install_stubs(const std::vector<ImportSlot>& slots, uint64_t stub_base,
                   uint64_t stub_size, std::string* err) {
    auto fail = [&](const char* s){ if (err) *err = s; return false; };
    if (stub_size < 96) return fail("stub_size too small for Windows ABI bridge (need >= 96)");
    if (!g_nid_db) g_nid_db = new NidDb();
    dispatch_init(&slots, g_nid_db);

    uint64_t n = slots.size();
    if (n == 0) { g_stub_base = stub_base; g_stub_size = stub_size; g_nstubs = 0; return true; }
    uint64_t region = page_up(n * stub_size);
    void* got = VirtualAlloc((void*)(uintptr_t)stub_base, region, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
    if (!got || got != (void*)(uintptr_t)stub_base) return fail("VirtualAlloc stub region failed");

    uint8_t* base = (uint8_t*)got;
    for (uint64_t i = 0; i < n; i++) {
        uint8_t* slot = base + i * stub_size;
        HleFn fn = Hle::lookup(slots[i].nid);
        if (fn) emit_impl(slot, (uint64_t)fn);
        else    emit_unimpl(slot, (uint32_t)i, (uint64_t)&prosper_on_unimpl);
    }
    g_stub_base = stub_base; g_stub_size = stub_size; g_nstubs = n;
    // Keep the stack-address-to-import diagnostic available on every execution host. The fingerprint
    // tool reports return addresses inside this table; index/off/name output makes those frames useful
    // without relying on a Linux-only boot.
    if (getenv("PROSPER_STUBDUMP")) {
        for (uint64_t i = 0; i < n; i++) {
            const std::string& nm = g_nid_db ? g_nid_db->resolve(slots[i].nid) : std::string();
            fprintf(stderr, "[stub] #%llu off=0x%llx %s::%s %s\n", (unsigned long long)i,
                    (unsigned long long)(i * stub_size), slots[i].lib.c_str(), slots[i].nid.c_str(), nm.c_str());
        }
    }
    return true;
}

void install_sigaltstack() {}   // Windows delivers exceptions on the runtime's own guard stack

void install_trap_handler() {
    static bool installed = false;
    if (installed) return;
    installed = true;
    AddVectoredExceptionHandler(1 /*first*/, veh);
}

int recovery_thunk_call_rsp_mod16() {
    return prosper_veh_recover_call_rsp_mod16 == ~uint64_t{0}
         ? -1 : (int)prosper_veh_recover_call_rsp_mod16;
}

uint64_t sse4a_fastpath_patch_count() {
    return (uint64_t)InterlockedCompareExchange(&g_sse4a_fastpath_count, 0, 0);
}

// Diagnostics that need Linux perf_event / int3 patching are unavailable on Windows.
void arm_hwbp_this_thread() {}

uint64_t stub_addr(uint64_t idx) { return g_stub_base + idx * g_stub_size; }
uint64_t hle_guest_return_address(uint64_t entry_rsp) {
    if (!entry_rsp) return 0;
    const uint64_t immediate = *(const uint64_t*)(uintptr_t)entry_rsp;
    const bool from_stub = g_stub_size && immediate >= g_stub_base &&
                           (immediate - g_stub_base) / g_stub_size < g_nstubs;
    // The SysV-to-MS bridge reserves 0x48 bytes and CALLS the HLE handler. Including that call's
    // return slot, the original guest return address is 0x50 bytes above the handler-entry RSP.
    if (from_stub) return *(const uint64_t*)(uintptr_t)(entry_rsp + 0x50);
    return immediate;
}
uint64_t invoke_stub(uint64_t idx) {
    if (idx >= g_nstubs) return 0;
    return ((HleFn)(uintptr_t)stub_addr(idx))(0, 0, 0, 0, 0, 0);
}

void register_thread_stack(uint64_t tid, void* base, uint64_t size) {
    std::lock_guard<std::mutex> lk(g_smx); g_stacks[tid] = { (uint64_t)base, size };
}
void unregister_thread_stack(uint64_t tid) {
    std::lock_guard<std::mutex> lk(g_smx); g_stacks.erase(tid);
}
bool guest_stack_for_thread(uint64_t tid, void** base, size_t* size) {
    auto lookup = [&](uint64_t key) {
        std::lock_guard<std::mutex> lk(g_smx);
        auto it = g_stacks.find(key);
        if (it == g_stacks.end()) return false;
        if (base) *base = (void*)(uintptr_t)it->second.first;
        if (size) *size = (size_t)it->second.second;
        return true;
    };
    if (lookup(tid)) return true;

    // Frontends enter the guest from std::thread. winpthreads cannot always translate that
    // implicit handle with pthread_gethandle(), but a self-query has an unambiguous native ID.
    if (tid == (uint64_t)pthread_self() && lookup(cur_tid())) return true;

    // Windows workers are registered by native thread id because exception delivery and stack
    // limits use Win32 APIs, while the guest sees winpthreads' small pthread_t handle. Translate
    // that handle before looking up another thread's stack. Without this, scePthreadAttrGet(target)
    // missed every worker and IL2CPP's collector scanned the caller's stack instead.
    HANDLE thread = (HANDLE)pthread_gethandle((pthread_t)tid);
    if (!thread || thread == INVALID_HANDLE_VALUE) return false;
    const DWORD native_tid = GetThreadId(thread);
    return native_tid != 0 && native_tid != tid && lookup(native_tid);
}
bool guest_stack_for_current_thread(void** base, size_t* size) {
    return guest_stack_for_thread(cur_tid(), base, size);
}

void set_module_start_param_ranges(const std::vector<std::pair<uint64_t, uint64_t>>& ranges) {
    g_modstart_param_ranges = ranges;
}

size_t run_guest_inits(const std::vector<uint64_t>& fns) {
    ULONG_PTR stack_lo = 0, stack_hi = 0;
    GetCurrentThreadStackLimits(&stack_lo, &stack_hi);
    if (stack_lo && stack_hi > stack_lo) {
        const uint64_t native_tid = cur_tid();
        register_thread_stack(native_tid, (void*)stack_lo, (uint64_t)(stack_hi - stack_lo));
        t_init_stack_registration.native_tid = native_tid;
    }
    guest_tls_activate_thread();   // give this (main) thread its guest %fs TCB before running guest code
    size_t ok = 0;
    for (uint64_t f : fns) {
        g_trap_kind = 0; t_armed = 1;
        uint64_t argc = 0, argp = 0;
        for (auto& r : g_modstart_param_ranges)
            if (f >= r.first && f < r.second) { argc = 0x10; argp = (uint64_t)&g_modstart_desc; break; }
        if (__builtin_setjmp(t_jb) == 0) { prosper_call_guest_sysv(f, argc, argp); ok++; }
        t_armed = 0;
        if (g_trap_kind) {
            fprintf(stderr, "[prosper] init fn 0x%llx faulted (%s); continuing\n",
                    (unsigned long long)f, trap_detail().c_str());
            fprintf(stderr, "[prosper]   rax=%016llx rbx=%016llx rcx=%016llx rdx=%016llx\n"
                            "[prosper]   rsi=%016llx rdi=%016llx rbp=%016llx rsp=%016llx\n"
                            "[prosper]   r8 =%016llx r9 =%016llx r10=%016llx r11=%016llx\n"
                            "[prosper]   r12=%016llx r13=%016llx r14=%016llx r15=%016llx\n",
                    (unsigned long long)g_rax,(unsigned long long)g_rbx,(unsigned long long)g_rcx,
                    (unsigned long long)g_rdx,(unsigned long long)g_rsi,(unsigned long long)g_rdi,
                    (unsigned long long)g_rbp,(unsigned long long)g_rsp,(unsigned long long)g_r8,
                    (unsigned long long)g_r9,(unsigned long long)g_r10,(unsigned long long)g_r11,
                    (unsigned long long)g_r12,(unsigned long long)g_r13,(unsigned long long)g_r14,
                    (unsigned long long)g_r15);
            fprintf(stderr, "[prosper]   bytes@rip-8:"); dump_fault_bytes(g_fault_rip - 8, 24);
            fprintf(stderr, "  (rip=0x%llx)\n", (unsigned long long)g_fault_rip);
            fprintf(stderr, "[prosper]   bytes@entry:"); dump_fault_bytes(f, 24);
            fprintf(stderr, "  (entry=0x%llx)\n", (unsigned long long)f);
        }
    }
    return ok;
}

BootResult run_entry(const LoadedImage& img) {
    const size_t STK = 16 * 1024 * 1024;
    void* stk = VirtualAlloc(nullptr, STK, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    BootResult r;
    if (!stk) { r.kind = 2; r.detail = "guest stack VirtualAlloc failed"; return r; }
    register_thread_stack(cur_tid(), stk, STK);
    arm_diag_breakpoints();
    trace_guest_thread_lifecycle(true, (uint64_t)pthread_self(), cur_tid(), stk, STK);

    uint64_t top = ((uint64_t)(uintptr_t)stk + STK) & ~(uint64_t)0xf;
    std::vector<std::string> args = { "/app0/eboot.bin" };
    if (const char* extra = getenv("PROSPER_GUEST_ARGS")) {
        const char* p = extra;
        while (*p) { while (*p == ' ') p++; if (!*p) break; const char* s = p;
                     while (*p && *p != ' ') p++; args.emplace_back(s, (size_t)(p - s)); }
    }
    std::vector<uint64_t> argptrs;
    for (const auto& a : args) { size_t n = a.size() + 1; top -= n; memcpy((void*)(uintptr_t)top, a.c_str(), n); argptrs.push_back(top); }
    std::vector<uint64_t> envptrs;
    if (const char* ge = getenv("PROSPER_GUEST_ENV")) {
        std::string s(ge); size_t i = 0;
        while (i < s.size()) {
            size_t j = s.find(';', i);
            std::string kv = s.substr(i, j == std::string::npos ? std::string::npos : j - i);
            if (!kv.empty()) { size_t n = kv.size() + 1; top -= n; memcpy((void*)(uintptr_t)top, kv.c_str(), n); envptrs.push_back(top); }
            if (j == std::string::npos) break; i = j + 1;
        }
        top &= ~(uint64_t)0xf;
    }
    top &= ~(uint64_t)0xf;
    // crt0 vector: argc, argv[], NULL, envp[], NULL, auxv AT_NULL(0,0). rsp ≡ 8 (mod 16) at entry.
    std::vector<uint64_t> vecv;
    vecv.push_back(args.size());
    for (uint64_t pp : argptrs) vecv.push_back(pp);
    vecv.push_back(0);
    for (uint64_t pp : envptrs) vecv.push_back(pp);
    vecv.push_back(0);
    vecv.push_back(0); vecv.push_back(0);
    if (vecv.size() & 1) vecv.push_back(0);
    size_t vecsz = vecv.size() * sizeof(uint64_t);
    top -= vecsz; top &= ~(uint64_t)0xf;
    top -= 8;
    memcpy((void*)(uintptr_t)top, vecv.data(), vecsz);
    uint64_t sp = top, rdi = sp, rsi = 0;

    // Describe the switched guest stack in the TEB. We `mov %rsp` to `stk` below, but Windows still
    // thinks the thread's stack is the original host-thread stack (NT_TIB.StackBase/StackLimit). When
    // a guest fault (e.g. the SSE4a INSERTQ/EXTRQ trap) dispatches, ntdll's KiUserExceptionDispatcher
    // compares the (guest) rsp against the (host) StackLimit, decides the stack is exhausted, and
    // raises STATUS_STACK_OVERFLOW (0xC00000FD) instead of delivering the exception to our VEH — so the
    // very first trap after boot killed the process. Point the TEB at the real guest stack for the
    // duration of guest execution; restore it on the recovery path (back on the host stack). This is
    // the standard requirement for running on a manually-allocated stack on Windows.
    NT_TIB* tib = (NT_TIB*)NtCurrentTeb();
    void* teb_save_base = tib->StackBase, *teb_save_limit = tib->StackLimit;

    g_trap_kind = 0; g_fault_addr = 0; g_fault_rip = 0; t_armed = 1;
    if (__builtin_setjmp(t_jb) == 0) {
        tib->StackBase  = (void*)((uintptr_t)stk + STK);   // high end (grows down toward StackLimit)
        tib->StackLimit = stk;                             // low end of the committed guest stack
        guest_tls_activate_thread();   // (re-)apply this thread's guest %fs base immediately before entry
                                       // (idempotent; no syscall on the re-apply path, so fs survives to the jmp)
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
        tib->StackBase = teb_save_base; tib->StackLimit = teb_save_limit;   // back on the host stack
        r.kind = g_trap_kind; r.detail = trap_detail();
        r.fault_addr = g_fault_addr; r.fault_rip = g_fault_rip;
        r.rbp = g_rbp; r.rsp = g_rsp; r.rax = g_rax; r.rdi = g_rdi;
        r.rsi = g_rsi; r.rdx = g_rdx; r.rbx = g_rbx;
        t_armed = 1;
        if (__builtin_setjmp(t_jb) == 0) {
            uint64_t bp = g_rbp;
            for (int i = 0; i < 24 && bp > 0x10000; i++) {
                if (!addr_readable(bp) || !addr_readable(bp + 8)) break;
                uint64_t ret = *(uint64_t*)(uintptr_t)(bp + 8);
                if (ret) r.backtrace.push_back(ret);
                uint64_t nbp = *(uint64_t*)(uintptr_t)bp;
                if (nbp <= bp) break;
                bp = nbp;
            }
        }
        t_armed = 0;
    }
    trace_guest_thread_lifecycle(false, (uint64_t)pthread_self(), cur_tid(), stk, STK);
    unregister_thread_stack(cur_tid());
    return r;
}

} // namespace prosper
#endif // _WIN32
