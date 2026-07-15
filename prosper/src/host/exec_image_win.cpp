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
// that depend on Linux perf_event / ptrace (PROSPER_HWBP/HWWATCH/BP/PEEK/DUMPAT) remain absent.
#ifdef _WIN32

#include "exec_image.hpp"
#include "sse4a.hpp"
#include "../hle/nid.hpp"
#include "../hle/dispatch.hpp"

#include <windows.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <csetjmp>
#include <map>
#include <mutex>
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

    // Thread-stack registry (portable; mirrors the Linux one) so GC/thread code gets real bounds.
    std::map<uint64_t, std::pair<uint64_t, uint64_t>> g_stacks;
    std::mutex g_smx;

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
    void emit_impl(uint8_t* p, uint64_t fn) {
        size_t o = emit_sysv_to_ms_prologue(p);
        p[o++] = 0x48; p[o++] = 0xB8; memcpy(p + o, &fn, 8); o += 8;   // movabs rax,fn
        p[o++] = 0xFF; p[o++] = 0xD0;                                  // call rax
        p[o++] = 0x48; p[o++] = 0x83; p[o++] = 0xC4; p[o++] = 0x48;    // add rsp,0x48
        p[o++] = 0xC3;                                                 // ret
    }
    void emit_unimpl(uint8_t* p, uint32_t idx, uint64_t fn) {
        size_t o = 0;
        p[o++] = 0x48; p[o++] = 0x83; p[o++] = 0xEC; p[o++] = 0x28;    // sub rsp,0x28 (shadow + align)
        p[o++] = 0xB9; memcpy(p + o, &idx, 4); o += 4;                 // mov ecx,idx  (MS 1st arg)
        p[o++] = 0x48; p[o++] = 0xB8; memcpy(p + o, &fn, 8); o += 8;   // movabs rax,fn
        p[o++] = 0xFF; p[o++] = 0xD0;                                  // call rax  (prosper_on_unimpl)
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

    // AMD SSE4a INSERTQ/EXTRQ emulation (Zen2-only; #UD on the Intel ISA Rosetta/native Intel expose,
    // and generally on the host). Decode from Rip, mutate the xmm regs in CONTEXT, advance Rip.
    // Mirrors the Linux try_emulate_sse4a but reads/writes CONTEXT.Xmm* instead of fpregs.
    bool try_emulate_sse4a(CONTEXT* c) {
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
        else { int dr = (op == 0x78) ? rm : reg; set(dr, sse4a_extrq(lo(dr), len, idx)); }
        c->Rip += i;
        return true;
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
        if (code == EXCEPTION_ILLEGAL_INSTRUCTION && try_emulate_sse4a(c))
            return EXCEPTION_CONTINUE_EXECUTION;
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
            if (a >= 0x1000000000ull &&
                prosper_try_commit_dmem(
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
    if (!g_base) g_base = img.base;
    return true;
}

bool install_stubs(const std::vector<ImportSlot>& slots, uint64_t stub_base,
                   uint64_t stub_size, std::string* err) {
    auto fail = [&](const char* s){ if (err) *err = s; return false; };
    if (stub_size < 80) return fail("stub_size too small for Windows ABI bridge (need >= 80)");
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

// Diagnostics that need Linux perf_event / int3 patching are unavailable on Windows.
void arm_hwbp_this_thread() {}

uint64_t stub_addr(uint64_t idx) { return g_stub_base + idx * g_stub_size; }
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
    std::lock_guard<std::mutex> lk(g_smx);
    auto it = g_stacks.find(tid);
    if (it == g_stacks.end()) return false;
    if (base) *base = (void*)(uintptr_t)it->second.first;
    if (size) *size = (size_t)it->second.second;
    return true;
}
bool guest_stack_for_current_thread(void** base, size_t* size) {
    return guest_stack_for_thread(cur_tid(), base, size);
}

void set_module_start_param_ranges(const std::vector<std::pair<uint64_t, uint64_t>>& ranges) {
    g_modstart_param_ranges = ranges;
}

size_t run_guest_inits(const std::vector<uint64_t>& fns) {
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
    return r;
}

} // namespace prosper
#endif // _WIN32
