// guest_tls.cpp — guest initial-exec TLS support (Linux). GATED behind PROSPER_GUEST_FS (default off).
//
// The guest (FreeBSD/PS5, x86-64 Variant II TLS) accesses static/initial-exec thread-locals directly via
// %fs: `mov %fs:0x0,%rax; mov -0xa8(%rax),%rdx`. We normally run guest code on the HOST pthread's %fs (so
// the real libc.prx + our C++ HLE handlers share one glibc TCB), which means those guest tpoffs alias host
// glibc's static TLS -> garbage (the crash at eboot+0xa9c0bb: [TP-0xa8] read host junk 0x2). General-dynamic
// TLS (__tls_get_addr) is already handled elsewhere; THIS backs the initial-exec path.
//
// Fix: give each guest thread its own guest TCB with the modules' static TLS laid out below the thread
// pointer per Variant II, and run guest code with %fs = guest TP. HLE handlers (which use host libc/TLS)
// swap %fs back to the host TCB for the duration of the call — done in the emitted import stubs
// (exec_image_linux.cpp), which read the stashed host %fs from [guestTP + GUEST_TCB_HOSTFS_OFF].
#if defined(__linux__) || defined(__APPLE__)
#include "../hle/dispatch.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <sys/mman.h>

namespace prosper {

namespace {
    std::vector<TlsModuleDesc> g_mods;   // index 0 reserved; 1 = main exe (eboot), then deps
    std::vector<uint64_t> g_below;       // per-module distance below TP (parallel to g_mods; [0] unused)
    uint64_t g_total_below = 0;          // bytes of static TLS below the thread pointer
    bool     g_enabled = false;
    bool     g_configured = false;

    inline uint64_t align_up(uint64_t v, uint64_t a) { return a ? (v + a - 1) & ~(a - 1) : v; }
    inline uint64_t rd_fsbase() { uint64_t v; __asm__ volatile("rdfsbase %0" : "=r"(v)); return v; }
    inline void     wr_fsbase(uint64_t v) { __asm__ volatile("wrfsbase %0" : : "r"(v)); }
}

// TCB (at/above the thread pointer): [TP+0]=self pointer (the `mov %fs:0x0` idiom), and we stash this
// thread's host %fs at [TP+GUEST_TCB_HOSTFS_OFF] so the import stubs can swap back to it. Keep these
// offsets in sync with exec_image_linux.cpp's stub emitter.
static constexpr uint64_t TCB_SIZE        = 0x200;
static constexpr uint64_t HOSTFS_OFF      = 0x100;   // == GUEST_TCB_HOSTFS_OFF in the stub emitter
static constexpr uint64_t MAGIC_OFF       = 0x108;   // == GUEST_TCB_MAGIC_OFF in the stub emitter
static constexpr uint32_t TCB_MAGIC       = 0x50524F53u;  // "PROS" — marks OUR guest TCB (== GUEST_TCB_MAGIC)

void guest_tls_set_templates(const TlsModuleDesc* descs, size_t count) {
    g_enabled = getenv("PROSPER_GUEST_FS") != nullptr;
#ifdef __APPLE__
    // Rosetta 2 does not implement rdfsbase/wrfsbase (verified: SIGILL on an M2, macOS 26) and
    // Darwin has no fsbase API, so guest initial-exec %fs TLS cannot be enabled on this host yet.
    // The default (host-%fs-aliasing) boot path is unaffected. See docs/PORTING.md.
    if (g_enabled) {
        fprintf(stderr, "[guest-tls] PROSPER_GUEST_FS is not supported on macOS (Rosetta lacks "
                        "wrfsbase); continuing with it DISABLED\n");
        g_enabled = false;
    }
#endif
    g_mods.assign(descs, descs + count);
    // x86-64 Variant II static-TLS layout (glibc _dl_determine_tlsoffset, TLS_TCB_AT_TP): each
    // module's block sits BELOW the thread pointer at a distance rounded to that module's REAL
    // PT_TLS p_align — off[i] = round_up(off[i-1] + memsz[i], align[i]) — so its start (TP - off[i])
    // is p_align-aligned and a symbol at byte X reads at %fs:(X - off[i]), exactly the tpoff the
    // guest static linker baked into its initial-exec accesses. The old code rounded each module's
    // SIZE to a hardcoded 16 (and the total to 64), which for a module with p_align > 16 placed the
    // block at the wrong offset -> every %fs:-N read resolved off (#143). For the common p_align<=16
    // case the per-module offsets are identical to before (round_up up a 16-multiple running total by
    // 16 == the old per-size rounding), so nothing changes there.
    g_below.assign(g_mods.size(), 0);
    uint64_t off = 0, max_align = 16;
    for (size_t i = 1; i < g_mods.size(); i++) {
        uint64_t a = g_mods[i].align ? g_mods[i].align : 16;
        if (a > max_align) max_align = a;
        off = align_up(off + g_mods[i].memsz, a);
        g_below[i] = off;
    }
    // Round the total (hence the TP position within the page-aligned block) up to the max module
    // align, with a 64-byte floor preserved from the original (TCB/cache-line headroom).
    g_total_below = align_up(off, max_align < 64 ? 64 : max_align);
    g_configured = true;
    if (getenv("PROSPER_TLSLOG"))
        fprintf(stderr, "[tls] guest-fs %s; static TLS below TP = 0x%llx bytes\n",
                g_enabled ? "ENABLED" : "disabled", (unsigned long long)g_total_below);
}

bool guest_tls_enabled() { return g_enabled && g_configured; }

// Allocate + initialize this thread's guest TLS block and switch %fs to the guest TP. Returns the guest
// TP (fs base), or 0 if disabled. Idempotent-ish: always makes a fresh block (called once per guest thread
// at its entry). The block is intentionally leaked for the thread's lifetime (freed by process exit).
uint64_t guest_tls_activate_thread() {
    if (!guest_tls_enabled()) return 0;
    uint64_t host_fs = rd_fsbase();
    size_t total = (size_t)g_total_below + TCB_SIZE;
    uint8_t* block = (uint8_t*)mmap(nullptr, total, PROT_READ | PROT_WRITE,
                                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (block == MAP_FAILED) return 0;
    memset(block, 0, total);                    // tbss + TCB start zeroed
    uint64_t tp = (uint64_t)block + g_total_below;
    // Lay each module's TLS below TP at the precomputed Variant II offset (#143). Copy tdata (filesz),
    // leave tbss zero. g_below[i] is the module's distance below TP, so its block start is tp-g_below[i].
    for (size_t i = 1; i < g_mods.size(); i++) {
        if (g_mods[i].filesz && g_mods[i].init_va)
            memcpy((void*)(tp - g_below[i]), (const void*)(uintptr_t)g_mods[i].init_va, g_mods[i].filesz);
    }
    *(uint64_t*)(tp + 0)          = tp;         // TCB self-pointer
    *(uint64_t*)(tp + HOSTFS_OFF) = host_fs;    // stash for the stub swap-back
    *(uint32_t*)(tp + MAGIC_OFF)  = TCB_MAGIC;  // marks this as OUR guest TCB (stubs swap only if present)
    // Replicate the HOST glibc stack-guard (%fs:0x28) + pointer-guard (%fs:0x30) into the guest TCB. HOST
    // code compiled with -fstack-protector can transiently run under the guest %fs (our SIGSEGV/SIGTRAP
    // signal handlers fire while the guest %fs is live); its prologue/epilogue canary reads come from
    // [fs+0x28], so without this the epilogue would read a zeroed guest slot, mismatch, and abort via
    // __stack_chk_fail (observed: single-stepping under guest-fs "stack smashing detected"). Copying the
    // host canary makes those reads correct and re-enables HWBP/single-step-based tracing under guest-fs.
    *(uint64_t*)(tp + 0x28) = *(volatile uint64_t*)(host_fs + 0x28);   // stack canary
    *(uint64_t*)(tp + 0x30) = *(volatile uint64_t*)(host_fs + 0x30);   // pointer guard
    wr_fsbase(tp);
    return tp;
}

// Called at the entry of the CRASH signal handler (fault_handler). If the faulting thread is running on
// OUR guest %fs, switch to the stashed host %fs so the handler's host-libc calls (snprintf/write) and any
// siglongjmp-return into host C++ run on the correct TCB — otherwise they read guest TLS as glibc's TCB and
// double-fault (an uncaught SIGSEGV -> core dump). No-op if fs isn't one of our guest TCBs (host thread, or
// gate off): reading [fs+MAGIC_OFF] is safe on the host TCB (positive offset into the allocated TLS/dtv
// area) and simply won't match the magic. NOT used by the GC RT-signal handler, which must KEEP the guest
// %fs to run the guest's own exception handler. Safe to call unconditionally.
void guest_fs_enter_host_for_signal() {
#ifdef __APPLE__
    return;   // guest-fs is force-disabled on Darwin, and rdfsbase itself SIGILLs under Rosetta
#endif
    uint64_t fs = rd_fsbase();
    if (fs && *(volatile uint32_t*)(fs + MAGIC_OFF) == TCB_MAGIC)
        wr_fsbase(*(volatile uint64_t*)(fs + HOSTFS_OFF));
}

// Scoped variant for diagnostic signal handlers that RETURN to the guest (HWBP/int3 logging): swap to the
// host %fs for the handler's host-libc calls and return the previous (guest) fs so the caller can restore
// it before returning to guest code. Returns 0 if not on a guest TCB (nothing to restore).
uint64_t guest_fs_to_host_scoped() {
#ifdef __APPLE__
    return 0;   // see guest_fs_enter_host_for_signal
#endif
    uint64_t fs = rd_fsbase();
    if (fs && *(volatile uint32_t*)(fs + MAGIC_OFF) == TCB_MAGIC) {
        wr_fsbase(*(volatile uint64_t*)(fs + HOSTFS_OFF));
        return fs;
    }
    return 0;
}
void guest_fs_restore_scoped(uint64_t prev_fs) { if (prev_fs) wr_fsbase(prev_fs); }

// This thread's guest TP if it is running on our guest TCB, else 0. (Linux/macOS query the live
// %fs; the magic marks it as ours.) Used by the Windows fault handler; harmless on Linux.
uint64_t guest_fs_current_tp() {
    if (!guest_tls_enabled()) return 0;
#ifdef __APPLE__
    return 0;   // guest-fs force-disabled; rd_fsbase SIGILLs under Rosetta
#else
    uint64_t fs = rd_fsbase();
    return (fs && *(volatile uint32_t*)(fs + MAGIC_OFF) == TCB_MAGIC) ? fs : 0;
#endif
}
// Linux/macOS preserve the FS base across kernel transitions, so there is never drift to correct.
bool guest_fs_reapply() { return false; }

// Diagnostic/test accessors for the Variant II static-TLS layout (#143): the distance below the
// thread pointer at which module `modid`'s block starts (0 if out of range), and the total static
// TLS below TP. Computed by guest_tls_set_templates regardless of the PROSPER_GUEST_FS gate, so a
// test can verify the per-module p_align offsets without wrfsbase / an enabled guest-fs.
uint64_t guest_tls_module_below(uint32_t modid) { return modid < g_below.size() ? g_below[modid] : 0; }
uint64_t guest_tls_total_below() { return g_total_below; }

} // namespace prosper

#elif defined(_WIN32)
// Windows: guest initial-exec %fs TLS, via user-mode FSGSBASE (rdfsbase/wrfsbase, enabled on
// Win10 1709+). This differs from the Linux path in two ways that together make it SIMPLER here:
//   (1) The HOST uses %gs (TEB) and only the GUEST uses %fs, so there is NO per-HLE-call %fs swap —
//       host handlers and guest code never collide on %fs (the Linux stub swap dance is unneeded).
//   (2) But Windows RESETS the user %fs base to 0 on every kernel transition (context switch /
//       syscall) — it restores the thread's kernel-saved FS base (0), and wrfsbase updates only the
//       live register. So "set once and leave it" is impossible. We set the base on guest entry and
//       the VEH (exec_image_win.cpp) re-applies it and retries whenever a guest %fs access faults
//       because the base drifted (guest_fs_reapply). One fault per kernel-transition boundary, not
//       per access. See docs/PORTING.md "The Windows frontier: guest %fs TLS".
// Because host-%fs aliasing (the Linux default fallback) cannot work here, guest-fs is enabled
// whenever templates are configured (opt out with PROSPER_NO_GUEST_FS for bisection).
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include "../hle/dispatch.hpp"
#include <windows.h>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace prosper {
namespace {
    std::vector<TlsModuleDesc> g_mods;
    std::vector<uint64_t> g_below;
    uint64_t g_total_below = 0;
    bool g_enabled = false, g_configured = false;
    thread_local uint64_t t_guest_tp = 0;

    inline uint64_t align_up(uint64_t v, uint64_t a) { return a ? (v + a - 1) & ~(a - 1) : v; }
    inline uint64_t rd_fsbase() { uint64_t v; __asm__ volatile("rdfsbase %0" : "=r"(v)); return v; }
    inline void     wr_fsbase(uint64_t v) { __asm__ volatile("wrfsbase %0" : : "r"(v)); }
}

static constexpr uint64_t TCB_SIZE  = 0x200;
static constexpr uint64_t MAGIC_OFF = 0x108;
static constexpr uint32_t TCB_MAGIC = 0x50524F53u;   // "PROS"

void guest_tls_set_templates(const TlsModuleDesc* descs, size_t count) {
    g_enabled = getenv("PROSPER_NO_GUEST_FS") == nullptr;
    g_mods.assign(descs, descs + count);
    // x86-64 Variant II static-TLS layout (same math as the Linux path): each module's block sits
    // BELOW the thread pointer at a distance rounded to its PT_TLS p_align, so a symbol at byte X
    // reads at %fs:(X - off[i]) — exactly the tpoff the guest static linker baked in.
    g_below.assign(g_mods.size(), 0);
    uint64_t off = 0, max_align = 16;
    for (size_t i = 1; i < g_mods.size(); i++) {
        uint64_t a = g_mods[i].align ? g_mods[i].align : 16;
        if (a > max_align) max_align = a;
        off = align_up(off + g_mods[i].memsz, a);
        g_below[i] = off;
    }
    g_total_below = align_up(off, max_align < 64 ? 64 : max_align);
    g_configured = true;
    if (getenv("PROSPER_TLSLOG"))
        fprintf(stderr, "[tls] guest-fs %s (win/fsgsbase); static TLS below TP = 0x%llx bytes\n",
                g_enabled ? "ENABLED" : "disabled", (unsigned long long)g_total_below);
}

bool guest_tls_enabled() { return g_enabled && g_configured; }

uint64_t guest_tls_activate_thread() {
    if (!guest_tls_enabled()) return 0;
    if (t_guest_tp) { wr_fsbase(t_guest_tp); return t_guest_tp; }   // idempotent: re-apply, never re-alloc
    size_t total = (size_t)g_total_below + TCB_SIZE;
    uint8_t* block = (uint8_t*)VirtualAlloc(nullptr, total, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!block) return 0;
    memset(block, 0, total);
    uint64_t tp = (uint64_t)block + g_total_below;
    for (size_t i = 1; i < g_mods.size(); i++)
        if (g_mods[i].filesz && g_mods[i].init_va)
            memcpy((void*)(tp - g_below[i]), (const void*)(uintptr_t)g_mods[i].init_va, g_mods[i].filesz);
    *(uint64_t*)(tp + 0)         = tp;            // TCB self-pointer (the `mov %fs:0x0` idiom)
    *(uint32_t*)(tp + MAGIC_OFF) = TCB_MAGIC;     // marks OUR guest TCB
    // Guest stack-guard canary at %fs:0x28: a fixed non-zero value (the guest checks it for
    // self-consistency; MinGW's host protector uses a global __stack_chk_guard, not %fs, so there
    // is no host canary to mirror here — unlike the Linux path).
    *(uint64_t*)(tp + 0x28) = 0x00000aff0c2b1d5eull;
    t_guest_tp = tp;
    wr_fsbase(tp);   // MUST be last — an intervening syscall (e.g. VirtualAlloc) resets the FS base
    return tp;
}

uint64_t guest_fs_current_tp() { return t_guest_tp; }
bool guest_fs_reapply() {
    if (!t_guest_tp) return false;
    if (rd_fsbase() == t_guest_tp) return false;   // already correct -> genuine fault, don't loop
    wr_fsbase(t_guest_tp);
    return true;
}

// Host uses %gs on Windows, so no guest<->host %fs swap is ever needed.
void guest_fs_enter_host_for_signal() {}
uint64_t guest_fs_to_host_scoped() { return 0; }
void guest_fs_restore_scoped(uint64_t) {}
uint64_t guest_tls_module_below(uint32_t modid) { return modid < g_below.size() ? g_below[modid] : 0; }
uint64_t guest_tls_total_below() { return g_total_below; }
} // namespace prosper
#endif
