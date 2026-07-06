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
#ifdef __linux__
#include "../hle/dispatch.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <sys/mman.h>

namespace prosper {

namespace {
    std::vector<TlsModuleDesc> g_mods;   // index 0 reserved; 1 = main exe (eboot), then deps
    uint64_t g_total_below = 0;          // bytes of static TLS below the thread pointer
    bool     g_enabled = false;
    bool     g_configured = false;

    inline uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }
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
    g_mods.assign(descs, descs + count);
    // Variant II: sum each module's aligned static-TLS size. Module 1 (main exe) ends up closest to TP.
    g_total_below = 0;
    for (size_t i = 1; i < g_mods.size(); i++)
        g_total_below += align_up(g_mods[i].memsz, 16);
    g_total_below = align_up(g_total_below, 64);
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
    // Lay each module's TLS below TP, main exe (module 1) closest. Copy tdata (filesz), leave tbss zero.
    uint64_t off = 0;
    for (size_t i = 1; i < g_mods.size(); i++) {
        uint64_t sz = align_up(g_mods[i].memsz, 16);
        off += sz;
        if (g_mods[i].filesz && g_mods[i].init_va)
            memcpy((void*)(tp - off), (const void*)(uintptr_t)g_mods[i].init_va, g_mods[i].filesz);
    }
    *(uint64_t*)(tp + 0)          = tp;         // TCB self-pointer
    *(uint64_t*)(tp + HOSTFS_OFF) = host_fs;    // stash for the stub swap-back
    *(uint32_t*)(tp + MAGIC_OFF)  = TCB_MAGIC;  // marks this as OUR guest TCB (stubs swap only if present)
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
    uint64_t fs = rd_fsbase();
    if (fs && *(volatile uint32_t*)(fs + MAGIC_OFF) == TCB_MAGIC)
        wr_fsbase(*(volatile uint64_t*)(fs + HOSTFS_OFF));
}

} // namespace prosper
#endif
