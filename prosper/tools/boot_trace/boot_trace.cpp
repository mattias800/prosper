// boot_trace — link the game's modules, boot the guest, and report how far it got:
// the unimplemented-call trace (via stderr from dispatch) plus, on a fault, the register
// state and an rbp-chain backtrace classified by module. The primary bring-up debugging
// tool. Linux only. Usage: boot_trace <dump-root>
#include "loader/linker.hpp"
#include "host/exec_image.hpp"
#include "host/boot_program.hpp"          // shared guest-boot path (also used by prosper-app)
#include "hle/dispatch.hpp"
#include <cstdio>
#include <string>
#include <cstdint>
#include <exception>          // std::set_terminate / std::current_exception (PROSPER_TERM_BT)
#ifndef _WIN32
#include <execinfo.h>         // backtrace / backtrace_symbols (host crash backtrace)
#endif
extern "C" int prosper_reserved_range_state(uint64_t);   // memory-HLE mapping classifier (diagnostic)
#ifdef PROSPER_AUDIO_SDL3
#include "audio_sdl3.hpp"                 // optional SDL3 audio frontend (-DPROSPER_AUDIO_SDL3=ON)
#endif
#include <cstdlib>                        // getenv (PROSPER_PAD gate)
#ifdef PROSPER_PAD_SDL3
#include "pad_sdl3.hpp"                   // optional SDL3 gamepad frontend (-DPROSPER_PAD_SDL3=ON)
#endif
#ifdef PROSPER_PAD_EVDEV
#include "pad_evdev.hpp"                  // zero-dep Linux evdev gamepad frontend
#endif
#ifdef PROSPER_HAVE_VULKAN
#include "gpu/gpu_execute.hpp"
#include "gpu/tile.hpp"                   // render-target de-swizzle (detile_surface, tiled_surface_bytes)
#include "gpu/bc_decode.hpp"              // BC1/2/3 block decompression -> RGBA8 (#121)
#include "gpu/shader_resources.hpp"       // ShaderResourceTable / ResourceClass (bind the shaders' resources)
#include "gpu/rdna2_to_spirv.hpp"         // recompile_fragment (diagnostic solid-color PS)
#include "../../tests/render_runner.h"   // offscreen Vulkan backend (render_triangle_rgba) + dump_bmp
#include "../../frontends/shared/live_renderer.hpp"   // shared live renderer (also used by prosper-app)
#include "../../frontends/shared/live_compute.hpp"    // synchronous AGC compute execution
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#endif
#endif

// PROSPER_CRASHPEEK (below) needs these in every config — the gpu/*.hpp includes above are gated on
// PROSPER_HAVE_VULKAN, but guest_readable is a core (non-Vulkan) symbol always linked into prosper_core,
// and strchr needs <cstring>. Declare/include them unconditionally so the no-Vulkan build compiles.
// PROSPER_NULLGUARD's cave mmap likewise needs <sys/mman.h> in every non-Windows config (the Vulkan
// block above only transitively provided it in some local configs — CI's no-Vulkan build broke).
#include <cstring>
#ifndef _WIN32
#include <sys/mman.h>
#endif
namespace prosper::gpu { bool guest_readable(uint64_t addr, uint32_t bytes); }
// Lazy-commit reserved-range probe: 0 = not in any guest mapping (a memcpy would SIGSEGV), 1 = reserved
// (the fault handler will commit it on touch), 2 = already committed. Used to bound texture/buffer copies
// to their real backing — a resource's declared size can run past what the game actually mapped.
extern "C" int prosper_reserved_range_state(uint64_t addr);

using namespace prosper;


// Module bases (keep in sync with the inputs below).
static const uint64_t EBOOT = 0x400000000ull, IL2CPP = 0x440000000ull, PS5UTIL = 0x4c0000000ull,
                      PSN = 0x4e0000000ull, SAVEDATA = 0x4f0000000ull, LIBC = 0x500000000ull,
                      STUB = 0x600000000ull;
static const char* cls(uint64_t a) {
    if (a >= EBOOT    && a < IL2CPP)   return "eboot";
    if (a >= IL2CPP   && a < PS5UTIL)  return "Il2cpp";
    if (a >= PS5UTIL  && a < PSN)      return "PS5Util";
    if (a >= PSN      && a < SAVEDATA) return "PSN.prx";
    if (a >= SAVEDATA && a < LIBC)     return "SaveData.prx";
    if (a >= LIBC     && a < STUB)     return "libc.prx";
    if (a >= STUB     && a < 0x610000000ull) return "STUB";
    return "mapped/host";
}
static uint64_t bof(uint64_t a) {
    if (a >= EBOOT    && a < IL2CPP)   return a - EBOOT;
    if (a >= IL2CPP   && a < PS5UTIL)  return a - IL2CPP;
    if (a >= PS5UTIL  && a < PSN)      return a - PS5UTIL;
    if (a >= PSN      && a < SAVEDATA) return a - PSN;
    if (a >= SAVEDATA && a < LIBC)     return a - SAVEDATA;
    if (a >= LIBC     && a < STUB)     return a - LIBC;
    return a;
}

int main(int argc, char** argv) {
    // PROSPER_TERM_BT: install a std::terminate handler that dumps the HOST backtrace + exception
    // message. Diagnostic for the macOS std::mutex EINVAL crash — the guest rbp-walker only shows
    // guest frames, but an uncaught host C++ exception (e.g. a mutex lock throwing) terminates on
    // the throwing thread, so backtrace() here captures the real host call chain to the bad lock.
    if (getenv("PROSPER_TERM_BT")) {
        std::set_terminate([]{
            fprintf(stderr, "\n=== PROSPER_TERM_BT: std::terminate ===\n");
            if (auto ep = std::current_exception()) {
                try { std::rethrow_exception(ep); }
                catch (const std::exception& ex) { fprintf(stderr, "  what(): %s\n", ex.what()); }
                catch (...) { fprintf(stderr, "  (non-std exception)\n"); }
            }
#ifndef _WIN32
            void* frames[64];
            int n = backtrace(frames, 64);
            char** syms = backtrace_symbols(frames, n);
            for (int i = 0; i < n; ++i) fprintf(stderr, "  #%02d %s\n", i, syms ? syms[i] : "?");
#else
            // No execinfo on MinGW; the exception message above is still the useful part.
            fprintf(stderr, "  (host backtrace unavailable on this platform)\n");
#endif
            fflush(stderr);
            _Exit(42);
        });
    }
    std::string d = (argc >= 2) ? argv[1] : "../../PPSA24651-app0";
    Program p; std::string e;
    // Boot the title via the shared path (also used by prosper-app): link the fixed module set,
    // map, TLS/unwind/procparam, stubs, trap handler, plugin ranges, and the dependent-module
    // init_arrays. The lambda installs boot_trace's host frontends at the same point it always did
    // — right after the built-in HLE is registered, before the images are mapped: sceAudioOut
    // output to the host, and (gated by PROSPER_PAD) a controller backend (SDL3, else evdev).
    if (!boot_program(d, p, &e, [&]{
#ifdef PROSPER_AUDIO_SDL3
        prosper::install_sdl3_audio_sink();
#endif
        if (getenv("PROSPER_PAD")) {
            bool pad_ok = false; (void)pad_ok;
#ifdef PROSPER_PAD_SDL3
            pad_ok = prosper::install_sdl3_pad_backend();
#endif
#ifdef PROSPER_PAD_EVDEV
            if (!pad_ok) pad_ok = prosper::install_evdev_pad_backend();
#endif
        }
    })) { printf("%s\n", e.c_str()); return 1; }

    // PROSPER_PATCH_RET=addr[,addr...]: write 0xC3 (ret) at each absolute guest address, neutralizing that
    // function (returns immediately, stack balanced). Bring-up diagnostic to bisect a crashing subsystem —
    // e.g. skip the per-frame incremental-GC pump (Il2cpp+0x5df0 => 0x440005df0) to test whether the GC is
    // the wall to scene activation. Guest code is RWX; write before the guest starts.
    if (const char* pr = getenv("PROSPER_PATCH_RET")) {
        const char* s = pr;
        while (*s) {
            char* end = nullptr; uint64_t a = strtoull(s, &end, 0);
            if (a && end != s) { *(volatile uint8_t*)(uintptr_t)a = 0xC3;
                fprintf(stderr, "[patch] ret @ 0x%llx\n", (unsigned long long)a); }
            // Always make progress: skip the comma if present, else advance past what we parsed; if
            // strtoull consumed nothing (end == s, malformed) step one char so we never spin forever.
            s = (*end == ',') ? end + 1 : (end != s) ? end : s + 1;
        }
    }
    // PROSPER_PATCH_BYTE=addr=val[,addr=val...]: write an arbitrary byte at a guest address (finer than
    // PATCH_RET — e.g. flip a conditional 0x74 je -> 0xEB jmp to force a branch). Same bring-up bisection use.
    if (const char* pb = getenv("PROSPER_PATCH_BYTE")) {
        const char* s = pb;
        while (*s) {
            char* e1 = nullptr; uint64_t a = strtoull(s, &e1, 0);
            if (a && e1 && *e1 == '=') {
                char* e2 = nullptr; unsigned long v = strtoul(e1 + 1, &e2, 0);
                *(volatile uint8_t*)(uintptr_t)a = (uint8_t)v;
                fprintf(stderr, "[patch] byte 0x%02x @ 0x%llx\n", (unsigned)(v & 0xff), (unsigned long long)a);
                s = (e2 && *e2 == ',') ? e2 + 1 : (e2 ? e2 : s + 1);
            } else s = (e1 && *e1) ? e1 + 1 : s + 1;
        }
    }
    // PROSPER_NULLGUARD=addr,len : install a null-receiver guard trampoline on the guest method at `addr`.
    // If its first arg (rdi = the C# `this`) is null, return 0 immediately; otherwise run the original.
    // Motivation: the cutscene-load crash is a property getter on `WorkerThread.field_0x40`, a
    // System.Diagnostics.Stopwatch that Unity's async loader has not yet created (see
    // CUTSCENE_PROGRESSION.md) — the getter dereferences the null Stopwatch in every path. Returning 0 for a
    // null/unstarted timer is a defensible, non-faking behavior (an unstarted Stopwatch reads 0 elapsed), so
    // this lets the REAL cutscene render past the crash while the engine-level "create the Stopwatch"
    // fix is developed. `len` = number of whole prologue bytes to relocate (>=5). Requires an executable
    // guest-adjacent cave within +/-2GB so a 5-byte jmp rel32 reaches it.
#ifndef _WIN32   // PROSPER_NULLGUARD is a Messenger-specific mmap+patch diagnostic; not ported to Windows yet.
    if (const char* ng = getenv("PROSPER_NULLGUARD")) {
        char* e1 = nullptr; uint64_t addr = strtoull(ng, &e1, 0);
        long len = (e1 && *e1 == ',') ? strtol(e1 + 1, nullptr, 0) : 13;
        if (addr && len >= 5) {
            // Cave near the guest modules (within 2GB of Il2cpp at 0x440000000) so jmp rel32 reaches.
            void* cave = mmap((void*)0x460000000ull, 0x1000, PROT_READ | PROT_WRITE | PROT_EXEC,
                              MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
            if (cave == MAP_FAILED) { fprintf(stderr, "[nullguard] cave mmap failed\n"); }
            else {
                uint8_t* c = (uint8_t*)cave; size_t o = 0;
                c[o++] = 0x48; c[o++] = 0x85; c[o++] = 0xff;                 // test rdi,rdi
                c[o++] = 0x74; c[o++] = (uint8_t)(len + 5);                  // jz ret0 (past prologue+jmp)
                memcpy(c + o, (void*)(uintptr_t)addr, len); o += len;        // relocated original prologue
                c[o++] = 0xe9;                                               // jmp addr+len
                int32_t rel = (int32_t)((int64_t)(addr + len) - (int64_t)(uintptr_t)(c + o + 4));
                memcpy(c + o, &rel, 4); o += 4;
                // ret0: value returned for a null receiver. PROSPER_NULLGUARD_RET selects the model:
                //   "tsc" (default) = raw rdtsc counter; "zero" = 0; "ms" = rdtsc>>22 (~milliseconds @4GHz).
                // WHY it matters: if the guarded method is Unity's async-load time-budget Stopwatch, returning
                // raw TSC cycles where the caller expects MILLISECONDS makes every elapsed check overshoot the
                // budget -> it integrates 0 objects/frame -> the load stalls forever on a black screen. A
                // scaled ~ms value lets the budget see plausible small elapsed and actually integrate.
                const char* ngret = getenv("PROSPER_NULLGUARD_RET");
                if (ngret && !strcmp(ngret, "zero")) {
                    c[o++] = 0x31; c[o++] = 0xc0;                               // xor eax,eax  (rax=0)
                    c[o++] = 0xc3;                                             // ret
                } else {
                    c[o++] = 0x0f; c[o++] = 0x31;                              // rdtsc  (edx:eax = tsc)
                    c[o++] = 0x48; c[o++] = 0xc1; c[o++] = 0xe2; c[o++] = 0x20; // shl rdx, 32
                    c[o++] = 0x48; c[o++] = 0x09; c[o++] = 0xd0;               // or rax, rdx
                    if (ngret && !strcmp(ngret, "ms")) {
                        c[o++] = 0x48; c[o++] = 0xc1; c[o++] = 0xe8; c[o++] = 0x16; // shr rax, 22 (~ns->ms scale)
                    }
                    c[o++] = 0xc3;                                             // ret
                }
                // Patch method entry: jmp cave (5 bytes). Remaining relocated bytes stay as harmless tail.
                uint8_t* g = (uint8_t*)(uintptr_t)addr;
                int32_t jrel = (int32_t)((int64_t)(uintptr_t)cave - (int64_t)(uintptr_t)(g + 5));
                g[0] = 0xe9; memcpy(g + 1, &jrel, 4);
                fprintf(stderr, "[nullguard] installed on 0x%llx (len=%ld) cave=%p\n",
                        (unsigned long long)addr, len, cave);
            }
        }
    }
#endif  // !_WIN32 (PROSPER_NULLGUARD)

#ifdef PROSPER_HAVE_VULKAN
    // Compute is part of command submission even when frame rendering/dumping is disabled.
    // PROSPER_NO_COMPUTE=1 is a progression diagnostic only: semantic timelines still retain the
    // dispatches, but neither graphics nor compute mutates guest GPU resources. This distinguishes
    // host compute throughput from guest/HLE progression; it is never a correctness mode.
    if (!getenv("PROSPER_NO_COMPUTE")) prosper::frontend::register_live_compute();
    // PROSPER_RENDER=1: register the live Vulkan renderer (shared with prosper-app via
    // frontends/shared/live_renderer) so execute_and_present composites every submitted Dcb with
    // draws and hands the frame to the present path; periodic BMP screenshots go to PROSPER_FRAME_DIR
    // (default cwd). Set PROSPER_NO_FRAME_DUMPS=1 for renderer-equivalence diagnostics without that
    // boot_trace-only disk I/O. llvmpipe renders headless in WSL.
    if (getenv("PROSPER_RENDER")) {
        std::string fdir = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
        const bool dump_bmps = getenv("PROSPER_NO_FRAME_DUMPS") == nullptr;
        prosper::frontend::register_live_renderer(fdir, dump_bmps);
    }
#endif

    BootResult r = run_entry(p.imgs[0]);
    printf("\n=== RUN ENDED: kind=%d  %s ===\n", r.kind, r.detail.c_str());
    printf("  rip=%s+0x%llx  fault_addr=0x%llx\n  rax=0x%llx rbx=? rdi=0x%llx rsi=0x%llx rdx=0x%llx rbp=0x%llx rsp=0x%llx\n",
           cls(r.fault_rip), (unsigned long long)bof(r.fault_rip), (unsigned long long)r.fault_addr,
           (unsigned long long)r.rax, (unsigned long long)r.rdi, (unsigned long long)r.rsi,
           (unsigned long long)r.rdx, (unsigned long long)r.rbp, (unsigned long long)r.rsp);
    printf("  backtrace (%zu frames):\n", r.backtrace.size());
    for (uint64_t a : r.backtrace) printf("    %-12s +0x%llx\n", cls(a), (unsigned long long)bof(a));
    // Classify the fault address vs our tracked guest mappings (0=untracked/gap, 1=reserved-uncommitted,
    // 2=committed). A write at a page boundary whose predecessor byte is committed but which is itself
    // reserved/untracked pinpoints a guest run-off-the-end-of-a-region (Windows memory-model diagnosis).
    if (r.kind == 2 && r.fault_addr) {
        auto st = [](int s){ return s==2?"committed":s==1?"reserved":"untracked/gap"; };
        printf("  [memclass] fault_addr=%s  fault_addr-8=%s  fault_addr-0x1000=%s\n",
               st(prosper_reserved_range_state(r.fault_addr)),
               st(prosper_reserved_range_state(r.fault_addr - 8)),
               st(prosper_reserved_range_state(r.fault_addr - 0x1000)));
    }
    // PROSPER_CRASHPEEK: after a recovered main-thread fault, dump guest memory at the fault registers
    // (still 1:1-mapped) so a null-source deref (e.g. an IL2CPP memcpy from a null field) can be traced
    // to the object + field that is null. Reads are bounded by guest_readable (never re-faults).
    if (getenv("PROSPER_CRASHPEEK") && r.kind == 2) {
        auto dump = [&](const char* tag, uint64_t addr) {
            printf("  [peek] %s=0x%llx:", tag, (unsigned long long)addr);
            if (!addr || !prosper::gpu::guest_readable(addr, 8)) { printf(" <unmapped>\n"); return; }
            for (int i = 0; i < 8; i++) {
                uint64_t a = addr + (uint64_t)i * 8;
                if (prosper::gpu::guest_readable(a, 8)) printf(" %016llx", (unsigned long long)*(const uint64_t*)(uintptr_t)a);
                else { printf(" ...."); break; }
            }
            printf("\n");
        };
        dump("rax", r.rax); dump("rdi", r.rdi);
        // The dest rdi is unaligned; also show the object rax as bytes (type tags / vtable ptr @+0).
        // Scan rax's first 8 qwords for the null field (the likely memcpy source).
        for (int i = 0; i < 8; i++) {
            uint64_t a = r.rax + (uint64_t)i * 8;
            if (prosper::gpu::guest_readable(a, 8) && *(const uint64_t*)(uintptr_t)a == 0)
                printf("  [peek] rax+0x%x is NULL (candidate memcpy source field)\n", i * 8);
        }
        // Dump 24 instruction bytes at the fault rip and each backtrace frame's return site
        // (return addr - 5, the call instruction) for offline `objdump -b binary -m i386:x86-64`.
        auto insn = [&](const char* tag, uint64_t addr) {
            printf("  [insn] %s 0x%llx:", tag, (unsigned long long)addr);
            for (int i = 0; i < 24; i++)
                if (prosper::gpu::guest_readable(addr + i, 1)) printf(" %02x", *(const uint8_t*)(uintptr_t)(addr + i));
                else { printf(" ??"); break; }
            printf("\n");
        };
        insn("rip", r.fault_rip);
        for (size_t i = 0; i < r.backtrace.size() && i < 5; i++)
            insn("call@", r.backtrace[i] >= 5 ? r.backtrace[i] - 5 : r.backtrace[i]);

        // --- IL2CPP managed-object class resolver ---------------------------------------
        // A managed object is { Il2CppClass* klass; void* monitor; ... }. Il2CppClass (Unity
        // 2022.3 il2cpp) starts { Il2CppImage* image; void* gc_desc; const char* name;
        // const char* namespaze; ... } so name = *(char**)(klass+0x10), ns = *(klass+0x18).
        // Given a candidate object pointer, validate klass + name look sane and print the name.
        auto rd8 = [&](uint64_t a) -> uint64_t {
            return prosper::gpu::guest_readable(a, 8) ? *(const uint64_t*)(uintptr_t)a : 0;
        };
        auto ascii_at = [&](uint64_t p, char* out, int cap) -> int {
            if (!p || !prosper::gpu::guest_readable(p, 1)) return 0;
            int n = 0;
            for (; n < cap - 1; n++) {
                if (!prosper::gpu::guest_readable(p + n, 1)) return 0;
                char c = *(const char*)(uintptr_t)(p + n);
                if (c == 0) break;
                if (c < 0x20 || (unsigned char)c > 0x7e) return 0;   // not a clean C identifier/name
                out[n] = c;
            }
            if (n == 0) return 0;
            out[n] = 0; return n;
        };
        auto klass_name = [&](uint64_t klass, char* nm, int ncap, char* ns, int nscap) -> bool {
            if (!klass || !prosper::gpu::guest_readable(klass, 0x20)) return false;
            uint64_t namep = rd8(klass + 0x10), nsp = rd8(klass + 0x18);
            if (!ascii_at(namep, nm, ncap)) return false;
            if (!ascii_at(nsp, ns, nscap)) ns[0] = 0;   // namespace may legitimately be empty
            return true;
        };
        auto try_obj = [&](const char* tag, uint64_t obj) {
            if (!obj || !prosper::gpu::guest_readable(obj, 8)) return false;
            uint64_t klass = rd8(obj);
            char nm[128], ns[128];
            if (!klass_name(klass, nm, sizeof nm, ns, sizeof ns)) return false;
            printf("  [class] %s obj=0x%llx klass=0x%llx name=\"%s%s%s\" [obj+0x40]=0x%llx\n",
                   tag, (unsigned long long)obj, (unsigned long long)klass,
                   ns[0] ? ns : "", ns[0] ? "." : "", nm, (unsigned long long)rd8(obj + 0x40));
            return true;
        };
        // PROSPER_PEEK_CLASS=0xADDR[,..]: treat each ADDR as an Il2CppClass* and print its name;
        // also chase its static-fields singleton ([class+0xb8] -> statics -> [statics+0] = obj) and name it.
        if (const char* pk = getenv("PROSPER_PEEK_CLASS")) {
            for (const char* p = pk; *p; ) {
                uint64_t k = strtoull(p, nullptr, 0);
                char nm[128], ns[128];
                if (klass_name(k, nm, sizeof nm, ns, sizeof ns))
                    printf("  [peekclass] klass=0x%llx name=\"%s%s%s\" statics=0x%llx\n",
                           (unsigned long long)k, ns[0]?ns:"", ns[0]?".":"", nm, (unsigned long long)rd8(k + 0xb8));
                else printf("  [peekclass] klass=0x%llx <no name>\n", (unsigned long long)k);
                uint64_t statics = rd8(k + 0xb8);
                if (statics) { uint64_t obj = rd8(statics);
                    try_obj("  singleton[statics+0]", obj);
                    // also scan first 8 static slots for managed objects
                    for (int i = 0; i < 16; i++) { uint64_t o = rd8(statics + i*8);
                        char t[32]; snprintf(t,sizeof t,"  static+0x%x", i*8); try_obj(t, o); }
                }
                const char* c = strchr(p, ','); if (!c) break; p = c + 1;
            }
        }
        printf("  --- il2cpp class scan (rbx=0x%llx) ---\n", (unsigned long long)r.rbx);
        try_obj("rax", r.rax);
        try_obj("rbx", r.rbx);
        try_obj("rdi", r.rdi);
        // Scan the crashing frame's stack for managed-object pointers (callerObj lives here as a
        // saved/spilled register). Report each qword whose target resolves to an Il2CppClass name.
        for (uint64_t sp = r.rsp; sp && sp <= r.rbp + 0x80 && sp < r.rsp + 0x400; sp += 8) {
            uint64_t v = rd8(sp);
            char lbl[48]; snprintf(lbl, sizeof lbl, "stack[rbp%+lld]", (long long)((int64_t)sp - (int64_t)r.rbp));
            try_obj(lbl, v);
        }
        // PROSPER_PEEK_CODE=0xADDR[,0xADDR...]: dump 512 code bytes at each guest address (still mapped
        // post-fault) for offline `objdump -b binary -m i386:x86-64 --adjust-vma=ADDR`. Used to
        // disassemble the guest GC suspend handler etc. without extracting the SELF module.
        if (const char* pc = getenv("PROSPER_PEEK_CODE")) {
            for (const char* p = pc; *p; ) {
                uint64_t a = strtoull(p, nullptr, 0);
                printf("  [code] 0x%llx:", (unsigned long long)a);
                for (int i = 0; i < 512; i++) {
                    if (prosper::gpu::guest_readable(a + i, 1)) printf(" %02x", *(const uint8_t*)(uintptr_t)(a + i));
                    else { printf(" ??"); break; }
                }
                printf("\n");
                const char* c = strchr(p, ','); if (!c) break; p = c + 1;
            }
        }
    }
    return 0;
}
