// boot_trace — link the game's modules, boot the guest, and report how far it got:
// the unimplemented-call trace (via stderr from dispatch) plus, on a fault, the register
// state and an rbp-chain backtrace classified by module. The primary bring-up debugging
// tool. Linux only. Usage: boot_trace <dump-root>
#include "loader/linker.hpp"
#include "host/exec_image.hpp"
#include "hle/dispatch.hpp"
#include <cstdio>
#include <string>
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
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
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
    std::string d = (argc >= 2) ? argv[1] : "../../PPSA24651-app0";
    Program p; std::string e;
    // libc.prx loaded last => its init_array runs first (deepest dependency), before eboot's entry.
    // Experimental (branch libc-prx-integration): route eboot's 145 libc imports to the REAL Sony
    // libc instead of our HLE. Cross-module export beats the HLE stub slot (see linker.cpp pass 2).
    std::vector<LinkInput> in = {
        { d + "/eboot.bin", EBOOT },
        { d + "/Media/Modules/Il2cppUserAssemblies.prx", IL2CPP },
        { d + "/Media/Modules/PS5Util.prx", PS5UTIL },
        // Native PSN Unity plugin. Loading it as a guest module (its imports resolve to our HLE)
        // makes its exports — PSN_PrxInitialize / PSN_PrxUpdate / UnityPluginLoad — reachable via
        // sceKernelDlsym, so the managed PSN bootstrap (Unity.PSN.PS5.Main.Initialize) can start the
        // async worker pool. Gate behind PROSPER_NO_PSN=1 to fall back to the 4-module boot.
        { d + "/Media/Plugins/PSN.prx", PSN },
        // Native SaveData Unity plugin — exports PrxSaveDataInitialize / PrxSaveDataMount / etc. The
        // managed SaveData layer dlsym's these at startup to read the save slot (new-game vs. continue),
        // which gates the transition from the loading screen into the intro. Same load path as PSN.prx.
        { d + "/Media/Plugins/SaveData.prx", SAVEDATA },
        { d + "/sce_module/libc.prx", LIBC },
    };
    if (getenv("PROSPER_NO_PSN"))
        for (size_t i = in.size(); i-- > 0; )
            if (in[i].path.find("PSN.prx") != std::string::npos ||
                in[i].path.find("SaveData.prx") != std::string::npos) in.erase(in.begin() + (ptrdiff_t)i);
    // Cross-title tolerance (docs/CROSS_ENGINE_UE4.md step 1, minimal form): drop dependent modules
    // whose file doesn't exist in this dump — e.g. the UE4 title ships no Il2cpp/PS5Util, so it links
    // eboot + libc.prx only. The eboot (index 0) is always kept; each module keeps its fixed base.
    // For the primary target every file exists, so its link is byte-for-byte unchanged.
    for (size_t i = in.size(); i-- > 1; ) {
        if (FILE* f = fopen(in[i].path.c_str(), "rb")) fclose(f);
        else { printf("skipping absent module: %s\n", in[i].path.c_str()); in.erase(in.begin() + (ptrdiff_t)i); }
    }
    if (!link_program(in, STUB, p, &e)) { printf("link failed: %s\n", e.c_str()); return 1; }
    printf("linked %zu modules; %zu imports (%zu cross-module, %zu stub slots); %zu init fns\n",
           p.mods.size(), p.total_imports, p.resolved_cross_module, p.slots.size(), p.init_fns.size());

    // Let sceKernelDlsym resolve exported symbols by name against all loaded modules (e.g. the
    // PSN.prx plugin's PSN_PrxInitialize / UnityPluginLoad), so Unity's native-plugin loader binds
    // real export addresses instead of getting ESRCH.
    set_module_exports(&p.exports);

    register_builtin_hle();
#ifdef PROSPER_AUDIO_SDL3
    prosper::install_sdl3_audio_sink();   // route the guest's sceAudioOut output to the host via SDL3
#endif
    // Controller input, gated by PROSPER_PAD (default: core's neutral/disconnected pad, deterministic).
    // Prefer the cross-platform SDL3 gamepad backend; fall back to the zero-dep Linux evdev reader.
    if (getenv("PROSPER_PAD")) {
        bool pad_ok = false; (void)pad_ok;
#ifdef PROSPER_PAD_SDL3
        pad_ok = prosper::install_sdl3_pad_backend();
#endif
#ifdef PROSPER_PAD_EVDEV
        if (!pad_ok) pad_ok = prosper::install_evdev_pad_backend();
#endif
    }
    set_app0_root(d);
    for (auto& img : p.imgs) if (!map_image(img, &e)) { printf("map failed: %s\n", e.c_str()); return 1; }
    { std::vector<TlsModuleDesc> td; for (auto& t : p.tls_templates) td.push_back({t.init_va, t.filesz, t.memsz});
      set_tls_modules(td.data(), td.size());      // enable __tls_get_addr for loaded modules (real libc.prx)
      guest_tls_set_templates(td.data(), td.size()); }   // gated PROSPER_GUEST_FS: guest initial-exec %fs TLS
    // C++ exception unwinding: give sceKernelGetModuleInfoForUnwind each module's .eh_frame_hdr + text seg.
    { static std::vector<std::string> names; names.reserve(p.imgs.size());   // stable name storage
      std::vector<UnwindModuleDesc> um;
      for (size_t i = 0; i < p.imgs.size() && i < p.mods.size(); i++) {
          auto& img = p.imgs[i]; auto& mod = *p.mods[i];
          UnwindModuleDesc dd; dd.lo = img.base + img.min_vaddr; dd.hi = img.base + img.max_vaddr;
          for (auto& s : mod.segments) if (s.type == 0x6474e550u) { dd.ehframe_hdr = img.base + s.vaddr; dd.ehframe_hdr_sz = s.memsz; }
          if (!mod.loads.empty()) { dd.seg0 = img.base + mod.loads[0].vaddr; dd.seg0_sz = mod.loads[0].memsz; }
          std::string nm = mod.path; auto sl = nm.find_last_of("/\\"); if (sl != std::string::npos) nm = nm.substr(sl + 1);
          names.push_back(nm); dd.name = names.back().c_str(); um.push_back(dd);
      }
      set_unwind_modules(um.data(), um.size()); }
    // sceKernelGetProcParam -> eboot's SCE_PROCPARAM (real libc reads its heap/malloc config here).
    for (auto& s : p.mods[0]->segments)
        if (s.type == PT_SCE_PROCPARAM) { set_proc_param(EBOOT + s.vaddr); break; }
    if (!install_stubs(p.slots, p.stub_base, p.stub_size, &e)) { printf("stubs failed: %s\n", e.c_str()); return 1; }
    install_trap_handler();
    // PSN.prx / SaveData.prx are SCE native plugins whose user module_start validates a module-param
    // descriptor {size=0x10, version=0x200, cb=NULL} and null-faults if started with (argc=0, argp=NULL)
    // — the fault that left PSN unregistered ("PSN is an old version"). Register their guest ranges so
    // run_guest_inits starts them with the real descriptor. Skipped when PROSPER_NO_PSN drops them.
    if (!getenv("PROSPER_NO_PSN"))
        set_module_start_param_ranges({ { PSN, SAVEDATA }, { SAVEDATA, LIBC } });
    run_guest_inits(p.init_fns);

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

#ifdef PROSPER_HAVE_VULKAN
    // PROSPER_RENDER=1: register the live Vulkan renderer so execute_and_present fires on every
    // submitted Dcb with draws (Stage A of GPU_EXECUTOR_DESIGN.md, now live). Each rendered frame
    // goes to the present path; the first few (and then every 60th) are also dumped as BMP
    // screenshots under PROSPER_FRAME_DIR (default cwd). llvmpipe renders headless in WSL.
    if (getenv("PROSPER_RENDER")) {
        static std::atomic<int> frame_no{0};
        static std::string fdir = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
        prosper::gpu::set_submit_renderer(
            [](const std::vector<prosper::gpu::DrawItem>& items, uint32_t w, uint32_t h) -> std::vector<uint8_t> {
                using RC = prosper::gpu::ResourceClass;
                // PROSPER_RENDER_FIRST=<N>: skip the slow (~400x) Vulkan render for the first N GPU submits, so
                // the game reaches a LATE scene (e.g. the level1 cutscene, which only starts submitting after
                // ~5000 title-loop submits) at native speed before we begin rendering/dumping. Returning {}
                // means "not rendered this submit". Without this, rendering from boot is far too slow to ever
                // reach a post-loading-screen scene.
                static std::atomic<int> g_submit_idx{0};
                static int g_render_first = getenv("PROSPER_RENDER_FIRST") ? atoi(getenv("PROSPER_RENDER_FIRST")) : 0;
                if (g_submit_idx++ < g_render_first) return {};
                // Dump the FIRST item's recompiled SPIR-V (diagnostic; survives a mid-render crash).
                if (getenv("PROSPER_SHADER_DUMP") && !items.empty()) {
                    std::string d = getenv("PROSPER_SHADER_DUMP");
                    if (FILE* f = fopen((d + "/frame_vs.spv").c_str(), "wb")) { fwrite(items[0].vs.data(), 4, items[0].vs.size(), f); fclose(f); }
                    if (FILE* f = fopen((d + "/frame_fs.spv").c_str(), "wb")) { fwrite(items[0].fs.data(), 4, items[0].fs.size(), f); fclose(f); }
                    fprintf(stderr, "[render] dumped SPIR-V vs=%zu fs=%zu dwords\n", items[0].vs.size(), items[0].fs.size()); fflush(stderr);
                }
                // Copy [a, a+n) into dst, but stop at the first 64KB block that is NOT within a reserved guest
                // mapping (prosper_reserved_range_state == 0). A resource's declared size (e.g. a 2048x1024
                // atlas = 8 MB) can run past its real committed backing; an unguarded memcpy then walks off
                // the mapping and SIGSEGVs the render thread mid-copy — /dev/null-write "readable" can't catch
                // it because /dev/null discards without faulting. Returns bytes copied; dst is pre-zeroed so a
                // short copy just leaves a transparent/black tail (only the missing region degrades).
                auto safe_copy = [](uint8_t* dst, uint64_t a, size_t n) -> size_t {
                    const size_t PG = 0x10000;   // lazy-commit granularity (64 KB)
                    size_t done = 0;
                    while (done < n) {
                        uint64_t cur = a + done;
                        if (cur < 0x1000 || prosper_reserved_range_state(cur) == 0) break;
                        size_t chunk = std::min(n - done, PG - (size_t)(cur & (PG - 1)));
                        std::memcpy(dst + done, (const void*)(uintptr_t)cur, chunk);
                        done += chunk;
                    }
                    return done;
                };
                static std::vector<std::vector<uint8_t>> texstore;  // keep every item's texture bytes alive
                texstore.clear();
                // Build one draw's set-tagged resources from its VS (set 0) + PS (set 1) tables — read the
                // bytes from 1:1-mapped guest memory, detile textures. (Each constant/vertex buffer + texture
                // gets its own binding; the recompiler declared a storage buffer / image sampler at each.)
                auto build_R = [&](const prosper::gpu::ShaderResourceTable* vrt, const prosper::gpu::ShaderResourceTable* prt) {
                  std::vector<prosper::test::FrameResource> R;
                  auto add = [&](const prosper::gpu::ShaderResourceTable* t, uint32_t set){
                    if (!t) return;
                    for (auto& r : t->resources) {
                        prosper::test::FrameResource fr; fr.binding = r.binding; fr.set = set;
                        if (r.cls == RC::Texture || r.cls == RC::StorageImage) {
                            uint32_t tw = r.width ? r.width : 4, th = r.height ? r.height : 4;
                            size_t nb = (size_t)tw * th * 4;
                            texstore.emplace_back(nb, 0x00);
                            // Block-compressed (BC1/2/3): read the (possibly tiled) compressed blocks, block-
                            // detile, and decode to RGBA8 in-place. The blocks are the tiled element (BC3 =
                            // 16 bytes -> SW_4KB_S 16x16-block micro-tiles). #121.
                            const uint32_t bcb = prosper::gpu::bc_block_bytes(r.format);
                            if (bcb) {
                                uint32_t bw = (tw + 3) / 4, bh = (th + 3) / 4;
                                // CONFIDENCE: MED — SW_4KB_S keeps a 4KB micro-tile, so element count =
                                // 4096/bpe arranged square: 16-byte blocks (BC2/3/5/7) -> 16x16, 8-byte
                                // (BC1/4) -> 32x16 (approximated as 32 here). Verified against the 32-bpp
                                // path (bpe=4 -> 32x32) and llvmpipe re-tiling, but NOT yet against a
                                // structured (non-uniform) BC surface — the one live BC3 tex is ~solid white.
                                uint32_t tside = (bcb == 16) ? 16u : 32u;
                                size_t comp_bytes = (size_t)bw * bh * bcb;
                                std::vector<uint8_t> lin(comp_bytes, 0);
                                bool tiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode) && !getenv("PROSPER_NODETILE");
                                if (tiled) {
                                    size_t tbytes = prosper::gpu::tiled_elements_bytes(bw, bh, bcb, tside, r.tile_mode);
                                    std::vector<uint8_t> traw(tbytes, 0);
                                    safe_copy(traw.data(), r.gpu_addr, tbytes);
                                    prosper::gpu::detile_elements(lin.data(), traw.data(), tbytes, bw, bh, bcb, tside, r.tile_mode);
                                } else {
                                    safe_copy(lin.data(), r.gpu_addr, comp_bytes);
                                }
                                prosper::gpu::bc_decode_surface(texstore.back().data(), lin.data(), lin.size(), tw, th, r.format);
                            } else {
                                safe_copy(texstore.back().data(), r.gpu_addr, nb);
                            }
                            // PROSPER_DUMP_RAWTEX: write the raw tiled RGBA bytes (pre-detile) to a .bin for
                            // offline swizzle experimentation.
                            if (getenv("PROSPER_DUMP_RAWTEX") && !texstore.back().empty()) {
                                std::string d = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
                                char fn[512]; snprintf(fn, sizeof fn, "%s/rawtex_%ux%u.bin", d.c_str(), tw, th);
                                if (FILE* f = fopen(fn, "wb")) { fwrite(texstore.back().data(), 1, nb, f); fclose(f);
                                    fprintf(stderr, "[render] dumped raw tiled bytes -> %s (%zu)\n", fn, nb); fflush(stderr); }
                            }
                            // PROSPER_TESTTEX: overwrite with a recognizable gradient/checker to prove the
                            // sample path (the game's real render-target texture is empty until we execute
                            // the scene draws that fill it).
                            if (getenv("PROSPER_TESTTEX")) {
                                for (uint32_t y = 0; y < th; y++) for (uint32_t x = 0; x < tw; x++) {
                                    uint8_t* p = &texstore.back()[((size_t)y * tw + x) * 4];
                                    bool ck = ((x / 64) ^ (y / 64)) & 1;
                                    p[0] = (uint8_t)(255 * x / tw); p[1] = (uint8_t)(255 * y / th);
                                    p[2] = ck ? 200 : 40; p[3] = 255;
                                }
                            }
                            if (getenv("PROSPER_GFXLOG")) { const uint8_t* b = texstore.back().data();
                                size_t nz = 0; for (size_t i = 0; i < nb && i < (1u<<16); i++) nz += (b[i] != 0);
                                fprintf(stderr, "[render] tex binding=%u %ux%u first64k-nonzero=%zu\n", r.binding, tw, th, nz); }
                            // AUTO-DETILE: de-swizzle a GPU-tiled sampled surface into the linear texstore,
                            // driven by the T# tile_mode now threaded through the resource table (r.tile_mode).
                            // PROSPER_DETILE forces it (for surfaces whose tiling we don't yet flag); PROSPER_NODETILE
                            // disables it. Reads the PADDED tiled buffer (height rounded to whole 32-texel tile rows).
                            bool auto_tiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode);
                            const char* dt = getenv("PROSPER_DETILE");
                            // BC textures already block-detiled + decoded above; the 32-bpp detiler must not touch them.
                            if (!bcb && !getenv("PROSPER_NODETILE") && (auto_tiled || (dt && atoi(dt) != 0))) {
                                const uint32_t tmode = auto_tiled ? r.tile_mode : (uint32_t)prosper::gpu::TileMode::Sw4KbS;
                                // PROSPER_PITCH: padded row pitch in texels for surfaces whose pitch > width.
                                const uint32_t pitch = getenv("PROSPER_PITCH") ? (uint32_t)atoi(getenv("PROSPER_PITCH")) : 0;
                                size_t tiled_bytes = prosper::gpu::tiled_surface_bytes(tw, th, tmode, pitch);
                                std::vector<uint8_t> tiled(tiled_bytes, 0);
                                size_t got = safe_copy(tiled.data(), r.gpu_addr, tiled_bytes);
                                if (got < nb)
                                    // The padded tiled buffer's tail (th rounded to whole 32-row tiles) runs
                                    // past the real backing: fall back to the width*height linear bytes copied
                                    // above rather than an all-zero buffer (which would BLANK the texture).
                                    // Only the bottom tile-row degrades.
                                    std::memcpy(tiled.data(), texstore.back().data(), std::min(nb, tiled_bytes));
                                // PROSPER_DUMP_RAWTILE: write the EXACT padded tiled bytes to a .bin (no lossy BMP
                                // round-trip) so the de-swizzle can be reversed offline against a known image (#101).
                                if (getenv("PROSPER_DUMP_RAWTILE") && frame_no < 200) {
                                    std::string dd = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
                                    char bn[512]; snprintf(bn, sizeof bn, "%s/tiled_f%04d_b%u_%ux%u.bin",
                                                           dd.c_str(), (int)frame_no, r.binding, tw, th);
                                    if (FILE* bf = fopen(bn, "wb")) { fwrite(tiled.data(), 1, tiled.size(), bf); fclose(bf); }
                                }
                                prosper::gpu::detile_surface(texstore.back().data(), tiled.data(), tw, th, tmode, pitch);
                            }
                            // PROSPER_DUMP_TEX: write the RAW texture memory (interpreted linearly) to a BMP,
                            // bypassing the shader — reveals whether the render target is tiled (scrambled) or
                            // linear (recognizable), and the tiling structure.
                            if (getenv("PROSPER_DUMP_TEX") && !texstore.back().empty() && frame_no < 200) {
                                std::string d = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
                                // Frame-numbered so a specific frame's texture (loading screen, cutscene) can be
                                // recovered — a bare per-binding name overwrites every frame. Bounded to the first
                                // 200 frames (loading + intro) so a long run doesn't fill the disk.
                                char fn[512]; snprintf(fn, sizeof fn, "%s/rawtex_f%04d_b%u.bmp", d.c_str(), (int)frame_no, r.binding);
                                prosper::test::dump_bmp(fn, texstore.back(), tw, th);
                                fprintf(stderr, "[render] dumped raw texture -> %s\n", fn); fflush(stderr);
                            }
                            fr.tex_rgba = texstore.back().data(); fr.tw = tw; fr.th = th;
                        } else {
                            uint32_t nb = std::min(r.size ? r.size : 256u, 1u << 20) & ~3u;   // cap 1 MB, dword-aligned
                            if (nb >= 4) {
                                std::vector<uint8_t> tmp(nb, 0);
                                if (safe_copy(tmp.data(), r.gpu_addr, nb) > 0)
                                    fr.dwords.assign((const uint32_t*)tmp.data(), (const uint32_t*)(tmp.data() + nb));
                            }
                            if (fr.dwords.empty()) fr.dwords.assign(64, 0);
                        }
                        R.push_back(std::move(fr));
                    }
                  };
                  add(vrt, 0); add(prt, 1);   // VS resources -> descriptor set 0, PS -> set 1
                  return R;
                };
                // Diagnostic shader/state overrides (computed once, applied to EVERY draw item):
                //   REFVS  -> a known-good fullscreen-triangle VS (isolates the game's real VS).
                //   TESTPS -> a solid-magenta PS (isolates VS geometry from PS shading).
                //   FS_SPV -> a caller-supplied PS SPIR-V (e.g. a UV visualizer).
                //   NOPS   -> bypass the resolved pipeline state (default state).
                #include "refvs.inc"
                const bool refvs = getenv("PROSPER_RENDER_REFVS");
                std::vector<uint32_t> refvs_spv(kRefVs, kRefVs + sizeof(kRefVs) / 4);
                std::vector<uint32_t> ps_override;
                if (getenv("PROSPER_RENDER_TESTPS")) {
                    static const uint32_t kMagentaPs[] = {   // v0=1.0(R) v1=0.0(G) v2=1.0(B) v3=1.0(A); exp mrt0; endpgm
                        0x7E0002F2u, 0x7E020280u, 0x7E0402F2u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u };
                    ps_override = prosper::gpu::recompile_fragment(kMagentaPs, sizeof(kMagentaPs) / 4, nullptr);
                }
                if (const char* fsp = getenv("PROSPER_FS_SPV")) {
                    if (FILE* f = fopen(fsp, "rb")) {
                        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
                        std::vector<uint32_t> m((size_t)sz / 4);
                        if (sz >= 20 && fread(m.data(), 4, m.size(), f) == m.size()) ps_override = std::move(m);
                        fclose(f);
                    }
                }
                const bool nops = getenv("PROSPER_RENDER_NOPS");
                // Assemble the backend draws — one per realized DrawItem, each with its own resources +
                // fixed-function state (or the diagnostic overrides above). render_draws_rgba composites
                // them into ONE framebuffer.
                std::vector<prosper::test::BackendDraw> bds;
                for (const auto& it : items) {
                    prosper::test::BackendDraw bd;
                    bd.vs     = refvs ? refvs_spv : it.vs;
                    bd.fs     = ps_override.empty() ? it.fs : ps_override;
                    bd.vcount = refvs ? 3u : it.vertex_count;
                    bd.ps     = nops ? nullptr : &it.ps;
                    bd.R      = build_R(it.vrt.get(), it.prt.get());
                    // Indexed draw: hand the executor-fetched index data to the backend (vkCmdDrawIndexed).
                    // Skipped under REFVS — the reference VS is a 3-vertex non-indexed fullscreen triangle.
                    if (!refvs) bd.indices = it.indices;
                    if (getenv("PROSPER_GFXLOG")) fprintf(stderr,
                        "[render] item %zu: %zu resources vcount=%u nidx=%zu topo=%u mask=0x%x blend=%d\n",
                        bds.size(), bd.R.size(), bd.vcount, bd.indices.size(), it.ps.topology,
                        it.ps.color_write_mask, (int)it.ps.blend_enable);
                    bds.push_back(std::move(bd));
                }
                std::vector<uint8_t> px = prosper::test::render_draws_rgba(bds, w, h);
                int n = frame_no++;
                if (px.empty()) {
                    fprintf(stderr, "[render] frame %d: Vulkan render FAILED (%ux%u)\n", n, w, h);
                } else if (n < 60 || n % 10 == 0) {   // widened to capture the title fade-in progression
                    char fn[512]; snprintf(fn, sizeof fn, "%s/frame_%04d.bmp", fdir.c_str(), n);
                    prosper::test::dump_bmp(fn, px, w, h);
                    fprintf(stderr, "[render] frame %d rendered (%ux%u) -> %s\n", n, w, h, fn);
                }
                return px;
            });
        fprintf(stderr, "[render] live Vulkan submit renderer registered (frames -> %s)\n", fdir.c_str());
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
