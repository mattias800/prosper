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
#endif

using namespace prosper;


// Module bases (keep in sync with the inputs below).
static const uint64_t EBOOT = 0x400000000ull, IL2CPP = 0x440000000ull, PS5UTIL = 0x4c0000000ull,
                      LIBC = 0x500000000ull, STUB = 0x600000000ull;
static const char* cls(uint64_t a) {
    if (a >= EBOOT   && a < IL2CPP)  return "eboot";
    if (a >= IL2CPP  && a < PS5UTIL) return "Il2cpp";
    if (a >= PS5UTIL && a < LIBC)    return "PS5Util";
    if (a >= LIBC    && a < STUB)    return "libc.prx";
    if (a >= STUB    && a < 0x610000000ull) return "STUB";
    return "mapped/host";
}
static uint64_t bof(uint64_t a) {
    if (a >= EBOOT   && a < IL2CPP)  return a - EBOOT;
    if (a >= IL2CPP  && a < PS5UTIL) return a - IL2CPP;
    if (a >= PS5UTIL && a < LIBC)    return a - PS5UTIL;
    if (a >= LIBC    && a < STUB)    return a - LIBC;
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
        { d + "/sce_module/libc.prx", LIBC },
    };
    if (!link_program(in, STUB, p, &e)) { printf("link failed: %s\n", e.c_str()); return 1; }
    printf("linked %zu modules; %zu imports (%zu cross-module, %zu stub slots); %zu init fns\n",
           p.mods.size(), p.total_imports, p.resolved_cross_module, p.slots.size(), p.init_fns.size());

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
    run_guest_inits(p.init_fns);

#ifdef PROSPER_HAVE_VULKAN
    // PROSPER_RENDER=1: register the live Vulkan renderer so execute_and_present fires on every
    // submitted Dcb with draws (Stage A of GPU_EXECUTOR_DESIGN.md, now live). Each rendered frame
    // goes to the present path; the first few (and then every 60th) are also dumped as BMP
    // screenshots under PROSPER_FRAME_DIR (default cwd). llvmpipe renders headless in WSL.
    if (getenv("PROSPER_RENDER")) {
        static std::atomic<int> frame_no{0};
        static std::string fdir = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
        prosper::gpu::set_submit_renderer(
            [](const std::vector<uint32_t>& vs, const std::vector<uint32_t>& fs,
               const prosper::gpu::ResolvedPipelineState& ps,
               const prosper::gpu::ShaderResourceTable* vrt, const prosper::gpu::ShaderResourceTable* prt,
               uint32_t w, uint32_t h, uint32_t draw_vcount) {
                // Dump the recompiled SPIR-V FIRST (before the slow Vulkan render), so it survives even if
                // a concurrent worker fault kills the process mid-render — lets us spirv-val it offline.
                if (getenv("PROSPER_SHADER_DUMP")) {
                    std::string d = getenv("PROSPER_SHADER_DUMP");
                    if (FILE* f = fopen((d + "/frame_vs.spv").c_str(), "wb")) { fwrite(vs.data(), 4, vs.size(), f); fclose(f); }
                    if (FILE* f = fopen((d + "/frame_fs.spv").c_str(), "wb")) { fwrite(fs.data(), 4, fs.size(), f); fclose(f); }
                    fprintf(stderr, "[render] dumped SPIR-V vs=%zu fs=%zu dwords\n", vs.size(), fs.size()); fflush(stderr);
                }
                // Bind EVERY resource each shader declares, each at its own descriptor binding, reading the
                // bytes from 1:1-mapped guest memory (an unbound image_sample/storage binding is UB). The
                // executor gave each constant/vertex buffer + texture a distinct binding; the recompiler
                // declared a storage buffer / image sampler at each, so we must bind them all.
                using RC = prosper::gpu::ResourceClass;
                int dn = open("/dev/null", O_WRONLY);
                auto readable = [&](uint64_t a, size_t n){ return a > 0x1000 && dn >= 0 && write(dn, (const void*)(uintptr_t)a, n) == (ssize_t)n; };
                std::vector<prosper::test::FrameResource> R;
                static std::vector<std::vector<uint8_t>> texstore;  // keep texture bytes alive across the call
                texstore.clear();
                auto add = [&](const prosper::gpu::ShaderResourceTable* t){
                    if (!t) return;
                    for (auto& r : t->resources) {
                        prosper::test::FrameResource fr; fr.binding = r.binding;
                        if (r.cls == RC::Texture || r.cls == RC::StorageImage) {
                            uint32_t tw = r.width ? r.width : 4, th = r.height ? r.height : 4;
                            size_t nb = (size_t)tw * th * 4;
                            texstore.emplace_back(nb, 0x00);
                            if (readable(r.gpu_addr, nb)) std::memcpy(texstore.back().data(), (const void*)(uintptr_t)r.gpu_addr, nb);
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
                            // PROSPER_DETILE=1: de-swizzle the GPU-tiled render target (SW_4KB_S) into the
                            // linear texstore via the shared, tested gpu module. Reads the PADDED tiled
                            // buffer (height rounded to whole 32-texel tile rows) so the bottom tile-row
                            // isn't dropped. (The T# tile_mode should drive this once threaded through the
                            // resource table; gated for now.)
                            if (const char* dt = getenv("PROSPER_DETILE"); dt && atoi(dt) != 0) {
                                const uint32_t tmode = (uint32_t)prosper::gpu::TileMode::Sw4KbS;
                                size_t tiled_bytes = prosper::gpu::tiled_surface_bytes(tw, th, tmode);
                                std::vector<uint8_t> tiled(tiled_bytes, 0);
                                if (readable(r.gpu_addr, tiled_bytes))
                                    std::memcpy(tiled.data(), (const void*)(uintptr_t)r.gpu_addr, tiled_bytes);
                                prosper::gpu::detile_surface(texstore.back().data(), tiled.data(), tw, th, tmode);
                            }
                            // PROSPER_DUMP_TEX: write the RAW texture memory (interpreted linearly) to a BMP,
                            // bypassing the shader — reveals whether the render target is tiled (scrambled) or
                            // linear (recognizable), and the tiling structure.
                            if (getenv("PROSPER_DUMP_TEX") && !texstore.back().empty()) {
                                std::string d = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
                                char fn[512]; snprintf(fn, sizeof fn, "%s/rawtex_b%u.bmp", d.c_str(), r.binding);
                                prosper::test::dump_bmp(fn, texstore.back(), tw, th);
                                fprintf(stderr, "[render] dumped raw texture -> %s\n", fn); fflush(stderr);
                            }
                            fr.tex_rgba = texstore.back().data(); fr.tw = tw; fr.th = th;
                        } else {
                            uint32_t nb = std::min(r.size ? r.size : 256u, 1u << 20);   // cap 1 MB
                            if (nb >= 4 && readable(r.gpu_addr, nb))
                                fr.dwords.assign((const uint32_t*)(uintptr_t)r.gpu_addr, (const uint32_t*)(uintptr_t)(r.gpu_addr + (nb & ~3u)));
                            if (fr.dwords.empty()) fr.dwords.assign(64, 0);
                        }
                        R.push_back(std::move(fr));
                    }
                };
                add(vrt); add(prt);
                if (dn >= 0) close(dn);
                if (getenv("PROSPER_GFXLOG")) { fprintf(stderr, "[render] binding %zu resources (draw %u verts)\n",
                    R.size(), draw_vcount); fflush(stderr); }
                // PROSPER_RENDER_REFVS: replace the game's (intricate) vertex shader with a known-good
                // fullscreen-triangle VS that exports uv in [0,1] across the screen (location 1) + white
                // (location 0). Paired with the game's REAL pixel shader + texture, this shows the game's
                // actual composited texture — isolating the VS from the rest of the pipeline.
                std::vector<uint32_t> vs_use = vs; uint32_t vcount_use = draw_vcount;
                if (getenv("PROSPER_RENDER_REFVS")) {
                    #include "refvs.inc"
                    vs_use.assign(kRefVs, kRefVs + sizeof(kRefVs) / 4); vcount_use = 3;
                }
                // PROSPER_RENDER_TESTPS: replace the real pixel shader with a solid MAGENTA one (recompiled
                // from a tiny RDNA2 EXP blob) to isolate VS geometry from PS shading — if the VS positions
                // are on-screen, magenta triangles appear regardless of the texture/PS math.
                std::vector<uint32_t> fs_use = fs;
                if (getenv("PROSPER_RENDER_TESTPS")) {
                    static const uint32_t kMagentaPs[] = {   // v0=1.0(R) v1=0.0(G) v2=1.0(B) v3=1.0(A); exp mrt0; endpgm
                        0x7E0002F2u, 0x7E020280u, 0x7E0402F2u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u };
                    auto m = prosper::gpu::recompile_fragment(kMagentaPs, sizeof(kMagentaPs) / 4, nullptr);
                    if (!m.empty()) fs_use = m;
                }
                if (getenv("PROSPER_GFXLOG")) fprintf(stderr,
                    "[render] ps: topo=%u fmt=%u depth(test=%d write=%d op=%u) blend(en=%d src=%u dst=%u op=%u) mask=0x%x\n",
                    ps.topology, ps.color0_format, ps.depth_test_enable, ps.depth_write_enable, ps.depth_compare_op,
                    ps.blend_enable, ps.src_color_blend_factor, ps.dst_color_blend_factor, ps.color_blend_op, ps.color_write_mask);
                // PROSPER_RENDER_NOPS: bypass the game's resolved pipeline state (default state instead) to
                // isolate whether blend/depth/color-mask is discarding the fragments.
                const prosper::gpu::ResolvedPipelineState* ps_use = getenv("PROSPER_RENDER_NOPS") ? nullptr : &ps;
                std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vs_use, fs_use, w, h, ps_use,
                    nullptr, nullptr, nullptr, R.empty() ? nullptr : &R, vcount_use);
                int n = frame_no++;
                if (px.empty()) {
                    fprintf(stderr, "[render] frame %d: Vulkan render FAILED (%ux%u)\n", n, w, h);
                } else if (n < 8 || n % 60 == 0) {
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
    return 0;
}
