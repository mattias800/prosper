// boot_trace — link the game's modules, boot the guest, and report how far it got:
// the unimplemented-call trace (via stderr from dispatch) plus, on a fault, the register
// state and an rbp-chain backtrace classified by module. The primary bring-up debugging
// tool. Linux only. Usage: boot_trace <dump-root>
#include "loader/linker.hpp"
#include "host/exec_image.hpp"
#include "hle/dispatch.hpp"
#include <cstdio>
#include <string>
#ifdef PROSPER_HAVE_VULKAN
#include "gpu/gpu_execute.hpp"
#include "../../tests/render_runner.h"   // offscreen Vulkan backend (render_triangle_rgba) + dump_bmp
#include <atomic>
#include <cstdlib>
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
               const prosper::gpu::ResolvedPipelineState& ps, uint32_t w, uint32_t h) {
                // Dump the recompiled SPIR-V FIRST (before the slow Vulkan render), so it survives even if
                // a concurrent worker fault kills the process mid-render — lets us spirv-val it offline.
                if (getenv("PROSPER_SHADER_DUMP")) {
                    std::string d = getenv("PROSPER_SHADER_DUMP");
                    if (FILE* f = fopen((d + "/frame_vs.spv").c_str(), "wb")) { fwrite(vs.data(), 4, vs.size(), f); fclose(f); }
                    if (FILE* f = fopen((d + "/frame_fs.spv").c_str(), "wb")) { fwrite(fs.data(), 4, fs.size(), f); fclose(f); }
                    fprintf(stderr, "[render] dumped SPIR-V vs=%zu fs=%zu dwords\n", vs.size(), fs.size()); fflush(stderr);
                }
                std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vs, fs, w, h, &ps);
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
