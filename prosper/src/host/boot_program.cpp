// boot_program.cpp — see boot_program.hpp. Extracted verbatim from boot_trace's boot sequence
// (behavior-preserving); the guest-execution substrate is Linux-only, so guard the whole body.
#include "boot_program.hpp"

#ifdef __linux__
#include "host/exec_image.hpp"
#include "hle/dispatch.hpp"
#include "self/module.hpp"     // PT_SCE_PROCPARAM
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace prosper {

bool boot_program(const std::string& d, Program& p, std::string* err,
                  const std::function<void()>& after_hle_registered) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };

    // libc.prx loaded last => its init_array runs first (deepest dependency), before eboot's entry.
    std::vector<LinkInput> in = {
        { d + "/eboot.bin", BOOT_EBOOT },
        { d + "/Media/Modules/Il2cppUserAssemblies.prx", BOOT_IL2CPP },
        { d + "/Media/Modules/PS5Util.prx", BOOT_PS5UTIL },
        // Native PSN / SaveData Unity plugins: loaded as guest modules so their exports
        // (PSN_PrxInitialize / UnityPluginLoad / PrxSaveDataInitialize ...) resolve via sceKernelDlsym.
        { d + "/Media/Plugins/PSN.prx", BOOT_PSN },
        { d + "/Media/Plugins/SaveData.prx", BOOT_SAVEDATA },
        { d + "/sce_module/libc.prx", BOOT_LIBC },
    };
    if (getenv("PROSPER_NO_PSN"))
        for (size_t i = in.size(); i-- > 0; )
            if (in[i].path.find("PSN.prx") != std::string::npos ||
                in[i].path.find("SaveData.prx") != std::string::npos) in.erase(in.begin() + (ptrdiff_t)i);
    // Cross-title tolerance: drop dependent modules whose file doesn't exist in this dump (the eboot
    // at index 0 is always kept; each module keeps its fixed base).
    for (size_t i = in.size(); i-- > 1; ) {
        if (FILE* f = fopen(in[i].path.c_str(), "rb")) fclose(f);
        else { printf("skipping absent module: %s\n", in[i].path.c_str()); in.erase(in.begin() + (ptrdiff_t)i); }
    }

    std::string e;
    if (!link_program(in, BOOT_STUB, p, &e)) return fail("link failed: " + e);
    printf("linked %zu modules; %zu imports (%zu cross-module, %zu stub slots); %zu init fns\n",
           p.mods.size(), p.total_imports, p.resolved_cross_module, p.slots.size(), p.init_fns.size());

    // sceKernelDlsym resolves exports by name against all loaded modules.
    set_module_exports(&p.exports);
    // Per-module tables (#147): LoadStartModule hands out real handles for linked-module paths and
    // dlsym consults the handle's module before the global first-definition-wins table.
    { std::vector<ModuleExportTable> mt;
      for (const auto& me : p.mod_exports) mt.push_back({ me.path, &me.nids });
      set_module_export_tables(std::move(mt)); }

    register_builtin_hle();
    if (after_hle_registered) after_hle_registered();   // caller installs host frontends here

    set_app0_root(d);
    for (auto& img : p.imgs) if (!map_image(img, &e)) return fail("map failed: " + e);
    { std::vector<TlsModuleDesc> td; for (auto& t : p.tls_templates) td.push_back({t.init_va, t.filesz, t.memsz});
      set_tls_modules(td.data(), td.size());              // __tls_get_addr for loaded modules (real libc.prx)
      guest_tls_set_templates(td.data(), td.size()); }    // gated PROSPER_GUEST_FS: guest initial-exec %fs TLS

    // C++ exception unwinding: give each module's .eh_frame_hdr + text segment to the unwinder.
    { static std::vector<std::string> names; names.reserve(p.imgs.size());
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
        if (s.type == PT_SCE_PROCPARAM) { set_proc_param(BOOT_EBOOT + s.vaddr); break; }

    if (!install_stubs(p.slots, p.stub_base, p.stub_size, &e)) return fail("stubs failed: " + e);
    install_trap_handler();

    // PSN.prx / SaveData.prx native plugins validate a module-param descriptor and null-fault if
    // started with (argc=0, argp=NULL). Register their guest ranges so run_guest_inits starts them
    // with the real descriptor. Skipped when PROSPER_NO_PSN drops them.
    if (!getenv("PROSPER_NO_PSN"))
        set_module_start_param_ranges({ { BOOT_PSN, BOOT_SAVEDATA }, { BOOT_SAVEDATA, BOOT_LIBC } });

    run_guest_inits(p.init_fns);
    return true;
}

} // namespace prosper

#else  // non-Linux: the guest-execution substrate doesn't exist.
namespace prosper {
bool boot_program(const std::string&, Program&, std::string* err, const std::function<void()>&) {
    if (err) *err = "boot_program is Linux-only"; return false;
}
} // namespace prosper
#endif
