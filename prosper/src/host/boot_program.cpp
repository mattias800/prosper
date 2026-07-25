// boot_program.cpp — see boot_program.hpp. Extracted verbatim from boot_trace's boot sequence
// (behavior-preserving); backed by whichever exec_image_<os> substrate the platform provides
// (Linux/macOS: exec_image_linux.cpp; Windows: exec_image_win.cpp).
#include "boot_program.hpp"

#include <cctype>
#include <filesystem>
#include <system_error>

namespace prosper {

// See boot_program.hpp. The fixed preload list names each module with ONE hard-coded casing, but
// titles disagree: The Messenger ships "Il2cppUserAssemblies.prx" while Blasphemous 2 / Evergate ship
// "Il2CppUserAssemblies.prx". On case-insensitive hosts (NTFS / WSL DrvFs / default APFS) fopen()
// opens either spelling — which is the only reason the wrong-case probe ever "worked". On a
// case-sensitive Linux filesystem it misses, the module is silently dropped as absent, and the
// guest's runtime sceKernelLoadStartModule gets ENOENT and null-jumps (SIGSEGV at rip=0, #1006).
// PS5 module paths are effectively case-insensitive (matching the runtime basename compare in
// hle_kernel.cpp), so correct each missing component against the real directory entry instead.
// Pure std::filesystem; never throws (error_code overloads), and an unreadable/missing parent just
// yields `want` unchanged so the caller's existing absence handling applies.
std::string resolve_host_path_case(const std::string& want) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (want.empty() || fs::exists(fs::path(want), ec)) return want;   // exact case (or case-insensitive FS)
    const fs::path p(want);
    const std::string parent = p.parent_path().string();
    if (parent.empty() || parent == want) return want;                 // no ancestor left to correct
    const std::string dir = resolve_host_path_case(parent);          // fix ancestor casing first
    const std::string base = p.filename().string();
    auto ieq = [](const std::string& a, const std::string& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); i++)
            if (std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i])) return false;
        return true;
    };
    for (const auto& e : fs::directory_iterator(dir, fs::directory_options::skip_permission_denied, ec)) {
        const std::string real = e.path().filename().string();
        if (ieq(real, base)) return (fs::path(dir) / real).string();
    }
    // No case-insensitive match for the leaf. If an ANCESTOR was case-corrected during the recursion
    // (dir != the original parent), recombine that corrected parent with the original leaf spelling so
    // the correction survives — a genuinely absent leaf still does not exist, so a read probe keeps
    // failing ENOENT, but an O_CREAT under an existing wrong-case directory now lands inside the real
    // directory (guest makes "/savedata0/Dir", later opens "/savedata0/dir/new" with O_CREAT — PS5's
    // case-insensitive namespace creates it inside the existing dir; #1236). When nothing was
    // corrected, return `want` UNCHANGED — byte-for-byte, preserving the caller's exact separators
    // (fs::path would otherwise rewrite them to the host-native form, e.g. '\' on Windows).
    if (dir != parent) return (fs::path(dir) / base).string();
    return want;
}

} // namespace prosper

#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
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
        // Some titles (e.g. PPSA02664) ship the Unity PSN plugin as PSNCore.prx (+ PSNCommon.prx
        // dependency) rather than PSN.prx. Preload them so the C# P/Invoke ("PSNCore"::"PrxInitialize")
        // resolves via sceKernelDlsym. Without this the il2cpp P/Invoke resolver tries to on-demand-load
        // PSNCore.prx and HANGS, stalling the whole app-init -> scene-load chain (#238). PSNCommon later
        // in the list => its init runs first (it's PSNCore's dependency). Absent-file titles skip these.
        { d + "/Media/Plugins/PSNCore.prx", BOOT_PSNCORE },
        { d + "/Media/Plugins/PSNCommon.prx", BOOT_PSNCOMMON },
        // FMOD's C# integration resolves these through P/Invoke only when its RuntimeManager is first
        // constructed. Until prosper has true runtime PRX loading (#639), link the optional pair up
        // front so LoadStartModule returns a real module handle. Studio precedes core in this list
        // because dependent-module init functions run in reverse order (core must initialize first).
        { d + "/Media/Plugins/libfmodstudio.prx", BOOT_FMODSTUDIO },
        { d + "/Media/Plugins/libfmod.prx", BOOT_FMOD },
        // Wwise's generated C# wrappers resolve AkSoundEngine and its optional motion/Vorbis
        // extensions lazily through P/Invoke. GRIS reaches that path while constructing
        // AkCallbackManager: without a real module handle, the first SWIG upcast raises an
        // EntryPointNotFoundException, aborts the static constructor, and its later async bank load
        // dereferences the uninitialized callback dictionary. As with FMOD above, preload these
        // optional native plugins until runtime PRX loading is implemented. Put the extensions
        // before the core because dependent-module init functions run in reverse link order.
        { d + "/Media/Plugins/AkMotion.prx", BOOT_AKMOTION },
        { d + "/Media/Plugins/AkVorbisHwAccelerator.prx", BOOT_AKVORBIS },
        { d + "/Media/Plugins/AkSoundEngine.prx", BOOT_AKSOUNDENGINE },
        // Sonic Origins ships libSceNpCppWebApi as a user-space support PRX. Its game code imports
        // the C++ object/thread implementation directly, while the module itself depends only on
        // the lower-level Json2/Rtc/NpWebApi2/Http/kernel/libc interfaces handled below. Runtime PRX
        // loading is not implemented yet (#639), so preload an optional copy from sce_module just as
        // we do for lazy Unity audio plugins. Keep it before libc so reverse init order initializes
        // libc first. Absent-file filtering leaves every title without this PRX unchanged.
        { d + "/sce_module/libSceNpCppWebApi.prx", BOOT_NPCPPWEBAPI },
        { d + "/sce_module/libc.prx", BOOT_LIBC },
    };
    if (getenv("PROSPER_NO_PSN"))
        for (size_t i = in.size(); i-- > 0; )
            if (in[i].path.find("PSN.prx") != std::string::npos ||
                in[i].path.find("PSNCore.prx") != std::string::npos ||
                in[i].path.find("PSNCommon.prx") != std::string::npos ||
                in[i].path.find("SaveData.prx") != std::string::npos) in.erase(in.begin() + (ptrdiff_t)i);
    // Cross-title tolerance: drop dependent modules whose file doesn't exist in this dump (the eboot
    // at index 0 is always kept; each module keeps its fixed base). Resolve each hard-coded casing to
    // the real on-disk entry first so a case-only mismatch (#1006) doesn't drop a present module on a
    // case-sensitive host filesystem.
    for (size_t i = in.size(); i-- > 1; ) {
        std::string resolved = resolve_host_path_case(in[i].path);
        if (resolved != in[i].path) {
            printf("module path case-corrected: %s -> %s\n", in[i].path.c_str(), resolved.c_str());
            in[i].path = std::move(resolved);
        }
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
    {
        // General-dynamic and initial-exec TLS consume the same descriptors and module-id order.
        set_tls_modules(p.tls_templates.data(), p.tls_templates.size());
        guest_tls_set_templates(p.tls_templates.data(), p.tls_templates.size());
    }

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
        set_module_start_param_ranges({ { BOOT_PSN, BOOT_SAVEDATA }, { BOOT_SAVEDATA, BOOT_LIBC },
                                        { BOOT_PSNCORE, 0x490000000ull }, { BOOT_PSNCOMMON, 0x4b0000000ull } });

    run_guest_inits(p.init_fns);
    return true;
}

} // namespace prosper

#else  // no guest-execution substrate for this platform.
namespace prosper {
bool boot_program(const std::string&, Program&, std::string* err, const std::function<void()>&) {
    if (err) *err = "boot_program: no guest-execution substrate for this platform"; return false;
}
} // namespace prosper
#endif
