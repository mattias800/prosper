// boot_program.cpp — see boot_program.hpp. Extracted verbatim from boot_trace's boot sequence
// (behavior-preserving); backed by whichever exec_image_<os> substrate the platform provides
// (Linux/macOS: exec_image_linux.cpp; Windows: exec_image_win.cpp).
#include "boot_program.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <system_error>
#include <vector>

// Optional diagnostics (zero-cost when disabled — stubs inline to no-ops).
#include "diagnostics/diagnostics.hpp"

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

// See boot_program.hpp. Pure filesystem query so it can be unit-tested without a guest image.
std::vector<std::string> discover_extra_plugin_modules(
    const std::string& dump_root, const std::vector<std::string>& listed_basenames) {
    namespace fs = std::filesystem;
    std::vector<std::string> found;
    if (dump_root.empty()) return found;
    auto lower = [](std::string s) {
        for (auto& c : s) c = (char)std::tolower((unsigned char)c);
        return s;
    };
    std::vector<std::string> listed;
    listed.reserve(listed_basenames.size());
    for (const auto& b : listed_basenames) listed.push_back(lower(b));

    std::error_code ec;
    const std::string dir = resolve_host_path_case(dump_root + "/Media/Plugins");
    if (!fs::is_directory(fs::path(dir), ec)) return found;
    for (const auto& e : fs::directory_iterator(dir, fs::directory_options::skip_permission_denied, ec)) {
        if (!e.is_regular_file(ec)) continue;
        const std::string name = e.path().filename().string();
        const std::string lname = lower(name);
        if (lname.size() < 5 || lname.compare(lname.size() - 4, 4, ".prx") != 0) continue;
        bool already = false;
        for (const auto& l : listed) if (l == lname) { already = true; break; }
        if (!already) found.push_back(e.path().string());
    }
    // directory_iterator order is filesystem-defined; sort so a boot is reproducible. DESCENDING by
    // lowercased basename, because the caller appends this block to a link list whose init functions
    // run in reverse order — the guest therefore initializes them in ascending name order.
    std::sort(found.begin(), found.end(), [&](const std::string& a, const std::string& b) {
        return lower(fs::path(a).filename().string()) > lower(fs::path(b).filename().string());
    });
    return found;
}

} // namespace prosper

#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
#include "host/exec_image.hpp"
#include "host/runtime_module_load.hpp"
#include "hle/dispatch.hpp"
#include "self/module.hpp"     // PT_SCE_PROCPARAM
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace prosper {

// The loader's ACTUAL link set, exposed so it has exactly one definition (#2199).
//
// nid_census used to build its own module set by scanning the file tree, and the two differed in
// ways that silently removed rows from its report: only Media/Plugins is auto-discovered, .sprx is
// never auto-linked, sce_module contributes exactly two named files, and modules whose file is
// absent or that the linker refuses are dropped. The census excludes any import satisfied by a
// SIBLING module's export -- sound only if the sibling set matches what the loader links. Where the
// tool's set was larger, it excluded a binding that in fact falls to the dispatcher's `return 0` at
// runtime, i.e. a FALSE ABSENCE from a report whose whole purpose is to list what reaches the
// dispatcher. A missing row looks like nothing at all.
//
// Split out of boot_program() as a pure prefix: it depends only on `d`, has no early exit, and
// boot_program now calls it, so the two cannot drift. Note it PRINTS while it works (auto-link and
// case-correction lines) -- that output is part of the loader's existing behaviour and is now also
// visible to any other caller.
std::vector<LinkInput> boot_link_inputs(const std::string& d, bool verbose) {
    // `verbose` exists only so a TOOL can call this without corrupting its own stdout: the
    // four prints below are the loader's, and nid_census --tsv writes machine-readable rows to
    // the same stream. boot_program passes true, so loader output is byte-identical (#2199).
    const auto say = [&](auto&&... args) { if (verbose) printf(args...); };
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
        // Unity's PS5 platform layer calls CommonDialog::PrxCommonDialogUpdate before it services
        // deferred save/load work. Titles such as Space Adventure Cobra reach that P/Invoke only
        // after accepting input on the title screen; leaving the PRX to the unsupported runtime
        // loader therefore stalls the main thread with the save request still queued. Preload the
        // optional plugin just like PSN/SaveData so its export is available to il2cpp immediately.
        { d + "/Media/Plugins/CommonDialog.prx", BOOT_COMMONDIALOG },
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
    // #1609: the fixed list above only names plugins some earlier title needed. Link whatever else
    // this title ships in its own Media/Plugins directory, so a first P/Invoke into it resolves
    // instead of raising a silent DllNotFoundException (prosper has no runtime PRX loading, #639).
    // Inserted just before the sce_module entries so these plugins initialize after libc and the
    // bundled support PRX but before the hard-coded Unity plugins (init runs in reverse list order).
    if (!getenv("PROSPER_NO_PLUGIN_AUTOLINK")) {
        std::vector<std::string> listed;
        for (const auto& e : in) {
            const size_t slash = e.path.find_last_of("/\\");
            listed.push_back(slash == std::string::npos ? e.path : e.path.substr(slash + 1));
        }
        const std::vector<std::string> extra = discover_extra_plugin_modules(d, listed);
        size_t insert_at = in.size();
        for (size_t i = 0; i < in.size(); i++)
            if (in[i].path.find("/sce_module/") != std::string::npos) { insert_at = i; break; }
        unsigned slot = 0;
        for (const auto& path : extra) {
            if (slot >= BOOT_PLUGIN_AUTO_SLOTS) {
                say("plugin auto-link: no free base slot for %s (max %u) — NOT linked\n",
                       path.c_str(), BOOT_PLUGIN_AUTO_SLOTS);
                continue;
            }
            const uint64_t base = BOOT_PLUGIN_AUTO_BASE + (uint64_t)slot * BOOT_PLUGIN_AUTO_STRIDE;
            say("plugin auto-link: %s @ 0x%llx\n", path.c_str(), (unsigned long long)base);
            // skip_on_export_collision: never introduce a duplicate NID export. A title can ship two
            // builds of the same library (Evergate's libfmod.prx + libfmodL.prx export identical
            // NIDs); linking both would run two init_arrays and make dlsym answer differently per
            // module handle. Deduplicating on exports rather than on a filename suffix also degrades
            // correctly for a title that ships only the debug variant — nothing collides, so it links.
            in.insert(in.begin() + (ptrdiff_t)(insert_at + slot), { path, base, true });
            slot++;
        }
    }
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
            say("module path case-corrected: %s -> %s\n", in[i].path.c_str(), resolved.c_str());
            in[i].path = std::move(resolved);
        }
        if (FILE* f = fopen(in[i].path.c_str(), "rb")) fclose(f);
        else { say("skipping absent module: %s\n", in[i].path.c_str()); in.erase(in.begin() + (ptrdiff_t)i); }
    }
    return in;
}

bool boot_program(const std::string& d, Program& p, std::string* err,
                  const std::function<void()>& after_hle_registered) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };

    // Diagnostics: record boot start.
    diagnostics::record_boot_phase(diagnostics::BootPhase::PROCESS_START);

    // libc.prx loaded last => its init_array runs first (deepest dependency), before eboot's entry.
    std::vector<LinkInput> in = boot_link_inputs(d);

    std::string e;
    if (!link_program(in, BOOT_STUB, p, &e)) return fail("link failed: " + e);
    // Loud, self-describing report of every auto-linked plugin the linker refused: which module was
    // dropped, the exact NID that collided, and which already-linked module owns it.
    for (const auto& s : p.skipped_modules)
        printf("plugin auto-link: SKIPPED %s — it exports NID %s, already provided by %s\n",
               s.path.c_str(), s.nid.c_str(), s.owner_path.c_str());
    // Aliased exports (#1635). first-wins is the policy and is unchanged; being silent about it was
    // the defect. Both modules stay mapped and both init_arrays run, and sceKernelDlsym consults the
    // handle's own table first (#147) — so a NID listed here resolves to DIFFERENT addresses
    // depending on which handle asks. Bounded print, full count, so a busy pair cannot bury the line.
    if (!p.aliased_exports.empty()) {
        constexpr size_t kMaxShown = 8;
        const size_t shown = std::min(p.aliased_exports.size(), kMaxShown);
        for (size_t i = 0; i < shown; i++) {
            const auto& a = p.aliased_exports[i];
            printf("export alias: NID %s -> %s (0x%llx); %s also exports it (0x%llx), discarded\n",
                   a.nid.c_str(), a.winner_path.c_str(), (unsigned long long)a.winner,
                   a.loser_path.c_str(), (unsigned long long)a.loser);
        }
        if (p.aliased_exports.size() > shown)
            printf("export alias: ... and %zu more\n", p.aliased_exports.size() - shown);
    }
    printf("linked %zu modules; %zu imports (%zu cross-module, %zu stub slots); %zu init fns; "
           "%zu aliased exports\n",
           p.mods.size(), p.total_imports, p.resolved_cross_module, p.slots.size(),
           p.init_fns.size(), p.aliased_exports.size());

    // Diagnostics: linking complete.
    diagnostics::record_boot_phase(diagnostics::BootPhase::LINKING);

    // sceKernelDlsym resolves exports by name against all loaded modules.
    set_module_exports(&p.exports);
    // Per-module tables (#147): LoadStartModule hands out real handles for linked-module paths and
    // dlsym consults the handle's module before the global first-definition-wins table.
    { std::vector<ModuleExportTable> mt;
      for (const auto& me : p.mod_exports) mt.push_back({ me.path, &me.nids });
      set_module_export_tables(std::move(mt)); }

    register_builtin_hle();
    if (after_hle_registered) after_hle_registered();   // caller installs host frontends here

    // Diagnostics: HLE handlers registered.
    diagnostics::record_boot_phase(diagnostics::BootPhase::HLE_REGISTERED);

    set_app0_root(d);
    for (auto& img : p.imgs) if (!map_image(img, &e)) return fail("map failed: " + e);

    // Diagnostics: all modules mapped.
    diagnostics::record_boot_phase(diagnostics::BootPhase::MODULES_MAPPED);
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

    // A runtime sceKernelLoadStartModule appends import slots to this same vector (#639). Reserve
    // headroom so the common case never reallocates the buffer the dispatcher indexes; the append
    // is synchronised either way (dispatch_append_slots), this just avoids the copy.
    p.slots.reserve(p.slots.size() + 1024);
    if (!install_stubs(p.slots, p.stub_base, p.stub_size, &e)) return fail("stubs failed: " + e);
    install_trap_handler();

    // Diagnostics: stubs and trap handler installed.
    diagnostics::record_boot_phase(diagnostics::BootPhase::STUBS_INSTALLED);
    // Enable real runtime PRX loading now that the fixed set is linked, mapped and stubbed (#639).
    runtime_module_loader_init(&p);

    // PSN.prx / SaveData.prx native plugins validate a module-param descriptor and null-fault if
    // started with (argc=0, argp=NULL). Register their guest ranges so run_guest_inits starts them
    // with the real descriptor. Skipped when PROSPER_NO_PSN drops them.
    if (!getenv("PROSPER_NO_PSN"))
        set_module_start_param_ranges({ { BOOT_PSN, BOOT_SAVEDATA }, { BOOT_SAVEDATA, BOOT_LIBC },
                                        { BOOT_PSNCORE, 0x490000000ull }, { BOOT_PSNCOMMON, 0x4b0000000ull } });

    // Diagnostics: guest initialization running.
    diagnostics::record_boot_phase(diagnostics::BootPhase::GUEST_INITS_RUNNING);
    run_guest_inits(p.init_fns);

    // Diagnostics: boot complete.
    diagnostics::record_boot_phase(diagnostics::BootPhase::BOOT_COMPLETE);
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
