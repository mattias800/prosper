// boot_program.hpp — boot the standard PS5 title into a ready-to-run state.
//
// Extracted from boot_trace so every consumer (the diagnostic runner AND the prosper-app frontend)
// shares ONE boot path instead of duplicating the ~150 lines of link/map/TLS/unwind/stub/init glue.
// It does the full guest bring-up up to and including the dependent-module init_arrays; the caller
// then installs its own frontends (renderer / audio sink / pad backend) and calls run_entry().
#pragma once
#include "loader/linker.hpp"   // Program
#include <string>
#include <vector>
#include <functional>

namespace prosper {

// Fixed guest module bases (the PS5 title's fixed address map). Shared so callers that classify
// guest addresses (e.g. boot_trace's diagnostics) use the same constants the boot uses.
// Keep executable images out of the fixed direct-memory apertures used by current PS5 titles.
// Astro Bot maps [0x400000000,0x40b800000) and [0x500000000,0x587400000) before bringing up
// AGC; placing eboot/libc at the old 0x400/0x500 bases made those legitimate mappings alias code.
inline constexpr uint64_t BOOT_EBOOT    = 0x410000000ull;
inline constexpr uint64_t BOOT_IL2CPP   = 0x440000000ull;
inline constexpr uint64_t BOOT_PS5UTIL  = 0x4c0000000ull;
inline constexpr uint64_t BOOT_PSNCORE  = 0x480000000ull;   // PPSA02664's Unity PSN native plugin
inline constexpr uint64_t BOOT_PSNCOMMON = 0x4a0000000ull;  //   (PSNCore.prx + PSNCommon.prx)
inline constexpr uint64_t BOOT_COMMONDIALOG = 0x4b0000000ull; // Unity CommonDialog native plugin
inline constexpr uint64_t BOOT_PSN      = 0x4e0000000ull;
inline constexpr uint64_t BOOT_NPCPPWEBAPI = 0x4d0000000ull; // optional bundled PSN C++ support PRX
inline constexpr uint64_t BOOT_SAVEDATA = 0x4f0000000ull;
inline constexpr uint64_t BOOT_LIBC     = 0x5c0000000ull;
inline constexpr uint64_t BOOT_FMODSTUDIO = 0x520000000ull; // optional Unity FMOD native plugins
inline constexpr uint64_t BOOT_FMOD       = 0x540000000ull;
inline constexpr uint64_t BOOT_AKMOTION   = 0x560000000ull; // optional Unity Wwise native plugins
inline constexpr uint64_t BOOT_AKVORBIS   = 0x580000000ull;
inline constexpr uint64_t BOOT_AKSOUNDENGINE = 0x5a0000000ull;
// Base pool for auto-discovered Unity native plugins (see discover_extra_plugin_modules). Placed
// between libc and the import-stub region: every shipped plugin observed so far is under 8 MiB, so a
// 64 MiB stride leaves a wide margin, and the last slot ends 128 MiB below BOOT_STUB.
inline constexpr uint64_t BOOT_PLUGIN_AUTO_BASE  = 0x5d0000000ull;
inline constexpr uint64_t BOOT_PLUGIN_AUTO_STRIDE = 0x4000000ull;   // 64 MiB
inline constexpr unsigned BOOT_PLUGIN_AUTO_SLOTS  = 10;
inline constexpr uint64_t BOOT_STUB     = 0x600000000ull;
// Base pool for PRXs the guest loads at RUNTIME via sceKernelLoadStartModule (#639) — modules that
// are NOT in the fixed preload list above, because the path is only known while the title runs
// (R-Type Delta ships 24 per-stage modules under /app0/prx/). Placed ABOVE the import-stub
// aperture: callback_fs.hpp recognises a guest import-stub frame by a return address in
// [0x600000000,0x700000000), so nothing else may be mapped inside that window. 128 MiB per slot
// bounds each module (the whole local corpus's largest module image is ~251 MiB, and the loader
// refuses anything that does not fit rather than overlapping its neighbour).
// (Kept clear of the ad-hoc stub bases a few unit tests pass to install_stubs — 0x680000000,
// 0x710000000, 0x720000000, 0x730000000 — so a future test that boots a program and loads a module
// in one process cannot alias them.)
inline constexpr uint64_t BOOT_RUNTIME_MODULE_BASE   = 0x800000000ull;
inline constexpr uint64_t BOOT_RUNTIME_MODULE_STRIDE = 0x8000000ull;   // 128 MiB
inline constexpr unsigned BOOT_RUNTIME_MODULE_SLOTS  = 48;
inline constexpr uint64_t BOOT_RUNTIME_MODULE_END =
    BOOT_RUNTIME_MODULE_BASE + (uint64_t)BOOT_RUNTIME_MODULE_SLOTS * BOOT_RUNTIME_MODULE_STRIDE;

// --- Guest-address labelling, shared by every diagnostic that prints "<module>+0x<rva>" ----------
//
// These exist because the same label was being produced three different ways (#1659). The eboot base
// MOVED in #825 (0x400000000 -> 0x410000000, so Astro Bot's direct-memory mapping at
// [0x400000000,0x40b800000) would stop aliasing code), and the diagnostics that had hard-coded the old
// literal were never updated. They kept printing "eboot+0x…" against 0x400000000, so every offset was
// the real RVA plus 0x10000000 — a plausible-looking number that lands nowhere, and one that no longer
// round-trips through PROSPER_BP (which correctly adds the *mapped* base).
//
// Two further defects the literal caused, beyond the shift:
//   * [0x400000000,0x410000000) is now a direct-memory aperture, not code — a data pointer there was
//     classified as "eboot" and printed as a code offset.
//   * Sites testing a single wide range (e.g. [0x400000000,0x4c0000000)) labelled EVERY module in it
//     "eboot+", so an IL2CPP return address was reported as an eboot RVA. Wrong binary, not just wrong
//     offset.
//
// Use these instead of subtracting any literal. `guest_module_name` returns a stable short name, and
// `guest_module_offset` the offset within that module; for an address outside every fixed module the
// name is "mapped/host" and the offset is the address itself, so a caller can print it verbatim.
inline const char* guest_module_name(uint64_t a) {
    if (a >= BOOT_EBOOT         && a < BOOT_IL2CPP)        return "eboot";
    if (a >= BOOT_IL2CPP        && a < BOOT_PSNCORE)       return "Il2cpp";
    if (a >= BOOT_PSNCORE       && a < BOOT_PSNCOMMON)     return "PSNCore.prx";
    if (a >= BOOT_PSNCOMMON     && a < BOOT_COMMONDIALOG)  return "PSNCommon.prx";
    if (a >= BOOT_COMMONDIALOG  && a < BOOT_PS5UTIL)       return "CommonDialog.prx";
    if (a >= BOOT_PS5UTIL       && a < BOOT_NPCPPWEBAPI)   return "PS5Util";
    if (a >= BOOT_NPCPPWEBAPI   && a < BOOT_PSN)           return "libSceNpCppWebApi.prx";
    if (a >= BOOT_PSN           && a < BOOT_SAVEDATA)      return "PSN.prx";
    if (a >= BOOT_SAVEDATA      && a < BOOT_FMODSTUDIO)    return "SaveData.prx";
    if (a >= BOOT_FMODSTUDIO    && a < BOOT_FMOD)          return "libfmodstudio.prx";
    if (a >= BOOT_FMOD          && a < BOOT_AKMOTION)      return "libfmod.prx";
    if (a >= BOOT_AKMOTION      && a < BOOT_AKVORBIS)      return "AkMotion.prx";
    if (a >= BOOT_AKVORBIS      && a < BOOT_AKSOUNDENGINE) return "AkVorbisHwAccelerator.prx";
    if (a >= BOOT_AKSOUNDENGINE && a < BOOT_LIBC)          return "AkSoundEngine.prx";
    if (a >= BOOT_LIBC          && a < BOOT_PLUGIN_AUTO_BASE) return "libc.prx";
    if (a >= BOOT_PLUGIN_AUTO_BASE && a < BOOT_STUB)       return "plugin";
    if (a >= BOOT_STUB          && a < 0x610000000ull)     return "STUB";
    // Runtime-loaded PRX (#639). Like the plugin pool this is one label for the whole aperture:
    // which slot holds which module is a run-local fact, so the offset is reported modulo the
    // stride and the exact module comes from the loader's own [loadmod] line.
    if (a >= BOOT_RUNTIME_MODULE_BASE && a < BOOT_RUNTIME_MODULE_END) return "runtime-prx";
    return "mapped/host";
}
inline uint64_t guest_module_offset(uint64_t a) {
    if (a >= BOOT_EBOOT         && a < BOOT_IL2CPP)        return a - BOOT_EBOOT;
    if (a >= BOOT_IL2CPP        && a < BOOT_PSNCORE)       return a - BOOT_IL2CPP;
    if (a >= BOOT_PSNCORE       && a < BOOT_PSNCOMMON)     return a - BOOT_PSNCORE;
    if (a >= BOOT_PSNCOMMON     && a < BOOT_COMMONDIALOG)  return a - BOOT_PSNCOMMON;
    if (a >= BOOT_COMMONDIALOG  && a < BOOT_PS5UTIL)       return a - BOOT_COMMONDIALOG;
    if (a >= BOOT_PS5UTIL       && a < BOOT_NPCPPWEBAPI)   return a - BOOT_PS5UTIL;
    if (a >= BOOT_NPCPPWEBAPI   && a < BOOT_PSN)           return a - BOOT_NPCPPWEBAPI;
    if (a >= BOOT_PSN           && a < BOOT_SAVEDATA)      return a - BOOT_PSN;
    if (a >= BOOT_SAVEDATA      && a < BOOT_FMODSTUDIO)    return a - BOOT_SAVEDATA;
    if (a >= BOOT_FMODSTUDIO    && a < BOOT_FMOD)          return a - BOOT_FMODSTUDIO;
    if (a >= BOOT_FMOD          && a < BOOT_AKMOTION)      return a - BOOT_FMOD;
    if (a >= BOOT_AKMOTION      && a < BOOT_AKVORBIS)      return a - BOOT_AKMOTION;
    if (a >= BOOT_AKVORBIS      && a < BOOT_AKSOUNDENGINE) return a - BOOT_AKVORBIS;
    if (a >= BOOT_AKSOUNDENGINE && a < BOOT_LIBC)          return a - BOOT_AKSOUNDENGINE;
    if (a >= BOOT_LIBC          && a < BOOT_PLUGIN_AUTO_BASE) return a - BOOT_LIBC;
    if (a >= BOOT_PLUGIN_AUTO_BASE && a < BOOT_STUB)
        return (a - BOOT_PLUGIN_AUTO_BASE) % BOOT_PLUGIN_AUTO_STRIDE;
    if (a >= BOOT_STUB          && a < 0x610000000ull)     return a - BOOT_STUB;
    if (a >= BOOT_RUNTIME_MODULE_BASE && a < BOOT_RUNTIME_MODULE_END)
        return (a - BOOT_RUNTIME_MODULE_BASE) % BOOT_RUNTIME_MODULE_STRIDE;
    return a;
}
// End of the import-stub aperture. install_stubs reserves this whole window (never more), so a stub
// address is exactly [BOOT_STUB, BOOT_STUB_END) and nothing else may be mapped inside it —
// callback_fs.hpp identifies a guest import-stub frame by a return address in this range.
inline constexpr uint64_t BOOT_STUB_END = 0x610000000ull;

// True when `a` lies inside some fixed guest module, i.e. "<name>+0x<offset>" is meaningful.
inline bool guest_va_in_module(uint64_t a) {
    return (a >= BOOT_EBOOT && a < BOOT_STUB_END) ||
           (a >= BOOT_RUNTIME_MODULE_BASE && a < BOOT_RUNTIME_MODULE_END);
}
// True when `a` is guest MODULE CODE — in a module but not in the import-stub aperture. Stack scans
// that recover a guest callsite want this, not guest_va_in_module: a stub address is a real guest
// address but names prosper's own trampoline, not the caller. Written as one helper because the
// three sites that used to spell it `guest_va_in_module(a) && a < BOOT_STUB` silently stopped
// covering the runtime-module aperture the moment it was placed ABOVE the stubs (#639).
inline bool guest_va_in_module_code(uint64_t a) {
    return guest_va_in_module(a) && !(a >= BOOT_STUB && a < BOOT_STUB_END);
}

// Boot the title rooted at `dump_root` (the app0 directory): link the fixed module set (dropping any
// absent dependent module — cross-title tolerance; honors PROSPER_NO_PSN), register the built-in
// HLE, map the images, set up TLS / unwind / procparam, install the import stubs + trap handler,
// register the PSN/SaveData module-start ranges, and run the dependent-module init_arrays. On
// success `out` is fully set up and the caller should register its renderer/backends and call
// run_entry(out.imgs[0]).
//
// `after_hle_registered`, if provided, runs immediately after register_builtin_hle() and before the
// images are mapped — the point a caller installs host frontends (audio sink / pad backend) so they
// are in place before the guest runs. Returns false (with *err set) on a link/map/stub failure.
//
// Available on every platform with a guest-execution substrate (currently Linux, Windows, and macOS).
// Other platforms return false.
bool boot_program(const std::string& dump_root, Program& out, std::string* err,
                  const std::function<void()>& after_hle_registered = {});

// Resolve `want` to the real on-disk entry, correcting case-only mismatches component by component
// (#1006). The PS5's filesystem namespace is case-insensitive (see the runtime basename compare in
// hle_kernel.cpp), so neither module presence (the boot preload list hard-codes one casing per module
// path, but titles disagree: The Messenger ships "Il2cppUserAssemblies.prx"; Blasphemous 2 / Evergate
// ship "Il2CppUserAssemblies.prx") nor a guest file open (ArcRunner requests "Content/Movies/..."
// where the dump ships "content/movies/...", #1226) may be decided by exact-case probe on a
// case-sensitive host filesystem. Returns the corrected path, or `want` unchanged when the path
// already exists or no entry matches (the caller's absence handling then applies as before).
// Consumers: boot_program's preload loop, hle_file's translate(), and the unit test.
std::string resolve_host_path_case(const std::string& want);

// Enumerate the title's own Unity native plugins that the fixed preload list does not name (#1609).
//
// prosper has no runtime PRX loading (#639), so every native plugin a title P/Invokes must be linked
// at boot. The preload list above grew one plugin at a time (PSN, PSNCore/PSNCommon, CommonDialog,
// FMOD, Wwise), which means any title shipping a plugin nobody has hard-coded yet fails its very
// first P/Invoke with `DllNotFoundException` — and because IL2CPP raises that inside an async state
// machine, the exception is captured into a Task and never printed, so the title simply stops making
// progress with no diagnostic at all. Tales of Graces f Remastered (PPSA19991) loses three of the
// thirty singletons its boot state machine awaits that way and then renders an empty scene forever.
//
// `<dump_root>/Media/Plugins/*.prx` is the engine's own plugin directory, so its contents are exactly
// the set the guest may ask for. Returns the paths of the entries whose basename does not
// case-insensitively match any of `listed_basenames`, ordered so that the caller (which appends them
// to a link list whose init functions run in reverse order) initializes them in ascending
// case-insensitive name order. A missing directory yields an empty vector; never throws.
std::vector<std::string> discover_extra_plugin_modules(
    const std::string& dump_root, const std::vector<std::string>& listed_basenames);

} // namespace prosper
