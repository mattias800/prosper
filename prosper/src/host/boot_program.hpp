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
