// boot_program.hpp — boot the standard PS5 title into a ready-to-run state.
//
// Extracted from boot_trace so every consumer (the diagnostic runner AND the prosper-app frontend)
// shares ONE boot path instead of duplicating the ~150 lines of link/map/TLS/unwind/stub/init glue.
// It does the full guest bring-up up to and including the dependent-module init_arrays; the caller
// then installs its own frontends (renderer / audio sink / pad backend) and calls run_entry().
#pragma once
#include "loader/linker.hpp"   // Program
#include <string>
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
inline constexpr uint64_t BOOT_PSN      = 0x4e0000000ull;
inline constexpr uint64_t BOOT_SAVEDATA = 0x4f0000000ull;
inline constexpr uint64_t BOOT_LIBC     = 0x5c0000000ull;
inline constexpr uint64_t BOOT_FMODSTUDIO = 0x520000000ull; // optional Unity FMOD native plugins
inline constexpr uint64_t BOOT_FMOD       = 0x540000000ull;
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

} // namespace prosper
