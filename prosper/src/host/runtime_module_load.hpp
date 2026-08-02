// runtime_module_load.hpp — real runtime PRX loading for sceKernelLoadStartModule (#639).
//
// Before this, "loading" a PRX only ever resolved a path against the fixed set `boot_program`
// linked BEFORE guest entry; anything else was reported ENOENT (#146's honest answer to a
// capability prosper did not have). That is fine for a title that preloads every module it uses,
// and wrong for one whose code lives in modules chosen while it runs: R-Type Delta: HD Boosted
// (PPSA26414) ships 24 per-stage PRXs under /app0/prx/ and its eboot is only a shell, so the very
// first sceKernelLoadStartModule miss left it calling a NULL DLLLoadStart (#1591).
//
// This maps, relocates, links and starts such a module at the point the guest asks for it — the
// same passes `link_program` runs at boot, applied to one module against the already-linked
// program. What it deliberately does NOT do, and why:
//
//   * It never rebinds an ALREADY-RELOCATED module's imports. prosper binds eagerly at link time,
//     so a module loaded later cannot retroactively satisfy an earlier module's import; that
//     import already points at its stub slot. Sony's loader has the same property for a module
//     that is already started.
//   * It does not model unload. sceKernelStopUnloadModule is out of scope here; a repeat load of
//     the same path returns the cached handle rather than re-running module_start.
//   * A module needing INITIAL-EXEC TLS (R_X86_64_TPOFF64) cannot be given static TLS space after
//     threads exist. Those relocations are left unapplied by apply_relocations and reported here,
//     loudly, rather than silently baked to a wrong offset. General-dynamic TLS works.
//   * Module identity is the BASENAME, because that is what the PS5's own namespace uses and what
//     sceKernelLoadStartModule's caller checks first (module_handle_for_path, #147). Two modules
//     with the same basename in different directories therefore alias to one handle, and a runtime
//     path whose basename matches a pre-linked module resolves to that module instead of loading.
//     Pre-existing semantics; #639 only makes them reachable for guest-composed paths.
#pragma once
#include <cstdint>
#include <string>

namespace prosper {

struct Program;

// Enable runtime loading against the booted program. Called once by boot_program after the fixed
// module set is linked, mapped, stubbed and initialised. `p` must outlive the run: the loader
// appends to its import-slot table and its TLS templates.
void runtime_module_loader_init(Program* p);

// Load and start the PRX named by a GUEST path (e.g. "/app0/prx/title_Release.prx").
//
// Returns 0 on success with *out_handle set to the module handle sceKernelDlsym accepts, or a
// SCE_KERNEL_ERROR_* code:
//   0x80020002 ENOENT   — no such file (unchanged from the pre-#639 behaviour)
//   0x80020016 EINVAL   — bad arguments / the loader is not initialised
//   0x8002000c ENOMEM   — no free base slot, or the image does not fit one
//   0x80020008 ENOEXEC  — the file is not a loadable module
// `args`/`argp` are the guest's own sceKernelLoadStartModule arguments and are passed to
// module_start; `guest_fs` is the guest thread pointer to run guest code on (0 = run on the
// current %fs, which is correct when the guest-%fs gate is off). *out_res, when non-null,
// receives module_start's return value.
uint64_t runtime_load_start_module(const char* guest_path, uint64_t args, uint64_t argp,
                                   uint64_t guest_fs, int32_t* out_res, uint64_t* out_handle);

// Test/diagnostic: how many modules the runtime loader has loaded so far.
size_t runtime_loaded_module_count();

} // namespace prosper
