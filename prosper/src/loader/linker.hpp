// linker.hpp — multi-module dynamic linker. Loads the main executable plus its
// dependent PRX modules into one guest address space, builds a global export table,
// and resolves every import to either another module's export (real cross-module call)
// or an HLE stub slot (implemented handler or unimplemented logger). Host-agnostic.
#pragma once
#include "../self/module.hpp"
#include "hle/dispatch/dispatch.hpp"   // ImportSlot
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace prosper {

struct LinkInput {
    std::string path;
    uint64_t base;
    // #1609: this input is an OPTIONAL module discovered on disk rather than a named dependency.
    // Skip it entirely — do not build its image, do not contribute its exports, do not run its
    // init function — when any NID it exports is already exported by an earlier accepted module.
    //
    // Needed because the global export table below is first-definition-wins and SILENT: a second
    // module exporting the same NIDs is linked, mapped and initialized, but every guest import
    // aliases to whichever module happens to come first in this list. Evergate (PPSA01885) ships
    // both `libfmod.prx` and `libfmodL.prx` — FMOD's release and logging builds of the SAME library,
    // exporting the SAME NIDs. Auto-linking the second one would run a second full FMOD core's
    // init_array and make `sceKernelDlsym` return a different implementation depending on which
    // module handle it was asked through. Deduplicating on filename cannot see that (the names
    // genuinely differ); deduplicating on exports is what the hazard actually is.
    bool skip_on_export_collision = false;

    // This input is a bundled SUPPORT PRX that is only worth loading when another module actually
    // imports it by name. Skip it entirely — no image, no exports, no init function — when no other
    // accepted input names its library in its own import table.
    //
    // Distinct from `skip_on_export_collision`, and NOT a rule that may be generalised to the other
    // optional preloads above it. The Unity FMOD/Wwise/PSN plugins are preloaded precisely BECAUSE
    // nothing imports them statically: they are reached through `sceKernelDlsym` P/Invoke at
    // runtime, so they appear in no import table and this test would wrongly drop every one of them.
    // It applies only where the preload exists to satisfy a real static import.
    //
    // The hazard it removes is that loading a module is not free and not silent: prosper runs its
    // `module_start`, which is guest code. *Sniper Ghost Warrior Contracts 2* (PPSA03130) ships
    // `sce_module/libSceNpCppWebApi.prx` and never imports it — its own NP library is the unrelated
    // `libSceNpWebApi2` — and that module's `module_start` deadlocks in a condition wait 81 ms into
    // the boot, before `run_entry`, taking the whole title to rung 0. The preload was added for
    // *Sonic Origins*, which does import it and keeps it.
    //
    // **Setting this flag changes every title that ships the file, so census them.** For this one:
    // of 47 dumps, 42 ship it, 40 keep it, and **two** lose it — PPSA03130 and `PPSA03831`
    // *Sonic Frontiers*, which is a tracked title with no snapshot guard. See
    // `loader/support_modules.hpp` for the full census and the NID-overlap analysis behind
    // "believed safe".
    bool only_if_imported = false;
};

struct Program {
    std::vector<std::unique_ptr<Module>> mods;   // unique_ptr: stable addresses for imports[]
    std::vector<LoadedImage>             imgs;    // parallel to mods
    std::vector<ImportSlot>              slots;   // unresolved FUNCTION imports -> code stub slots
    std::vector<uint64_t>                init_fns; // dependent-module init fns, in call order
    std::vector<TlsModuleDesc>           tls_templates; // indexed by module TLS id (0 = unused)
    uint64_t entry = 0;                            // main module entry
    uint64_t stub_base = 0, stub_size = 96;   // 96 contains the largest guest-%fs swap stub (94 bytes)

    // Unresolved DATA imports (ELF STT_OBJECT) -> slots in a separate, writable, NON-executable,
    // zero-initialised aperture. A data import names a VARIABLE, not an entry point: the guest
    // dereferences it, and it may STORE through it (libc initialises `__stack_chk_guard` that way).
    //
    // Before #3529 these shared the code-stub aperture with function imports, which produced two
    // distinct defects. A read returned the first bytes of an emitted trampoline -- on NINJA GAIDEN
    // 4 (PPSA25258) the stack canary read 0x000000bf, the `mov edi, 0` that opens `emit_unimpl` --
    // so every `-fstack-protector` epilogue compared a machine-code word against a saved 0 and
    // called `__stack_chk_fail`. And a WRITE landed on prosper's own executable pages, silently
    // rewriting whichever stub occupied the slot, after which that import jumped into altered code
    // at an arbitrarily later moment.
    //
    // The population is not exotic, and these figures are the TOOL'S OWN -- `nid_census --data-only`
    // over the 60 local dump roots, whose default scope is already the loader's link set:
    //
    //   scope: 2284 distinct imported NIDs over 338 module(s) read, 0 unreadable
    //   424 DATA binding(s) unresolved by any sibling module, over 60 of 60 input(s);
    //   a further 1566 were satisfied cross-module
    //
    //   f7uOxY9mM1U  __stack_chk_guard   libkernel            60 of 60 titles
    //   djxxOmW6-aw  __progname          libkernel            60 of 60 titles
    //   ZT4ODD2Ts9o  (unnamed)           libSceLibcInternal   51 titles
    //   GAtITrgxKDE  (unnamed)           libSceNet             1 title
    //
    // QUOTE LINK-SET FIGURES, NOT DISK FIGURES, and take them from the tool. Two earlier revisions
    // of this comment got it wrong in the same way and the second was written while correcting the
    // first: "596 shipped modules" came from passing MODULE PATHS to nid_census individually (the
    // mechanism is verified -- that scan sees 808 modules, 728 importing the guard -- though the
    // exact figure 596 has not been reproduced), where
    // each .prx becomes its own "title" and cross-module exclusion degenerates -- a recursive
    // population of 808 modules the linker never links. The replacement then said the link set is
    // "roughly 303 modules", which is a BINDING count written where a module count goes. The
    // per-title counts survived both errors intact, which is exactly what made the numbers under
    // them easy to repeat unchecked. Corrected in review of #3541.
    // Whether a given one reaches a stub depends on the title's own libc: an import a sibling module
    // defines is bound to that definition and never comes here.
    std::vector<ImportSlot> data_slots;
    uint64_t data_base = 0;
    // One page per data object. The size of an imported variable is NOT knowable here -- an
    // undefined symbol carries st_size 0 throughout this corpus -- so the stride is chosen to be
    // larger than any plausible scalar or small struct rather than derived. Untouched pages cost no
    // physical memory, so a generous stride is close to free and keeps an over-long store inside its
    // own slot instead of on its neighbour's.
    uint64_t data_stride = 4096;

    // Global export table: NID -> absolute guest address (first definition wins). Retained so the
    // HLE can serve sceKernelDlsym by name (nid_hash(name)) against loaded modules — e.g. resolve
    // the native PSN.prx plugin's PSN_PrxInitialize / UnityPluginLoad exports for Unity's plugin loader.
    std::unordered_map<std::string, uint64_t> exports;

    // Per-module export tables (parallel to mods): the module's load path + its own NID -> guest
    // address map. sceKernelDlsym resolves against the HANDLE'S module first (#147) — with only the
    // global first-definition-wins table, two modules exporting the same NID alias to the first.
    struct ModuleExports { std::string path; std::unordered_map<std::string, uint64_t> nids; };
    std::vector<ModuleExports> mod_exports;

    // Inputs dropped by `LinkInput::skip_on_export_collision`, with the exact NID that collided and
    // the already-accepted module that owns it. Retained (rather than only logged) so the caller can
    // report the decision in its own vocabulary and a test can assert it.
    struct SkippedModule { std::string path, nid, owner_path; };
    std::vector<SkippedModule> skipped_modules;

    // Exports that were ALIASED rather than skipped: a module linked by name contributed a NID the
    // global table already had, so `emplace` kept the first definition and discarded this one
    // silently (#1635). First-wins is the intended policy and is unchanged — going unreported was
    // not. The loser stays mapped and its init_array still runs, and sceKernelDlsym consults the
    // handle's own table first (#147), so the same NID can resolve to two different addresses
    // depending on which handle is asked. Retained rather than only logged so the caller can report
    // it and a test can assert it.
    //
    // This is not hypothetical: a census over 30 local dumps found 41 aliased NIDs across 7 titles,
    // all among modules linked BY NAME — PSNCommon.prx + PSNCore.prx share 4 on six titles,
    // libfmod.prx + libfmodstudio.prx share 16, and AkMotion/AkSoundEngine/AkVorbisHwAccelerator
    // share one three ways. See tools/re/dup_exports.py.
    //
    // MEASURED SEVERITY TODAY: nil. Cross-referencing self_dump's [IMPORTS BY LIBRARY] for every
    // linked module of those 7 titles, NONE of the 21 distinct aliased NIDs is imported by anything —
    // so no import currently resolves through an alias, and no title's behaviour depends on which
    // module won. That is why this is a diagnostic and not a policy change. It is also why the
    // diagnostic matters: the moment a title does import one, the winner becomes load-order-dependent
    // and this is the only thing that would say so. Re-check with tools/re/dup_exports.py before
    // assuming it still holds for a newly added dump.
    struct AliasedExport { std::string nid, winner_path, loser_path; uint64_t winner, loser; };
    std::vector<AliasedExport> aliased_exports;

    // Stats for reporting. Every import lands in exactly one bucket:
    // total_imports == resolved_cross_module + stubbed + bound_data.
    size_t total_imports = 0, resolved_cross_module = 0, stubbed = 0, bound_data = 0;
};

// The exported NIDs a module contributes to the global export table: defined (non-import),
// NID-bearing symbols with a nonzero value. Kept as one helper so the collision check and the table
// build below cannot drift apart — if they disagreed, a module could be accepted as collision-free
// and then still alias someone else's NID.
std::vector<std::string> module_export_nids(const Module& m);

// First NID of `m` already present in `claimed` (NID -> owning module path), or an empty `nid` when
// there is no collision. Deterministic: `m`'s symbols are scanned in file order.
struct ExportCollision { std::string nid, owner_path; };
ExportCollision find_export_collision(
    const Module& m, const std::unordered_map<std::string, std::string>& claimed);

// How much of a candidate module an already-linked module already exports. `subsumed()` is the test
// that separates two BUILDS of one library (which must not both link) from two DIFFERENT libraries
// that share a symbol (which must). See the measured corpus at measure_export_subsumption().
struct ExportSubsumption {
    size_t exported = 0;
    size_t already_claimed = 0;
    // 0.90: the corpus splits 99.7-100% (duplicate builds) against 0.4-19.2% (distinct libraries),
    // so this sits in an 80-point empty band rather than near any observed value.
    static constexpr double kSubsumedFraction = 0.90;
    bool subsumed() const {
        return exported != 0 && (double)already_claimed / (double)exported >= kSubsumedFraction;
    }
};

// Link the given modules (the first is the main executable). Applies relocations.
// Returns false with *err on failure.
//
// `stub_base` roots the executable import-stub aperture and `data_base` the writable import-DATA
// aperture; the two must not overlap and neither may be 0. `data_base` is a required argument rather
// than a defaulted one on purpose: there is no safe fallback for a data import (binding it into the
// code aperture is the defect #3529 records, and binding it near address 0 is worse), so a caller
// that has not mapped an aperture must be a compile error rather than a silent regression.
bool link_program(const std::vector<LinkInput>& inputs, uint64_t stub_base, uint64_t data_base,
                  Program& out, std::string* err);

} // namespace prosper
