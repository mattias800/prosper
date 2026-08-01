// linker.hpp — multi-module dynamic linker. Loads the main executable plus its
// dependent PRX modules into one guest address space, builds a global export table,
// and resolves every import to either another module's export (real cross-module call)
// or an HLE stub slot (implemented handler or unimplemented logger). Host-agnostic.
#pragma once
#include "../self/module.hpp"
#include "../hle/dispatch.hpp"   // ImportSlot
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
};

struct Program {
    std::vector<std::unique_ptr<Module>> mods;   // unique_ptr: stable addresses for imports[]
    std::vector<LoadedImage>             imgs;    // parallel to mods
    std::vector<ImportSlot>              slots;   // unresolved imports -> stub slots
    std::vector<uint64_t>                init_fns; // dependent-module init fns, in call order
    std::vector<TlsModuleDesc>           tls_templates; // indexed by module TLS id (0 = unused)
    uint64_t entry = 0;                            // main module entry
    uint64_t stub_base = 0, stub_size = 96;   // 96 contains the largest guest-%fs swap stub (94 bytes)

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

    // Stats for reporting.
    size_t total_imports = 0, resolved_cross_module = 0, stubbed = 0;
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

// Link the given modules (the first is the main executable). Applies relocations.
// Returns false with *err on failure.
bool link_program(const std::vector<LinkInput>& inputs, uint64_t stub_base,
                  Program& out, std::string* err);

} // namespace prosper
