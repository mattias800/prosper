#pragma once
// module_path_policy — which files inside a game dump prosper is willing to link as a guest module.
//
// REJECT BY DEFAULT. A module may be linked from `kPermittedModuleDirs` below, or from the dump ROOT
// subject to the platform-name and extension rules further down (#3497). Together those are the
// COMPLETE set of dump-relative locations; anything else is refused and reported, whatever the file
// is called and whatever produced it.
//
// The root is the loosest of them and the one to read carefully: it is auto-scanned wholesale, and
// it is not a directory a title owns by convention -- `eboot.bin` and `sce_module/` live there too.
//
// Why this is an enforced policy and not merely a property of how boot_link_inputs happens to be
// written today: some dumps in circulation ship a `fakelib/` directory of replacement Sony
// libraries — libSceAppContent, libSceNpEntitlementAccess, libSceGameUpdate — whose purpose is to
// answer ownership queries with an unconditional yes. prosper answers those queries itself, from
// the locally declared inventory (`src/hle/service/hle_addcontent.cpp`), and the project charter
// draws the line exactly here: reimplementing a platform query faithfully is the job; manufacturing
// a positive answer to an ownership query is circumvention performed by the emulator. Linking a
// third-party module that does it on our behalf lands on the wrong side of that line, and it would
// also silently invalidate every entitlement result we measure — the guest would be answered by
// somebody else's stub while our own implementation sat unused.
//
// Before this policy existed prosper already did not load those files, but only as a side effect of
// three unrelated details of `discover_extra_plugin_modules()`: it scans one fixed directory, it is
// not recursive, and it matches `.prx` where `fakelib` ships `.sprx`. Any one of those changing
// would have started loading them with nothing anywhere to notice — and #1609 widened module
// discovery once already. A property that no test asserts is not a property.
//
// Deliberately NOT switchable. There is no environment variable and no flag, because a flag is
// precisely the thing that would end up set. Widening the allowlist is a source change, and so gets
// a reviewer.
//
// CONFIDENCE: HIGH — the permitted set is enumerated directly from every dump-relative path
// `boot_link_inputs()` can construct.
#include "loader/linker.hpp"

#include <string>
#include <vector>

namespace prosper {

// The complete set of dump-relative directories a linkable module may live in. Compared
// case-insensitively, with `/` and `\` treated alike.
inline constexpr const char* kPermittedModuleDirs[] = {
    "Media/Modules",
    "Media/Plugins",
    "sce_module",
};

// Files permitted directly in the dump root REGARDLESS of extension. Module files in the root are
// handled separately (see the platform-prefix rules below): a title's own native middleware may sit
// beside eboot.bin -- Darksiders II ships four such modules there -- so the root is not the
// eboot-only location this list alone would suggest.
inline constexpr const char* kPermittedRootFiles[] = {
    "eboot.bin",
};

// A Sony-named module (`libSce*`) may be linked ONLY from `sce_module/`, never from the two Media
// directories. Those two are auto-linked wholesale by #1609 — every `.prx` a dump ships in
// `Media/Plugins` gets linked — so without this a dump could substitute any Sony library simply by
// dropping it in the Unity plugin folder, and the `fakelib/` rejection above would be trivially
// sidestepped by moving the file. `Media/*` is where a title's OWN native plugins live; a Sony
// library there is not something a legitimate build produces.
//
// CONFIDENCE: HIGH for the corpus — measured across all 50 local dumps, zero ship a `libSce*` under
// `Media/Plugins` or `Media/Modules`, while three ship exactly such files under `fakelib/`. MED as a
// universal claim about PS5 titles.
//
// That measurement covers the two Media directories and was NOT re-run for the dump root when the
// root was opened (#3497). What the root measurement does say: of 60 local dumps, exactly two ship a
// root-level module at all — PPSA23806 (four, its own middleware) and PPSA16901 (one) — and neither
// is a platform name. So the root rule is unexercised as a refusal across the corpus, which is a
// weaker statement than the Media one and is why the platform-prefix list below exists rather than
// relying on `libSce*` alone. If a real title ever does need one, the refusal is loud and
// names the file, and widening the rule is a reviewed source change — which is the right direction
// for the error to point, because the alternative failure is silent.
inline constexpr const char* kSonyLibraryPrefix = "libsce";  // compared lowercased

// The dump ROOT is auto-scanned wholesale (#3497), and unlike `Media/*` it is not a directory a
// title owns by convention -- `eboot.bin` and `sce_module/` live there too. `libSce*` alone is NOT
// sufficient cover for it: at least 23 system modules do not carry that prefix -- 21 of the 275 in
// the 3.20 reference, plus `libwvoec.sprx` and `ulobjmgr.sprx`, which are on-disk firmware and absent
// from that reference (so 23 is a floor, not a count). `linker.cpp`'s "a cross-module export beats a
// stub slot" rule means a root module shadowing one of those would silently displace prosper's HLE.
//
// WHICH of them matter is the narrower and more useful question: a module can only shadow prosper
// where prosper registers the NID, and `src/hle/` implements exactly two non-`libSce` surfaces --
// `kernel/` and `libc/`. The list below therefore covers the whole shadowing-relevant surface, and
// the other non-`libSce` names (`libwvoec`, `ulobjmgr`, `libcurl`, `libpng16`, …) are inert here
// because prosper implements none of them.
//
// So the root additionally refuses the PLATFORM modules prosper itself provides or implements.
// Deliberately NOT the whole non-`libSce` list: most of it is third-party (`libcurl`, `libpng16`,
// `libfreetype`, `libicu`, `libharfbuzz`) and a title may legitimately ship its own build of those
// -- refusing them would break exactly the case this widening exists to serve.
//
// HONEST LIMIT, because the previous version of this comment overstated it and was corrected in
// review: this is a name heuristic over the platform surface, not a proof about every Sony-authored
// module. What actually makes the root safe is the pair -- this list plus the extension gate plus
// the fact that `fakelib/` and every other directory remain refused outright. A dump that ships a
// platform replacement under a name matched by neither rule would still be linked, and the defence
// against that is the link-time HLE ownership, not this function.
//
// `dump_hygiene.py` mirrors this list (`PLATFORM_ROOT_PREFIXES` there) so the tool and the loader
// agree on what the root refuses. They are two implementations of one rule and can drift; if you
// add a prefix here, add it there. Before the re-review of #3508 they HAD drifted -- this header
// cited the tool as a backstop for names `libSce*` misses, while the tool matched `libsce` only.
inline constexpr const char* kRootRefusedPlatformPrefixes[] = {
    "libkernel",   // libkernel, libkernel_sys, libkernel_web
    "libc.",       // libc.prx exactly -- NOT libcairo/libcurl, which a title may ship
    "libc_",
    "libmdbg",     // libmdbg_syscore
    "gaikai",      // gaikai-player
};

enum class ModulePathVerdict {
    Permitted,
    OutsideDumpRoot,             // not under the dump root at all, or escapes it via `..`
    DirectoryNotPermitted,       // inside the dump, in a directory prosper never links from
    SonyLibraryOutsideSceModule, // a platform module in an auto-linked location (Media/* or the root)
};

struct ModulePathDecision {
    ModulePathVerdict verdict = ModulePathVerdict::DirectoryNotPermitted;
    // Human-readable and specific: this string is what a person debugging a missing module reads.
    std::string reason;

    bool permitted() const { return verdict == ModulePathVerdict::Permitted; }
};

// Pure, filesystem-free, and therefore testable: decides from the two strings alone. `path` is
// judged lexically against `dump_root`, so a `..` component is refused rather than resolved.
ModulePathDecision classify_module_path(const std::string& dump_root, const std::string& path);

struct RejectedModule {
    std::string path;
    std::string reason;
};

// Removes every entry of `in` whose path the policy refuses, and returns what was removed so the
// caller can report it. Reporting is the caller's job because a dropped module is invisible
// afterwards, and silence here is the failure mode this whole file exists to prevent.
std::vector<RejectedModule> enforce_module_path_policy(const std::string& dump_root,
                                                       std::vector<LinkInput>& in);

}  // namespace prosper
