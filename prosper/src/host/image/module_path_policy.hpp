#pragma once
// module_path_policy — which files inside a game dump prosper is willing to link as a guest module.
//
// REJECT BY DEFAULT. `kPermittedModuleDirs` below is the COMPLETE set of dump-relative locations a
// module may be linked from; anything else is refused and reported, whatever the file is called and
// whatever produced it.
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

// Files permitted directly in the dump root.
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
// universal claim about PS5 titles. If a real title ever does need one, the refusal is loud and
// names the file, and widening the rule is a reviewed source change — which is the right direction
// for the error to point, because the alternative failure is silent.
inline constexpr const char* kSonyLibraryPrefix = "libsce";  // compared lowercased

enum class ModulePathVerdict {
    Permitted,
    OutsideDumpRoot,             // not under the dump root at all, or escapes it via `..`
    DirectoryNotPermitted,       // inside the dump, in a directory prosper never links from
    SonyLibraryOutsideSceModule, // a libSce* module in an auto-linked Media directory
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
