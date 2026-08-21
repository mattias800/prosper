// support_modules.hpp — which bundled support PRXs are worth loading at all.
//
// prosper has no runtime PRX loading (#639), so `boot_link_inputs()` preloads a fixed list of
// optional modules that some title needed. Preloading is not free and not silent: a linked module
// contributes exports AND prosper runs its `module_start`, which is guest code that can block
// forever. So a module preloaded for title A can wedge the boot of unrelated title B, which merely
// ships the same file.
//
// That is not hypothetical. `sce_module/libSceNpCppWebApi.prx` is preloaded because *Sonic Origins*
// statically imports it. *Sniper Ghost Warrior Contracts 2* (PPSA03130) ships the identical file,
// imports `libSceNpWebApi2` instead (a different library), and its `module_start` deadlocks in a
// condition wait 81 ms into the boot — before `run_entry`, before a single asset is read.
//
// The rule here is deliberately NARROW: an input opts in with `LinkInput::only_if_imported`, and
// only then is it required to be named by some other module's import table. It must not be
// generalised to the Unity FMOD/Wwise/PSN plugins in the same list, which are preloaded PRECISELY
// because nothing imports them statically — they are reached at runtime through `sceKernelDlsym`
// P/Invoke, appear in no import table, and this test would drop every one of them.
//
// Split out of boot_program.cpp as a pure function over already-parsed data so the policy can be
// tested without a game dump, a SELF parser, or a filesystem.

#pragma once

#include <string>
#include <vector>

#include "loader/linker.hpp"

namespace prosper {

// The library name an importer would use for this module: the path's basename with its extension
// removed. `/dump/sce_module/libSceNpCppWebApi.prx` -> `libSceNpCppWebApi`. Handles both separators
// so a Windows-spelled path resolves identically.
std::string support_module_lib_name(const std::string& path);

// Indices of `in` that are `only_if_imported` candidates nobody imports, in DESCENDING order so a
// caller can erase them one by one without invalidating the indices it has not used yet.
//
// `imports_by_index` is parallel to `in`: entry i lists the library names module i imports. A
// candidate's own entry is IGNORED rather than trusted — a module must not be able to justify its
// own preload, whether by importing itself or by importing a sibling candidate. A module that could
// not be parsed contributes an empty list, which is why an unreadable importer must be reported by
// the caller: silence here would look identical to "imports nothing".
std::vector<size_t> unimported_support_module_indices(
    const std::vector<LinkInput>& in,
    const std::vector<std::vector<std::string>>& imports_by_index);

} // namespace prosper
