// save_paths.hpp — where one title's persistent save data lives on the host, and why it may not
// share a directory with another title's (#2734).
//
// prosper serves two independent PS5 save APIs, and BOTH used to resolve to a single flat host
// directory shared by every title on the machine:
//
//   * the guest "/savedata0" file mount (sceSaveDataMount3 -> hle_file.cpp), whose per-save
//     subdirectory name is the GUEST's own dirName, and
//   * SaveDataMemory (sceSaveDataMemorySet2/Get2 -> hle_service.cpp), keyed by (userId, slotId).
//
// Neither key is unique across titles. UE4 titles routinely pick generic dirNames — the observed
// inventory on one developer machine held 34 slot directories from many titles, including
// `OptionSettings`, `GameUserSettingsini`, `Profile`, `Global`, `Inputini` and `Settings` — so two
// installed titles read and overwrote each other's saves. *Little Nightmares III* read an
// `OptionSettings` slot another Unreal title had written, correctly concluded it was not its own
// save format, **deleted it**, and held on a modal reading "Your options save has corrupted and has
// been deleted" that an unattended run cannot dismiss. That is real data loss caused by prosper and
// reported to the user in the game's own words, so it reads as the title's defect rather than ours.
//
// Every save path therefore carries a title component: <root>/<TITLE_ID>/<guest key>. The title id
// comes from ONE place — the single sce_sys/param.json parse in hle_addcontent.cpp, published as
// AppParamDeclaration::title_id — so no second reader can disagree with it about who is running.
//
// ## PROSPER_SAVE0 / PROSPER_SAVEDATA_DIR are ROOTS, not final directories
//
// Both variables used to name the flat directory that saves were written directly into. They now
// name the root that the per-title directories are created UNDER, and there is deliberately no way
// to opt out of the title component. The reason is that the ~15 documented run recipes in docs/ and
// scripts/ all set these variables, and they are what agents and users copy: an "escape hatch"
// spelling would have left the collision reachable in precisely the configurations people actually
// use, which is the whole defect. `savedata_policy: "fresh"` in tools/snapshot keeps working and now
// means only "start from an empty save state" rather than doubling as collision avoidance.
//
// ## The default location moved off /tmp
//
// The old defaults (`/tmp/prosper-savedata0`, `/tmp/prosper-savedata-mem`) put persistent user data
// on what is, on the Linux development box, a RAM-backed tmpfs with a per-user quota shared by every
// concurrent agent — so a save could evaporate on reboot or be evicted by an unrelated capture run.
// Saves are user DATA, so the default follows the platform's per-user data location, matching the
// shape prosper-app already uses for its settings file (an env override in front of a sensible
// per-user default): $XDG_DATA_HOME or ~/.local/share on Unix, %APPDATA% on Windows.
//
// ## Existing saves are NOT migrated, on purpose
//
// See the `## Ruled out` notes in docs/SAVE_DATA_LAYOUT.md. Briefly: nothing on disk records which
// title wrote a flat slot directory, so neither of the obvious migrations is safe. A move-on-first-
// use hands `OptionSettings` to whichever title happens to boot first, destructively and
// irreversibly — the same data-loss bug wearing a fix's clothes. A read-fallback to the flat path
// re-creates #2734 exactly: it is what let Little Nightmares III read a foreign save in the first
// place. Legacy directories are therefore left untouched where they are, and their presence is
// announced once per process with the manual move to make, so the failure is loud rather than a
// silent orphaning.
#pragma once

#include <string>

namespace prosper {

// The path component separating one title's saves from another's: the running application's
// validated title id, or kUnknownTitleNamespace when there is no local declaration to derive one
// from (an eboot-only dump has no sce_sys/param.json). Never empty, never a path separator, and
// never a name a valid title id could take.
std::string save_title_namespace();

// The placeholder used when the running application declares no usable title id. The leading
// underscore cannot begin a valid title id (`PPSA` + 5 digits), so it can never collide with one.
inline constexpr const char* kUnknownTitleNamespace = "_unknown-title";

// The roots, without the title component: the environment override in front of the per-user default.
// Nothing is created, and the value is deliberately re-derived on each call rather than frozen in a
// static -- a snapshot taken by whichever caller happened to run first is the same "first caller
// wins" failure the title component avoids, and it would make the override untestable in-process.
// The one thing that does happen at most once per distinct root is the legacy-save report below.
std::string savedata0_root();
std::string savedata_mem_root();

// <root>/<title namespace>. Pure: safe to call from a diagnostic that must not touch the disk.
std::string savedata0_dir();
std::string savedata_mem_dir();

// The same paths, with the directory created. Returns the path on success and an empty string if it
// could not be created, so a caller cannot mistake an unwritable location for a working one.
std::string savedata0_ensure_dir();
std::string savedata_mem_ensure_dir();

}   // namespace prosper
