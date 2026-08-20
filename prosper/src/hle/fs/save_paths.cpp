// save_paths.cpp — see save_paths.hpp for the contract and for why existing saves are not migrated.
#include "hle/fs/save_paths.hpp"

#include "hle/service/hle_addcontent.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <set>
#include <string>
#include <system_error>
#include <vector>

namespace prosper {
namespace {

namespace fs = std::filesystem;

// The pre-#2734 flat defaults. Retained ONLY so a machine that has saves under them is told where
// they are; nothing is ever written to these paths again.
constexpr const char* kLegacyFlatSave0Root = "/tmp/prosper-savedata0";
constexpr const char* kLegacyFlatMemRoot   = "/tmp/prosper-savedata-mem";

const char* env_or_null(const char* name) {
    const char* value = std::getenv(name);
    return (value && *value) ? value : nullptr;
}

// The platform's per-user DATA location — the sibling of the per-user config location prosper-app
// already uses for its settings file. Empty when the environment names nowhere sensible.
std::string user_data_root() {
#ifdef _WIN32
    if (const char* appdata = env_or_null("APPDATA")) return std::string(appdata) + "\\prosper";
    if (const char* profile = env_or_null("USERPROFILE"))
        return std::string(profile) + "\\AppData\\Roaming\\prosper";
#else
    if (const char* xdg = env_or_null("XDG_DATA_HOME")) return std::string(xdg) + "/prosper";
    if (const char* home = env_or_null("HOME")) return std::string(home) + "/.local/share/prosper";
#endif
    return {};
}

std::string resolve_root(const char* env_name, const char* leaf, const char* last_resort) {
    if (const char* configured = env_or_null(env_name)) return configured;
    const std::string data = user_data_root();
    if (!data.empty()) {
#ifdef _WIN32
        return data + "\\" + leaf;
#else
        return data + "/" + leaf;
#endif
    }
    // No HOME and no XDG_DATA_HOME: a bare container or a service account. Fall back to the historic
    // path so saving still works, and say so — on the development box that location is a RAM-backed
    // tmpfs, so "your saves are somewhere that does not survive a reboot" is exactly the sort of
    // thing that must not be discovered later.
    std::fprintf(stderr,
                 "[savedata] no per-user data location (neither %s is set); falling back to %s, "
                 "which may not survive a reboot. Set %s to keep saves somewhere durable.\n",
#ifdef _WIN32
                 "APPDATA nor USERPROFILE",
#else
                 "XDG_DATA_HOME nor HOME",
#endif
                 last_resort, env_name);
    return last_resort;
}

bool is_title_namespace(const std::string& name) {
    return valid_title_id(name) || name == kUnknownTitleNamespace;
}

// One-time report of save data left over from the pre-#2734 flat layout. Deliberately prints the
// host paths: unlike the add-content diagnostics, whose whole content is a directory layout nobody
// needs, this message is useless without them — its entire purpose is to tell the person sitting at
// the machine where their old saves are and what to type. It is a runtime message on their own
// terminal about their own files, not a published log.
void announce_legacy_flat_saves(const std::string& active_root, const char* historic_root,
                                bool memory_store) {
    std::vector<std::string> roots{active_root};
    if (active_root != historic_root) roots.emplace_back(historic_root);

    for (const std::string& root : roots) {
        std::error_code ec;
        fs::directory_iterator it(root, fs::directory_options::skip_permission_denied, ec);
        if (ec) continue;
        size_t stranded = 0;
        std::string example;
        for (const fs::directory_entry& entry : it) {
            const std::string name = entry.path().filename().string();
            if (name.empty() || name[0] == '.') continue;
            if (memory_store) {
                // SaveDataMemory's legacy shape is `savemem_<user>_<slot>.bin` sitting directly in
                // the root; post-fix it is one level down, under the title.
                if (!entry.is_regular_file(ec)) continue;
                if (name.rfind("savemem_", 0) != 0) continue;
            } else {
                if (!entry.is_directory(ec)) continue;
                if (is_title_namespace(name)) continue;
            }
            if (example.empty()) example = name;
            ++stranded;
        }
        if (stranded == 0) continue;

        // "1 save directory" / "2 save directories" -- the message names real user data, so it is
        // worth reading as a sentence rather than as a template.
        const char* what = memory_store ? (stranded == 1 ? "SaveDataMemory slot file"
                                                         : "SaveDataMemory slot files")
                                        : (stranded == 1 ? "save directory" : "save directories");
        std::fprintf(stderr,
            "[savedata] %s: %zu %s from prosper's old shared layout (for example \"%s\").\n"
            "[savedata]   Saves are now kept per title, under <root>/<TITLE_ID>/, because the guest\n"
            "[savedata]   chooses these names and several titles pick the same one -- so they used to\n"
            "[savedata]   overwrite each other (#2734).\n"
            "[savedata]   These are LEFT UNTOUCHED and are not visible to any title. prosper cannot\n"
            "[savedata]   move them for you: nothing on disk records which title wrote one, and\n"
            "[savedata]   guessing would hand one title another title's save. To restore one, move it\n"
            "[savedata]   yourself into the title that owns it:\n"
            "[savedata]     %s/%s  ->  %s/<TITLE_ID>/%s\n",
            root.c_str(), stranded, what, example.c_str(),
            root.c_str(), example.c_str(), active_root.c_str(), example.c_str());
    }
}

// At most one report per distinct root, so a guest that polls sceSaveDataDirNameSearch does not
// repeat it, while a process that legitimately changes root (a frontend loading a second title,
// a test) still gets the report for the new one.
void announce_once(const std::string& root, const char* historic_root, bool memory_store) {
    static std::mutex reported_mx;
    static std::set<std::string> reported;
    {
        std::lock_guard<std::mutex> lock(reported_mx);
        if (!reported.insert(std::string(memory_store ? "m:" : "f:") + root).second) return;
    }
    announce_legacy_flat_saves(root, historic_root, memory_store);
}

}   // namespace

std::string save_title_namespace() {
    const std::string id = app_param_declaration().title_id;
    if (!id.empty() && valid_title_id(id)) return id;
    return kUnknownTitleNamespace;
}

std::string savedata0_root() {
    std::string root = resolve_root("PROSPER_SAVE0", "savedata0", kLegacyFlatSave0Root);
    announce_once(root, kLegacyFlatSave0Root, /*memory_store=*/false);
    return root;
}

std::string savedata_mem_root() {
    std::string root = resolve_root("PROSPER_SAVEDATA_DIR", "savedata-mem", kLegacyFlatMemRoot);
    announce_once(root, kLegacyFlatMemRoot, /*memory_store=*/true);
    return root;
}

namespace {
std::string join_title(const std::string& root) {
    return (fs::path(root) / save_title_namespace()).string();
}
std::string ensure(const std::string& dir) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (!fs::is_directory(dir, ec)) {
        std::fprintf(stderr, "[savedata] cannot create the save directory '%s': saves will not persist\n",
                     dir.c_str());
        return {};
    }
    return dir;
}
}   // namespace

std::string savedata0_dir()   { return join_title(savedata0_root()); }
std::string savedata_mem_dir(){ return join_title(savedata_mem_root()); }

std::string savedata0_ensure_dir()    { return ensure(savedata0_dir()); }
std::string savedata_mem_ensure_dir() { return ensure(savedata_mem_dir()); }

}   // namespace prosper
