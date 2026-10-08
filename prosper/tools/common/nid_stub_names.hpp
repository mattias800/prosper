// nid_stub_names.hpp — read <NID> ↔ <funcName> pairs out of the PS5 3.20 firmware stub dump.
//
// Part of the `prosper` PS5->PC compatibility layer.
//
// The stub dump (CLAUDE.md: "PS5 3.20 firmware library reference", a gitignored sibling of the
// repository) is one generated `libSceXxx.c` per system library, each carrying one loader line per
// export:
//
//     if(sprx_dlsym(__handle, "PI7jIZj4pcE", &__ptr_sceRandomGetRandomNumber)) return;
//
// so the NID/name pair is read off the line directly — no hashing is involved in BUILDING the map,
// which is what lets a caller check prosper's own `nid_hash` AGAINST the dump rather than using the
// hash to produce it (nid_census --self-check does exactly that, through `on_pair`).
//
// Header-only and free of any prosper_core dependency on purpose: `self_dump` is a deliberately
// standalone single-translation-unit tool, and this is the one piece it needs to share with
// `nid_census` so the two cannot drift into two different readings of the same dump.
#pragma once

#include <cctype>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <string>

namespace prosper_tools {

// Extract the NID and function name from one generated `sprx_dlsym(...)` loader line.
// Returns false for every other line in the file.
inline bool parse_stub_line(const std::string& line, std::string* nid, std::string* name) {
    const size_t call = line.find("sprx_dlsym(");
    if (call == std::string::npos) return false;
    const size_t q1 = line.find('"', call);
    if (q1 == std::string::npos) return false;
    const size_t q2 = line.find('"', q1 + 1);
    if (q2 == std::string::npos) return false;
    const size_t ptr = line.find("&__ptr_", q2);
    if (ptr == std::string::npos) return false;
    size_t end = ptr + 7;
    while (end < line.size() && (isalnum((unsigned char)line[end]) || line[end] == '_')) ++end;
    *nid = line.substr(q1 + 1, q2 - q1 - 1);
    *name = line.substr(ptr + 7, end - (ptr + 7));
    return !nid->empty() && !name->empty();
}

// Parse one flat `NID name` line from a community NID database (aerolib.csv / ps5rs stubs.txt):
//
//     PI7jIZj4pcE sceRandomGetRandomNumber
//
// The NID and the name are separated by whitespace; everything after the first run of whitespace is
// the name, trimmed. Blank lines and `#` comment lines return false. This is the format the
// ps5-payload SDK's genstub flow and the zecoxao/ps5rs databases use, so a caller can name NIDs from
// the aggregated community map without the per-library `sprx_dlsym(...)` firmware dump.
inline bool parse_nid_csv_line(const std::string& line, std::string* nid, std::string* name) {
    size_t b = 0;
    while (b < line.size() && std::isspace((unsigned char)line[b])) ++b;
    if (b >= line.size() || line[b] == '#') return false;   // blank or comment
    size_t e = b;
    while (e < line.size() && !std::isspace((unsigned char)line[e])) ++e;  // end of NID
    const std::string n = line.substr(b, e - b);
    size_t s = e;
    while (s < line.size() && std::isspace((unsigned char)line[s])) ++s;   // skip gap
    size_t t = line.size();
    while (t > s && std::isspace((unsigned char)line[t - 1])) --t;         // rtrim name
    if (n.empty() || s >= t) return false;
    *nid = n;
    *name = line.substr(s, t - s);
    return true;
}

struct StubNames {
    std::map<std::string, std::string> by_nid;   // nid -> function name
    std::map<std::string, std::string> lib_of;   // nid -> library file stem
    size_t pairs = 0;                            // parsed loader lines (duplicates included)
    size_t files = 0;                            // .c files read
    bool   dir_ok = false;                       // the directory itself was readable
};

// Read every `*.c` in `dir`. `on_pair`, when set, is invoked for each parsed pair in file order so a
// caller can run its own control over the table (e.g. re-derive the NID from the name).
//
// A missing or unreadable directory yields `dir_ok == false` with an empty table rather than an
// exception — callers report that state explicitly instead of presenting an empty table as "this
// module imports nothing nameable".
inline StubNames load_stub_names(
    const std::string& dir,
    const std::function<void(const std::string& nid, const std::string& name)>& on_pair = {}) {
    namespace fs = std::filesystem;
    StubNames t;
    std::error_code ec;
    fs::directory_iterator it(dir, ec);
    if (ec) return t;
    t.dir_ok = true;
    for (const auto& e : it) {
        if (!e.is_regular_file() || e.path().extension() != ".c") continue;
        t.files++;
        std::ifstream in(e.path());
        std::string line, nid, name;
        while (std::getline(in, line)) {
            if (!parse_stub_line(line, &nid, &name)) continue;
            t.pairs++;
            t.by_nid[nid] = name;
            t.lib_of[nid] = e.path().stem().string();
            if (on_pair) on_pair(nid, name);
        }
    }
    return t;
}

// Read one flat `NID name` database file (aerolib.csv / ps5rs stubs.txt). A `# libName` comment
// line sets the library attributed to the pairs that follow it (the ps5rs stubs.txt convention);
// aerolib.csv has no such comments, so `lib_of` is simply left empty there. `on_pair` fires per
// parsed pair in file order, exactly as `load_stub_names` does, so --self-check works identically
// whichever source named the NID. A missing/unreadable file yields `dir_ok == false`.
inline StubNames load_nid_csv(
    const std::string& path,
    const std::function<void(const std::string& nid, const std::string& name)>& on_pair = {}) {
    StubNames t;
    std::ifstream in(path);
    if (!in) return t;
    t.dir_ok = true;
    t.files = 1;
    std::string line, nid, name, cur_lib;
    while (std::getline(in, line)) {
        // Track a leading `# libName` so a grouped dump attributes names to libraries.
        size_t b = 0;
        while (b < line.size() && std::isspace((unsigned char)line[b])) ++b;
        if (b < line.size() && line[b] == '#') {
            // Only a `# libName` comment (a single whitespace-free token beginning "lib") sets the
            // current library, as ps5rs stubs.txt writes it. A prose comment is just skipped, so it
            // does not mis-attribute the pairs that follow it to a bogus library.
            size_t s = b + 1;
            while (s < line.size() && std::isspace((unsigned char)line[s])) ++s;
            size_t e = line.size();
            while (e > s && std::isspace((unsigned char)line[e - 1])) --e;
            const std::string tok = (s < e) ? line.substr(s, e - s) : std::string();
            if (tok.rfind("lib", 0) == 0 &&
                tok.find_first_of(" \t") == std::string::npos) {
                cur_lib = tok;
            }
            continue;
        }
        if (!parse_nid_csv_line(line, &nid, &name)) continue;
        t.pairs++;
        t.by_nid[nid] = name;
        if (!cur_lib.empty()) t.lib_of[nid] = cur_lib;
        if (on_pair) on_pair(nid, name);
    }
    return t;
}

// Name NIDs from either source: a DIRECTORY is the per-library `sprx_dlsym(...)` firmware dump
// (load_stub_names); a regular FILE is a flat `NID name` database (load_nid_csv). Anything else
// yields an empty table with `dir_ok == false`, which the caller reports rather than presenting as
// "nothing is nameable".
inline StubNames load_nid_names(
    const std::string& path,
    const std::function<void(const std::string& nid, const std::string& name)>& on_pair = {}) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (fs::is_directory(path, ec)) return load_stub_names(path, on_pair);
    if (fs::is_regular_file(path, ec)) return load_nid_csv(path, on_pair);
    return {};
}

}  // namespace prosper_tools
