#pragma once
// game_library.hpp — describe the PS5 titles under a games directory: each one's app0 root, content
// id, human-readable name, and icon. This is the data the library view draws and what --list-games
// prints; the UI is a separate concern layered on top (#1471).
//
// Filesystem access is injected (GameLibraryIo) and nothing here touches SDL or ImGui, so the whole
// scan is unit-tested against a fixed set of names and file contents — no window, no real dump. Same
// pure regression seam as game_path.hpp, which this builds on for "is this a title" decisions.
//
// The param.json readers are deliberately dependency-free string scans: there is no JSON library in
// the tree, and the two fields wanted here (titleId, localizedParameters.<lang>.titleName) are
// simple enough that pulling one in would be disproportionate.

#include "game_path.hpp"

#include <algorithm>
#include <cctype>
#include <functional>
#include <string>
#include <vector>

namespace prosper::frontend {

// Injected IO. `list_dir` returns the immediate child names of a directory (no "." / ".."), in any
// order; `read_file` returns a file's whole contents, or "" when it cannot be read.
struct GameLibraryIo {
    std::function<std::vector<std::string>(const std::string&)> list_dir;
    std::function<std::string(const std::string&)> read_file;
    // Optional: map a path to the real on-disk spelling, correcting case-only differences the way
    // boot_program does (#1006/#1226). Used so a recorded icon_path is one that actually opens on a
    // case-sensitive host. Identity when unset.
    std::function<std::string(const std::string&)> resolve_case;
};

struct GameEntry {
    std::string app0_root;   // what boot_program() wants
    std::string title_id;    // e.g. "PPSA24651"; "" when param.json has none
    std::string title_name;  // e.g. "The Messenger"; falls back to the app0 basename
    std::string icon_path;   // sce_sys/icon0.png, or "" when absent
};

// The JSON string value following the first "titleName" in [from, limit). Returns "" when absent.
//
// `limit` is what keeps a bounded lookup honest: without it a language object that omits titleName
// would silently yield the NEXT language's name, which is the same class of quiet mislabelling this
// file exists to fix. Callers that legitimately want a document-wide search pass npos.
inline std::string param_title_name_after(const std::string& json, size_t from,
                                          size_t limit = std::string::npos) {
    size_t k = json.find("\"titleName\"", from);
    if (k == std::string::npos || k >= limit) return "";
    k = json.find(':', k); if (k == std::string::npos) return "";
    k = json.find('"', k); if (k == std::string::npos) return "";
    std::string out;
    for (size_t e = k + 1; e < json.size() && json[e] != '"'; ++e) {
        if (json[e] == '\\' && e + 1 < json.size()) { ++e; out += json[e]; }  // \" \\ etc.
        else out += json[e];
    }
    return out;
}

// Where `"<lang>"` is used as an OBJECT KEY — `"en-US": { ... }` — rather than as a value. The
// distinction matters because the language name also appears as defaultLanguage's own value
// (`"defaultLanguage": "en-US"`), and matching that instead reads the wrong title. A key is identified
// by what follows it: optional space, a colon, optional space, then `{`.
// Reports the object's opening brace in `brace_out` so the caller can bound its search to that object.
inline size_t find_language_object(const std::string& json, const std::string& lang,
                                   size_t from = 0, size_t* brace_out = nullptr) {
    if (lang.empty()) return std::string::npos;
    const std::string needle = "\"" + lang + "\"";
    for (size_t k = json.find(needle, from); k != std::string::npos;
         k = json.find(needle, k + 1)) {
        size_t c = k + needle.size();
        while (c < json.size() && std::isspace((unsigned char)json[c])) ++c;
        if (c >= json.size() || json[c] != ':') continue;
        ++c;
        while (c < json.size() && std::isspace((unsigned char)json[c])) ++c;
        if (c < json.size() && json[c] == '{') {
            if (brace_out) *brace_out = c;
            return k;
        }
    }
    return std::string::npos;
}

// The offset just past the object that opens at `brace` (which must index a '{'), or npos if the
// braces never balance. String contents are skipped so a '}' inside a value cannot end the object
// early.
inline size_t json_object_end(const std::string& json, size_t brace) {
    if (brace >= json.size() || json[brace] != '{') return std::string::npos;
    int depth = 0;
    for (size_t i = brace; i < json.size(); ++i) {
        const char ch = json[i];
        if (ch == '"') {                                  // skip a string, honouring escapes
            for (++i; i < json.size(); ++i) {
                if (json[i] == '\\') { ++i; continue; }
                if (json[i] == '"') break;
            }
            continue;
        }
        if (ch == '{') ++depth;
        else if (ch == '}' && --depth == 0) return i + 1;
    }
    return std::string::npos;
}

// The title's display name. PS5 param.json stores it per language under localizedParameters; prefer
// the entry for defaultLanguage, falling back to the first titleName present only when that entry
// cannot be found at all.
//
// The language object is located by SHAPE, not by position. The previous version of this logic (which
// lived in main.cpp and drove the window title) required the object key to appear textually before the
// "defaultLanguage" line, and silently produced the wrong name on any dump that declares
// defaultLanguage first: the first `"en-US"` match was then defaultLanguage's own value, the position
// guard rejected it, and the fallback picked whichever language happened to come first in the file.
// The Plucky Squire (PPSA15319) ships exactly that order and read as "DER KÜHNE KNAPPE" (de-DE)
// instead of "The Plucky Squire" (#1471).
inline std::string parse_param_title_name(const std::string& json) {
    // Anchor both lookups inside localizedParameters when it is present, so a language-keyed object
    // elsewhere in the document cannot be mistaken for a title entry. Falls back to the whole document
    // if that key is absent, which keeps hand-written or trimmed metadata working.
    const size_t lp = json.find("\"localizedParameters\"");
    size_t scope_begin = 0, scope_end = std::string::npos;
    if (lp != std::string::npos) {
        const size_t brace = json.find('{', lp);
        if (brace != std::string::npos) {
            scope_begin = brace;
            scope_end = json_object_end(json, brace);
        }
    }
    std::string title;
    const size_t dl = json.find("\"defaultLanguage\"", scope_begin);
    if (dl != std::string::npos && dl < scope_end) {
        const size_t c = json.find(':', dl);
        const size_t q = (c == std::string::npos) ? std::string::npos : json.find('"', c);
        if (q != std::string::npos) {
            const size_t qe = json.find('"', q + 1);
            if (qe != std::string::npos) {
                const std::string lang = json.substr(q + 1, qe - q - 1);
                size_t brace = std::string::npos;
                const size_t lk = find_language_object(json, lang, scope_begin, &brace);
                // Bounded to that language's own object: if it carries no titleName we must report
                // nothing for it rather than borrow the next language's name.
                if (lk != std::string::npos && lk < scope_end)
                    title = param_title_name_after(json, lk, json_object_end(json, brace));
            }
        }
    }
    if (title.empty()) title = param_title_name_after(json, scope_begin, scope_end);
    return title;
}

// The title's content id ("titleId": "PPSA24651"). Returns "" when absent.
inline std::string parse_param_title_id(const std::string& json) {
    size_t k = json.find("\"titleId\"");
    if (k == std::string::npos) return "";
    k = json.find(':', k); if (k == std::string::npos) return "";
    k = json.find('"', k); if (k == std::string::npos) return "";
    const size_t end = json.find('"', k + 1);
    if (end == std::string::npos) return "";
    return json.substr(k + 1, end - k - 1);
}

// The last path component, used as the display name when param.json gives none.
inline std::string path_basename(const std::string& path) {
    const std::string p = strip_trailing_separators(path);
    const size_t slash = p.find_last_of("/\\");
    return slash == std::string::npos ? p : p.substr(slash + 1);
}

// Describe one already-resolved app0 root. Never fails: a title whose param.json is missing or
// unreadable still gets an entry, named after its directory — the whole point of the library is to
// show what is bootable, and boot_program only needs eboot.bin.
inline GameEntry describe_game(const std::string& app0_root, const GamePathProbe& probe,
                               const GameLibraryIo& io) {
    GameEntry entry;
    entry.app0_root = strip_trailing_separators(app0_root);
    // Ask the probe before reading. A title accepted on eboot.bin alone has no param.json, and the
    // host reader case-corrects — which, on a miss, enumerates the directories it searched. That is a
    // scan of a whole title root to learn something the probe already knows.
    const std::string param = entry.app0_root + "/sce_sys/param.json";
    const std::string json = (io.read_file && probe.is_file && probe.is_file(param))
                                 ? io.read_file(param)
                                 : std::string();
    if (!json.empty()) {
        entry.title_id = parse_param_title_id(json);
        entry.title_name = parse_param_title_name(json);
    }
    if (entry.title_name.empty()) entry.title_name = path_basename(entry.app0_root);
    // Record the resolved spelling, not the requested one: the probe corrects case, so storing the
    // literal path could hand the caller something that will not open.
    const std::string icon = entry.app0_root + "/sce_sys/icon0.png";
    if (probe.is_file && probe.is_file(icon))
        entry.icon_path = io.resolve_case ? io.resolve_case(icon) : icon;
    return entry;
}

// Case-insensitive display order, so the list reads the way a person would sort it. Ties break on the
// content id and then the path, so the order is total and a scan is reproducible whatever order
// list_dir returned.
inline bool game_entry_display_less(const GameEntry& a, const GameEntry& b) {
    const auto fold = [](const std::string& s) {
        std::string out = s;
        std::transform(out.begin(), out.end(), out.begin(),
                       [](unsigned char c) { return (char)std::tolower(c); });
        return out;
    };
    const std::string an = fold(a.title_name), bn = fold(b.title_name);
    if (an != bn) return an < bn;
    if (a.title_id != b.title_id) return a.title_id < b.title_id;
    return a.app0_root < b.app0_root;
}

// Every PS5 title directly inside `games_dir`, in display order.
//
// One level deep, deliberately: a title's own subdirectories hold its assets, so recursing would both
// waste time and risk presenting an inner directory as a separate game. The directory itself is not
// considered even when it is a title root — pointing this at a single app0 folder yields nothing
// rather than one oddly-named entry, and #1469's picker already covers opening one specific game.
inline std::vector<GameEntry> scan_game_library(const std::string& games_dir,
                                                const GamePathProbe& probe,
                                                const GameLibraryIo& io) {
    std::vector<GameEntry> games;
    const std::string dir = strip_trailing_separators(games_dir);
    if (dir.empty() || !io.list_dir || !probe.is_dir || !probe.is_dir(dir)) return games;
    for (const std::string& child : io.list_dir(dir)) {
        if (child.empty() || child == "." || child == "..") continue;
        const std::string path = dir + "/" + child;
        if (!is_app0_root(path, probe)) continue;
        games.push_back(describe_game(path, probe, io));
    }
    std::sort(games.begin(), games.end(), game_entry_display_less);
    return games;
}

} // namespace prosper::frontend
