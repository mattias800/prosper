#pragma once
// app_config.hpp — the small amount of state the app remembers between launches, currently just where
// the user keeps their games (#1471).
//
// The project's existing convention for host paths is a PROSPER_* environment override with a sensible
// default (see PROSPER_SAVEDATA_DIR, PROSPER_SAVE0, PROSPER_CAPTURE_DIR). That stays intact and keeps
// winning over the file: a persisted setting exists only so someone who opened a folder in the GUI is
// not asked again next launch, and it must never quietly override what a command line or a script
// asked for. Precedence is therefore: --games-dir > PROSPER_GAMES_DIR > config file > nothing.
//
// Parsing and serializing are pure and unit-tested; choosing the file's location and touching the disk
// stays in main.cpp.

#include <string>
#include <vector>

namespace prosper::frontend {

struct AppConfig {
    std::string games_dir;   // "" = not set

    // Settings this build does not understand, kept verbatim so a round trip does not destroy them.
    // Reading tolerating unknown keys is not enough on its own: --set-games-dir rewrites the whole
    // file, so without this a release build would silently delete whatever a newer build had stored.
    std::vector<std::string> unknown_lines;
};

// Trim ASCII spaces and tabs from both ends.
inline std::string config_trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t')) ++b;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r')) --e;
    return s.substr(b, e - b);
}

// Read a `key = value` file. Blank lines and `#` comments are ignored, unknown keys are ignored (so a
// newer build's config does not break an older one), and a later duplicate wins. Values are taken
// literally after the first `=`, so a path may contain spaces, `=`, or `#`.
inline AppConfig parse_app_config(const std::string& text) {
    AppConfig cfg;
    size_t pos = 0;
    while (pos <= text.size()) {
        size_t nl = text.find('\n', pos);
        if (nl == std::string::npos) nl = text.size();
        const std::string line = config_trim(text.substr(pos, nl - pos));
        pos = nl + 1;
        if (line.empty() || line[0] == '#') continue;
        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = config_trim(line.substr(0, eq));
        const std::string value = config_trim(line.substr(eq + 1));
        if (key == "games_dir") cfg.games_dir = value;
        else cfg.unknown_lines.push_back(line);   // preserved across a rewrite
        if (nl == text.size()) break;
    }
    return cfg;
}

inline std::string serialize_app_config(const AppConfig& cfg) {
    std::string out =
        "# prosper-app settings. Written by the app.\n"
        "# A --games-dir argument or PROSPER_GAMES_DIR in the environment overrides games_dir.\n"
        "# Editing by hand is fine; the app rewrites this file, so comments are not preserved.\n";
    if (!cfg.games_dir.empty()) out += "games_dir = " + cfg.games_dir + "\n";
    for (const std::string& line : cfg.unknown_lines) out += line + "\n";
    return out;
}

// Apply the precedence above. Each argument is "" when that source said nothing.
inline std::string resolve_games_dir(const std::string& flag, const std::string& env,
                                     const AppConfig& file) {
    if (!flag.empty()) return flag;
    if (!env.empty()) return env;
    return file.games_dir;
}

} // namespace prosper::frontend
