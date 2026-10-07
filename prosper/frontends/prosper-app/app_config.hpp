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

#include "game_path.hpp"   // strip_trailing_separators: recent roots compare canonically

#include <map>
#include <string>
#include <vector>

namespace prosper::frontend {

struct AppConfig {
    std::string games_dir;   // "" = not set

    // Whether the library plays the focused title's music (#1630). On by default — the console-like
    // presentation is the point of the feature — and PROSPER_LAUNCHER_MUSIC still overrides this the
    // way every other host setting here is overridden.
    bool launcher_music = true;

    // Settings this build does not understand, kept verbatim so a round trip does not destroy them.
    // Reading tolerating unknown keys is not enough on its own: --set-games-dir rewrites the whole
    // file, so without this a release build would silently delete whatever a newer build had stored.
    std::vector<std::string> unknown_lines;

    // Guest launch arguments (PROSPER_GUEST_ARGS), applied when the user's environment does not
    // already set one (#2973): Unity titles need `-force-gfx-direct` to reach their frame loop in
    // this app (the MT gfx-jobs handshake is not emulated yet — see RENDER_LOOP.md), while some
    // titles must NOT receive it. `guest_args_default` applies to every title;
    // `guest_args_by_title` overrides per TITLE_ID (config key spelling: `guest_args.PPSA02664`).
    std::string guest_args_default;                                  // "" = none
    std::map<std::string, std::string> guest_args_by_title;

    // Recently opened titles, most-recent-first, as app0 roots. Backs File > Recent games.
    // Capped so the file cannot grow without bound; a root whose dump is gone stays listed
    // (opening it says so) rather than being silently forgotten.
    std::vector<std::string> recent_games;
    static constexpr size_t kRecentGamesMax = 8;

    // Host settings previously reachable only via a flag or the environment, so a released build
    // started without a terminal (desktop icon, library picker) could never set them. Each applies
    // ONLY to boots the library starts — never to a scripted `prosper-app <dump>` run, whose guest
    // answers must not depend on a per-user file. "" means unset (the flag/env/default path is
    // untouched); present/display spellings are validated at apply time, so a typo costs the
    // setting rather than a wrong behaviour.
    std::string savedata_dir;   // PROSPER_SAVEDATA_DIR; "" = not set
    std::string present_mode;   // --present-mode; "" = unset, else fifo|mailbox|immediate
    std::string display_mode;   // PROSPER_DISPLAY_MODE; "" = unset, else legacy|host|host-high-refresh

    // Process volume in percent (--volume). -1 = unset, so the flag and the default path are
    // untouched; 0 is a real choice (muted) and survives. Parsed strictly, clamped like the flag.
    int volume_percent = -1;
};

// Strict decimal percent, clamped to [0,100]; anything unparseable is -1 (unset), so a typo
// costs the setting rather than muting or deafening the run.
inline int parse_volume_percent(const std::string& value) {
    if (value.empty()) return -1;
    int v = 0;
    for (char c : value) {
        if (c < '0' || c > '9') return -1;
        v = v * 10 + (c - '0');
        if (v > 100) return 100;
    }
    return v;
}

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
        else if (key == "launcher_music")
            // Anything other than an explicit off is on, so a hand-edited "yes" or "1" behaves.
            cfg.launcher_music = !(value == "0" || value == "false" || value == "off" || value == "no");
        else if (key == "guest_args") cfg.guest_args_default = value;
        else if (key.rfind("guest_args.", 0) == 0)
            cfg.guest_args_by_title[key.substr(11)] = value;
        else if (key == "recent") {
            if (!value.empty()) cfg.recent_games.push_back(value);
        } else if (key == "savedata_dir") cfg.savedata_dir = value;
        else if (key == "present_mode") cfg.present_mode = value;
        else if (key == "display_mode") cfg.display_mode = value;
        else if (key == "volume") cfg.volume_percent = parse_volume_percent(value);
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
    // A value carrying a newline would inject a key into the file on rewrite. Paths cannot
    // contain one on any host here, so such a value is dropped rather than written. (Values
    // read from the file can never carry one — parsing is line-based — so this only fires for
    // hand-built configs.)
    const auto path_safe = [](const std::string& v) {
        return v.find('\n') == std::string::npos && v.find('\r') == std::string::npos;
    };
    if (!cfg.games_dir.empty() && path_safe(cfg.games_dir))
        out += "games_dir = " + cfg.games_dir + "\n";
    // Written unconditionally, unlike games_dir: "off" is a real choice and must survive a rewrite,
    // whereas an absent games_dir simply means nothing was chosen.
    out += std::string("launcher_music = ") + (cfg.launcher_music ? "1" : "0") + "\n";
    for (const std::string& line : cfg.unknown_lines) out += line + "\n";
    for (const auto& [title, args] : cfg.guest_args_by_title)
        out += "guest_args." + title + " = " + args + "\n";
    if (!cfg.guest_args_default.empty()) out += "guest_args = " + cfg.guest_args_default + "\n";
    for (const std::string& recent : cfg.recent_games)
        if (!recent.empty() && path_safe(recent)) out += "recent = " + recent + "\n";
    if (!cfg.savedata_dir.empty() && path_safe(cfg.savedata_dir))
        out += "savedata_dir = " + cfg.savedata_dir + "\n";
    if (!cfg.present_mode.empty()) out += "present_mode = " + cfg.present_mode + "\n";
    if (!cfg.display_mode.empty()) out += "display_mode = " + cfg.display_mode + "\n";
    if (cfg.volume_percent >= 0) out += "volume = " + std::to_string(cfg.volume_percent) + "\n";
    return out;
}

// Apply the precedence above. Each argument is "" when that source said nothing.
inline std::string resolve_games_dir(const std::string& flag, const std::string& env,
                                     const AppConfig& file) {
    if (!flag.empty()) return flag;
    if (!env.empty()) return env;
    return file.games_dir;
}

// Everything a library-started boot needs from the persisted host settings, resolved in one
// pure step so the precedence main.cpp ships is the precedence the tests pin — including the
// "environment wins" arms, which take the env values as inputs rather than reading them.
struct HostPolicyInputs {
    bool flag_present_mode = false;   // --present-mode was given
    bool flag_display_mode = false;   // --display-mode was given
    bool flag_volume = false;         // --volume was given
    std::string env_savedata_dir;     // PROSPER_SAVEDATA_DIR, or "" when unset
    std::string env_display_mode;     // PROSPER_DISPLAY_MODE, or "" when unset
    AppConfig file;
};

struct HostPolicy {
    std::string savedata_dir;   // "" = leave the environment alone
    std::string present_mode;   // "" = keep the default; validated (fifo|mailbox|immediate) by main
    std::string display_mode;   // "" = keep the default; validated (legacy|host|...) by main
    int volume_percent = -1;    // -1 = untouched
};

inline HostPolicy resolve_host_policy(const HostPolicyInputs& in) {
    HostPolicy out;
    // No --savedata-dir flag exists (the release scripts own that spelling), so this is env > file.
    if (in.env_savedata_dir.empty()) out.savedata_dir = in.file.savedata_dir;
    if (!in.flag_present_mode) out.present_mode = in.file.present_mode;
    if (!in.flag_display_mode && in.env_display_mode.empty())
        out.display_mode = in.file.display_mode;
    if (!in.flag_volume) out.volume_percent = in.file.volume_percent;
    return out;
}

// Record a boot at `app0_root` at the head of the recent list: deduped, canonicalized,
// capped at kRecentGamesMax. Pure over the struct, so the rule is unit-tested.
inline void note_recent_game(AppConfig& cfg, const std::string& app0_root) {
    const std::string root = strip_trailing_separators(app0_root);
    if (root.empty()) return;
    std::vector<std::string> kept;
    kept.reserve(AppConfig::kRecentGamesMax);
    kept.push_back(root);
    for (const std::string& have : cfg.recent_games) {
        if (kept.size() >= AppConfig::kRecentGamesMax) break;
        if (have.empty() || strip_trailing_separators(have) == root) continue;
        kept.push_back(have);
    }
    cfg.recent_games = std::move(kept);
}

// Per-title launch arguments for the guest: a `guest_args.<TITLE_ID>` entry wins over the global
// `guest_args` default; "" when neither is set. The caller decides precedence against the user's
// own environment (see main.cpp — the environment always wins over this file).
inline std::string guest_args_for(const AppConfig& cfg, const std::string& title_id) {
    const auto it = cfg.guest_args_by_title.find(title_id);
    if (it != cfg.guest_args_by_title.end()) return it->second;
    return cfg.guest_args_default;
}

} // namespace prosper::frontend
