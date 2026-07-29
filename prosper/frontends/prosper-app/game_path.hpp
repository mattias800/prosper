#pragma once
// game_path.hpp — turn a path the user handed the app (dropped on the window, or chosen in the host
// folder picker) into the app0 root boot_program() wants, and decide what the app can do with it.
//
// Filesystem access is injected and nothing here touches SDL, so the whole decision surface is
// unit-tested (test_game_path.cpp) without a window, a picker, or a real game dump — the same pure
// regression seam as present_mode.hpp / present_policy.hpp / window_controls.hpp. Argv keeps its
// own path into boot_program(); these helpers only serve the interactive sources (#1469).

#include <functional>
#include <string>
#include <vector>

namespace prosper::frontend {

// Injected filesystem queries. `is_file` answers for regular files, `is_dir` for directories.
struct GamePathProbe {
    std::function<bool(const std::string&)> is_file;
    std::function<bool(const std::string&)> is_dir;
};

// What the app can do with a path the user opened.
enum class GameOpenAction {
    ignore,           // not a title, or nothing was chosen — report it, change nothing
    boot_in_process,  // no guest started yet: boot it here, on this process
    relaunch,         // a guest is already running: only a fresh process can host another title
};

// Drop the trailing separators a folder picker or file manager may append. A filesystem root
// ("/", "C:\") keeps its separator — there it is the whole path, not a suffix.
inline std::string strip_trailing_separators(std::string path) {
    while (path.size() > 1 && (path.back() == '/' || path.back() == '\\')) {
        if (path.size() == 3 && path[1] == ':') break;   // "C:\" is a root, not "C:" plus a separator
        path.pop_back();
    }
    return path;
}

// The directory containing `path`, or "" when it has none (a bare name, or a root).
inline std::string parent_directory(const std::string& path) {
    const std::string p = strip_trailing_separators(path);
    if (p.size() <= 1) return "";                                  // "" or a POSIX root
    if (p.size() == 3 && p[1] == ':') return "";                   // a Windows drive root
    const size_t slash = p.find_last_of("/\\");
    if (slash == std::string::npos) return "";
    if (slash == 0) return p.substr(0, 1);                         // "/game" -> "/"
    if (slash == 2 && p[1] == ':') return p.substr(0, 3);          // "C:\game" -> "C:\"
    return p.substr(0, slash);
}

// A directory is a title root when it holds the guest executable boot_program() loads, or the
// metadata that names the title. Either alone is enough: this is a gate that keeps an unrelated
// folder from reaching the boot path, not a substitute for it. boot_program() still performs the
// authoritative (case-correcting, #1006) resolution of every module it needs.
inline bool is_app0_root(const std::string& dir, const GamePathProbe& probe) {
    if (dir.empty() || !probe.is_dir || !probe.is_file) return false;
    if (!probe.is_dir(dir)) return false;
    return probe.is_file(dir + "/eboot.bin") || probe.is_file(dir + "/sce_sys/param.json");
}

// Resolve whatever the user handed us to an app0 root, or "" when the path is not part of a title.
// A drop carries whichever entry the file manager happened to be showing, so accept the root
// itself, a file in it (eboot.bin), a directory in it (sce_sys), or a file one level deeper
// (sce_sys/param.json).
//
// The walk examines a starting directory and its two ancestors, and no further; a picked *file*
// starts at the directory holding it, so it reaches one level deeper than a picked directory does.
// The bound is the point: searching all the way up would let a drop from deep inside a title's
// assets silently boot whatever unrelated title happens to sit above it.
inline std::string resolve_app0_root(const std::string& picked, const GamePathProbe& probe) {
    const std::string p = strip_trailing_separators(picked);
    if (p.empty() || !probe.is_dir || !probe.is_file) return "";
    // Start at a directory: a picked file stands for the directory holding it.
    std::string dir = probe.is_dir(p) ? p : (probe.is_file(p) ? parent_directory(p) : std::string());
    for (int up = 0; up < 3 && !dir.empty(); ++up) {
        if (is_app0_root(dir, probe)) return dir;
        dir = parent_directory(dir);
    }
    return "";
}

// A guest cannot be torn down in-process: run_entry() never observes prosper_request_stop(), so the
// app exits rather than joining it (#352). Until that lands, a second title needs a second process.
//
// It is the boot ATTEMPT that is spent, not the success. boot_program() appends into the Program it
// is given and re-runs one-shot global setup (HLE registration, export table, trap handler, host
// backends), so calling it twice would link the second title behind the first attempt's leftover
// modules and leave imgs[0] — the image the guest thread actually enters — pointing at the failed
// title. Anything after the first attempt therefore goes to a fresh process.
inline GameOpenAction decide_open_action(const std::string& app0_root, bool boot_attempted) {
    if (app0_root.empty()) return GameOpenAction::ignore;
    return boot_attempted ? GameOpenAction::relaunch : GameOpenAction::boot_in_process;
}

// Quote one argument for a Windows command line, following the rule CommandLineToArgvW and the CRT
// parse: a backslash is literal except in the run immediately preceding a double quote, where each
// one must be doubled. Getting this wrong corrupts any argument that ENDS in a backslash, because
// the closing quote is then read as escaped and the argument swallows the tokens after it — and a
// tab-completed directory, or a title sitting at a drive root ("D:\"), ends in exactly that.
inline std::string quote_windows_argument(const std::string& arg) {
    std::string out = "\"";
    size_t backslashes = 0;
    for (const char c : arg) {
        if (c == '\\') { ++backslashes; continue; }
        if (c == '"') {
            out.append(2 * backslashes + 1, '\\');   // double the run, then escape the quote itself
            backslashes = 0;
        } else {
            out.append(backslashes, '\\');           // not before a quote: the run stays literal
            backslashes = 0;
        }
        out += c;
    }
    out.append(2 * backslashes, '\\');               // the run before the closing quote is doubled
    out += '"';
    return out;
}

// Join arguments into the single command line CreateProcess takes.
inline std::string windows_command_line(const std::vector<std::string>& args) {
    std::string cmdline;
    for (size_t i = 0; i < args.size(); i++) {
        if (i) cmdline += ' ';
        cmdline += quote_windows_argument(args[i]);
    }
    return cmdline;
}

struct StartupPickInputs {
    bool has_dump = false;      // --dump / a positional path was given
    bool test_pattern = false;  // --test-pattern
    bool forced = false;        // --pick
    bool suppressed = false;    // --no-pick
    bool bare_launch = false;   // the process was started with no arguments at all
};

// Whether to open the host folder picker at startup. A run that was already told what to show is
// never interrupted by a dialog, and --no-pick wins over --pick so a script can hard-disable it.
// The only implicit case is a launch with no arguments at all — a double-click, where argv cannot
// carry a game and an empty window would be a dead end.
inline bool should_pick_at_startup(const StartupPickInputs& in) {
    if (in.has_dump || in.test_pattern) return false;
    if (in.suppressed) return false;
    if (in.forced) return true;
    return in.bare_launch;
}

} // namespace prosper::frontend
