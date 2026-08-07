#pragma once
// frame_grab_naming.hpp — names the files ONE F9 frame grab writes, and claims those names before
// the capture starts producing them.
//
// A grab writes two files at two different moments: the .bmp screenshot is armed at one present and
// written at a later one, while the .prgbundle is written on the render thread when the capture
// window closes. Two incidents came out of naming them independently of each other and of the run
// (#1693):
//
//   * The old name was a per-process counter (frame_grab_001) in the cwd, so a second title played in
//     the same directory silently OVERWROTE the first title's captures — evidence destroyed in the
//     middle of an investigation that was reading them.
//   * A grab killed between the two writes left its .bmp beside a same-named .prgbundle from an
//     earlier boot 51 minutes before, and the two were picked up as "one frame in two states". A
//     1920x1080/1519-draw scene submit compared against a 3840x2160/97-draw composite would have
//     manufactured a large, entirely artefactual difference.
//
// So the contract here is not "add a timestamp" — it is that every artifact of one capture is bound
// to that capture BY NAME, and that no capture can ever write over another's file:
//
//   1. One stamp per capture, read ONCE when the grab is armed (FrameGrabNamer::reserve). A capture
//      that takes seconds cannot produce two different stamps, because there is only ever one.
//   2. The title id is in the name, so two titles in one directory cannot collide at all.
//   3. Both names are claimed together with an EXCLUSIVE create (O_CREAT|O_EXCL). Never overwriting
//      is a property of the syscall, not of a check-then-write — the app does run twice on one box.
//      A collision suffix applies to the WHOLE capture: choosing it per file would push a bundle to
//      "-2" while its screenshot stayed unsuffixed, which is incident 2 rebuilt out of the mechanism
//      meant to prevent it.
//
//   frame_grab_<TITLEID>_<YYYYMMDD>-<HHMMSS>-<mmm>[-<N>].{prgbundle,bmp}
//
// A capture that aborts therefore leaves a zero-byte artifact rather than nothing. That is the
// intended behaviour: it is visibly incomplete and it unambiguously belongs to its own capture,
// where the old scheme left a file that looked like a matched pair with someone else's.
//
// With one qualification, because a reader should not have to discover it: this holds when the app is
// NOT around to do better. When it is — a grab superseded by another press, or a bundle that failed —
// the frontend removes the reservation it owns and says so in the log, which is strictly better than
// a placeholder nobody explains. So a missing sibling is not necessarily an incomplete capture; the
// log says which happened.
//
// No SDL, no Vulkan, no globals: the whole scheme is unit-tested against a real temporary directory
// and an injected clock (test_frame_grab_naming.cpp).

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <system_error>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace prosper::frontend {

// Everything a capture's files are named after, reduced to what is safe in a filename on every host:
// [A-Za-z0-9._-], with anything else folded to '_'. Leading/trailing punctuation is trimmed so a name
// can never start with '.' (hidden) or be "." / ".." (not a name at all). "" when nothing survives —
// callers substitute their own placeholder, because an empty component would silently shorten the
// name into a different shape.
inline std::string sanitize_capture_component(std::string_view raw, size_t max_len = 32) {
    std::string out;
    out.reserve(std::min(raw.size(), max_len));
    for (char c : raw) {
        if (out.size() >= max_len) break;
        const unsigned char u = static_cast<unsigned char>(c);
        const bool keep = (u >= '0' && u <= '9') || (u >= 'A' && u <= 'Z') || (u >= 'a' && u <= 'z') ||
                          c == '.' || c == '_' || c == '-';
        out += keep ? c : '_';
    }
    const auto trim = [](const std::string& s) {
        size_t b = 0, e = s.size();
        while (b < e && (s[b] == '.' || s[b] == '_' || s[b] == '-')) ++b;
        while (e > b && (s[e - 1] == '.' || s[e - 1] == '_' || s[e - 1] == '-')) --e;
        return s.substr(b, e - b);
    };
    return trim(out);
}

// The content id from a dump's directory name: "<...>/PPSA25009-app0" -> "PPSA25009". This is the
// fallback for a dump whose param.json is missing or unreadable (parse_param_title_id in
// game_library.hpp is the primary source); "" when the path yields nothing usable.
inline std::string title_id_from_app0_path(std::string_view app0_root) {
    while (!app0_root.empty() && (app0_root.back() == '/' || app0_root.back() == '\\'))
        app0_root.remove_suffix(1);
    const size_t slash = app0_root.find_last_of("/\\");
    std::string name(slash == std::string_view::npos ? app0_root : app0_root.substr(slash + 1));
    const std::string suffix = "-app0";
    if (name.size() > suffix.size()) {
        const std::string tail = name.substr(name.size() - suffix.size());
        std::string folded;
        for (char c : tail) folded += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (folded == suffix) name.erase(name.size() - suffix.size());
    }
    return sanitize_capture_component(name);
}

// Local wall clock at millisecond resolution: YYYYMMDD-HHMMSS-mmm.
//
// Local rather than UTC because the reader correlates this against "when I pressed F9" and against
// the other local-time artifacts of the same session. Milliseconds because two presses inside one
// second is an ordinary thing for a person to do — a second-resolution stamp would collide there and
// send the capture down the suffix path on an ordinary double-press rather than on a real clash.
//
// Two consequences of local time, considered and accepted rather than overlooked: names from two
// machines in different timezones are not directly comparable, and a DST fall-back produces one hour
// in which lexical order and true chronology disagree. Neither affects what the stamp is FOR (binding
// one capture's files together, and separating it from every other capture), and uniqueness does not
// depend on the clock being monotonic — the exclusive create below is what guarantees that.
inline std::string format_frame_grab_stamp(std::chrono::system_clock::time_point t) {
    using namespace std::chrono;
    const auto secs = floor<seconds>(t);           // floor, so the millisecond part is never negative
    const auto ms = duration_cast<milliseconds>(t - secs).count();
    const std::time_t tt = system_clock::to_time_t(secs);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    char buf[40];
    std::snprintf(buf, sizeof buf, "%04d%02d%02d-%02d%02d%02d-%03d",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec, static_cast<int>(ms));
    return buf;
}

// The names one capture owns. `ok` means both were created exclusively and belong to this capture
// alone; every path in here is real on disk from that moment on (zero bytes until written).
struct FrameGrabPaths {
    bool ok = false;
    std::string stem;         // "frame_grab_PPSA25009_20260801-142233-471" (no directory, no extension)
    std::string bundle;       // <dir>/<stem>.prgbundle
    std::string screenshot;   // <dir>/<stem>.bmp
    unsigned suffix = 0;      // 0 = the preferred name was free; 2.. = a collision forced this suffix
    unsigned index = 0;       // this session's Nth grab — for the log line, never for the filename
    std::string error;        // why !ok
    // Non-fatal, but the caller must SAY it: a rollback that could not remove its own half-claim
    // strands a zero-byte .prgbundle with no matching .bmp — exactly the unexplained orphan this
    // file exists to prevent, so it cannot be left to be discovered in a directory listing.
    std::string warning;
};

// Create `path` only if it does not exist. Returns false with errno set; errno == EEXIST means the
// name is taken (try the next suffix), anything else is a real filesystem failure and must not be
// retried 999 times.
inline bool create_file_exclusive(const std::string& path) {
#if defined(_WIN32)
    const int fd = ::_open(path.c_str(), _O_CREAT | _O_EXCL | _O_WRONLY | _O_BINARY,
                           _S_IREAD | _S_IWRITE);
    if (fd < 0) return false;
    ::_close(fd);
#else
    const int fd = ::open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
    if (fd < 0) return false;
    ::close(fd);
#endif
    return true;
}

// Claim both names for one capture armed at `armed_at`. The stamp is a parameter, not a call to
// now(), precisely so a caller cannot accidentally take one per file.
//
// Both files are created, or neither is: a bundle name that was free but whose screenshot name was
// taken is released again and the suffix advances, so the two artifacts always share one stem.
inline FrameGrabPaths reserve_frame_grab(const std::string& dir, const std::string& title_id,
                                         std::chrono::system_clock::time_point armed_at,
                                         unsigned max_suffix = 999) {
    FrameGrabPaths out;
    std::string base_dir = dir.empty() ? std::string(".") : dir;
    while (base_dir.size() > 1 && (base_dir.back() == '/' || base_dir.back() == '\\'))
        base_dir.pop_back();
    std::error_code ec;
    // Best effort: if the directory cannot be made, the exclusive create below reports the real
    // errno, which is a better message than anything derived from ec here.
    std::filesystem::create_directories(base_dir, ec);

    std::string title = sanitize_capture_component(title_id);
    if (title.empty()) title = "notitle";
    const std::string stem_base = "frame_grab_" + title + "_" + format_frame_grab_stamp(armed_at);

    for (unsigned n = 0; n <= max_suffix; ++n) {
        std::string stem = stem_base;
        if (n) stem += "-" + std::to_string(n + 1);      // first collision reads "-2", not "-1"
        const std::string bundle = base_dir + "/" + stem + ".prgbundle";
        const std::string shot = base_dir + "/" + stem + ".bmp";
        errno = 0;
        if (!create_file_exclusive(bundle)) {
            if (errno == EEXIST) continue;
            out.error = "cannot create " + bundle + ": " + std::strerror(errno);
            return out;
        }
        errno = 0;
        if (!create_file_exclusive(shot)) {
            const int err = errno;                    // captured BEFORE remove(), which clobbers errno
            // Release the half-claim. This capture owns both names or neither; leaving the bundle
            // behind would strand an empty file that no capture is going to write.
            std::filesystem::remove(bundle, ec);
            if (ec)
                out.warning = "could not remove the released reservation " + bundle + ": " +
                              ec.message() + " (an empty .prgbundle with no .bmp is stranded there)";
            if (err == EEXIST) continue;
            out.error = "cannot create " + shot + ": " + std::strerror(err);
            return out;
        }
        out.ok = true;
        out.stem = std::move(stem);
        out.bundle = bundle;
        out.screenshot = shot;
        out.suffix = n ? n + 1 : 0;
        return out;
    }
    out.error = "every name from " + stem_base + " through suffix -" + std::to_string(max_suffix + 1) +
                " is already taken";
    return out;
}

// One capture's names, with the clock owned here so it is read exactly once per capture.
//
// The clock is injectable so the "all files of one capture share the stamp" property can be tested
// against a clock that ADVANCES between calls — with a fixed clock that test passes vacuously,
// including against the implementation that caused incident 2.
class FrameGrabNamer {
public:
    using Clock = std::function<std::chrono::system_clock::time_point()>;

    FrameGrabNamer() = default;
    FrameGrabNamer(std::string dir, std::string title_id, Clock clock = {})
        : dir_(std::move(dir)), title_id_(std::move(title_id)), clock_(std::move(clock)) {}

    void set_directory(std::string dir) { dir_ = std::move(dir); }
    // `label` is what the arming log line says (id plus the human name when known); the FILENAME only
    // ever uses `title_id`.
    void set_title(std::string title_id, std::string label = std::string()) {
        title_id_ = std::move(title_id);
        label_ = label.empty() ? title_id_ : std::move(label);
    }
    const std::string& title_id() const { return title_id_; }
    const std::string& title_label() const { return label_.empty() ? title_id_ : label_; }
    unsigned count() const { return count_; }

    // Claim the names for one capture. Reads the clock ONCE, here, for every artifact of this grab.
    FrameGrabPaths reserve() {
        const auto now = clock_ ? clock_() : std::chrono::system_clock::now();
        FrameGrabPaths paths = reserve_frame_grab(dir_, title_id_, now);
        paths.index = ++count_;   // counts presses, so a failed reservation still advances it
        return paths;
    }

private:
    std::string dir_ = ".";
    std::string title_id_;
    std::string label_;
    Clock clock_;
    unsigned count_ = 0;
};

// ---------------------------------------------------------------------------------------------
// Log lines.
//
// A log line asserts only what is true at the moment it is emitted. The arming line therefore names
// no file: at arm time the final path is not a fact — the capture may abort, the write may fail, and
// a name announced in the grammar of a result is one a reader (human or agent) cannot distinguish
// from one. Every artifact instead gets its own line AFTER it exists, carrying its own real path.
//
// Machine-readable contract for whoever is reading these logs to find artifacts: a write line ends
// with " -> <path>" and nothing follows the path. frame_grab_logged_path() is that read.
// ---------------------------------------------------------------------------------------------

// "[grab] F9 #2: arming a whole-frame capture for PPSA25009 (Blue Prince)"
//
// The label carries the title's DISPLAY name, copied out of the dump's param.json without
// sanitisation, so it can contain anything — including " -> ". An arm line that parses as a write
// line would hand a log reader a path that never existed, which is this file's whole subject, so the
// arrow is folded out of the label rather than trusted not to appear.
// `trigger` names what armed this capture. It defaults to the hotkey so existing callers and their
// tests are unaffected, but a SCHEDULED capture must not say "F9": there is no operator on that path,
// and a log line that names a keypress nobody made sends its reader looking for one. (#2233)
inline std::string frame_grab_arm_line(unsigned index, const std::string& title_label,
                                       std::string_view trigger = "F9") {
    std::string label = title_label.empty() ? std::string("an unidentified title") : title_label;
    for (size_t k = label.find(" -> "); k != std::string::npos; k = label.find(" -> ", k))
        label.replace(k, 4, " - ");
    return "[grab] " + std::string(trigger) + " #" + std::to_string(index) +
           ": arming a whole-frame capture for " + label;
}

// "[grab] bundle written (312 submits) [name collision: suffix -2] -> ./frame_grab_....prgbundle"
// `detail` is optional and parenthesised; `suffix` is 0 for the ordinary unsuffixed case, so a reader
// can tell a suffixed capture from an unsuffixed one without diffing filenames.
inline std::string frame_grab_write_line(std::string_view what, const std::string& path,
                                         unsigned suffix = 0, std::string_view detail = {}) {
    std::string line = "[grab] ";
    line += what;
    line += " written";
    if (!detail.empty()) { line += " ("; line += detail; line += ")"; }
    if (suffix) line += " [name collision: suffix -" + std::to_string(suffix) + "]";
    // "Everything after the first arrow is the path" only holds while nothing BEFORE the arrow
    // contains one. Both callers are safe by construction today; enforce it structurally anyway, so
    // the invariant does not depend on a future caller having read this comment.
    for (size_t k = line.find(" -> "); k != std::string::npos; k = line.find(" -> ", k))
        line.replace(k, 4, " - ");
    line += " -> " + path;
    return line;
}

// The path a write line points at, or "" when the line carries none. Everything after the FIRST
// " -> " is the path — the same mechanical read an agent scanning a run log performs, which is why
// tests parse it back out of the formatted line and stat the result instead of trusting the string
// they formatted.
//
// First rather than last, deliberately: the text before the arrow is ours and never contains one,
// while the path is the user's (PROSPER_CAPTURE_DIR is an arbitrary directory) and could. Taking the
// last arrow would silently truncate such a path into one that does not exist.
inline std::string frame_grab_logged_path(std::string_view line) {
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.remove_suffix(1);
    const size_t arrow = line.find(" -> ");
    if (arrow == std::string_view::npos) return "";
    return std::string(line.substr(arrow + 4));
}

} // namespace prosper::frontend
