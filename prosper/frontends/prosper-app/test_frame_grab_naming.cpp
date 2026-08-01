// test_frame_grab_naming — the F9 capture's file names and the log lines that report them.
//
// Every case here is derived from a way the OLD scheme (a per-process counter, frame_grab_001, in the
// cwd) destroyed or falsified evidence, not from what the new code happens to produce:
//
//   * Two titles played in one directory overwrote each other's captures.
//   * A grab killed between its two writes left a .bmp beside a same-named .prgbundle from an earlier
//     boot 51 minutes before, and the pair was picked up as one frame in two states.
//   * A write that silently succeeded over an existing file destroyed a capture an investigation was
//     actively reading.
//
// The clock is injected and ADVANCES between calls, because the interesting property — every artifact
// of one capture shares one stamp — is invisible to a test whose two writes land in the same second.
// The log assertions parse the path back OUT of the formatted line and stat it against a fresh
// directory listing: asserting a formatted string against the string that formatted it proves nothing
// about whether a file is there.
//
// A real directory is used (under the test's cwd, i.e. the build dir — deliberately not /tmp, which is
// a shared RAM tmpfs on this project's machines). The files are empty or a few bytes.

#include "frame_grab_naming.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <vector>

using prosper::frontend::FrameGrabNamer;
using prosper::frontend::FrameGrabPaths;
using prosper::frontend::format_frame_grab_stamp;
using prosper::frontend::frame_grab_arm_line;
using prosper::frontend::frame_grab_logged_path;
using prosper::frontend::frame_grab_write_line;
using prosper::frontend::reserve_frame_grab;
using prosper::frontend::sanitize_capture_component;
using prosper::frontend::title_id_from_app0_path;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

namespace {

using Clock = std::chrono::system_clock;

Clock::time_point at_epoch_ms(long long ms) {
    return Clock::time_point(std::chrono::milliseconds(ms));
}

void write_file(const std::string& path, const std::string& text) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(text.data(), static_cast<std::streamsize>(text.size()));
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// A FRESH listing of what is actually on disk — never the paths the code under test reported.
std::set<std::string> listing(const std::string& dir) {
    std::set<std::string> names;
    std::error_code ec;
    for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec))
        names.insert(it->path().filename().string());
    return names;
}

std::string stem_of(const std::string& path) {
    return std::filesystem::path(path).stem().string();
}

bool all_digits(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) if (c < '0' || c > '9') return false;
    return true;
}

} // namespace

int main() {
    const std::string dir = "frame_grab_naming_test";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);

    // --- the stamp ------------------------------------------------------------------------------
    // Shape, and the fields a timezone cannot move (seconds and milliseconds). The exact date/hour
    // depends on the host zone by design, so asserting them here would test the test machine.
    {
        const std::string stamp = format_frame_grab_stamp(at_epoch_ms(1'722'517'353'471LL));
        CHECK(stamp.size() == 8 + 1 + 6 + 1 + 3, "the stamp is YYYYMMDD-HHMMSS-mmm");
        CHECK(stamp[8] == '-' && stamp[15] == '-', "the stamp separates date, time and milliseconds");
        CHECK(all_digits(stamp.substr(0, 8)) && all_digits(stamp.substr(9, 6)) &&
              all_digits(stamp.substr(16, 3)), "every stamp field is digits only");
        CHECK(stamp.substr(16, 3) == "471", "the millisecond field is the instant's milliseconds");
        CHECK(stamp.substr(13, 2) == "33", "the seconds field is the instant's seconds");
        // Sub-second resolution is what keeps two presses inside one second apart.
        CHECK(format_frame_grab_stamp(at_epoch_ms(1'722'517'353'471LL)) !=
              format_frame_grab_stamp(at_epoch_ms(1'722'517'353'472LL)),
              "instants one millisecond apart get different stamps");
    }

    // --- the title component --------------------------------------------------------------------
    {
        CHECK(title_id_from_app0_path("/games/PPSA25009-app0") == "PPSA25009",
              "the app0 directory yields the content id");
        CHECK(title_id_from_app0_path("/games/PPSA25009-app0/") == "PPSA25009",
              "a trailing separator does not change it");
        CHECK(title_id_from_app0_path("/games/PPSA25009-APP0") == "PPSA25009",
              "the -app0 suffix is matched case-insensitively");
        CHECK(title_id_from_app0_path("/games/PPSA25009") == "PPSA25009",
              "a directory without the suffix is taken as-is");
        CHECK(title_id_from_app0_path("").empty(), "no path yields no id");
        // Display names are why the FILENAME uses the id. Both of these are real titleName values
        // from dumps in this project: PPSA20052 and PPSA08576.
        CHECK(sanitize_capture_component("Worms Armageddon: Anniversary Edition") ==
                  "Worms_Armageddon__Anniversary_Ed",
              "spaces and colons cannot reach a filename, and a long name is capped");
        CHECK(sanitize_capture_component("Asterix & Obelix Slap Them All!") ==
                  "Asterix___Obelix_Slap_Them_All",
              "ampersands and trailing punctuation cannot reach a filename");
        CHECK(sanitize_capture_component("../../etc/passwd") == "etc_passwd",
              "separators and leading dots cannot reach a filename");
        CHECK(sanitize_capture_component("..") .empty(), "a component that is only punctuation is empty");
        CHECK(sanitize_capture_component(std::string(64, 'x')).size() == 32,
              "a component is capped so one title cannot dominate the name");
    }

    // --- incident 1: two titles in one directory must not collide --------------------------------
    // Worst case on purpose: the same directory, the same instant, and both are each session's FIRST
    // grab — which is exactly the state in which the counter scheme overwrote Blue Prince's captures
    // with Dragon Quest VII's.
    {
        const auto t = at_epoch_ms(1'722'517'353'471LL);
        FrameGrabNamer blue(dir, "PPSA25009", [t] { return t; });
        FrameGrabNamer dq7(dir, "PPSA17942", [t] { return t; });
        const FrameGrabPaths a = blue.reserve();
        const FrameGrabPaths b = dq7.reserve();
        CHECK(a.ok && b.ok, "both captures reserved their names");
        CHECK(a.bundle != b.bundle && a.screenshot != b.screenshot,
              "two titles capturing at the same instant get different names");
        CHECK(a.suffix == 0 && b.suffix == 0, "different titles do not need a collision suffix");
        write_file(a.bundle, "blue-prince-bundle");
        write_file(b.bundle, "dq7-bundle");
        CHECK(read_file(a.bundle) == "blue-prince-bundle",
              "the second title's capture did not overwrite the first title's");
        const std::set<std::string> names = listing(dir);
        CHECK(names.size() == 4, "four files on disk: two captures, two artifacts each");
        std::filesystem::remove_all(dir, ec); std::filesystem::create_directories(dir, ec);
    }

    // --- incident 2: one capture, one stamp, however long the capture takes ----------------------
    // The clock advances 51 minutes on EVERY read — the real gap between the two files that were
    // handed to a lane as a matched pair. An implementation that stamped each artifact at its own
    // write time would produce two different names here; one that reads the clock once cannot.
    {
        int clock_reads = 0;
        auto advancing = [&clock_reads]() mutable {
            return at_epoch_ms(1'722'517'353'471LL + 51LL * 60 * 1000 * clock_reads++);
        };
        FrameGrabNamer namer(dir, "PPSA25009", advancing);
        const FrameGrabPaths grab = namer.reserve();
        CHECK(grab.ok, "the capture reserved its names");
        CHECK(clock_reads == 1, "the clock is read exactly once per capture, at arm time");
        CHECK(stem_of(grab.bundle) == stem_of(grab.screenshot),
              "the bundle and the screenshot share one stem even as the clock moves on");
        // Now write them at genuinely different times, bmp first, bundle much later — the real order
        // and the real gap. The names must be the ones reserved at arm time, not what the clock says
        // at each write.
        write_file(grab.screenshot, "shot");
        (void)advancing();                       // time passes: the capture is still running
        write_file(grab.bundle, "bundle");
        (void)advancing();
        const std::set<std::string> names = listing(dir);
        CHECK(names.count(stem_of(grab.bundle) + ".bmp") == 1 &&
              names.count(stem_of(grab.bundle) + ".prgbundle") == 1,
              "both files landed on the stem reserved at arm time");
        CHECK(names.size() == 2, "a capture whose writes are 51 minutes apart still writes two files");
        // A later capture in the same session is a different capture, and says so.
        const FrameGrabPaths later = namer.reserve();
        CHECK(later.ok && stem_of(later.bundle) != stem_of(grab.bundle),
              "the next capture gets its own stem");
        CHECK(later.index == 2 && grab.index == 1, "captures are numbered for the log line");
        std::filesystem::remove_all(dir, ec); std::filesystem::create_directories(dir, ec);
    }

    // --- an aborted capture must not pair with a different capture -------------------------------
    // Session A grabs, writes its bundle, and dies before the screenshot. Session B — a NEW process,
    // so a counter would restart at 001 — grabs the same title 51 minutes later and writes only its
    // screenshot. Under the old scheme B's .bmp landed on A's stem and the two read as one frame.
    {
        const auto t0 = at_epoch_ms(1'722'517'353'471LL);
        const auto t1 = at_epoch_ms(1'722'517'353'471LL + 51LL * 60 * 1000);
        FrameGrabNamer session_a(dir, "PPSA25009", [t0] { return t0; });
        const FrameGrabPaths a = session_a.reserve();
        write_file(a.bundle, "scene-submit-1519-draws");
        // ... process killed here: a.screenshot is still the empty file the reservation created.

        FrameGrabNamer session_b(dir, "PPSA25009", [t1] { return t1; });   // fresh process
        const FrameGrabPaths b = session_b.reserve();
        write_file(b.screenshot, "composite-3840x2160");

        CHECK(stem_of(a.bundle) != stem_of(b.screenshot),
              "the surviving artifacts of two captures never share a stem");
        CHECK(read_file(a.bundle) == "scene-submit-1519-draws",
              "the aborted capture's bundle is untouched by the later capture");
        CHECK(read_file(a.screenshot).empty(),
              "the aborted capture's screenshot is visibly empty, not somebody else's frame");
        CHECK(read_file(b.bundle).empty(),
              "the later capture's unwritten bundle is empty, not the earlier capture's");
        const std::set<std::string> names = listing(dir);
        CHECK(names.size() == 4, "each capture owns exactly its own two names");
        std::filesystem::remove_all(dir, ec); std::filesystem::create_directories(dir, ec);
    }

    // --- a collision gets a suffix, and nothing is ever overwritten ------------------------------
    {
        const auto t = at_epoch_ms(1'722'517'353'471LL);
        FrameGrabNamer first(dir, "PPSA15552", [t] { return t; });
        FrameGrabNamer second(dir, "PPSA15552", [t] { return t; });   // same title, same instant
        const FrameGrabPaths a = first.reserve();
        write_file(a.bundle, "first-bundle");
        write_file(a.screenshot, "first-shot");
        const FrameGrabPaths b = second.reserve();
        CHECK(b.ok && b.suffix == 2, "a name that is taken yields the next suffix, starting at -2");
        CHECK(stem_of(b.bundle) == stem_of(a.bundle) + "-2", "the suffix reads -2 in the name");
        CHECK(stem_of(b.bundle) == stem_of(b.screenshot),
              "the suffix belongs to the whole capture, not to one file");
        write_file(b.bundle, "second-bundle");
        write_file(b.screenshot, "second-shot");
        CHECK(read_file(a.bundle) == "first-bundle" && read_file(a.screenshot) == "first-shot",
              "both files of the first capture survive the second capture");
        CHECK(listing(dir).size() == 4, "both captures are on disk in full");

        // The log line is the only way anyone finds these files. Parse the path back out of the line
        // and check it against the directory — not against the string that was formatted.
        const std::string line = frame_grab_write_line("bundle", b.bundle, b.suffix, "312 submits");
        const std::string logged = frame_grab_logged_path(line);
        CHECK(!logged.empty(), "a write line carries a path a reader can extract");
        CHECK(listing(dir).count(std::filesystem::path(logged).filename().string()) == 1,
              "the path in the log line exists in the directory listing");
        CHECK(read_file(logged) == "second-bundle",
              "the path in the log line holds the bytes that capture wrote");
        CHECK(line.find("suffix -2") != std::string::npos,
              "a suffixed capture says so, so a reader need not diff filenames");
        // "Everything after the first arrow is the path" must not depend on a caller's discretion:
        // an arrow in the text BEFORE the path would truncate every extracted path on that line.
        CHECK(frame_grab_logged_path(
                  frame_grab_write_line("bundle", b.bundle, 0, "detail -> with an arrow")) == b.bundle,
              "an arrow in the write line's own text cannot displace the path");
        CHECK(frame_grab_write_line("bundle", a.bundle, a.suffix).find("suffix") == std::string::npos,
              "an unsuffixed capture claims no suffix");
        std::filesystem::remove_all(dir, ec); std::filesystem::create_directories(dir, ec);
    }

    // --- an existing file is never truncated, and a half-claim is released -----------------------
    // Only the SCREENSHOT name is taken. The reservation must not keep the bundle name it already
    // created for that stem: the capture owns both names or neither.
    {
        const auto t = at_epoch_ms(1'722'517'353'471LL);
        const std::string stamp = format_frame_grab_stamp(t);
        const std::string taken = dir + "/frame_grab_PPSA13579_" + stamp + ".bmp";
        write_file(taken, "PRE-EXISTING EVIDENCE");
        const FrameGrabPaths grab = reserve_frame_grab(dir, "PPSA13579", t);
        CHECK(grab.ok && grab.suffix == 2, "a taken screenshot name moves the whole capture to -2");
        CHECK(read_file(taken) == "PRE-EXISTING EVIDENCE",
              "the pre-existing file still holds its bytes: nothing was truncated");
        CHECK(!std::filesystem::exists(dir + "/frame_grab_PPSA13579_" + stamp + ".prgbundle"),
              "the released half-claim leaves no orphan bundle behind");
        CHECK(listing(dir).size() == 3, "the pre-existing file plus the new capture's two");
        std::filesystem::remove_all(dir, ec); std::filesystem::create_directories(dir, ec);
    }

    // --- the mirror collision: the BUNDLE name is taken, the screenshot name is free -------------
    // The simpler branch, and the one that must NOT roll anything back: the bundle create fails
    // before the screenshot name is ever touched, so no orphan can be left at the unsuffixed stem.
    {
        const auto t = at_epoch_ms(1'722'517'353'471LL);
        const std::string stamp = format_frame_grab_stamp(t);
        const std::string taken = dir + "/frame_grab_PPSA17942_" + stamp + ".prgbundle";
        write_file(taken, "AN EARLIER CAPTURE");
        const FrameGrabPaths grab = reserve_frame_grab(dir, "PPSA17942", t);
        CHECK(grab.ok && grab.suffix == 2, "a taken bundle name moves the whole capture to -2");
        CHECK(read_file(taken) == "AN EARLIER CAPTURE",
              "the pre-existing bundle still holds its bytes");
        CHECK(!std::filesystem::exists(dir + "/frame_grab_PPSA17942_" + stamp + ".bmp"),
              "the free screenshot name at the taken stem is never created");
        CHECK(listing(dir).size() == 3, "the pre-existing file plus the new capture's two");
        std::filesystem::remove_all(dir, ec); std::filesystem::create_directories(dir, ec);
    }

    // --- the capture directory need not exist yet ------------------------------------------------
    {
        const std::string nested = dir + "/made/by/the/reservation";
        const FrameGrabPaths grab = reserve_frame_grab(nested, "PPSA25009",
                                                       at_epoch_ms(1'722'517'353'471LL));
        CHECK(grab.ok, "PROSPER_CAPTURE_DIR is created when it is absent");
        CHECK(listing(nested).size() == 2, "both files land in the created directory");
        std::filesystem::remove_all(dir, ec); std::filesystem::create_directories(dir, ec);
    }

    // --- a capture with no title still gets a well-formed name ----------------------------------
    {
        const FrameGrabPaths grab = reserve_frame_grab(dir, "", at_epoch_ms(1'722'517'353'471LL));
        CHECK(grab.ok, "a run with no dump can still grab a frame");
        CHECK(stem_of(grab.bundle).rfind("frame_grab_notitle_", 0) == 0,
              "an absent title is named, not omitted, so the name keeps its shape");
        std::filesystem::remove_all(dir, ec); std::filesystem::create_directories(dir, ec);
    }

    // --- the arming line asserts only what arming knows ------------------------------------------
    {
        const std::string arm = frame_grab_arm_line(2, "PPSA25009 (Blue Prince)");
        CHECK(arm.find("PPSA25009") != std::string::npos && arm.find("Blue Prince") != std::string::npos,
              "the arming line says which title is being captured");
        CHECK(arm.find("#2") != std::string::npos, "the arming line numbers the press");
        // At arm time no file is written, the capture may abort, and a collision may change the name.
        // A path here would be an intention written in the grammar of a result.
        CHECK(arm.find(".prgbundle") == std::string::npos && arm.find(".bmp") == std::string::npos &&
              arm.find(" -> ") == std::string::npos,
              "the arming line names no file and carries no path");
        CHECK(frame_grab_logged_path(arm).empty(),
              "a reader extracting paths from the log finds none in an arming line");
        // The label carries the title's DISPLAY name, taken from the dump's param.json without
        // sanitisation. A name containing the artifact arrow would otherwise make the arming line
        // parse as a write line and hand the reader a path that never existed.
        const std::string hostile = frame_grab_arm_line(1, "PPSA00000 (Title -> Subtitle)");
        CHECK(frame_grab_logged_path(hostile).empty(),
              "an arrow inside the title name cannot turn an arming line into an artifact report");
        CHECK(hostile.find("Subtitle") != std::string::npos,
              "...and the name is still shown, not dropped");
        CHECK(frame_grab_logged_path("[grab] bundle written -> /out/x.prgbundle\n") == "/out/x.prgbundle",
              "a write line's path is everything after the arrow, newline excluded");
        // PROSPER_CAPTURE_DIR is an arbitrary directory, so the PATH can contain an arrow while the
        // text before it never does. Reading the last arrow instead of the first would hand back a
        // truncated path that does not exist.
        CHECK(frame_grab_logged_path(
                  frame_grab_write_line("bundle", "/out/a -> b/x.prgbundle")) == "/out/a -> b/x.prgbundle",
              "a path that itself contains an arrow survives the round trip");
    }

    std::filesystem::remove_all(dir, ec);
    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
