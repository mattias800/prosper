// An ARMED guest-log capture marker that never matches must say so (#1684). Pure/offline: no guest
// and no Vulkan device.
//
// The failure this guards is not a wrong bundle, it is silence. A marker that is a PREFIX of the
// real guest line — `LevelDocument Loaded: worldmap` against the printed
// `LevelDocument Loaded: worldmap [worldmap]` — never matches, and before this change nothing at all
// was reported: the run completed, exited 0, wrote no bundle, and the only evidence was a missing
// file. A ~7-minute routed Astro Bot capture was lost that way.
//
// This test lives apart from test_guest_log_capture because the gate is a process-wide one-shot:
// that test's marker MATCHES, and a fired gate has nothing to report by construction.
#include "../src/gpu/gpu_timeline.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include "test_scratch.h"
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); fails++; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

static void set_test_env(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

static void clear_test_env(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

static int stream_fd(FILE* stream) {
#ifdef _WIN32
    return _fileno(stream);
#else
    return fileno(stream);
#endif
}

// Runs `body` with stderr redirected to a scratch file and returns everything it wrote. The exit
// report's whole job is to put text on stderr, so asserting on a returned string alone would leave
// the delivery untested — an empty report and an unreachable report look identical from the caller.
static std::string capture_stderr(const std::filesystem::path& path, void (*body)()) {
#ifdef _WIN32
    const int saved = _dup(stream_fd(stderr));
#else
    const int saved = dup(stream_fd(stderr));
#endif
    FILE* sink = std::fopen(path.string().c_str(), "w");
    if (!sink || saved < 0) { if (sink) std::fclose(sink); return "<redirect failed>"; }
    std::fflush(stderr);
#ifdef _WIN32
    _dup2(stream_fd(sink), stream_fd(stderr));
#else
    dup2(stream_fd(sink), stream_fd(stderr));
#endif
    body();
    std::fflush(stderr);
#ifdef _WIN32
    _dup2(saved, stream_fd(stderr));
    _close(saved);
#else
    dup2(saved, stream_fd(stderr));
    close(saved);
#endif
    std::fclose(sink);

    std::string text;
    if (FILE* in = std::fopen(path.string().c_str(), "rb")) {
        char buf[4096];
        size_t got = 0;
        while ((got = std::fread(buf, 1, sizeof(buf), in)) > 0) text.append(buf, got);
        std::fclose(in);
    }
    return text;
}

static bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

int main(int argc, char** argv) {
    // Child mode. It arms the gate, feeds a line the marker is only a PREFIX of, and returns from
    // main. Nothing here calls the reporter, so any text on this process's stderr can only come from
    // the std::atexit registration — which is the half of the fix that an in-process assertion is
    // structurally unable to see. Removing that one registration leaves every direct-call check
    // below passing while a real run goes silent again, which is precisely the original defect.
    if (argc > 2 && std::string(argv[1]) == "--unfired-child") {
        set_test_env("PROSPER_CAPTURE_BUNDLE_AFTER_GUEST_LOG", "LevelDocument Loaded: worldmap");
        set_test_env("PROSPER_CAPTURE_BUNDLE", argv[2]);
        const char guest[] = "LevelDocument Loaded: worldmap [worldmap]\n";
        observe_guest_log_for_capture(guest, sizeof(guest) - 1);
        return 0;
    }

    std::printf("== test_guest_log_capture_miss ==\n");
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto scratch = prosper_test::test_scratch_dir();
    const auto bundle_path = scratch / ("prosper-guest-log-miss-" + std::to_string(nonce) + ".prgbundle");
    const auto stderr_path = scratch / ("prosper-guest-log-miss-" + std::to_string(nonce) + ".stderr");

    // --- the pure formatter: the diagnosis, without needing a title ---------------------------
    {
        GuestLogCaptureMiss miss;
        miss.marker = "LevelDocument Loaded: worldmap";
        miss.lines_observed = 812;
        miss.closest_line = "LevelDocument Loaded: worldmap [worldmap]";
        miss.closest_prefix_bytes = miss.marker.size();
        const std::string text = format_guest_log_capture_miss(miss);
        CHECK(contains(text, "never matched") && contains(text, "NO BUNDLE WAS WRITTEN"),
              "a never-matched marker is named as such, and the missing bundle is stated");
        CHECK(contains(text, "\"LevelDocument Loaded: worldmap\""),
              "the report quotes the marker that was demanded");
        CHECK(contains(text, "LevelDocument Loaded: worldmap [worldmap]"),
              "the report quotes the closest observed line verbatim, so it can be copied");
        CHECK(contains(text, "PREFIX"),
              "a marker that is a prefix of the real line is diagnosed as exactly that");
        CHECK(contains(text, "812"), "the report states how many lines were compared");
    }
    {
        // Zero observed lines is a DIFFERENT fault from a wrong marker, and conflating them is what
        // sent the original investigation looking at the marker text.
        GuestLogCaptureMiss miss;
        miss.marker = "phase-ready";
        const std::string text = format_guest_log_capture_miss(miss);
        CHECK(contains(text, "no completed line reached the observer"),
              "an unreached guest-log path is distinguished from a wrong marker");
        CHECK(!contains(text, "closest observed line"),
              "with nothing observed there is no closest line to claim");
    }
    {
        GuestLogCaptureMiss miss;
        miss.marker = std::string(kGuestLogCaptureMaxLineBytes + 1, 'x');
        miss.marker_too_long = true;
        CHECK(contains(format_guest_log_capture_miss(miss), "never been enabled") ||
              contains(format_guest_log_capture_miss(miss), "never enabled"),
              "an over-long marker reports that the gate was never enabled");
    }

    // --- the live gate: a prefix marker against the real observer ------------------------------
    clear_test_env("PROSPER_CAPTURE_BUNDLE_AT_PRESENT");
    clear_test_env("PROSPER_CAPTURE_BUNDLE_TRIGGER_FILE");
    clear_test_env("PROSPER_GPU_TIMELINE");
    set_test_env("PROSPER_CAPTURE_BUNDLE_AFTER_GUEST_LOG", "phase-ready");
    set_test_env("PROSPER_CAPTURE_BUNDLE", bundle_path.string());
    CHECK(guest_log_capture_bundle_enabled(), "the gate arms from a marker plus a bundle path");

    // A trailing tab is invisible when printed raw, and it is exactly the kind of byte that breaks
    // an exact match while the two strings look identical in a terminal.
    observe_guest_log_for_capture("phase-ready\t\n", 13);
    observe_guest_log_for_capture("phase-ready-suffix\n", 19);
    observe_guest_log_for_capture("unrelated\n", 10);
    CHECK(!interactive_capture_bundle_active(),
          "none of the near-miss lines arms a capture: the match stays EXACT");

    GuestLogCaptureMiss live;
    CHECK(guest_log_capture_miss_snapshot(live),
          "an armed marker that never matched is reportable");
    CHECK(live.marker == "phase-ready", "the snapshot carries the configured marker");
    CHECK(live.lines_observed == 3 && live.lines_discarded == 0,
          "every completed line is counted, so 'marker wrong' and 'never reached' stay separable");
    CHECK(live.closest_line == "phase-ready\t" && live.closest_prefix_bytes == 11,
          "the closest observed line is the first sharing the longest prefix with the marker");

    const std::string live_text = format_guest_log_capture_miss(live);
    CHECK(contains(live_text, "\\x09"),
          "an invisible trailing byte in the closest line is escaped, not reproduced invisibly");
    CHECK(contains(live_text, "PREFIX"),
          "the live near-miss is diagnosed as a prefix marker");

    // --- delivery: the report actually reaches stderr, and only when a gate is armed ------------
    clear_test_env("PROSPER_CAPTURE_BUNDLE_AFTER_GUEST_LOG");
    const std::string quiet = capture_stderr(stderr_path, &report_unfired_automatic_capture_gates);
    CHECK(quiet.empty(), "with no gate configured the exit report says nothing at all");

    set_test_env("PROSPER_CAPTURE_BUNDLE_AFTER_GUEST_LOG", "phase-ready");
    const std::string loud = capture_stderr(stderr_path, &report_unfired_automatic_capture_gates);
    CHECK(contains(loud, "never matched") && contains(loud, "phase-ready"),
          "with the gate armed and unmatched the exit report names the marker on stderr");
    CHECK(contains(loud, "phase-ready\\x09"),
          "the delivered report carries the escaped closest line");

    // A gate configured without an output path produced no message anywhere before #1684.
    clear_test_env("PROSPER_CAPTURE_BUNDLE");
    const std::string no_path = capture_stderr(stderr_path, &report_unfired_automatic_capture_gates);
    CHECK(contains(no_path, "without PROSPER_CAPTURE_BUNDLE"),
          "a gate configured without an output path is reported rather than ignored");

    // Mutually exclusive gates: rejected at startup, and the rejection is restated at exit because
    // its only other symptom is the same missing file.
    set_test_env("PROSPER_CAPTURE_BUNDLE", bundle_path.string());
    set_test_env("PROSPER_CAPTURE_BUNDLE_AT_PRESENT", "500");
    const std::string conflict = capture_stderr(stderr_path, &report_unfired_automatic_capture_gates);
    CHECK(contains(conflict, "mutually exclusive"),
          "a conflicting gate configuration is restated at exit");

    // A present-ordinal gate that the run never reached names what the run DID reach.
    clear_test_env("PROSPER_CAPTURE_BUNDLE_AFTER_GUEST_LOG");
    record_gpu_timeline_present(7, 0, 0, 4, 4);
    record_gpu_timeline_present(8, 0, 0, 4, 4);
    const std::string at_present = capture_stderr(stderr_path, &report_unfired_automatic_capture_gates);
    CHECK(contains(at_present, "PROSPER_CAPTURE_BUNDLE_AT_PRESENT=500") &&
          contains(at_present, "never fired"),
          "an unreached present ordinal is reported by name");
    CHECK(contains(at_present, "2 present(s)") && contains(at_present, "last present count was 8"),
          "the report states what the run reached, not only what was asked for");

    set_test_env("PROSPER_CAPTURE_BUNDLE_AT_PRESENT", "not-a-number");
    const std::string malformed = capture_stderr(stderr_path, &report_unfired_automatic_capture_gates);
    CHECK(contains(malformed, "is not a number"),
          "a malformed present ordinal is reported instead of silently disabling the gate");

    // Leave nothing armed, so this test's own process exit stays quiet and its ctest log cannot be
    // misread as a failure. Done before the child spawn so the child inherits a clean environment.
    clear_test_env("PROSPER_CAPTURE_BUNDLE_AT_PRESENT");
    clear_test_env("PROSPER_CAPTURE_BUNDLE_AFTER_GUEST_LOG");
    clear_test_env("PROSPER_CAPTURE_BUNDLE_TRIGGER_FILE");
    clear_test_env("PROSPER_CAPTURE_BUNDLE");

    // --- the wiring: a real process that merely RETURNS FROM MAIN must still explain itself -----
    // The defect was silence at exit, so the guard has to cross a process boundary. Every check
    // above calls the reporter directly and therefore cannot distinguish a registered exit handler
    // from an unregistered one.
    {
        std::string self = argv[0];
        if (self.find('/') == std::string::npos && self.find('\\') == std::string::npos)
            self = "./" + self;
        const auto child_err = scratch / ("prosper-guest-log-miss-child-" + std::to_string(nonce) + ".stderr");
        const std::string command = "\"" + self + "\" --unfired-child \"" + bundle_path.string() +
                                    "\" 2> \"" + child_err.string() + "\"";
        const int rc = std::system(command.c_str());
        std::string child_text;
        if (FILE* in = std::fopen(child_err.string().c_str(), "rb")) {
            char buf[4096];
            size_t got = 0;
            while ((got = std::fread(buf, 1, sizeof(buf), in)) > 0) child_text.append(buf, got);
            std::fclose(in);
        }
        CHECK(rc == 0, "the child process exits cleanly, exactly as the lost Astro Bot run did");
        CHECK(contains(child_text, "never matched") && contains(child_text, "PREFIX"),
              "a process that only returns from main still reports its unfired marker at exit");
        CHECK(contains(child_text, "LevelDocument Loaded: worldmap [worldmap]"),
              "the exit report names the exact line the guest actually printed");
        std::error_code child_ec;
        std::filesystem::remove(child_err, child_ec);
    }

    std::error_code ec;
    std::filesystem::remove(stderr_path, ec);
    std::filesystem::remove(bundle_path, ec);
    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
