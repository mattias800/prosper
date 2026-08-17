// Capture tunables and the detailed timeline selector must never fail in silence (#2564, #2565).
// Pure/offline: no guest, no Vulkan device.
//
// Two defects with one shape, both siblings of #1684. A malformed tunable used to substitute a
// DIFFERENT policy without a word — and for PROSPER_CAPTURE_MAX_SUBMITS the substitute is UNCAPPED,
// the exact inverse of what a cap is for, on precisely the runs that need it. A detailed timeline
// selector that never matched used to say nothing at all: the run exited 0 and the only evidence was
// a missing .prgcap.
//
// Two kinds of arm here, and the split is the point:
//   * in-process checks call the formatters and resolvers directly. They can assert the TEXT and the
//     VALUE IN FORCE, and they are structurally unable to see whether anything is registered with
//     std::atexit or validated at load — the reporter is being called by hand.
//   * child-process arms spawn this same binary and let it merely return from main (or, for
//     PROSPER_CAPTURE_MAX_SUBMITS, never reach main at all). Only these can see a missing
//     registration or a missing load-time refusal, which is what #1684's arm C proved the hard way.
#include "../src/gpu/gpu_timeline.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

static std::string slurp(const std::filesystem::path& path) {
    std::string text;
    if (FILE* in = std::fopen(path.string().c_str(), "rb")) {
        char buf[4096];
        size_t got = 0;
        while ((got = std::fread(buf, 1, sizeof(buf), in)) > 0) text.append(buf, got);
        std::fclose(in);
    }
    return text;
}

// Runs `body` with stderr redirected to a scratch file and returns everything it wrote. Asserting on
// a returned string alone would leave the DELIVERY untested, and an undelivered report is exactly
// the defect: from the caller's side an empty report and an unreachable one look identical.
static int stream_fd(FILE* stream) {
#ifdef _WIN32
    return _fileno(stream);
#else
    return fileno(stream);
#endif
}

template <typename Body>
static std::string capture_stderr(const std::filesystem::path& path, Body body) {
    // dup/dup2 rather than freopen: there is no portable name to reopen the original stderr with,
    // and under ctest it is a pipe rather than a terminal.
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
    return slurp(path);
}

// Spawns this same binary in `mode`, redirecting the child's stderr to a BARE filename in the
// current directory. ctest runs a test in the directory the binary was built into, and neither a
// CMake target name nor this filename contains a space or a separator, so the whole command line
// needs no quoting at all — which is the question two Windows CI runs were lost to on #1684.
struct ChildRun {
    int rc = -1;
    std::string stderr_text;
    std::string command;
};
static ChildRun run_child(const char* argv0, const std::string& mode, const std::string& err_name) {
    ChildRun out;
    const std::string prefix =
#ifdef _WIN32
        ".\\";
#else
        "./";
#endif
    const std::string self = std::filesystem::path(argv0).filename().string();
    out.command = prefix + self + " " + mode + " 2>" + err_name;
    if (!std::system(nullptr)) std::printf("  [note] no command processor available\n");
    out.rc = std::system(out.command.c_str());
    out.stderr_text = slurp(err_name);
    std::error_code ec;
    std::filesystem::remove(err_name, ec);
    return out;
}

int main(int argc, char** argv) {
    // ---- child modes ---------------------------------------------------------------------------
    // Neither of these calls a reporter. Anything on the child's stderr can only have come from an
    // exit handler registered at load, or from the load-time refusal — the halves an in-process
    // assertion cannot distinguish from their absence.
    if (argc > 1 && std::strcmp(argv[1], "--child-submits") == 0) {
        std::printf("  [child] started (submits)\n");
        std::fflush(stdout);
        for (uint64_t n = 1; n <= 12; ++n) {
            begin_gpu_timeline_submit(n);
            record_gpu_timeline_submit(GpuState{}, n);
        }
        return 0;
    }
    if (argc > 1 && std::strcmp(argv[1], "--child-bare") == 0) {
        // Reaching this line is itself an observation, so it is witnessed on BOTH streams: stdout
        // lands inline in the ctest log ("did the spawn happen at all?", which a failed spawn and a
        // missing exit report otherwise answer identically), and stderr lands in the captured file,
        // where the PROSPER_CAPTURE_MAX_SUBMITS arm asserts its ABSENCE — the refusal is supposed to
        // end the process during static initialization, before main runs a single line.
        std::printf("  [child] started (bare)\n");
        std::fflush(stdout);
        std::fprintf(stderr, "[child] reached main\n");
        std::fflush(stderr);
        return 0;
    }

    std::printf("== test_capture_tunables ==\n");
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto scratch = prosper_test::test_scratch_dir();
    const auto stderr_path = scratch / ("prosper-capture-tunables-" + std::to_string(nonce) + ".err");
    const std::string child_err = "captun-child-" + std::to_string(nonce) + ".stderr";
    const auto bundle_path =
        scratch / ("prosper-capture-tunables-" + std::to_string(nonce) + ".prgbundle");

    // A stale gate in the ambient environment would arm something and silence the reports below.
    clear_test_env("PROSPER_CAPTURE_BUNDLE");
    clear_test_env("PROSPER_CAPTURE_BUNDLE_AT_PRESENT");
    clear_test_env("PROSPER_CAPTURE_BUNDLE_AFTER_GUEST_LOG");
    clear_test_env("PROSPER_CAPTURE_BUNDLE_TRIGGER_FILE");
    clear_test_env("PROSPER_CAPTURE_BUNDLE_MAX_MB");
    clear_test_env("PROSPER_CAPTURE_FRAMES");
    clear_test_env("PROSPER_CAPTURE_MAX_SUBMITS");
    clear_test_env("PROSPER_GPU_TIMELINE");
    clear_test_env("PROSPER_GPU_TIMELINE_CAPTURE");
    clear_test_env("PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT");

    // ---- the strict parse (#2565) --------------------------------------------------------------
    {
        CHECK(parse_capture_tunable(nullptr, 1, 100).status == CaptureTunableStatus::Unset &&
                  parse_capture_tunable("", 1, 100).status == CaptureTunableStatus::Unset,
              "an absent or empty value is Unset, not a rejection");
        const auto ok = parse_capture_tunable("40", 1, 100);
        CHECK(ok.status == CaptureTunableStatus::Accepted && ok.value == 40,
              "a whole decimal number is accepted at its own value");
        CHECK(parse_capture_tunable("0x28", 1, 100).value == 40,
              "hexadecimal is accepted, as every other capture tunable already spells addresses");
        CHECK(parse_capture_tunable("4O", 1, 100).status == CaptureTunableStatus::Malformed,
              "a letter O in place of a zero is REJECTED, not read as 4");
        CHECK(parse_capture_tunable("40 ", 1, 100).status == CaptureTunableStatus::Malformed,
              "a trailing space is rejected: the whole value must be the number");
        CHECK(parse_capture_tunable(" 40", 1, 100).status == CaptureTunableStatus::Malformed,
              "a leading space is rejected too, so the two spellings cannot disagree");
        // strtoull NEGATES a leading '-' into a huge positive number. Left unchecked, `-1` would be
        // accepted as 18446744073709551615 — a cap so large it is indistinguishable from uncapped,
        // which is the very substitution this change exists to prevent.
        CHECK(parse_capture_tunable("-1", 1, UINT64_MAX).status == CaptureTunableStatus::Malformed,
              "a negative value is rejected rather than wrapping to an enormous positive cap");
        CHECK(parse_capture_tunable("10", 64, 3072).status == CaptureTunableStatus::BelowRange &&
                  parse_capture_tunable("10", 64, 3072).value == 10,
              "a value under the minimum is BelowRange and still reports what was asked for");
        CHECK(parse_capture_tunable("5000", 64, 3072).status == CaptureTunableStatus::AboveRange,
              "a value over the maximum is AboveRange");
    }

    // ---- PROSPER_CAPTURE_MAX_SUBMITS refuses; it does not substitute ---------------------------
    {
        CHECK(format_capture_max_submits_refusal(nullptr).empty() &&
                  format_capture_max_submits_refusal("40").empty(),
              "an unset or valid cap produces no refusal");
        const std::string typo = format_capture_max_submits_refusal("4O");
        CHECK(contains(typo, "PROSPER_CAPTURE_MAX_SUBMITS=\"4O\""),
              "the refusal quotes the exact rejected value");
        CHECK(contains(typo, "REFUSING TO RUN") && contains(typo, "UNCAPPED"),
              "the refusal states both that it refuses and WHY the fallback is unacceptable");
        // The whole point of refusing rather than defaulting: there is no honest substitute here,
        // so the refusal must not borrow the phrasing of the tunables that DO fall back. "<x> is in
        // force" is the substituting notices' contract; a refusal that used it would be describing
        // a run that is not going to happen.
        CHECK(!contains(typo, "in force"),
              "the refusal names no value in force, because it substitutes none");
        CHECK(!format_capture_max_submits_refusal("0").empty(),
              "zero is refused too: 'uncapped' is expressible only by leaving the variable unset");
        CHECK(contains(format_capture_max_submits_refusal("40 "), "40\\x20"),
              "an invisible trailing space is ESCAPED, not reproduced invisibly");
    }

    // ---- PROSPER_CAPTURE_BUNDLE_MAX_MB reports and falls back ----------------------------------
    {
        CHECK(format_capture_bundle_max_mb_notice(nullptr).empty() &&
                  format_capture_bundle_max_mb_notice("512").empty(),
              "an unset or in-range budget produces no notice");
        const std::string typo = format_capture_bundle_max_mb_notice("2O48");
        CHECK(contains(typo, "PROSPER_CAPTURE_BUNDLE_MAX_MB=\"2O48\""),
              "the notice names the variable and quotes the rejected value");
        CHECK(contains(typo, std::to_string(kInteractiveBundleDefaultMb) + " MiB") &&
                  contains(typo, "DISCARDED"),
              "the notice states the budget actually in force and that the raise was discarded");
        CHECK(contains(format_capture_bundle_max_mb_notice("10"),
                       std::to_string(kInteractiveBundleMinMb) + " MiB minimum"),
              "a below-range budget reports the minimum it was raised to");
        CHECK(contains(format_capture_bundle_max_mb_notice("5000"),
                       std::to_string(kInteractiveBundleMaxMb) + " MiB maximum"),
              "an above-range budget reports the maximum it was capped at");
        // 0 is this variable's long-standing sentinel for "keep the built-in default" — honouring
        // it is not a substitution, so it must draw neither a notice nor a changed budget.
        CHECK(format_capture_bundle_max_mb_notice("0").empty(),
              "zero is the documented 'keep the default' sentinel and is not reported as a fault");
    }

    // ---- PROSPER_CAPTURE_FRAMES reports and falls back ------------------------------------------
    {
        CHECK(format_capture_frames_notice(nullptr).empty() &&
                  format_capture_frames_notice("16").empty(),
              "an unset or in-range window produces no notice");
        const std::string typo = format_capture_frames_notice("abc");
        CHECK(contains(typo, "PROSPER_CAPTURE_FRAMES=\"abc\""),
              "the notice names the variable and quotes the rejected value");
        // This is the sharpest case of the family: widening the window is the remedy the empty-window
        // message itself prints, so a mistyped remedy silently reproduced the original failure.
        CHECK(contains(typo, "window had no submits"),
              "the notice connects the discarded value to the failure it was meant to fix");
        CHECK(contains(format_capture_frames_notice("300"),
                       "clamped to " + std::to_string(kCaptureFramesMax)),
              "an above-range window reports the width it was clamped to");
    }

    // ---- PROSPER_CAPTURE_BUNDLE alone is a complete no-op, and now says so ----------------------
    // Deliberately BEFORE anything arms a capture: a run that did arm one is not misconfigured, and
    // the suppression is checked immediately afterwards.
    {
        set_test_env("PROSPER_CAPTURE_BUNDLE", bundle_path.string());
        const std::string alone =
            capture_stderr(stderr_path, [] { report_unfired_automatic_capture_gates(); });
        CHECK(contains(alone, "PROSPER_CAPTURE_BUNDLE") && contains(alone, "arms nothing"),
              "PROSPER_CAPTURE_BUNDLE set with no gate is reported instead of doing nothing");
        CHECK(contains(alone, "PROSPER_CAPTURE_BUNDLE_AT_PRESENT") &&
                  contains(alone, "PROSPER_CAPTURE_BUNDLE_AFTER_GUEST_LOG") &&
                  contains(alone, "PROSPER_CAPTURE_BUNDLE_TRIGGER_FILE"),
              "the report names every gate that would actually arm one");
        clear_test_env("PROSPER_CAPTURE_BUNDLE");
        const std::string quiet =
            capture_stderr(stderr_path, [] { report_unfired_automatic_capture_gates(); });
        CHECK(quiet.empty(), "with nothing configured at all the report stays silent");
    }

    // ---- the frames notice reaches stderr through the REAL arm path -----------------------------
    // First call of resolve_capture_frames() in this process, so it also proves the notice is
    // delivered rather than merely formatted. It latches "something armed", which is why the
    // PROSPER_CAPTURE_BUNDLE-alone arm above had to run first.
    {
        set_test_env("PROSPER_CAPTURE_FRAMES", "abc");
        const std::string armed = capture_stderr(stderr_path, [&] {
            (void)request_interactive_capture_bundle(bundle_path.string());
        });
        CHECK(contains(armed, "PROSPER_CAPTURE_FRAMES=\"abc\""),
              "arming a capture with a malformed window width says so on stderr");
        const std::string again = capture_stderr(stderr_path, [&] {
            (void)request_interactive_capture_bundle(bundle_path.string());
        });
        CHECK(again.empty(), "the notice is printed once, not once per arm");
        clear_test_env("PROSPER_CAPTURE_FRAMES");
        CHECK(resolve_capture_frames() == kCaptureFramesDefault,
              "an unset window resolves to the default");
        set_test_env("PROSPER_CAPTURE_FRAMES", "300");
        CHECK(resolve_capture_frames() == kCaptureFramesMax,
              "an above-range window resolves to the clamped value, unchanged from before");
        set_test_env("PROSPER_CAPTURE_FRAMES", "abc");
        CHECK(resolve_capture_frames() == kCaptureFramesDefault,
              "a malformed window still falls back to the default: only the silence changed");
        clear_test_env("PROSPER_CAPTURE_FRAMES");
    }
    // With a capture armed, PROSPER_CAPTURE_BUNDLE is no longer a no-op and must NOT be reported.
    {
        set_test_env("PROSPER_CAPTURE_BUNDLE", bundle_path.string());
        const std::string after_arm =
            capture_stderr(stderr_path, [] { report_unfired_automatic_capture_gates(); });
        CHECK(after_arm.empty(),
              "once something has armed, a set PROSPER_CAPTURE_BUNDLE is not called a no-op");
        clear_test_env("PROSPER_CAPTURE_BUNDLE");
    }

    // ---- the budget resolver: value in force, and one notice ------------------------------------
    {
        set_test_env("PROSPER_CAPTURE_BUNDLE_MAX_MB", "2O48");
        uint32_t resolved = 0;
        const std::string notice =
            capture_stderr(stderr_path, [&] { resolved = resolve_capture_bundle_max_mb(); });
        CHECK(contains(notice, "PROSPER_CAPTURE_BUNDLE_MAX_MB=\"2O48\""),
              "resolving a malformed budget says so on stderr");
        CHECK(resolved == 0,
              "a malformed budget still resolves to 0 = keep the built-in default, as before");
        set_test_env("PROSPER_CAPTURE_BUNDLE_MAX_MB", "5000");
        CHECK(resolve_capture_bundle_max_mb() == kInteractiveBundleMaxMb,
              "an above-range budget resolves to the maximum");
        set_test_env("PROSPER_CAPTURE_BUNDLE_MAX_MB", "10");
        CHECK(resolve_capture_bundle_max_mb() == kInteractiveBundleMinMb,
              "a below-range budget resolves to the minimum");
        set_test_env("PROSPER_CAPTURE_BUNDLE_MAX_MB", "512");
        CHECK(resolve_capture_bundle_max_mb() == 512, "an in-range budget resolves unchanged");
        set_test_env("PROSPER_CAPTURE_BUNDLE_MAX_MB", "0");
        CHECK(resolve_capture_bundle_max_mb() == 0,
              "zero still resolves to 0 = keep the built-in default, not to the 64 MiB minimum");
        clear_test_env("PROSPER_CAPTURE_BUNDLE_MAX_MB");
    }

    // ---- the cap is resolved ONCE, at load ------------------------------------------------------
    {
        CHECK(capture_max_submits() == 0, "an unset cap is uncapped");
        // Setting it now must not change the answer, and must not end this process: the refusal is a
        // load-time decision, so a value that appears mid-run is neither honoured nor fatal. The
        // child arm below is what proves the load-time half.
        set_test_env("PROSPER_CAPTURE_MAX_SUBMITS", "4O");
        CHECK(capture_max_submits() == 0,
              "the cap is resolved once at load; a later environment change does not re-open it");
        clear_test_env("PROSPER_CAPTURE_MAX_SUBMITS");
    }

    // ---- the detailed timeline selector (#2564) -------------------------------------------------
    {
        TimelineCaptureSelectorMiss none;
        CHECK(!timeline_capture_selector_miss_snapshot(none),
              "with no selector configured there is nothing to report");
    }
    {
        // The pure formatter, over the outcomes that have DIFFERENT fixes. Conflating them is what
        // sends the next run at the wrong problem.
        TimelineCaptureSelectorMiss miss;
        miss.capture_path = "/w/out.prgcap";
        miss.submit_spec = "41000";
        miss.request_built = miss.request_valid = true;
        miss.submit_hook_reached = 12800;
        const std::string no_timeline = format_timeline_capture_selector_miss(miss);
        CHECK(contains(no_timeline, "PROSPER_GPU_TIMELINE is NOT set"),
              "a selector without PROSPER_GPU_TIMELINE is diagnosed as exactly that");

        miss.timeline_requested = true;
        miss.submits_examined = 12800;
        miss.last_submit_examined = 12800;
        const std::string ordinal = format_timeline_capture_selector_miss(miss);
        CHECK(contains(ordinal, "ORDINAL selector") && contains(ordinal, "at or below 12800"),
              "an unreached ordinal names what the run DID reach, not only what was asked for");
        CHECK(contains(ordinal, "41000"), "the report quotes the ordinal that was demanded");

        miss.semantic_selector = true;
        miss.semantic.emplace_back("PROSPER_GPU_TIMELINE_CAPTURE_WHEN_TARGET_DIM", "3840x2160");
        const std::string semantic = format_timeline_capture_selector_miss(miss);
        CHECK(contains(semantic, "3840x2160") && contains(semantic, "RUN-LOCAL") &&
                  contains(semantic, "--select"),
              "a semantic miss quotes the predicate and points at re-deriving it, not at running "
              "longer");
        CHECK(!contains(semantic, "ORDINAL selector"),
              "the two outcomes are reported distinctly, because their fixes differ");

        TimelineCaptureSelectorMiss quiet_run;
        quiet_run.capture_path = "/w/out.prgcap";
        quiet_run.submit_spec = "1";
        quiet_run.timeline_requested = true;
        const std::string no_submits = format_timeline_capture_selector_miss(quiet_run);
        CHECK(contains(no_submits, "no GPU submits at all"),
              "a run that never submitted is a routing fault, and is named as one");

        TimelineCaptureSelectorMiss phase;
        phase.capture_path = "/w/out.prgcap";
        phase.submit_spec = "1";
        phase.timeline_requested = true;
        phase.request_built = phase.request_valid = true;
        phase.submit_hook_reached = 900;
        phase.after_compute_gated = true;
        phase.phase_submits_observed = 900;
        const std::string unarmed = format_timeline_capture_selector_miss(phase);
        CHECK(contains(unarmed, "phase gate never armed") && contains(unarmed, "900"),
              "an unarmed after-compute phase gate is separated from an unmatched endpoint");
    }
    {
        // The live selector, against the real submit path. One shot per process: the request object
        // is a singleton, so this is the only configuration this process can exercise.
        const auto timeline_path =
            scratch / ("prosper-capture-tunables-" + std::to_string(nonce) + ".prgtl");
        const auto capture_path =
            scratch / ("prosper-capture-tunables-" + std::to_string(nonce) + ".prgcap");
        set_test_env("PROSPER_GPU_TIMELINE", timeline_path.string());
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE", capture_path.string());
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT", "41000");
        for (uint64_t n = 1; n <= 5; ++n) {
            begin_gpu_timeline_submit(n);
            record_gpu_timeline_submit(GpuState{}, n);
        }
        TimelineCaptureSelectorMiss live;
        CHECK(timeline_capture_selector_miss_snapshot(live),
              "a configured selector that captured nothing is reportable");
        CHECK(live.submit_spec == "41000" && live.request_valid,
              "the snapshot carries the configured ordinal and the request's own verdict");
        CHECK(live.submits_examined == 5 && live.last_submit_examined == 5,
              "the snapshot counts the submits the selector actually judged");
        CHECK(live.detail_submits_captured == 0, "nothing was captured, which is why a report is due");
        const std::string delivered =
            capture_stderr(stderr_path, [] { report_unfired_timeline_capture_selector(); });
        CHECK(contains(delivered, "never matched a submit") && contains(delivered, "41000"),
              "the selector report reaches stderr and names the ordinal that was demanded");
        CHECK(contains(delivered, "at or below 5"),
              "the delivered report states the ordinal the run actually reached");

        clear_test_env("PROSPER_GPU_TIMELINE_CAPTURE");
        clear_test_env("PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT");
        const std::string silent =
            capture_stderr(stderr_path, [] { report_unfired_timeline_capture_selector(); });
        CHECK(silent.empty(), "with no selector configured the report says nothing at all");
        clear_test_env("PROSPER_GPU_TIMELINE");
        close_gpu_timeline();
    }

    // ---- the wiring: real processes that never call a reporter ----------------------------------
    // Everything above calls the reporters by hand and therefore cannot tell a registered exit
    // handler from an unregistered one, nor a load-time refusal from its absence.
    {
        const auto timeline_path =
            scratch / ("prosper-capture-tunables-child-" + std::to_string(nonce) + ".prgtl");
        const auto capture_path =
            scratch / ("prosper-capture-tunables-child-" + std::to_string(nonce) + ".prgcap");
        set_test_env("PROSPER_GPU_TIMELINE", timeline_path.string());
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE", capture_path.string());
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT", "41000");
        const ChildRun child = run_child(argv[0], "--child-submits", child_err);
        // A spawn or redirection fault leaves the same empty file the defect does. Name the
        // apparatus so the two can never be confused.
        if (child.rc != 0 || child.stderr_text.empty())
            std::printf("  [note] child rc=%d, stderr bytes=%zu, command was: %s\n", child.rc,
                        child.stderr_text.size(), child.command.c_str());
        CHECK(child.rc == 0, "the child exits cleanly, exactly as a real unmatched run does");
        CHECK(contains(child.stderr_text, "never matched a submit"),
              "a process that only returns from main still reports its unmatched selector at exit");
        CHECK(contains(child.stderr_text, "at or below 12"),
              "the exit report states the submit ordinal the child actually reached");
        CHECK(!std::filesystem::exists(capture_path),
              "no capture was written, which is the outcome the report exists to explain");
        clear_test_env("PROSPER_GPU_TIMELINE");
        clear_test_env("PROSPER_GPU_TIMELINE_CAPTURE");
        clear_test_env("PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT");
        std::error_code ec;
        std::filesystem::remove(timeline_path, ec);
    }
    {
        set_test_env("PROSPER_CAPTURE_BUNDLE", bundle_path.string());
        const ChildRun child = run_child(argv[0], "--child-bare", child_err);
        if (child.rc != 0 || child.stderr_text.empty())
            std::printf("  [note] child rc=%d, stderr bytes=%zu, command was: %s\n", child.rc,
                        child.stderr_text.size(), child.command.c_str());
        CHECK(child.rc == 0 && contains(child.stderr_text, "arms nothing"),
              "a process that sets only PROSPER_CAPTURE_BUNDLE explains itself at exit");
        clear_test_env("PROSPER_CAPTURE_BUNDLE");
    }
    {
        // The load-time refusal. This child never reaches main, so its exit status is the assertion:
        // without the refusal it would return 0 and run UNCAPPED, which is the defect.
        set_test_env("PROSPER_CAPTURE_MAX_SUBMITS", "4O");
        const ChildRun child = run_child(argv[0], "--child-bare", child_err);
        clear_test_env("PROSPER_CAPTURE_MAX_SUBMITS");
        if (child.rc == 0 || child.stderr_text.empty())
            std::printf("  [note] child rc=%d, stderr bytes=%zu, command was: %s\n", child.rc,
                        child.stderr_text.size(), child.command.c_str());
        CHECK(child.rc != 0,
              "a malformed submit cap ends the process with a nonzero status instead of running "
              "uncapped");
        CHECK(contains(child.stderr_text, "REFUSING TO RUN") &&
                  contains(child.stderr_text, "PROSPER_CAPTURE_MAX_SUBMITS=\"4O\""),
              "the refusal reaches stderr of a process that never reached main");
        CHECK(!contains(child.stderr_text, "[child] reached main"),
              "the refusal happens at LOAD: the child never ran a line of its own");
    }

    std::error_code ec;
    std::filesystem::remove(stderr_path, ec);
    std::filesystem::remove(bundle_path, ec);
    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
