// test_frame_breakdown.cpp -- the run-level frame breakdown line of the `[perf-alarm]` exit summary.
//
// The failure it guards: a summary that LOOKS like a partition of the 16.7 ms frame, or reports a mean
// over no flips, or lets a SAMPLED cost (texture references are timed 1 in 32) understate itself by the
// sampling period. Pure arithmetic and text; no GPU, no engine thread.
#include "diagnostics/perf/perf_alarm_rules.hpp"
#include "diagnostics/perf/perf_alarms.hpp"
#include "diagnostics/perf/perf_ledger.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>

using namespace prosper::diagnostics::perf;

namespace {

int g_failures = 0;

void check(const char* what, bool ok) {
    std::printf("  %s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

std::string summary_of(AlarmEngine& engine) {
    // write_summary takes a FILE*; a tmpfile keeps the test off the real disk's named paths.
    FILE* s = std::tmpfile();
    if (!s) return {};
    engine.write_summary(s);
    std::fflush(s);
    std::rewind(s);
    std::string text;
    char buffer[4096];
    size_t n;
    while ((n = std::fread(buffer, 1, sizeof buffer, s)) > 0) text.append(buffer, n);
    std::fclose(s);
    return text;
}

void formatting() {
    std::puts("format_frame_breakdown");
    check("zero flips prints nothing (a mean over no flips is undefined, not zero)",
          format_frame_breakdown(FrameBreakdownTotals{}).empty());

    // 3000 flips over 60 s: a 20 ms interval (50 fps) against a 60 Hz budget.
    FrameBreakdownTotals t;
    t.seconds = 60.0;
    t.flips = 3000;
    t.target_hz = 60;
    auto set = [&](Cost c, uint64_t ns, uint64_t events) {
        t.cost_ns[static_cast<size_t>(c)] = ns;
        t.cost_events[static_cast<size_t>(c)] = events;
    };
    set(Cost::GpuWaitGraphics, 12'000'000'000ull, 3000);   // 4 ms per flip
    set(Cost::PresentCpu, 3'000'000'000ull, 3000);          // 1 ms per flip
    set(Cost::FrontendBuild, 600'000'000ull, 600);           // 0.2 ms per flip, 0.2 events
    // A huge SAMPLED cost: a per-flip mean of it would understate the truth by the sampling period, so
    // it must stay out of the table however large it is.
    set(Cost::TextureRefSample, 50'000'000'000ull, 90);
    t.device_ns_graphics = 9'000'000'000ull;                 // 3 ms per flip
    t.device_samples_graphics = 2400;                        // 80% of the waits carried a pair
    const std::string line = format_frame_breakdown(t);

    check("the line carries the tag, flips, interval and the budget",
          line.find("[perf-alarm] summary observer=frame-breakdown flips=3000") == 0 &&
              line.find("interval=20.00ms (50.0 fps)") != std::string::npos &&
              line.find("budget=16.67ms") != std::string::npos);
    check("...says outright that it is summed thread time and not a partition",
          line.find("summed THREAD time, NOT a partition of the frame") != std::string::npos);
    check("per-flip means are time over FLIPS (12 s over 3000 flips is 4.00 ms)",
          line.find("gpu-wait-graphics 4.00ms (1.00 events)") != std::string::npos &&
              line.find("present-cpu 1.00ms (1.00 events)") != std::string::npos &&
              line.find("frontend-build 0.20ms (0.20 events)") != std::string::npos);
    check("rows are ordered by cost, largest first",
          line.find("gpu-wait-graphics") < line.find("present-cpu") &&
              line.find("present-cpu") < line.find("frontend-build"));
    check("the sampled texture-ref cost is excluded from the table, however large",
          line.find("texture-ref-sample 16") == std::string::npos &&
              line.find("texture-ref-sample is sampled 1-in-32 and is reported by its own rule") !=
                  std::string::npos);
    check("GPU device time is reported against how many waits carried a timestamp pair",
          line.find("gpu-device-graphics 3.00ms (timestamp pair on 2400 of 3000 waits)") !=
              std::string::npos);
    check("costs that never recorded an event are listed as such, not as zeros",
          line.find("no events: ") != std::string::npos &&
              line.find("surface-readback") != std::string::npos &&
              line.find("hle-blocking-wait") != std::string::npos &&
              line.find("shader-compile") != std::string::npos &&
              line.find("pipeline-create") != std::string::npos &&
              line.find("gpu-wait-compute") != std::string::npos);
    check("...and no zero-valued row is printed for them",
          line.find("surface-readback 0.00ms") == std::string::npos);

    t.target_hz = 30;   // the guest's 30 Hz request changes the budget the line states
    check("the budget follows the guest's flip rate",
          format_frame_breakdown(t).find("budget=33.33ms") != std::string::npos);

    FrameBreakdownTotals empty;
    empty.seconds = 10.0;
    empty.flips = 600;
    check("flips with no cost events say that no cost recorded any",
          format_frame_breakdown(empty).find("no cost recorded any event") != std::string::npos);
}

void through_the_engine() {
    std::puts("engine accumulation");
    EngineConfig config;
    config.log = nullptr;
    AlarmEngine engine(std::move(config));
    WindowSample w;
    w.seconds = 5.0;
    w.flips = 150;
    w.target_hz = 30;
    w.cost_ns[static_cast<size_t>(Cost::GpuWaitGraphics)] = 150'000'000ull;   // 1 ms per flip
    w.cost_events[static_cast<size_t>(Cost::GpuWaitGraphics)] = 150;
    engine.close_window(w, 5.0);
    engine.close_window(w, 10.0);
    const std::string summary = summary_of(engine);
    check("the engine sums its windows (300 flips over 10 s, 1.00 ms per flip)",
          summary.find("observer=frame-breakdown flips=300 span=10.0s") != std::string::npos &&
              summary.find("gpu-wait-graphics 1.00ms (1.00 events)") != std::string::npos);
    check("...at the budget of the guest's last flip rate",
          summary.find("budget=33.33ms") != std::string::npos);
    // This window has graphics waits with no timestamp pair, so gpu-device-time-coverage fires: the
    // breakdown is printed whether or not a rule fired.
    check("...and is printed when a rule fired, beside that rule's summary line",
          summary.find("summary rule=gpu-device-time-coverage fired") != std::string::npos &&
              summary.find("observer=frame-breakdown") != std::string::npos);

    EngineConfig quiet_config;
    quiet_config.log = nullptr;
    AlarmEngine quiet(std::move(quiet_config));
    WindowSample idle;
    idle.seconds = 5.0;
    idle.flips = 300;
    quiet.close_window(idle, 5.0);
    const std::string quiet_summary = summary_of(quiet);
    check("...and when no rule fired, beside the quiet-run line",
          quiet_summary.find("no rule fired in 1 windows") != std::string::npos &&
              quiet_summary.find("observer=frame-breakdown flips=300") != std::string::npos);
    check("a run whose costs recorded nothing says so, not a table of zeros",
          quiet_summary.find("no cost recorded any event") != std::string::npos);

    EngineConfig fresh_config;
    fresh_config.log = nullptr;
    AlarmEngine fresh(std::move(fresh_config));
    check("an engine that closed no window prints no breakdown",
          summary_of(fresh).find("frame-breakdown") == std::string::npos);
}

}  // namespace

int main() {
    formatting();
    through_the_engine();
    std::printf(g_failures ? "FAILED (%d)\n" : "PASSED\n", g_failures);
    return g_failures ? 1 : 0;
}
