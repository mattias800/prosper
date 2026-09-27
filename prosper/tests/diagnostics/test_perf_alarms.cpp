// `[perf-alarm]` rules and engine (#3891).
//
// The design rule the issue sets: build each alarm so it would have caught a problem the
// 2026-09-27 pass found by hand, and use that problem as the test. So every rule here gets a
// POSITIVE window constructed by hand from the numbers that pass measured before its fix, and a
// NEGATIVE window with the numbers after it -- plus a boundary arm just under the threshold, so a
// rule that fired on everything, or on nothing, cannot pass. The live half (the rule fires on the
// pre-fix commit and stays quiet on main) is recorded in the PR; this is the half CI can run.
//
// The engine arms check what a unit of the rules cannot: that the window closes on the flip that
// crosses it and not before, that the per-rule log de-duplication follows diag_ratelimit's
// contract while the JSONL keeps every window, that the exit summary distinguishes "nothing fired"
// from "never evaluated", and that the frame budget follows the guest's SetFlipRate.
#include "diagnostics/perf/perf_alarm_rules.hpp"
#include "diagnostics/perf/perf_alarms.hpp"
#include "diagnostics/perf/perf_ledger.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace prosper::diagnostics::perf;

namespace {

int g_failures = 0;

void check(const char* what, bool ok) {
    std::printf("  %s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

bool fired(const std::vector<AlarmFiring>& v, const char* rule) {
    for (const auto& a : v)
        if (std::strcmp(a.rule, rule) == 0) return true;
    return false;
}

// Exactly `rule` fired and nothing else: a positive arm must not pass because some OTHER rule
// happened to fire on the same window.
bool only(const std::vector<AlarmFiring>& v, const char* rule) {
    return v.size() == 1 && fired(v, rule);
}

// A healthy 5 s window at 30 fps against a 30 Hz target.
WindowSample healthy() {
    WindowSample w;
    w.seconds = 5.0;
    w.flips = 150;
    w.target_hz = 30;
    return w;
}

void set_cost(WindowSample& w, Cost c, double total_ms, uint64_t events, double max_ms) {
    const size_t i = static_cast<size_t>(c);
    w.cost_ns[i] = static_cast<uint64_t>(total_ms * 1e6);
    w.cost_events[i] = events;
    w.cost_max_ns[i] = static_cast<uint64_t>(max_ms * 1e6);
}

void set_count(WindowSample& w, Counter c, uint64_t n) {
    w.counters[static_cast<size_t>(c)] = n;
}

void set_gauge(WindowSample& w, Gauge g, uint64_t v) {
    w.gauges[static_cast<size_t>(g)] = v;
}

const RuleThresholds kDefault{};

void test_quiet_baseline() {
    std::puts("baseline");
    check("an empty healthy window raises nothing", evaluate_rules(healthy(), kDefault).empty());
    WindowSample zero_length = healthy();
    zero_length.seconds = 0;
    set_count(zero_length, Counter::DroppedDrawsFrontend, 5);
    check("a zero-length window is not evaluated", evaluate_rules(zero_length, kDefault).empty());
}

// #3876: Outer Wilds, cache at 4095/4096 MiB, 0 evictions, ~2,400 refusals per 5 s.
void test_texture_cache_thrash() {
    std::puts("texture-cache-thrash");
    WindowSample bad = healthy();
    bad.flips = 26;  // ~5 flips/s
    set_count(bad, Counter::TextureCacheMisses, 2400);
    set_count(bad, Counter::TextureCacheRefusals, 2389);
    set_count(bad, Counter::TextureCacheRefusedBytes, 2389ull * 4 * 1024 * 1024);
    set_count(bad, Counter::TextureCacheEvictions, 0);
    set_count(bad, Counter::DeviceAllocations, 3450);
    set_gauge(bad, Gauge::TextureCacheBytes, 4294846464ull);
    set_gauge(bad, Gauge::TextureCacheLimit, 4294967296ull);
    const auto a = evaluate_rules(bad, kDefault);
    check("pre-#3876 window fires texture-cache-thrash alone", only(a, "texture-cache-thrash"));
    check("the line names evictions=0 (the state that names the mechanism)",
          !a.empty() && a[0].detail.find("evictions=0") != std::string::npos);

    // After #3876: the same misses are admitted by evicting idle images -- no refusals.
    WindowSample good = bad;
    set_count(good, Counter::TextureCacheRefusals, 0);
    set_count(good, Counter::TextureCacheRefusedBytes, 0);
    set_count(good, Counter::TextureCacheEvictions, 2350);
    check("post-#3876 window (misses admitted by eviction) is quiet",
          evaluate_rules(good, kDefault).empty());

    WindowSample edge = good;
    set_count(edge, Counter::TextureCacheRefusals, 249);  // 49.8/s against 50/s
    check("249 refusals in 5 s (just under 50/s) is quiet", evaluate_rules(edge, kDefault).empty());
    set_count(edge, Counter::TextureCacheRefusals, 250);
    check("250 refusals in 5 s (50/s) fires", fired(evaluate_rules(edge, kDefault), "texture-cache-thrash"));
}

// #3882: Sonic Frontiers, ~65 depth-array readbacks per 5 s at ~10 ms each, ~8 flips/s.
void test_surface_readback() {
    std::puts("surface-readback");
    WindowSample bad = healthy();
    bad.flips = 40;
    set_cost(bad, Cost::SurfaceReadback, 650.0, 65, 24.0);
    const auto a = evaluate_rules(bad, kDefault);
    check("pre-#3882 window fires surface-readback alone", only(a, "surface-readback"));
    check("its cost is the window's readback time", !a.empty() && a[0].cost_ms > 649 && a[0].cost_ms < 651);

    WindowSample good = bad;
    set_cost(good, Cost::SurfaceReadback, 0, 0, 0);
    check("post-#3882 window (no readbacks) is quiet", evaluate_rules(good, kDefault).empty());

    // A few big readbacks during a load: expensive per event, but not a pattern.
    WindowSample load = bad;
    set_cost(load, Cost::SurfaceReadback, 400.0, 3, 200.0);
    check("3 readbacks in 5 s (under the 2/s floor) is quiet even at a large share",
          evaluate_rules(load, kDefault).empty());

    // Many sub-millisecond readbacks at a low frame rate: GTA V on main, ~75 ms per 5 s over 40 flips
    // is 11% of a 60 Hz budget. Real, second-order, below the quarter-budget alarm.
    WindowSample second_order = bad;
    second_order.target_hz = 60;
    set_cost(second_order, Cost::SurfaceReadback, 75.0, 440, 1.0);
    check("GTA-like 11%-of-budget readbacks are quiet", evaluate_rules(second_order, kDefault).empty());

    // Frequent but cheap: 100 readbacks at 0.1 ms each is 0.25 ms per flip, <1% of a 33 ms budget.
    WindowSample cheap = bad;
    set_cost(cheap, Cost::SurfaceReadback, 10.0, 100, 0.2);
    check("frequent cheap readbacks (<1% of budget) are quiet", evaluate_rules(cheap, kDefault).empty());
}

// Texture references: `refs` in the window, one in kTextureRefSamplePeriod timed at `mean_us`.
void set_texrefs(WindowSample& w, uint64_t refs, double mean_us, uint64_t samples = 0) {
    if (!samples) samples = refs / kTextureRefSamplePeriod;
    set_count(w, Counter::TextureReferences, refs);
    set_cost(w, Cost::TextureRefSample, static_cast<double>(samples) * mean_us / 1000.0, samples,
             mean_us / 1000.0 * 3);
}

// #3877: GTA V walked the whole RTT cache per texture reference, ~7-8 us each over ~150k references
// per 5 s at ~8 fps.
void test_texture_reference_cost() {
    std::puts("texture-reference-cost");
    WindowSample bad = healthy();
    bad.flips = 35;
    set_texrefs(bad, 150000, 7.0);  // est 1050 ms = 90% of a 33 ms budget per flip
    const auto a = evaluate_rules(bad, kDefault);
    check("pre-#3877 window (7 us/ref, 90% of budget) fires texture-reference-cost alone",
          only(a, "texture-reference-cost"));
    check("its cost is mean x references, not the sampled time alone",
          !a.empty() && a[0].cost_ms > 1000 && a[0].cost_ms < 1100);

    WindowSample good = bad;
    set_texrefs(good, 150000, 1.2);
    check("post-#3877 window (1.2 us/ref) is quiet", evaluate_rules(good, kDefault).empty());

    // A slow mean over a population that costs nothing.
    WindowSample tiny = healthy();
    set_texrefs(tiny, 6000, 10.0);  // est 60 ms over 150 flips: ~1% of budget
    check("a slow mean over a cheap population (<25% of budget) is quiet",
          evaluate_rules(tiny, kDefault).empty());

    // A heavy population that is fast per reference: the cost is volume, not resolution.
    WindowSample heavy = bad;
    set_texrefs(heavy, 900000, 1.17);
    check("a heavy population that is fast per reference is quiet",
          evaluate_rules(heavy, kDefault).empty());

    // Too few samples for a mean: 50 in 5 s against a 20/s floor.
    WindowSample sparse = bad;
    set_texrefs(sparse, 150000, 7.0, 50);
    check("a mean from 50 samples in 5 s (under 20/s) is quiet", evaluate_rules(sparse, kDefault).empty());
}

// #3879: The Messenger, main thread blocked 85% of wall time in sceVideoOutSetFlipRate, avg 14.2 ms.
void test_hle_blocking_wait() {
    std::puts("hle-blocking-wait");
    WindowSample bad = healthy();
    bad.target_hz = 60;
    bad.flips = 300;
    set_cost(bad, Cost::HleBlockingWait, 4250.0, 299, 16.6);
    bad.cost_label[static_cast<size_t>(Cost::HleBlockingWait)] = "VideoOut handle lock";
    const auto a = evaluate_rules(bad, kDefault);
    check("pre-#3879 window (85% of a thread blocked) fires hle-blocking-wait alone",
          only(a, "hle-blocking-wait"));
    check("the line names the lock",
          !a.empty() && a[0].detail.find("VideoOut handle lock") != std::string::npos);

    WindowSample good = bad;
    set_cost(good, Cost::HleBlockingWait, 3.0, 40, 0.2);  // occasional brief contention
    check("post-#3879 window (brief contention) is quiet", evaluate_rules(good, kDefault).empty());

    WindowSample edge = good;
    set_cost(edge, Cost::HleBlockingWait, 990.0, 100, 12.0);  // 19.8%
    check("19.8% of a thread blocked is quiet", evaluate_rules(edge, kDefault).empty());
    set_cost(edge, Cost::HleBlockingWait, 1000.0, 100, 12.0);  // 20%
    check("20% of a thread blocked fires", fired(evaluate_rules(edge, kDefault), "hle-blocking-wait"));
}

// #3875: --fps content sample read from uncached memory at ~3 ms per present.
void test_present_cpu_overhead() {
    std::puts("present-cpu-overhead");
    WindowSample bad = healthy();
    set_cost(bad, Cost::PresentCpu, 3.0 * 600, 600, 4.1);  // 120 presents/s
    const auto a = evaluate_rules(bad, kDefault);
    check("pre-#3875 window (3 ms/present) fires present-cpu-overhead alone",
          only(a, "present-cpu-overhead"));

    WindowSample good = bad;
    set_cost(good, Cost::PresentCpu, 0.05 * 600, 600, 0.1);
    check("post-#3875 window (0.05 ms/present) is quiet", evaluate_rules(good, kDefault).empty());

    WindowSample rare = bad;
    set_cost(rare, Cost::PresentCpu, 3.0 * 20, 20, 4.0);  // 4 presents/s
    check("a slow present at 4/s (under the 5/s floor) is quiet", evaluate_rules(rare, kDefault).empty());
}

// #3889: GTA V dropped every draw sampling its colour-grading LUT (34 per run on the menu route).
void test_dropped_draws() {
    std::puts("dropped-draws");
    WindowSample bad = healthy();
    set_count(bad, Counter::DroppedDrawsFrontend, 34);
    const auto a = evaluate_rules(bad, kDefault);
    check("pre-#3889 window (34 frontend rejects) fires dropped-draws alone", only(a, "dropped-draws"));

    WindowSample one = healthy();
    set_count(one, Counter::DroppedDrawsBackend, 1);
    check("a single backend drop fires (correctness: any)",
          fired(evaluate_rules(one, kDefault), "dropped-draws"));
    check("post-#3889 window (no drops) is quiet", evaluate_rules(healthy(), kDefault).empty());
    check("lowering sensitivity does not raise the correctness threshold",
          fired(evaluate_rules(one, RuleThresholds::scaled(1000)), "dropped-draws"));
}

void test_threshold_scaling() {
    std::puts("PROSPER_PERF_ALARM_THRESHOLD_PCT");
    WindowSample w = healthy();
    set_count(w, Counter::TextureCacheRefusals, 100);  // 20/s: quiet at 100%
    check("20 refusals/s is quiet at the default thresholds", evaluate_rules(w, kDefault).empty());
    check("...and fires at 25% (the investigation setting)",
          fired(evaluate_rules(w, RuleThresholds::scaled(25)), "texture-cache-thrash"));
    check("a zero percentage keeps the defaults rather than alarming on everything",
          evaluate_rules(w, RuleThresholds::scaled(0)).empty());
}

void test_budget_follows_flip_rate() {
    std::puts("frame budget from SetFlipRate");
    // 5 ms of readback per flip: 30% of a 60 Hz budget, 15% of 30 Hz, 10% of 20 Hz.
    WindowSample w = healthy();
    set_cost(w, Cost::SurfaceReadback, 5.0 * 150, 150, 6.0);
    w.target_hz = 60;
    check("5 ms/flip fires against a 60 Hz budget", fired(evaluate_rules(w, kDefault), "surface-readback"));
    w.target_hz = 20;
    check("the same 5 ms/flip is quiet against a 20 Hz budget", evaluate_rules(w, kDefault).empty());
}

void test_cost_scope_nesting() {
    std::puts("ledger CostScope");
    Ledger& l = ledger();
    const size_t i = static_cast<size_t>(Cost::SurfaceReadback);
    const uint64_t events0 = l.cost_events[i].load();
    {
        CostScope outer(Cost::SurfaceReadback);
        CostScope inner(Cost::SurfaceReadback);  // a helper calling a helper
    }
    check("nested scopes of one category are ONE event", l.cost_events[i].load() == events0 + 1);
    { CostScope again(Cost::SurfaceReadback); }
    check("a later sibling scope is a second event", l.cost_events[i].load() == events0 + 2);
}

void test_texture_reference_sample() {
    std::puts("ledger TextureReferenceSample");
    Ledger& l = ledger();
    const size_t ts = static_cast<size_t>(Cost::TextureRefSample);
    const size_t rb = static_cast<size_t>(Cost::SurfaceReadback);
    const size_t refs = static_cast<size_t>(Counter::TextureReferences);
    flush_thread_texture_references();
    const uint64_t refs0 = l.counters[refs].load(), samples0 = l.cost_events[ts].load();
    const uint64_t sample_ns0 = l.cost_ns[ts].load(), readback_ns0 = l.cost_ns[rb].load();
    { TextureReferenceSample not_a_texture(false); not_a_texture.finish(); }
    for (uint64_t i = 0; i < 2 * kTextureRefSamplePeriod; ++i) {
        TextureReferenceSample ref(true);
        {
            // A depth readback nested inside the reference's resolution: 2 ms charged to readback.
            CostScope readback(Cost::SurfaceReadback);
            const uint64_t until = now_ns() + 2'000'000;
            while (now_ns() < until) {}
        }
        ref.finish();
    }
    flush_thread_texture_references();
    check("every texture reference is counted, non-textures are not",
          l.counters[refs].load() - refs0 == 2 * kTextureRefSamplePeriod);
    check("one reference in kTextureRefSamplePeriod is timed",
          l.cost_events[ts].load() - samples0 == 2);
    const double readback_ms = (l.cost_ns[rb].load() - readback_ns0) / 1e6;
    const double sampled_ms = (l.cost_ns[ts].load() - sample_ns0) / 1e6;
    check("the nested readback is charged to surface-readback", readback_ms >= 2.0 * 2 * kTextureRefSamplePeriod);
    check("...and subtracted from the sampled reference (one cause, one alarm)", sampled_ms < 1.0);
}

std::string slurp(const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

size_t count_of(const std::string& text, const std::string& needle) {
    size_t n = 0;
    for (size_t p = text.find(needle); p != std::string::npos; p = text.find(needle, p + 1)) ++n;
    return n;
}

void test_engine() {
    std::puts("engine");
    const std::string dir = std::getenv("TMPDIR") ? std::getenv("TMPDIR") : ".";
    const std::string jsonl = dir + "/test_perf_alarms.jsonl";
    const std::string logp = dir + "/test_perf_alarms.log";
    FILE* log = std::fopen(logp.c_str(), "w+");
    {
        EngineConfig config;
        config.window_ns = 1'000'000'000ull;
        config.jsonl_path = jsonl;
        config.log = log;
        AlarmEngine engine(std::move(config));
        Ledger l;  // a private ledger: the process one is shared with the hooks

        // Boot residue before the first flip must not land in the first window.
        l.counters[static_cast<size_t>(Counter::DroppedDrawsFrontend)] = 999;
        uint64_t t = 10'000'000'000ull;
        check("the first flip only sets the baseline", engine.on_flip(t, l, 60).empty());

        std::vector<AlarmFiring> last;
        t += 500'000'000ull;
        check("a flip before the window is due evaluates nothing", engine.on_flip(t, l, 60).empty() &&
                                                                   engine.windows_evaluated() == 0);
        t += 500'000'000ull;
        last = engine.on_flip(t, l, 60);
        check("the first due window ignores pre-baseline residue (999 drops)", last.empty() &&
                                                                              engine.windows_evaluated() == 1);

        // Ten consecutive windows each dropping 3 draws: every one fires.
        for (int k = 0; k < 10; ++k) {
            l.counters[static_cast<size_t>(Counter::DroppedDrawsFrontend)] += 3;
            t += 1'000'000'000ull;
            last = engine.on_flip(t, l, 60);
        }
        check("the tenth window fires dropped-draws with the window's delta (3), not the total",
              last.size() == 1 && last[0].value == 3.0);
        check("dropped-draws fired in all 10 windows", engine.times_fired("dropped-draws") == 10);

        // A window that is quiet again.
        t += 1'000'000'000ull;
        check("a window with no new drops is quiet", engine.on_flip(t, l, 60).empty());

        std::fflush(log);
        const std::string text = slurp(logp);
        // diag_ratelimit: ordinals 1, 2, 3 (first three) then 4 and 8 (powers of two) = 5 lines.
        check("log de-duplicates per rule: 5 lines for 10 firings (1,2,3,4,8)",
              count_of(text, "[perf-alarm] #") == 5 && text.find("[perf-alarm] #8 ") != std::string::npos &&
                  text.find("[perf-alarm] #5 ") == std::string::npos);
        check("each line carries rule, value, threshold and a hint",
              text.find("rule=dropped-draws") != std::string::npos &&
                  text.find("threshold=1.00") != std::string::npos &&
                  text.find("hint=") != std::string::npos);

        std::rewind(log);
        engine.write_summary(log);
    }
    std::fflush(log);
    const std::string j = slurp(jsonl);
    check("JSONL keeps every firing window (10 alarm objects, no rate limit)",
          count_of(j, "{\"type\":\"alarm\"") == 10 &&
              count_of(j, "\"rule\":\"dropped-draws\"") == 10);
    check("JSONL records every evaluated window (12), fired or not",
          count_of(j, "{\"type\":\"window\"") == 12 &&
              j.find("\"dropped_frontend\":3,") != std::string::npos);
    check("JSONL records the ordinal and the target rate",
          j.find("\"ordinal\":10") != std::string::npos && j.find("\"target_hz\":60") != std::string::npos);
    const std::string summary = slurp(logp);
    check("exit summary names the rule and its window count",
          summary.find("summary rule=dropped-draws fired in 10 of 12 windows") != std::string::npos);
    std::fclose(log);

    // Sustain: texture-reference-cost must hold three consecutive windows; a quiet window resets.
    {
        EngineConfig config;
        config.log = nullptr;
        AlarmEngine engine(std::move(config));
        WindowSample slow = healthy();
        slow.flips = 35;
        set_texrefs(slow, 150000, 7.0);
        const bool first = engine.close_window(slow, 5).empty();
        const bool second = engine.close_window(slow, 10).empty();
        check("a slow-reference window is not reported until it has held for 3 windows",
              first && second && fired(engine.close_window(slow, 15), "texture-reference-cost"));
        engine.close_window(healthy(), 20);
        check("a quiet window resets the streak",
              engine.close_window(slow, 25).empty() && engine.close_window(slow, 30).empty() &&
                  engine.times_fired("texture-reference-cost") == 1);
        WindowSample readback = healthy();
        set_cost(readback, Cost::SurfaceReadback, 3000.0, 100, 40.0);
        check("a performance rule needs two windows",
              engine.close_window(readback, 35).empty() &&
                  fired(engine.close_window(readback, 40), "surface-readback"));
        check("sustain_windows: correctness 1, texture-reference-cost 3, others 2",
              sustain_windows("dropped-draws") == 1 && sustain_windows("texture-reference-cost") == 3 &&
                  sustain_windows("hle-blocking-wait") == 2);
    }

    // Summary forms: never evaluated prints nothing; evaluated and quiet says so.
    FILE* s = std::fopen(logp.c_str(), "w+");
    {
        EngineConfig config;
        config.log = nullptr;
        AlarmEngine engine(std::move(config));
        check("an engine that never closed a window prints no summary", !engine.write_summary(s));
        engine.close_window(healthy(), 5.0);
        check("an evaluated, quiet engine prints a summary", engine.write_summary(s));
    }
    std::fflush(s);
    check("...saying no rule fired", slurp(logp).find("no rule fired in 1 windows") != std::string::npos);
    std::fclose(s);
    std::remove(jsonl.c_str());
    std::remove(logp.c_str());
}

}  // namespace

int main() {
    test_quiet_baseline();
    test_texture_cache_thrash();
    test_surface_readback();
    test_texture_reference_cost();
    test_hle_blocking_wait();
    test_present_cpu_overhead();
    test_dropped_draws();
    test_threshold_scaling();
    test_budget_follows_flip_rate();
    test_cost_scope_nesting();
    test_texture_reference_sample();
    test_engine();
    check("rule_names lists the six phase-2 rules", rule_names().size() == 6);
    std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "ok", g_failures);
    return g_failures ? 1 : 0;
}
