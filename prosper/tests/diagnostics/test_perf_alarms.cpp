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
#include "diagnostics/transfer_pressure.hpp"
#include "gpu/diagnostics/draw_disposition.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
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

    // Boundary on kReadbackBudgetShare (25%): 40 flips of a 33.3 ms budget is 1333 ms.
    WindowSample edge = bad;
    set_cost(edge, Cost::SurfaceReadback, 1333.33 * 0.249, 65, 10.0);
    check("24.9% of budget is quiet", evaluate_rules(edge, kDefault).empty());
    set_cost(edge, Cost::SurfaceReadback, 1333.34 * 0.25, 65, 10.0);
    check("25.0% of budget fires", fired(evaluate_rules(edge, kDefault), "surface-readback"));

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

    // Boundary on kTextureReferenceNs (4 us), at a population large enough that share is not the gate.
    WindowSample edge = bad;
    set_texrefs(edge, 150000, 3.99);
    check("3.99 us/ref is quiet", evaluate_rules(edge, kDefault).empty());
    set_texrefs(edge, 150000, 4.0);
    check("4.00 us/ref fires", fired(evaluate_rules(edge, kDefault), "texture-reference-cost"));

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

    WindowSample edge = bad;
    set_cost(edge, Cost::PresentCpu, 0.99 * 600, 600, 1.2);
    check("0.99 ms/present is quiet", evaluate_rules(edge, kDefault).empty());
    set_cost(edge, Cost::PresentCpu, 1.0 * 600, 600, 1.2);
    check("1.00 ms/present fires", fired(evaluate_rules(edge, kDefault), "present-cpu-overhead"));

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


// ---- #3891 phase 3: per-site drop reasons ----------------------------------------------------

size_t reason_index(DropReason r) { return static_cast<size_t>(r); }

void test_drop_reasons() {
    std::puts("dropped-draws reasons");
    // The ledger hook: the reason AND the matching coarse counter, from one call, so the total the
    // rule reads and the breakdown it prints cannot disagree.
    Ledger& l = ledger();
    const auto counter = [&](Counter c) { return l.counters[static_cast<size_t>(c)].load(); };
    const auto reason = [&](DropReason r) { return l.drop_reasons[reason_index(r)].load(); };
    const uint64_t f0 = counter(Counter::DroppedDrawsFrontend);
    const uint64_t c0 = counter(Counter::DroppedDrawsContract);
    const uint64_t b0 = counter(Counter::DroppedDrawsBackend);
    const uint64_t v0 = reason(DropReason::VolumeNoRendererImage);
    const uint64_t k0 = reason(DropReason::ContractMismatch);
    const uint64_t s0 = reason(DropReason::BackendShaderRejected);
    drop_draw(DropReason::VolumeNoRendererImage);
    drop_draw(DropReason::VolumeNoRendererImage);
    drop_draw(DropReason::ContractMismatch);
    drop_draw(DropReason::BackendShaderRejected);
    check("drop_draw(volume-no-renderer-image) x2 counts two frontend drops with that reason",
          counter(Counter::DroppedDrawsFrontend) - f0 == 2 &&
              reason(DropReason::VolumeNoRendererImage) - v0 == 2);
    check("drop_draw(contract-mismatch) counts a CONTRACT drop, not a frontend-unresolved one",
          counter(Counter::DroppedDrawsContract) - c0 == 1 &&
              reason(DropReason::ContractMismatch) - k0 == 1);
    check("drop_draw(backend/shader-rejected) counts a BACKEND drop",
          counter(Counter::DroppedDrawsBackend) - b0 == 1 &&
              reason(DropReason::BackendShaderRejected) - s0 == 1);

    // Names: stable, unique, and the backend range mirrors draw_disposition's own names in order
    // (draw_disposition.cpp static_asserts the COUNT; this pins the MEANING of each slot).
    std::set<std::string> names;
    bool all_named = true;
    for (size_t i = 0; i < kDropReasonCount; ++i) {
        all_named &= kDropReasonNames[i] && *kDropReasonNames[i];
        if (kDropReasonNames[i]) names.insert(kDropReasonNames[i]);
    }
    check("every drop reason has a distinct non-empty name", all_named && names.size() == kDropReasonCount);
    bool mirrored = true;
    for (size_t i = 0; i < static_cast<size_t>(prosper::gpu::DrawDrop::Count); ++i)
        mirrored &= std::string(kDropReasonNames[reason_index(kFirstBackendDropReason) + i]) ==
                    std::string("backend/") +
                        prosper::gpu::draw_drop_name(static_cast<prosper::gpu::DrawDrop>(i));
    check("backend/* reasons are draw_disposition's reasons, slot for slot", mirrored);

    // The rule names the top sites. #3893's shape: ~2 frontend drops per flip from one site, plus a
    // rarer second site, so the ranking (not declaration order) is what the reader must see first.
    WindowSample w = healthy();
    set_count(w, Counter::DroppedDrawsFrontend, 270);
    w.drop_reasons[reason_index(DropReason::ArrayDepthView)] = 10;
    w.drop_reasons[reason_index(DropReason::VolumeNoRendererImage)] = 260;
    const auto a = evaluate_rules(w, kDefault);
    check("a window with reasons still fires dropped-draws alone", only(a, "dropped-draws"));
    check("detail names the reasons, largest first",
          !a.empty() && a[0].detail.find("reasons=volume-no-renderer-image:260,"
                                         "render-array-reject/depth-view:10") != std::string::npos);
    check("breakdown carries the same ranking for the JSONL",
          !a.empty() && a[0].breakdown.size() == 2 &&
              std::string(a[0].breakdown[0].first) == "volume-no-renderer-image" &&
              a[0].breakdown[0].second == 260);
    WindowSample none = healthy();
    set_count(none, Counter::DroppedDrawsBackend, 1);
    const auto b = evaluate_rules(none, kDefault);
    check("a drop with no recorded reason still fires, and says reasons=none",
          fired(b, "dropped-draws") && b[0].detail.find("reasons=none") != std::string::npos);

    uint64_t counts[5] = {1, 9, 0, 5, 7};
    const char* labels[5] = {"a", "b", "c", "d", "e"};
    const auto r = ranked(counts, labels, 5);
    check("ranked drops zeros and sorts by count", r.size() == 4 && r[0].second == 9 &&
                                                      r[3].second == 1);
    check("top_entries shows the top three and how many more",
          top_entries(r) == "b:9,e:7,d:5(+1 more)");
}

// ---- new rules ------------------------------------------------------------------------------

void test_skipped_dispatches() {
    std::puts("skipped-dispatches");
    Ledger& l = ledger();
    const auto skips = [&] { return l.counters[static_cast<size_t>(Counter::SkippedDispatches)].load(); };
    const auto reason = [&](DispatchSkip r) { return l.dispatch_skips[static_cast<size_t>(r)].load(); };
    const uint64_t n0 = skips(), d0 = reason(DispatchSkip::DescriptorContract);
    skip_dispatch(DispatchSkip::DescriptorContract);
    check("skip_dispatch counts the skip and its reason",
          skips() - n0 == 1 && reason(DispatchSkip::DescriptorContract) - d0 == 1);
    {
        const SuppressDispatchSkipCounting capture_re_realization;
        skip_dispatch(DispatchSkip::DescriptorContract);
    }
    check("a capture's re-realization does not count again", skips() - n0 == 1);
    skip_dispatch(DispatchSkip::MissingProgram);
    check("...and counting resumes after it", skips() - n0 == 2);

    // GTA V's missing world (#2481) was one declined compute program per frame.
    WindowSample bad = healthy();
    set_count(bad, Counter::SkippedDispatches, 150);
    bad.dispatch_skips[static_cast<size_t>(DispatchSkip::BackendDeclined)] = 150;
    const auto a = evaluate_rules(bad, kDefault);
    check("a window skipping one dispatch per flip fires skipped-dispatches alone",
          only(a, "skipped-dispatches"));
    check("...naming the reason", !a.empty() &&
                                      a[0].detail.find("reasons=backend-declined:150") != std::string::npos);
    WindowSample one = healthy();
    set_count(one, Counter::SkippedDispatches, 1);
    check("a single skip fires (correctness: any)", fired(evaluate_rules(one, kDefault), "skipped-dispatches"));
    check("...even at a lowered sensitivity", fired(evaluate_rules(one, RuleThresholds::scaled(1000)),
                                                    "skipped-dispatches"));
    check("no skips is quiet", evaluate_rules(healthy(), kDefault).empty());
    bool named = true;
    for (size_t i = 0; i < kDispatchSkipCount; ++i) named &= kDispatchSkipNames[i] && *kDispatchSkipNames[i];
    check("every dispatch-skip reason has a name", named);
}

void set_transfer(WindowSample& w, prosper::diagnostics::Transfer t, double mib_per_s) {
    w.transfer_bytes[static_cast<size_t>(t)] =
        static_cast<uint64_t>(mib_per_s * w.seconds * 1024.0 * 1024.0);
}

void test_host_copy_pressure() {
    std::puts("host-copy-pressure");
    using prosper::diagnostics::Transfer;
    // Sonic Frontiers on main (a6b9ee3f6): 1,427 MiB/s over the whole route -- detile 376 GB and
    // rtt-snapshot 77 GB in 318 s.
    WindowSample sonic = healthy();
    set_transfer(sonic, Transfer::Detile, 1183.0);
    set_transfer(sonic, Transfer::RenderTargetSnapshot, 241.0);
    set_transfer(sonic, Transfer::StorageMaterialize, 2.0);
    const auto a = evaluate_rules(sonic, kDefault);
    check("Sonic's 1,426 MiB/s host copy fires host-copy-pressure alone", only(a, "host-copy-pressure"));
    check("...naming detile as the largest site", !a.empty() &&
              a[0].detail.find("MiB-by-site=detile:") != std::string::npos &&
              std::string(a[0].breakdown[0].first) == "detile");
    // GTA V on main: ~80 MiB/s.
    WindowSample gta = healthy();
    set_transfer(gta, Transfer::StorageMaterialize, 32.0);
    set_transfer(gta, Transfer::RenderTargetSnapshot, 21.0);
    set_transfer(gta, Transfer::Detile, 27.0);
    check("GTA V's 80 MiB/s is quiet", evaluate_rules(gta, kDefault).empty());
    WindowSample under = healthy();
    set_transfer(under, Transfer::Detile, kHostCopyMiBPerSecond - 1.0);
    WindowSample over = healthy();
    set_transfer(over, Transfer::Detile, kHostCopyMiBPerSecond + 1.0);
    check("just under the threshold is quiet, just over fires",
          evaluate_rules(under, kDefault).empty() &&
              fired(evaluate_rules(over, kDefault), "host-copy-pressure"));
    check("...and the investigation setting lowers it",
          fired(evaluate_rules(gta, RuleThresholds::scaled(25)), "host-copy-pressure"));
}

void test_shader_compile() {
    std::puts("shader-compile");
    // A pipeline cache that never hits: 400 pipelines in 5 s at ~4 ms each while the title flips at
    // 30 Hz against a 30 Hz budget -- 1.6 s of compile per 5 s, 32% of the budget per flip.
    WindowSample thrash = healthy();
    set_cost(thrash, Cost::PipelineCreate, 1600.0, 400, 9.0);
    const auto a = evaluate_rules(thrash, kDefault);
    check("a steadily compiling window fires shader-compile alone", only(a, "shader-compile"));
    // Recompiles count toward the same budget.
    WindowSample mixed = healthy();
    set_cost(mixed, Cost::PipelineCreate, 800.0, 200, 9.0);
    set_cost(mixed, Cost::ShaderCompile, 800.0, 100, 30.0);
    check("recompiles and pipeline creation are summed",
          fired(evaluate_rules(mixed, kDefault), "shader-compile"));
    WindowSample mixed_half = healthy();
    set_cost(mixed_half, Cost::PipelineCreate, 800.0, 200, 9.0);
    check("...either half alone (16%) is quiet", evaluate_rules(mixed_half, kDefault).empty());
    // One enormous shader: much time, too few compiles to be a pattern.
    WindowSample one_big = healthy();
    set_cost(one_big, Cost::ShaderCompile, 2000.0, 40, 400.0);   // 8/s
    check("40 compiles in 5 s (below the 20/s floor) is quiet however long they take",
          evaluate_rules(one_big, kDefault).empty());
    // A healthy steady state compiles now and then.
    WindowSample steady = healthy();
    set_cost(steady, Cost::PipelineCreate, 60.0, 15, 8.0);
    check("15 pipelines in 5 s costing 60 ms is quiet", evaluate_rules(steady, kDefault).empty());
    // The tail of a cold-cache load: still expensive per flip (the title flips slowly while it
    // streams) but the compile COUNT has decayed -- Outer Wilds on main, 71 compiles in the sixth
    // window of its burst, 407 ms against 40 flips (61% of a 60 Hz budget).
    WindowSample tail = healthy();
    tail.flips = 40;
    tail.target_hz = 60;
    set_cost(tail, Cost::PipelineCreate, 393.0, 39, 19.0);
    set_cost(tail, Cost::ShaderCompile, 14.0, 32, 1.0);
    check("a decayed cold-cache tail (71 compiles, 61% of budget) is quiet on the count floor",
          evaluate_rules(tail, kDefault).empty());
    check("shader-compile needs eight sustained windows (cold-cache loads measured 3-5)",
          sustain_windows("shader-compile") == 8);
}

void test_sampler_seed() {
    std::puts("texture-reference sampler seed");
    // The one address whose term cancels the constant exactly. Without `| 1` this seed is 0, and
    // xorshift of 0 is 0 forever: every reference would be timed.
    const uintptr_t cancelling = static_cast<uintptr_t>(0x9e3779b9u) << 4;
    check("the seed is never zero, even when the address cancels the constant",
          texture_sample_seed(cancelling) != 0);
    check("...and still differs between threads' state addresses",
          texture_sample_seed(0x1000) != texture_sample_seed(0x2000));
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

    // Rate and slot coverage: 16,384 references laid out as draws of 8. A counter with period 32
    // would time slot 7 of every fourth draw and NEVER slots 0-6 (#3894 review); the sampler must
    // reach every slot at roughly 1 in 32.
    uint64_t refs0 = l.counters[refs].load();
    uint64_t samples0 = l.cost_events[ts].load();
    { TextureReferenceSample not_a_texture(false); not_a_texture.finish(); }
    constexpr uint64_t kRefs = 16384, kSlots = 8;
    uint64_t per_slot[kSlots] = {};
    for (uint64_t i = 0; i < kRefs; ++i) {
        const uint64_t before = l.cost_events[ts].load();
        TextureReferenceSample ref(true);
        ref.finish();
        if (l.cost_events[ts].load() != before) ++per_slot[i % kSlots];
    }
    flush_thread_texture_references();
    const uint64_t sampled = l.cost_events[ts].load() - samples0;
    check("every texture reference is counted, non-textures are not",
          l.counters[refs].load() - refs0 == kRefs);
    check("about one reference in kTextureRefSamplePeriod is timed (16384/32 = 512, within 25%)",
          sampled >= 384 && sampled <= 640);
    bool every_slot = true;
    for (uint64_t k = 0; k < kSlots; ++k) every_slot = every_slot && per_slot[k] >= 20;
    check("every slot of an 8-reference draw is sampled (no aliasing)", every_slot);

    // Nested readback: until two references have been sampled, each resolving through a 0.5 ms
    // readback. The readback time must go to surface-readback and NOT to the reference.
    samples0 = l.cost_events[ts].load();
    const uint64_t sample_ns0 = l.cost_ns[ts].load(), readback_ns0 = l.cost_ns[rb].load();
    uint64_t iterations = 0;
    while (l.cost_events[ts].load() - samples0 < 2 && iterations < 4000) {
        ++iterations;
        TextureReferenceSample ref(true);
        {
            CostScope readback(Cost::SurfaceReadback);
            const uint64_t until = now_ns() + 500'000;
            while (now_ns() < until) {}
        }
        ref.finish();
    }
    flush_thread_texture_references();
    const double readback_ms = (l.cost_ns[rb].load() - readback_ns0) / 1e6;
    const double sampled_ms = (l.cost_ns[ts].load() - sample_ns0) / 1e6;
    check("two references with a nested readback were sampled", l.cost_events[ts].load() - samples0 >= 2);
    check("the nested readback is charged to surface-readback", readback_ms >= 0.5 * iterations);
    check("...and subtracted from the sampled reference (one cause, one alarm)", sampled_ms < 0.5);
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
    // Unique per run: concurrent ctest invocations share TMPDIR.
    const std::string tag = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
        std::to_string(reinterpret_cast<uintptr_t>(&dir) & 0xffffff);
    const std::string jsonl = dir + "/test_perf_alarms_" + tag + ".jsonl";
    const std::string logp = dir + "/test_perf_alarms_" + tag + ".log";
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
                  sustain_windows("hle-blocking-wait") == 2 &&
                  sustain_windows("skipped-dispatches") == 1);
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
    const std::string quiet = slurp(logp);
    check("...saying no rule fired", quiet.find("no rule fired in 1 windows") != std::string::npos);
    check("...and listing the rules that had no data as NOT quiet",
          quiet.find("7 of 9 rules had data") != std::string::npos &&
              quiet.find("NO DATA (not measured in any window, so not quiet): "
                         "texture-reference-cost,present-cpu-overhead") != std::string::npos);
    std::fclose(s);
    std::remove(jsonl.c_str());
    std::remove(logp.c_str());
}


// The engine half of phase 3: per-window deltas of the reason arrays and of the external
// transfer totals, the JSONL breakdowns, the run-total breakdown in the summary, and the active
// set the --fps marker reads.
void test_engine_breakdowns() {
    std::puts("engine breakdowns");
    const std::string dir = std::getenv("TMPDIR") ? std::getenv("TMPDIR") : ".";
    const std::string tag = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()) + "_b" +
        std::to_string(reinterpret_cast<uintptr_t>(&dir) & 0xffffff);
    const std::string jsonl = dir + "/test_perf_alarms_" + tag + ".jsonl";
    const std::string logp = dir + "/test_perf_alarms_" + tag + ".log";
    FILE* log = std::fopen(logp.c_str(), "w+");
    {
        EngineConfig config;
        config.window_ns = 1'000'000'000ull;
        config.jsonl_path = jsonl;
        config.log = log;
        AlarmEngine engine(std::move(config));
        Ledger l;
        AlarmEngine::ExternalTotals ext;
        const size_t volume = static_cast<size_t>(DropReason::VolumeNoRendererImage);
        const size_t detile = static_cast<size_t>(prosper::diagnostics::Transfer::Detile);
        // Pre-baseline residue in both the reasons and the external totals.
        l.drop_reasons[volume] = 500;
        ext.transfer_bytes[detile] = 9ull << 30;
        uint64_t t = 1'000'000'000ull;
        engine.on_flip(t, l, 60, &ext);
        // Window 1: 2 drops from one site, 1 from another; 600 MiB detiled (600 MiB/s: fires only
        // after two windows).
        l.counters[static_cast<size_t>(Counter::DroppedDrawsFrontend)] += 3;
        l.drop_reasons[volume] += 2;
        l.drop_reasons[static_cast<size_t>(DropReason::ArrayShortBacking)] += 1;
        ext.transfer_bytes[detile] += 600ull << 20;
        t += 1'000'000'000ull;
        auto w1 = engine.on_flip(t, l, 60, &ext);
        check("window 1 reports dropped-draws with the WINDOW's reasons (2+1), not the residue",
              w1.size() == 1 && w1[0].breakdown.size() == 2 && w1[0].breakdown[0].second == 2);
        check("the active set is the rules reported in the last window",
              engine.active_rules().size() == 1 &&
                  std::string(engine.active_rules()[0]) == "dropped-draws");
        // Window 2: detile continues, no drops.
        ext.transfer_bytes[detile] += 600ull << 20;
        t += 1'000'000'000ull;
        auto w2 = engine.on_flip(t, l, 60, &ext);
        check("window 2: host-copy-pressure (sustained) fires from the external delta (600 MiB/s)",
              w2.size() == 1 && std::string(w2[0].rule) == "host-copy-pressure" &&
                  w2[0].value > 590 && w2[0].value < 610);
        check("...and the active set follows it", engine.active_rules().size() == 1 &&
                  std::string(engine.active_rules()[0]) == "host-copy-pressure");
        // Window 3: quiet.
        t += 1'000'000'000ull;
        check("a quiet window empties the active set",
              engine.on_flip(t, l, 60, &ext).empty() && engine.active_rules().empty());
        // Window 4: the same site again, to check the run total sums across windows.
        l.counters[static_cast<size_t>(Counter::DroppedDrawsFrontend)] += 4;
        l.drop_reasons[volume] += 4;
        t += 1'000'000'000ull;
        engine.on_flip(t, l, 60, &ext);
        std::rewind(log);
        engine.write_summary(log);
    }
    std::fflush(log);
    const std::string j = slurp(jsonl);
    check("the JSONL alarm record carries the breakdown object",
          j.find("\"breakdown\":{\"volume-no-renderer-image\":2,"
                 "\"render-array-reject/short-backing\":1}") != std::string::npos);
    check("every JSONL window carries drop_reasons, dispatch_skips and host copy by site",
          count_of(j, "\"drop_reasons\":{") == 4 && count_of(j, "\"dispatch_skips\":{}") == 4 &&
              j.find("\"host_copy_mib_by_site\":{\"detile\":600.0}") != std::string::npos &&
              j.find("\"drop_reasons\":{}") != std::string::npos);
    const std::string summary = slurp(logp);
    check("the exit summary sums a rule's breakdown over every window it fired in (2+4)",
          summary.find("summary rule=dropped-draws breakdown over fired windows: "
                       "volume-no-renderer-image:6,render-array-reject/short-backing:1") !=
              std::string::npos);
    std::fclose(log);
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
    test_drop_reasons();
    test_skipped_dispatches();
    test_host_copy_pressure();
    test_shader_compile();
    test_sampler_seed();
    test_threshold_scaling();
    test_budget_follows_flip_rate();
    test_cost_scope_nesting();
    test_texture_reference_sample();
    test_engine();
    test_engine_breakdowns();
    check("rule_names lists the six phase-2 rules and the three added with phase 3",
          rule_names().size() == 9);
    std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "ok", g_failures);
    return g_failures ? 1 : 0;
}
