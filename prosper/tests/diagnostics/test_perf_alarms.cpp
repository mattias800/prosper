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
#include "hle/dispatch/dispatch.hpp"

#include <algorithm>
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

// What one window REPORTS once every rule's sustain has held: the candidates after the engine's
// reporting deferrals. evaluate_rules alone returns every candidate, deferred or not, because a
// deferred rule must still count its streak.
std::vector<AlarmFiring> reported(const WindowSample& w, const RuleThresholds& t = kDefault) {
    std::vector<AlarmFiring> out = evaluate_rules(w, t);
    apply_reporting_deferrals(out);
    return out;
}

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

    // The backend outcome bracket: a refusal counts, a deliberate decline (the selector) does not.
    {
        const uint64_t before = skips(), declined = reason(DispatchSkip::BackendDeclined);
        const BackendDispatchOutcome refused;
        const bool refused_deliberate = refused.finish(false);
        check("a backend refusal counts backend-declined",
              !refused_deliberate && skips() - before == 1 &&
                  reason(DispatchSkip::BackendDeclined) - declined == 1);
        const BackendDispatchOutcome selector;
        note_deliberate_dispatch_decline();
        check("a deliberate decline inside the call is not counted, and is reported as deliberate",
              selector.finish(false) && skips() - before == 1);
        const BackendDispatchOutcome ran;
        check("an executed dispatch is not counted", !ran.finish(true) && skips() - before == 1);
    }

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

void test_gpu_memory_off_device() {
    std::puts("gpu-memory-off-device");
    // The ledger hook: counts, bytes and the named per-class slot move together.
    Ledger& l = ledger();
    const auto n = [&](Counter c) { return l.counters[static_cast<size_t>(c)].load(); };
    const uint64_t off0 = n(Counter::GpuMemoryOffDevice), bytes0 = n(Counter::GpuMemoryOffDeviceBytes);
    const uint64_t slot0 = l.gpu_memory_off_device[3].load();
    note_gpu_memory_off_device(3, "depth-target", 4096);
    check("note_gpu_memory_off_device counts the placement, its bytes and its class",
          n(Counter::GpuMemoryOffDevice) - off0 == 1 &&
              n(Counter::GpuMemoryOffDeviceBytes) - bytes0 == 4096 &&
              l.gpu_memory_off_device[3].load() - slot0 == 1 &&
              std::strcmp(l.gpu_memory_class_names[3].load(), "depth-target") == 0);
    note_gpu_memory_off_device(kGpuMemoryClassSlots, "out-of-range", 1);
    check("an out-of-range slot is ignored, not written past the array",
          n(Counter::GpuMemoryOffDevice) - off0 == 1);

    // An 8 GB card running out of VRAM mid-scene: 40 depth targets and 12 sampled textures retried
    // on a host type in one window.
    WindowSample bad = healthy();
    set_count(bad, Counter::GpuMemoryOffDevice, 52);
    set_count(bad, Counter::GpuMemoryOffDeviceBytes, 52ull * 8 * 1024 * 1024);
    set_count(bad, Counter::GpuMemoryFallbacks, 52);
    bad.gpu_memory_off_device[3] = 40;
    bad.gpu_memory_class_names[3] = "depth-target";
    bad.gpu_memory_off_device[6] = 12;
    bad.gpu_memory_class_names[6] = "sampled-texture";
    const auto a = evaluate_rules(bad, kDefault);
    check("off-device placements fire gpu-memory-off-device alone", only(a, "gpu-memory-off-device"));
    check("...naming the classes, largest first, the bytes and the fallbacks",
          !a.empty() &&
              a[0].detail.find("classes=depth-target:40,sampled-texture:12") != std::string::npos &&
              a[0].detail.find("bytes=416.0MiB") != std::string::npos &&
              a[0].detail.find("oom-fallbacks=52") != std::string::npos &&
              a[0].breakdown.size() == 2);
    WindowSample one = healthy();
    set_count(one, Counter::GpuMemoryOffDevice, 1);
    check("a single placement fires (any)", fired(evaluate_rules(one, kDefault), "gpu-memory-off-device"));
    check("...even at a lowered sensitivity",
          fired(evaluate_rules(one, RuleThresholds::scaled(1000)), "gpu-memory-off-device"));
    WindowSample fallback_on_device = healthy();
    set_count(fallback_on_device, Counter::GpuMemoryFallbacks, 5);
    check("a fallback that stayed device-local is quiet (only off-device placements alarm)",
          evaluate_rules(fallback_on_device, kDefault).empty());
    check("no placements is quiet", evaluate_rules(healthy(), kDefault).empty());
    check("it is a correctness-class rule: reported on the first window",
          sustain_windows("gpu-memory-off-device") == 1);
}

// ---- #3891 queue rules (2026-09-28) -----------------------------------------------------------

void test_unaccounted_draws() {
    std::puts("unaccounted-draws");
    WindowSample bad = healthy();
    set_count(bad, Counter::DrawsUnaccounted, 18);   // Sonic Frontiers' exit census, 2026-09-27
    const auto a = evaluate_rules(bad, kDefault);
    check("an unaccounted draw fires unaccounted-draws alone", only(a, "unaccounted-draws"));
    check("...naming the count", !a.empty() && a[0].detail.find("unaccounted=18") != std::string::npos);
    WindowSample one = healthy();
    set_count(one, Counter::DrawsUnaccounted, 1);
    check("one is enough (correctness), even at lowered sensitivity",
          fired(evaluate_rules(one, RuleThresholds::scaled(1000)), "unaccounted-draws"));
    WindowSample dropped = healthy();
    set_count(dropped, Counter::DroppedDrawsBackend, 3);
    check("named drops alone do not raise unaccounted-draws",
          !fired(evaluate_rules(dropped, kDefault), "unaccounted-draws"));
    check("correctness sustain", sustain_windows("unaccounted-draws") == 1);
}

void test_unimplemented_hle_calls() {
    std::puts("unimplemented-hle-calls");
    // The ledger hook, through the real dispatcher: two imports, three calls.
    Ledger& l = ledger();
    const auto n = [&](Counter c) { return l.counters[static_cast<size_t>(c)].load(); };
    const uint64_t calls0 = n(Counter::HleUnimplementedCalls), first0 = n(Counter::HleUnimplementedFirst);
    static const std::vector<prosper::ImportSlot> slots = {{"libSceTest", "AAAAAAAAAAA"},
                                                            {"libSceTest", "BBBBBBBBBBB"}};
    prosper::dispatch_init(&slots, nullptr);
    prosper::prosper_on_unimpl(0);
    prosper::prosper_on_unimpl(0);
    prosper::prosper_on_unimpl(1);
    check("prosper_on_unimpl counts every call and each import's first call",
          n(Counter::HleUnimplementedCalls) - calls0 == 3 &&
              n(Counter::HleUnimplementedFirst) - first0 == 2);
    prosper::dispatch_init(nullptr, nullptr);

    WindowSample bad = healthy();
    set_count(bad, Counter::HleUnimplementedFirst, 1);
    set_count(bad, Counter::HleUnimplementedCalls, 40);
    const auto a = evaluate_rules(bad, kDefault);
    check("a new unregistered NID after boot fires unimplemented-hle-calls alone",
          only(a, "unimplemented-hle-calls"));
    check("...with the new count and the window's calls",
          !a.empty() && a[0].detail.find("new-unimplemented=1 calls=40") != std::string::npos);
    WindowSample repeat = healthy();
    set_count(repeat, Counter::HleUnimplementedCalls, 400);
    check("repeated calls to an ALREADY-seen unregistered NID stay quiet (it was reported once)",
          evaluate_rules(repeat, kDefault).empty());
    check("correctness sustain", sustain_windows("unimplemented-hle-calls") == 1);
}

void test_diagnostic_path_active() {
    std::puts("diagnostic-path-active");
    WindowSample bad = healthy();
    set_gauge(bad, Gauge::DiagnosticPathSwitches,
              (1ull << static_cast<unsigned>(DiagnosticPathSwitch::NoLiveTargets)) |
                  (1ull << static_cast<unsigned>(DiagnosticPathSwitch::PerPassPixelDump)));
    const auto a = evaluate_rules(bad, kDefault);
    check("a diagnostic switch that turned live targets off fires diagnostic-path-active alone",
          only(a, "diagnostic-path-active"));
    check("...naming every switch",
          !a.empty() &&
              a[0].detail.find("switches=PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS,per-pass-pixel-dump") !=
                  std::string::npos);
    check("the production path (gauge 0) is quiet", evaluate_rules(healthy(), kDefault).empty());
    check("state rule: first window", sustain_windows("diagnostic-path-active") == 1);
}

void test_present_path_fallback() {
    std::puts("present-path-fallback");
    WindowSample bad = healthy();
    set_count(bad, Counter::PresentCpuFallbacks, 140);
    set_cost(bad, Cost::PresentCpu, 1.0, 10, 0.2);   // 10 GPU scanouts, cheap
    const auto a = evaluate_rules(bad, kDefault);
    check("most presents from the CPU fallback fires present-path-fallback alone",
          only(a, "present-path-fallback"));
    check("...with both counts", !a.empty() &&
              a[0].detail.find("cpu-fallback=140 gpu-scanout=10") != std::string::npos);
    // #3915: the renderer's decline reasons ride in the detail and the breakdown, largest first; a
    // window with none recorded says so rather than printing an empty list.
    check("...and says when no renderer decline was recorded", !a.empty() &&
              a[0].detail.find("declines=none-recorded") != std::string::npos);
    WindowSample named = bad;
    named.present_declines[6] = 686; named.present_decline_names[6] = "no-render-target";
    named.present_declines[12] = 3; named.present_decline_names[12] = "compute-scanout-stale";
    const auto n = evaluate_rules(named, kDefault);
    check("...and names the renderer's decline reasons, largest first", !n.empty() &&
              n[0].detail.find("declines=no-render-target:686,compute-scanout-stale:3") !=
                  std::string::npos &&
              n[0].breakdown.size() == 2);
    {
        using namespace prosper::diagnostics::perf;
        const uint64_t total = ledger().counters[static_cast<size_t>(Counter::PresentGpuDeclines)];
        const uint64_t slot = ledger().present_declines[6];
        note_present_decline(6, "no-render-target");
        note_present_decline(kPresentDeclineSlots, "out-of-range");   // ignored, not a crash
        check("note_present_decline bumps the total and its reason together",
              ledger().counters[static_cast<size_t>(Counter::PresentGpuDeclines)] == total + 1 &&
                  ledger().present_declines[6] == slot + 1);
    }
    WindowSample startup = healthy();
    set_count(startup, Counter::PresentCpuFallbacks, 40);    // 8/s, before the first GPU publish
    set_cost(startup, Cost::PresentCpu, 10.0, 130, 0.2);
    check("a minority of fallback presents (startup, 23%, above the rate floor) is quiet",
          evaluate_rules(startup, kDefault).empty());
    WindowSample few = healthy();
    set_count(few, Counter::PresentCpuFallbacks, 20);         // 4/s, all fallback
    check("below the 5/s floor is quiet", evaluate_rules(few, kDefault).empty());
    WindowSample none = healthy();
    check("no presents of either kind: no data", !rule_has_data("present-path-fallback", none));
    set_count(none, Counter::PresentCpuFallbacks, 1);
    check("a fallback present is data", rule_has_data("present-path-fallback", none));
    check("performance sustain (2)", sustain_windows("present-path-fallback") == 2);
}

void test_pipeline_cache_thrash() {
    std::puts("pipeline-cache-thrash");
    WindowSample bad = healthy();
    set_count(bad, Counter::PipelineEvictions, 60);
    set_count(bad, Counter::DescriptorSetLayoutEvictions, 5);
    const auto a = evaluate_rules(bad, kDefault);
    check("13 evictions/s fires pipeline-cache-thrash alone", only(a, "pipeline-cache-thrash"));
    check("...by cache, largest first",
          !a.empty() && a[0].detail.find("by-cache=pipeline:60,descriptor-set-layout:5") !=
                            std::string::npos);
    WindowSample layouts = healthy();
    set_count(layouts, Counter::DescriptorSetLayoutEvictions, 30);
    set_count(layouts, Counter::PipelineLayoutEvictions, 30);
    check("layout caches alone count too (12 evictions/s)",
          fired(evaluate_rules(layouts, kDefault), "pipeline-cache-thrash"));
    WindowSample dsl = healthy();
    set_count(dsl, Counter::DescriptorSetLayoutEvictions, 40);
    check("the descriptor-set-layout cache alone counts (8/s)",
          fired(evaluate_rules(dsl, kDefault), "pipeline-cache-thrash"));
    WindowSample churn = healthy();
    set_count(churn, Counter::PipelineLayoutEvictions, 20);   // 4/s: a scene change
    check("4 evictions/s is quiet", evaluate_rules(churn, kDefault).empty());
    check("...but fires at a quarter of the threshold",
          fired(evaluate_rules(churn, RuleThresholds::scaled(25)), "pipeline-cache-thrash"));
}

void test_texture_validation_churn() {
    std::puts("texture-validation-churn");
    // The counter carries bytes the compare READ, not texture sizes. A source whose change sits
    // near its end: 150 failures, each reading ~8 MiB before finding it -> 240 MiB/s.
    WindowSample bad = healthy();
    set_count(bad, Counter::TextureValidationFailures, 150);
    set_count(bad, Counter::TextureValidationFailedBytes, 150ull * 8 * 1024 * 1024);
    const auto a = evaluate_rules(bad, kDefault);
    check("240 MiB/s of bytes read by failed validations fires texture-validation-churn alone",
          only(a, "texture-validation-churn"));
    check("...with the count and bytes", !a.empty() &&
              a[0].detail.find("failed-validations=150 validated=1200MiB") != std::string::npos);
    // A movie plane rewritten wholesale every frame: 8 MiB textures, but each failed compare stops
    // at its first 64 KiB chunk. 150 failures read ~9.4 MiB, not 1.2 GiB: quiet.
    WindowSample movie = healthy();
    set_count(movie, Counter::TextureValidationFailures, 150);
    set_count(movie, Counter::TextureValidationFailedBytes, 150ull * 64 * 1024);
    check("wholesale rewrites (each compare stops at its first chunk) are quiet",
          evaluate_rules(movie, kDefault).empty());
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
    // 47.5 MiB per flip at 30 fps: both forms hold. Both are CANDIDATES (so both streaks count),
    // and only the per-flip one is REPORTED (one cause, one line).
    const auto candidates = evaluate_rules(sonic, kDefault);
    check("Sonic's 1,426 MiB/s at 30 fps: both host-copy forms are candidates",
          candidates.size() == 2 && fired(candidates, "host-copy-per-flip") &&
              fired(candidates, "host-copy-pressure"));
    const auto a = reported(sonic);
    check("...and host-copy-per-flip alone is reported (the rate form defers at report time)",
          only(a, "host-copy-per-flip"));
    // The same bytes over too few flips to form a per-frame figure (a stall): the rate form is
    // the only one that can see it, and it names the site.
    WindowSample stalled = sonic;
    stalled.flips = kHostCopyPerFlipMinFlips - 1;
    const auto st = evaluate_rules(stalled, kDefault);
    check("...but over fewer flips than the per-flip floor, host-copy-pressure fires alone",
          only(st, "host-copy-pressure"));
    check("...naming detile as the largest site", !st.empty() &&
              st[0].detail.find("MiB-by-site=detile:") != std::string::npos &&
              std::string(st[0].breakdown[0].first) == "detile");
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

void set_transfer_per_flip(WindowSample& w, prosper::diagnostics::Transfer t, double mib_per_flip,
                           uint64_t calls) {
    const size_t i = static_cast<size_t>(t);
    w.transfer_bytes[i] = static_cast<uint64_t>(mib_per_flip * w.flips * 1024.0 * 1024.0);
    w.transfer_calls[i] = calls;
}

// A window at `fps` over 5 s against a 30 Hz target.
WindowSample at_fps(double fps) {
    WindowSample w = healthy();
    w.flips = static_cast<uint64_t>(fps * w.seconds);
    return w;
}

void test_host_copy_per_flip() {
    std::puts("host-copy-per-flip");
    using prosper::diagnostics::Transfer;
    // GTA V's heavy regime on unmodified code (#3926), as measured on main 2026-09-29 (run 2 of 3,
    // t=151 s): 27.1 MiB per flip at 7 fps -- storage-materialize 2.23 MiB/call, rtt-snapshot 3.52,
    // detile 0.23. 190 MiB/s: far UNDER host-copy-pressure's 256 MiB/s, which never fired on it.
    WindowSample heavy = at_fps(7.0);                     // 35 flips
    set_transfer_per_flip(heavy, Transfer::StorageMaterialize, 12.3, 193);
    set_transfer_per_flip(heavy, Transfer::RenderTargetSnapshot, 10.5, 105);
    set_transfer_per_flip(heavy, Transfer::Detile, 4.3, 648);
    const auto a = evaluate_rules(heavy, kDefault);
    check("GTA V's heavy regime (27.1 MiB/flip at 7 fps) fires host-copy-per-flip alone",
          only(a, "host-copy-per-flip"));
    check("...value is MiB per flip", !a.empty() && a[0].value > 27.0 && a[0].value < 27.2);
    check("...naming the top sites with MiB/flip and MiB per call, largest first",
          !a.empty() &&
              a[0].detail.find("sites=storage-materialize:12.3MiB/flip@2.23MiB/call(193 calls),"
                               "rtt-snapshot:10.5MiB/flip@3.50MiB/call(105 calls),"
                               "detile:4.3MiB/flip@0.23MiB/call(648 calls)") != std::string::npos);
    check("...and the per-second figure the rate form would have read", !a.empty() &&
              a[0].detail.find("rate=190MiB/s") != std::string::npos);
    // The light regime of the same route and binary (run 3, t=129 s): the same sites and similar
    // call counts at half the bytes per call, 13.3 MiB/flip at 10.4 fps -- quiet.
    WindowSample light = healthy();
    light.flips = 52;
    set_transfer_per_flip(light, Transfer::StorageMaterialize, 5.40, 288);
    set_transfer_per_flip(light, Transfer::RenderTargetSnapshot, 3.64, 156);
    set_transfer_per_flip(light, Transfer::Detile, 4.27, 964);
    check("GTA V's light regime (13.3 MiB/flip) is quiet", evaluate_rules(light, kDefault).empty());
    // Frame-rate independence -- the reason this rule exists: the SAME per-flip cost at four frame
    // rates gets the same verdict, while the per-second figure crosses 256 MiB/s between them.
    bool same_verdict = true;
    for (double fps : {5.0, 9.0, 30.0, 60.0}) {
        WindowSample w = at_fps(fps);
        set_transfer_per_flip(w, Transfer::StorageMaterialize, 27.0, 100);
        same_verdict &= only(reported(w), "host-copy-per-flip");
    }
    check("27 MiB/flip fires host-copy-per-flip alone at 5, 9, 30 and 60 fps", same_verdict);
    bool quiet_everywhere = true;
    for (double fps : {5.0, 9.0, 30.0, 60.0}) {
        WindowSample w = at_fps(fps);
        set_transfer_per_flip(w, Transfer::StorageMaterialize, 1.0, 100);
        quiet_everywhere &= !fired(evaluate_rules(w, kDefault), "host-copy-per-flip");
    }
    check("1 MiB/flip is quiet at every frame rate", quiet_everywhere);
    WindowSample under = at_fps(9.0);
    set_transfer_per_flip(under, Transfer::RenderTargetSnapshot, kHostCopyMiBPerFlip * 0.98, 10);
    WindowSample over = at_fps(9.0);
    set_transfer_per_flip(over, Transfer::RenderTargetSnapshot, kHostCopyMiBPerFlip * 1.02, 10);
    check("just under the per-flip threshold is quiet, just over fires",
          evaluate_rules(under, kDefault).empty() &&
              only(evaluate_rules(over, kDefault), "host-copy-per-flip"));
    // The flip floor: a window with too few flips has no per-frame figure. It is NO DATA for this
    // rule (the rate form covers it), not a quiet window.
    WindowSample few = healthy();
    few.flips = kHostCopyPerFlipMinFlips - 1;
    set_transfer_per_flip(few, Transfer::StorageMaterialize, 40.0, 10);   // 72 MiB/s
    check("below the flip floor: quiet, and no data",
          !fired(evaluate_rules(few, kDefault), "host-copy-per-flip") &&
              !rule_has_data("host-copy-per-flip", few));
    // Sonic Frontiers' 3 fps gameplay tail (2026-09-29): 16 flips per window, two 4K rtt-snapshots
    // per flip. Above the floor, so it fires -- and at 203 MiB/s the rate form would have missed it.
    WindowSample tail = healthy();
    tail.flips = 16;
    set_transfer_per_flip(tail, Transfer::RenderTargetSnapshot, 63.3, 32);
    const auto st = evaluate_rules(tail, kDefault);
    check("Sonic's 3 fps tail (16 flips, 63 MiB/flip, 203 MiB/s) fires host-copy-per-flip alone",
          only(st, "host-copy-per-flip") &&
              st[0].detail.find("rtt-snapshot:63.3MiB/flip@31.65MiB/call(32 calls)") !=
                  std::string::npos);
    check("at the floor with nothing copied: data (the healthy answer)",
          rule_has_data("host-copy-per-flip", at_fps(kHostCopyPerFlipMinFlips / 5.0)));
    // A fast title copying a little per frame: the per-flip form is quiet and the rate form still
    // reports the absolute cost (120 fps x 3 MiB = 360 MiB/s).
    WindowSample fast = at_fps(120.0);
    set_transfer_per_flip(fast, Transfer::Detile, 3.0, 600);
    check("a small per-flip copy at a high frame rate is left to host-copy-pressure",
          only(reported(fast), "host-copy-pressure"));
    check("apply_reporting_deferrals keeps host-copy-pressure when host-copy-per-flip is absent",
          [] {
              std::vector<AlarmFiring> v(1);
              v[0].rule = "host-copy-pressure";
              apply_reporting_deferrals(v);
              return v.size() == 1;
          }());
    check("...host_copy_per_flip_holds agrees with the rule on each window",
          host_copy_per_flip_holds(heavy, kDefault) && !host_copy_per_flip_holds(fast, kDefault) &&
              !host_copy_per_flip_holds(few, kDefault) && !host_copy_per_flip_holds(under, kDefault));
    check("the investigation setting lowers the per-flip threshold",
          fired(evaluate_rules(under, RuleThresholds::scaled(50)), "host-copy-per-flip"));
    check("performance sustain (2)", sustain_windows("host-copy-per-flip") == 2);
}

void test_present_slot_trouble() {
    std::puts("present-slot-trouble");
    // Slot indices are GpuPresentOutcome values (PublishFailed = 11, ComputeScanoutUnwatched = 13);
    // the rule matches by NAME, so an index is only where the recorder happened to put it.
    WindowSample bad = healthy();                        // 150 flips
    set_cost(bad, Cost::PresentCpu, 10.0, 120, 0.2);
    bad.present_declines[11] = 30; bad.present_decline_names[11] = "publish-failed";
    bad.present_declines[6] = 5; bad.present_decline_names[6] = "no-render-target";
    set_count(bad, Counter::PresentGpuDeclines, 35);
    const auto a = evaluate_rules(bad, kDefault);
    check("publish-failed on 20% of flips fires present-slot-trouble alone",
          only(a, "present-slot-trouble"));
    check("...counting only the slot reasons, and naming them",
          !a.empty() && a[0].detail.find("slot-declines=30 flips=150 all-declines=35") !=
                            std::string::npos &&
              a[0].detail.find("reasons=publish-failed:30") != std::string::npos &&
              a[0].breakdown.size() == 1);
    WindowSample unwatched = healthy();
    unwatched.present_declines[13] = 20;
    unwatched.present_decline_names[13] = "compute-scanout-unwatched";
    check("compute-scanout-unwatched on 13% of flips fires it too",
          only(evaluate_rules(unwatched, kDefault), "present-slot-trouble"));
    WindowSample both = healthy();
    both.present_declines[11] = 9; both.present_decline_names[11] = "publish-failed";
    both.present_declines[13] = 9; both.present_decline_names[13] = "compute-scanout-unwatched";
    check("the two reasons add (6% + 6% = 12%)",
          only(evaluate_rules(both, kDefault), "present-slot-trouble") &&
              present_slot_trouble_declines(both) == 18);
    WindowSample rare = healthy();
    rare.present_declines[11] = 15; rare.present_decline_names[11] = "publish-failed";
    check("exactly 10% of flips is quiet (the condition is strictly over)",
          evaluate_rules(rare, kDefault).empty());
    // A frame that is ineligible for GPU present is not slot trouble, however many there are.
    WindowSample ineligible = healthy();
    ineligible.present_declines[6] = 686; ineligible.present_decline_names[6] = "no-render-target";
    ineligible.present_declines[12] = 90;
    ineligible.present_decline_names[12] = "compute-scanout-stale";
    check("other decline reasons (no-render-target, compute-scanout-stale) are quiet",
          evaluate_rules(ineligible, kDefault).empty() &&
              present_slot_trouble_declines(ineligible) == 0);
    WindowSample few = healthy();
    few.flips = kPresentSlotTroubleMinFlips - 1;
    few.present_declines[11] = few.flips; few.present_decline_names[11] = "publish-failed";
    check("below the flip floor is quiet", evaluate_rules(few, kDefault).empty());
    WindowSample none = healthy();
    check("no GPU presents and no declines: no data", !rule_has_data("present-slot-trouble", none));
    set_count(none, Counter::PresentGpuDeclines, 1);
    check("a decline is data", rule_has_data("present-slot-trouble", none));
    check("performance sustain (2)", sustain_windows("present-slot-trouble") == 2);
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
          quiet.find("15 of 23 rules had data") != std::string::npos &&
              quiet.find("NO DATA (not measured in any window, so not quiet): "
                         "texture-reference-cost,present-cpu-overhead,present-path-fallback,"
                         "present-slot-trouble,color-target-count-ceiling,gpu-sync-wait,"
                         "gpu-present-stalled,gpu-device-time-coverage") != std::string::npos);
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

// host-copy-per-flip through the engine: the per-window deltas of the external CALL totals (bytes
// per call must come from the window, not the process lifetime), the flip count as denominator,
// and the JSONL fields that keep both denominators visible.
void test_engine_host_copy_per_flip() {
    std::puts("engine host-copy-per-flip");
    const std::string dir = std::getenv("TMPDIR") ? std::getenv("TMPDIR") : ".";
    const std::string tag = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()) + "_f" +
        std::to_string(reinterpret_cast<uintptr_t>(&dir) & 0xffffff);
    const std::string jsonl = dir + "/test_perf_alarms_" + tag + ".jsonl";
    std::vector<AlarmFiring> w1, w2;
    {
        EngineConfig config;
        config.window_ns = 1'000'000'000ull;
        config.jsonl_path = jsonl;
        config.log = nullptr;
        AlarmEngine engine(std::move(config));
        Ledger l;
        AlarmEngine::ExternalTotals ext;
        const size_t storage = static_cast<size_t>(prosper::diagnostics::Transfer::StorageMaterialize);
        // Pre-baseline residue: a huge lifetime total at a very different bytes/call.
        ext.transfer_bytes[storage] = 50ull << 30;
        ext.transfer_calls[storage] = 10;
        uint64_t t = 1'000'000'000ull;
        engine.on_flip(t, l, 30, &ext);
        const uint64_t flip_ns = 1'000'000'000ull / 30;
        const auto run_window = [&](std::vector<AlarmFiring>& out) {
            // 30 flips in the window, each copying 24 MiB in 12 calls of 2 MiB (~720 MiB/s).
            for (int f = 0; f < 30; ++f) {
                ext.transfer_bytes[storage] += 24ull << 20;
                ext.transfer_calls[storage] += 12;
                t += flip_ns + 1;
                auto fired_now = engine.on_flip(t, l, 30, &ext);
                if (!fired_now.empty() || f == 29) out = std::move(fired_now);
            }
        };
        run_window(w1);
        run_window(w2);
    }
    check("window 1 holds the condition but is not reported (sustain 2)", w1.empty());
    check("window 2 reports host-copy-per-flip alone -- host-copy-pressure (~720 MiB/s) defers",
          w2.size() == 1 && std::string(w2[0].rule) == "host-copy-per-flip" &&
              w2[0].value > 23.0 && w2[0].value < 25.0);
    check("...with bytes per call from the WINDOW's calls (2 MiB), not the lifetime residue",
          !w2.empty() && w2[0].detail.find("storage-materialize:") != std::string::npos &&
              w2[0].detail.find("@2.00MiB/call") != std::string::npos);
    const std::string j = slurp(jsonl);
    check("every JSONL window carries the per-flip and per-second figures and calls by site",
          count_of(j, "\"host_copy_mib_per_flip\":") == 2 &&
              count_of(j, "\"host_copy_mib_per_s\":") == 2 &&
              count_of(j, "\"host_copy_calls_by_site\":{\"storage-materialize\":") == 2 &&
              count_of(j, "\"present_slot_trouble_declines\":0") == 2);
    std::remove(jsonl.c_str());
}

// The review blocker on #3928: a deferral decided at EVALUATION withheld host-copy-pressure's
// candidate whenever the per-flip form held, and the engine resets a rule's streak when it has no
// candidate. A per-flip value alternating across the threshold (25 / 15 MiB per flip at 30 fps --
// 750 / 450 MiB/s, both far above 256) then reset BOTH streaks every window and neither rule ever
// printed, where main reported host-copy-pressure from the second window. The deferral is a
// reporting decision only.
void test_engine_host_copy_alternating() {
    std::puts("engine host-copy deferral");
    using prosper::diagnostics::Transfer;
    const auto window = [](double mib_per_flip) {
        WindowSample w = healthy();              // 150 flips in 5 s
        set_transfer_per_flip(w, Transfer::StorageMaterialize, mib_per_flip, 1000);
        return w;
    };
    {
        EngineConfig config;
        config.log = nullptr;
        AlarmEngine engine(std::move(config));
        size_t reported_windows = 0, pressure = 0, per_flip = 0;
        for (int i = 0; i < 8; ++i) {
            const auto r = engine.close_window(window(i % 2 ? 15.0 : 25.0), 5.0 * (i + 1));
            reported_windows += !r.empty();
            pressure += fired(r, "host-copy-pressure");
            per_flip += fired(r, "host-copy-per-flip");
        }
        check("a per-flip value alternating 25/15 MiB/flip at 450-750 MiB/s is still reported",
              reported_windows > 0);
        check("...by host-copy-pressure in every window from the second (7 of 8), as on main",
              pressure == 7 && per_flip == 0);
    }
    {
        EngineConfig config;
        config.log = nullptr;
        AlarmEngine engine(std::move(config));
        std::vector<AlarmFiring> r1 = engine.close_window(window(25.0), 5.0);
        std::vector<AlarmFiring> r2 = engine.close_window(window(25.0), 10.0);
        check("a steady 25 MiB/flip reports host-copy-per-flip alone from the second window",
              r1.empty() && only(r2, "host-copy-per-flip"));
        // The per-flip value drops under threshold in window 3 at the same high rate: the rate
        // form's streak kept counting through the deferred windows, so it reports at once.
        std::vector<AlarmFiring> r3 = engine.close_window(window(15.0), 15.0);
        check("...and when it dips under, host-copy-pressure reports in that same window "
              "(its streak counted while deferred)", only(r3, "host-copy-pressure"));
        check("...deferred windows are not counted as fired",
              engine.times_fired("host-copy-pressure") == 1);
    }
}


// rtt-destination-refused (#3891, 2026-09-29): GTA V's heavy host-copy regime refuses one
// 2560x1440 RGBA8 result (14.06 MiB) destination-creation-refused on every other flip (#3873);
// the light regime refuses nothing.
void add_refusal(WindowSample& w, size_t slot, const char* name, uint64_t n, uint64_t bytes_each) {
    w.rtt_destination_refusal_names[slot] = name;
    w.rtt_destination_refused_bytes[slot] += n * bytes_each;
    w.rtt_destination_refusals[slot] += n;
    w.counters[static_cast<size_t>(Counter::RttDestinationRefusals)] += n;
    w.counters[static_cast<size_t>(Counter::RttDestinationRefusedBytes)] += n * bytes_each;
}

void test_rtt_destination_refused() {
    std::puts("rtt-destination-refused");
    constexpr uint64_t kResult = 2560ull * 1440 * 4;   // 14.06 MiB
    constexpr size_t kCreation = 18, kExtent = 15;
    WindowSample heavy = healthy();
    heavy.flips = 40;   // 8 fps
    add_refusal(heavy, kCreation, "destination-creation-refused", 20, kResult);
    const auto a = evaluate_rules(heavy, kDefault);
    check("GTA V's heavy regime (14 MiB every other flip, ~7 MiB/flip) fires "
          "rtt-destination-refused alone", only(a, "rtt-destination-refused"));
    check("...value is MiB per flip", !a.empty() && a[0].value > 7.0 && a[0].value < 7.1);
    check("...and the line names the reason and bytes per result",
          !a.empty() && a[0].detail.find("destination-creation-refused:281") != std::string::npos &&
              a[0].detail.find("(14.06MiB/result)") != std::string::npos);
    check("...breakdown carries MiB by reason for the JSONL",
          !a.empty() && a[0].breakdown.size() == 1 && a[0].breakdown[0].second == 281);
    check("the light regime (no refusals) is quiet", evaluate_rules(healthy(), kDefault).empty());
    {
        WindowSample edge = healthy();   // 150 flips
        add_refusal(edge, kExtent, "extent-mismatch", 1, 599ull << 20);   // 3.99 MiB/flip
        check("3.99 MiB/flip is quiet", evaluate_rules(edge, kDefault).empty());
        add_refusal(edge, kExtent, "extent-mismatch", 1, 1ull << 20);     // 4.00 MiB/flip
        check("4.00 MiB/flip fires", fired(evaluate_rules(edge, kDefault), "rtt-destination-refused"));
    }
    {
        WindowSample load = healthy();
        load.flips = 9;   // a load hitch: under the flip floor
        add_refusal(load, kCreation, "destination-creation-refused", 9, kResult);
        check("a window under the 10-flip floor is quiet and has no data",
              evaluate_rules(load, kDefault).empty() &&
                  !rule_has_data("rtt-destination-refused", load));
    }
    check("its sustain is two windows (a performance rule)",
          sustain_windows("rtt-destination-refused") == 2);
    // The ledger hook bumps both totals and the per-reason slot together.
    Ledger& l = ledger();
    const uint64_t before_n = l.counters[static_cast<size_t>(Counter::RttDestinationRefusals)].load();
    const uint64_t before_b = l.counters[static_cast<size_t>(Counter::RttDestinationRefusedBytes)].load();
    note_rtt_destination_refusal(kCreation, "destination-creation-refused", 1000);
    note_rtt_destination_refusal(kRttDestinationRefusalSlots, "out-of-range", 5);   // ignored
    check("note_rtt_destination_refusal counts once, adds bytes, names its slot, and ignores an "
          "out-of-range slot",
          l.counters[static_cast<size_t>(Counter::RttDestinationRefusals)].load() == before_n + 1 &&
              l.counters[static_cast<size_t>(Counter::RttDestinationRefusedBytes)].load() ==
                  before_b + 1000 &&
              l.rtt_destination_refused_bytes[kCreation].load() >= 1000 &&
              std::string(l.rtt_destination_refusal_names[kCreation].load()) ==
                  "destination-creation-refused");
}

// color-target-count-ceiling (#3891, 2026-09-29): GTA V's persistent colour-target cache peaks at
// 320 of 256 entries at 27-29% of its 4 GiB budget, evicting ~665 targets per run (#3873).
WindowSample gta_color_target_window(uint64_t peak_entries, uint64_t peak_mib, uint64_t evictions) {
    WindowSample w = healthy();
    set_gauge(w, Gauge::PersistentTargetEntryLimit, 256);
    set_gauge(w, Gauge::PersistentTargetByteLimit, 4096ull << 20);
    w.peaks[static_cast<size_t>(Peak::PersistentTargetEntries)] = peak_entries;
    w.peaks[static_cast<size_t>(Peak::PersistentTargetBytes)] = peak_mib << 20;
    set_count(w, Counter::PersistentTargetEvictions, evictions);
    set_count(w, Counter::PersistentTargetEvictedBytes, evictions * (3ull << 20));
    return w;
}

void test_color_target_count_ceiling() {
    std::puts("color-target-count-ceiling");
    const auto a = evaluate_rules(gta_color_target_window(320, 1150, 30), kDefault);
    check("GTA V (320 of 256 entries at 28% of the byte budget, 30 evictions) fires "
          "color-target-count-ceiling alone", only(a, "color-target-count-ceiling"));
    check("...naming the peak against both bounds",
          !a.empty() && a[0].detail.find("peak-entries=320/256") != std::string::npos &&
              a[0].detail.find("(28% of budget)") != std::string::npos);
    check("exactly at the count bound (256 of 256) fires",
          fired(evaluate_rules(gta_color_target_window(256, 1150, 30), kDefault),
                "color-target-count-ceiling"));
    check("one under the count bound (255) is quiet",
          evaluate_rules(gta_color_target_window(255, 1150, 30), kDefault).empty());
    check("at the count bound with the byte budget half used is quiet (memory binds too)",
          evaluate_rules(gta_color_target_window(320, 2048, 30), kDefault).empty());
    check("...49.9% of the byte budget fires",
          fired(evaluate_rules(gta_color_target_window(320, 2044, 30), kDefault),
                "color-target-count-ceiling"));
    check("10 evictions in 5 s (2/s) and no refusal is quiet: at the bound but costing nothing",
          evaluate_rules(gta_color_target_window(320, 1150, 10), kDefault).empty());
    check("11 evictions in 5 s fires",
          fired(evaluate_rules(gta_color_target_window(320, 1150, 11), kDefault),
                "color-target-count-ceiling"));
    {
        // GTA V gameplay as measured: exactly at 256 of 256, no evictions, one 14 MiB compute
        // result refused destination-creation-refused per flip.
        WindowSample g = gta_color_target_window(256, 988, 0);
        g.flips = 48;
        add_refusal(g, 18, "destination-creation-refused", 48, 2560ull * 1440 * 4);
        const auto r = evaluate_rules(g, kDefault);
        check("GTA V gameplay (256/256, 0 evictions, compute creation refused) fires it, beside "
              "rtt-destination-refused",
              r.size() == 2 && fired(r, "color-target-count-ceiling") &&
                  fired(r, "rtt-destination-refused"));
        // The refusal arm reports the refusal COUNT, not a 0.00 evictions/s (#3891 review).
        const AlarmFiring* c = nullptr;
        for (const auto& x : r) if (std::string(x.rule) == "color-target-count-ceiling") c = &x;
        check("...and, firing on the refusal arm, reports 48 creation-refusals rather than "
              "evictions/s",
              c && std::string(c->unit) == "creation-refusals" && c->value == 48.0 &&
                  c->detail.find("compute-creation-refused=48 (") != std::string::npos);
        const auto churn = evaluate_rules(gta_color_target_window(320, 1150, 30), kDefault);
        check("...while the eviction arm still reports evictions/s",
              !churn.empty() && std::string(churn[0].unit) == "evictions/s");
        WindowSample other = gta_color_target_window(256, 988, 0);
        other.flips = 48;
        add_refusal(other, 15, "extent-mismatch", 48, 2560ull * 1440 * 4);
        check("...a refusal for another reason (a shape mismatch) does not make the COUNT bind",
              !fired(evaluate_rules(other, kDefault), "color-target-count-ceiling"));
        WindowSample full = gta_color_target_window(256, 2500, 0);
        full.flips = 48;
        add_refusal(full, 18, "destination-creation-refused", 48, 2560ull * 1440 * 4);
        check("...and creation refused with the byte budget over half used is memory, not count",
              !fired(evaluate_rules(full, kDefault), "color-target-count-ceiling"));
    }
    check("a run that never consulted the cache (no entry bound) has no data",
          !rule_has_data("color-target-count-ceiling", healthy()) &&
              rule_has_data("color-target-count-ceiling", gta_color_target_window(0, 0, 0)));
}

// gpu-present-stalled (#3891, #3951): a GPU-present frontend that presented nothing while the guest
// flipped. GTA V on the #3951 regression: ~30 guest flips/s, no [app] fps line in 60+ s.
void test_gpu_present_stalled() {
    std::puts("gpu-present-stalled");
    WindowSample stalled = healthy();                    // 150 flips, no presents of either kind
    set_gauge(stalled, Gauge::GpuPresentActive, 1);
    const auto a = evaluate_rules(stalled, kDefault);
    check("GPU present on, 150 guest flips, 0 presents fires gpu-present-stalled alone",
          only(a, "gpu-present-stalled"));
    check("...as guest flips per second, saying no decline was recorded",
          !a.empty() && std::string(a[0].unit) == "guest-flips/s" && a[0].value == 30.0 &&
              a[0].detail.find("declines=none-recorded") != std::string::npos);
    WindowSample named = stalled;
    named.present_declines[6] = 150; named.present_decline_names[6] = "no-render-target";
    const auto n = evaluate_rules(named, kDefault);
    check("...and names the renderer's decline reasons when there are some",
          fired(n, "gpu-present-stalled") &&
              n[0].detail.find("declines=no-render-target:150") != std::string::npos);
    WindowSample presenting = stalled;
    set_cost(presenting, Cost::PresentCpu, 5.0, 1, 5.0);  // a single GPU present
    check("one GPU present in the window is quiet (a slow title is not a stalled one)",
          !fired(evaluate_rules(presenting, kDefault), "gpu-present-stalled"));
    WindowSample fallback = stalled;
    set_count(fallback, Counter::PresentCpuFallbacks, 1);
    check("a CPU-fallback present is a present too",
          !fired(evaluate_rules(fallback, kDefault), "gpu-present-stalled"));
    WindowSample hidden = stalled;                        // minimized / occluded / recreating
    set_count(hidden, Counter::PresentWindowUnavailable, 40);
    check("a window the frontend could not present to: no data, and quiet",
          !rule_has_data("gpu-present-stalled", hidden) &&
              !fired(evaluate_rules(hidden, kDefault), "gpu-present-stalled"));
    check("...and the hint does not claim a frozen or black window outright",
          !a.empty() && std::string(a[0].hint).find("frozen or black") == std::string::npos &&
              std::string(a[0].hint).find("presentable") != std::string::npos);
    WindowSample no_consumer = healthy();                 // tools/screenshot: never presents
    check("no GPU-present consumer: no data, and quiet",
          !rule_has_data("gpu-present-stalled", no_consumer) &&
              evaluate_rules(no_consumer, kDefault).empty());
    WindowSample loading = stalled;
    loading.flips = kGpuPresentStalledMinFlips - 1;       // the guest itself is not flipping
    check("below the flip floor (a loading pause): no data, and quiet",
          !rule_has_data("gpu-present-stalled", loading) &&
              !fired(evaluate_rules(loading, kDefault), "gpu-present-stalled"));
    loading.flips = kGpuPresentStalledMinFlips;
    check("at the flip floor it fires", fired(evaluate_rules(loading, kDefault),
                                             "gpu-present-stalled"));
    check("performance sustain (2 windows, one long hitch is not a stall)",
          sustain_windows("gpu-present-stalled") == 2);
    check("the rule is in the summary's list",
          std::count_if(rule_names().begin(), rule_names().end(), [](const char* r) {
              return std::string(r) == "gpu-present-stalled"; }) == 1);
}

// The exact full-overwrite shape census: one verdict per tested compute result.
void test_exact_result_census() {
    std::puts("exact-full-result census");
    std::set<std::string> names;
    bool all_named = true;
    for (const char* n : kExactResultDeclineNames) {
        all_named &= n && *n;
        if (n) names.insert(n);
    }
    check("every exact-result verdict has a distinct non-empty name",
          all_named && names.size() == kExactResultDeclineCount);
    check("Accepted is \"accepted\" and img-dim names Sonic's #3929 decline",
          std::string(kExactResultDeclineNames[0]) == "accepted" &&
              std::string(kExactResultDeclineNames[static_cast<size_t>(
                  ExactResultDecline::ImageDim)]) == "img-dim");
    Ledger& l = ledger();
    const size_t dim = static_cast<size_t>(ExactResultDecline::ImageDim);
    const uint64_t before = l.counters[static_cast<size_t>(Counter::ExactResultCandidates)].load();
    const uint64_t before_dim = l.exact_result_declines[dim].load();
    note_exact_result(ExactResultDecline::ImageDim);
    note_exact_result(ExactResultDecline::Accepted);
    note_exact_result(ExactResultDecline::Count);   // not a verdict: ignored
    check("note_exact_result counts candidates and the verdict, and ignores Count",
          l.counters[static_cast<size_t>(Counter::ExactResultCandidates)].load() == before + 2 &&
              l.exact_result_declines[dim].load() == before_dim + 1);
}

// The engine half: per-window deltas of the refusals and verdicts, per-window PEAKS that reset,
// the JSONL fields, and the exit census line.
void test_engine_2026_09_29_censuses() {
    std::puts("engine 2026-09-29 censuses");
    const std::string dir = std::getenv("TMPDIR") ? std::getenv("TMPDIR") : ".";
    const std::string tag = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()) + "_c" +
        std::to_string(reinterpret_cast<uintptr_t>(&dir) & 0xffffff);
    const std::string jsonl = dir + "/test_perf_alarms_" + tag + ".jsonl";
    const std::string logp = dir + "/test_perf_alarms_" + tag + ".log";
    FILE* log = std::fopen(logp.c_str(), "w+");
    std::vector<AlarmFiring> w1, w2, w3;
    constexpr size_t kCreation = 18;
    const size_t dim = static_cast<size_t>(ExactResultDecline::ImageDim);
    const size_t accepted = static_cast<size_t>(ExactResultDecline::Accepted);
    {
        EngineConfig config;
        config.window_ns = 1'000'000'000ull;
        config.jsonl_path = jsonl;
        config.log = log;
        AlarmEngine engine(std::move(config));
        Ledger l;
        AlarmEngine::ExternalTotals ext;
        // Pre-baseline residue: refusals and a peak from boot must not reach window 1.
        l.rtt_destination_refused_bytes[kCreation] = 9ull << 30;
        l.peaks[static_cast<size_t>(Peak::PersistentTargetEntries)] = 999;
        l.exact_result_declines[dim] = 7;
        uint64_t t = 1'000'000'000ull;
        engine.on_flip(t, l, 30, &ext);
        const uint64_t flip_ns = 1'000'000'000ull / 30;
        const auto run_window = [&](std::vector<AlarmFiring>& out, uint64_t refuse_every,
                                    uint64_t peak_entries) {
            l.gauges[static_cast<size_t>(Gauge::PersistentTargetEntryLimit)] = 256;
            l.gauges[static_cast<size_t>(Gauge::PersistentTargetByteLimit)] = 4096ull << 20;
            for (int f = 0; f < 30; ++f) {
                if (refuse_every && f % refuse_every == 0) {
                    l.rtt_destination_refusal_names[kCreation] = "destination-creation-refused";
                    l.rtt_destination_refused_bytes[kCreation] += 14ull << 20;
                    l.counters[static_cast<size_t>(Counter::RttDestinationRefusals)] += 1;
                    l.counters[static_cast<size_t>(Counter::RttDestinationRefusedBytes)] += 14ull << 20;
                }
                l.exact_result_declines[dim] += 1;
                l.exact_result_declines[accepted] += 2;
                l.counters[static_cast<size_t>(Counter::ExactResultCandidates)] += 3;
                std::atomic<uint64_t>& peak = l.peaks[static_cast<size_t>(Peak::PersistentTargetEntries)];
                if (peak.load() < peak_entries) peak.store(peak_entries);
                t += flip_ns + 1;
                auto fired_now = engine.on_flip(t, l, 30, &ext);
                if (!fired_now.empty() || f == 29) out = std::move(fired_now);
            }
        };
        run_window(w1, 2, 300);   // 15 x 14 MiB over 30 flips = 7 MiB/flip
        run_window(w2, 2, 300);
        run_window(w3, 0, 100);   // quiet, and the peak must have reset to this window's
        std::rewind(log);
        engine.write_summary(log);
    }
    std::fflush(log);
    check("window 1 holds rtt-destination-refused but is not reported (sustain 2)", w1.empty());
    check("window 2 reports rtt-destination-refused at the WINDOW's 7 MiB/flip, not the residue",
          fired(w2, "rtt-destination-refused") && w2[0].value > 6.9 && w2[0].value < 7.1);
    check("window 3 (no refusals) is quiet", w3.empty());
    const std::string j = slurp(jsonl);
    check("every JSONL window carries the refusal MiB by reason",
          count_of(j, "\"rtt_destination_refused_mib_by_reason\":{") == 3 &&
              j.find("\"rtt_destination_refused_mib_by_reason\":{\"destination-creation-refused\":210}") !=
                  std::string::npos);
    check("...the colour-target peaks per window, reset between windows (300 then 100, never the "
          "999 boot residue)",
          count_of(j, "\"persistent_target_peak_entries\":300,") == 2 &&
              count_of(j, "\"persistent_target_peak_entries\":100,") == 1 &&
              j.find("\"persistent_target_peak_entries\":999") == std::string::npos &&
              count_of(j, "\"persistent_target_entry_limit\":256,") == 3);
    check("...and the window's exact-result verdicts",
          count_of(j, "\"exact_result_verdicts\":{\"accepted\":60,\"img-dim\":30}") == 3 &&
              count_of(j, "\"exact_result_candidates\":90,") == 3);
    const std::string summary = slurp(logp);
    check("the exit summary prints the run's exact-result census and refusals (incl. residue)",
          summary.find("first failing field: accepted:180,img-dim:97; destination refusals of "
                       "accepted results (MiB): destination-creation-refused:9636") !=
              std::string::npos);
    std::fclose(log);
    std::remove(jsonl.c_str());
    std::remove(logp.c_str());
}

// gpu-sync-wait (#3948 stage 0): executor fence waits as a share of the budget, GPU headroom.
WindowSample sync_wait_window(double compute_wait_ms, uint64_t dispatches, double graphics_wait_ms,
                              uint64_t batches, double device_ms, uint64_t samples) {
    WindowSample w = healthy();
    w.flips = 46;
    w.target_hz = 60;
    set_cost(w, Cost::GpuWaitCompute, compute_wait_ms, dispatches, 1.0);
    set_cost(w, Cost::GpuWaitGraphics, graphics_wait_ms, batches, 1.0);
    set_count(w, Counter::GpuDeviceNsCompute, static_cast<uint64_t>(device_ms * 0.5 * 1e6));
    set_count(w, Counter::GpuDeviceNsGraphics, static_cast<uint64_t>(device_ms * 0.5 * 1e6));
    const uint64_t compute_samples = samples * dispatches / (dispatches + batches);
    set_count(w, Counter::GpuDeviceSamplesCompute, compute_samples);
    set_count(w, Counter::GpuDeviceSamplesGraphics, samples - compute_samples);
    return w;
}

void test_gpu_sync_wait() {
    std::puts("gpu-sync-wait");
    // GTA V gameplay as measured: 1,059 ms of waits (513 + 546) over 46 flips at 60 Hz, 859 ms of
    // device time inside, 3,729 of 4,005 waits sampled.
    const auto a = evaluate_rules(sync_wait_window(513, 1520, 546, 2485, 859, 3729), kDefault);
    check("GTA V gameplay (138% of the budget, GPU 17% busy) fires gpu-sync-wait alone",
          only(a, "gpu-sync-wait"));
    check("...value is %budget and the line names idle-in-wait and gpu-busy",
          !a.empty() && a[0].value > 137 && a[0].value < 139 &&
              a[0].detail.find("idle-in-wait=19%") != std::string::npos &&
              a[0].detail.find("gpu-busy=17%") != std::string::npos);
    check("its cost is the whole wait (what overlap could recover)",
          !a.empty() && a[0].cost_ms > 1058 && a[0].cost_ms < 1060);
    // 74% vs 75% of a 16.67 ms budget over 46 flips: 567.3 vs 575.0 ms of waits.
    check("74% of the budget (Sonic's menus sit near 50%) is quiet",
          evaluate_rules(sync_wait_window(284, 800, 283, 800, 300, 1600), kDefault).empty());
    check("75% of the budget fires",
          fired(evaluate_rules(sync_wait_window(288, 800, 288, 800, 300, 1600), kDefault),
                "gpu-sync-wait"));
    check("a GPU-saturated window (device time over half of wall) is quiet: overlap cannot help",
          evaluate_rules(sync_wait_window(1500, 1500, 1500, 2500, 2600, 4000), kDefault).empty());
    const WindowSample unsampled = sync_wait_window(513, 1520, 546, 2485, 0, 1000);
    check("with device time on under half of the waits there is no data, and no firing",
          !rule_has_data("gpu-sync-wait", unsampled) &&
              !fired(evaluate_rules(unsampled, kDefault), "gpu-sync-wait"));
    WindowSample few = sync_wait_window(513, 1520, 546, 2485, 859, 3729);
    few.flips = 9;
    check("under the 10-flip floor there is no data", !rule_has_data("gpu-sync-wait", few));
    check("its sustain is two windows", sustain_windows("gpu-sync-wait") == 2);
}

void test_gpu_device_time_coverage() {
    std::puts("gpu-device-time-coverage");
    // #3964's missing deferred envelope: graphics coverage stayed near 68-71% for
    // 38 gameplay windows. Corrected arms stay near 88-89%. Construct both populations
    // independently of the rule, including the all-compute masking case below.
    WindowSample broken = sync_wait_window(513, 1500, 546, 2500, 600, 4000);
    set_count(broken, Counter::GpuDeviceSamplesCompute, 1500);
    set_count(broken, Counter::GpuDeviceSamplesGraphics, 1750);
    const auto a = evaluate_rules(broken, kDefault);
    check("the missing-envelope population fires coverage alone, withholding headroom",
          only(a, "gpu-device-time-coverage") && !rule_has_data("gpu-sync-wait", broken));
    check("...the value is the worst stream's missing share and detail names both streams",
          !a.empty() && a[0].value > 29.9 && a[0].value < 30.1 &&
              std::string(a[0].unit) == "%unsampled" &&
              a[0].detail.find("compute=1500/1500 graphics=1750/2500") != std::string::npos);
    WindowSample fixed = broken;
    set_count(fixed, Counter::GpuDeviceSamplesGraphics, 2225);
    check("corrected 89% graphics coverage is quiet and restores gpu-sync-wait data",
          !fired(evaluate_rules(fixed, kDefault), "gpu-device-time-coverage") &&
              rule_has_data("gpu-sync-wait", fixed));

    WindowSample masked = sync_wait_window(513, 10000, 546, 1000, 600, 11000);
    set_count(masked, Counter::GpuDeviceSamplesCompute, 10000);
    set_count(masked, Counter::GpuDeviceSamplesGraphics, 0);
    check("91% aggregate coverage cannot mask completely missing graphics timestamps",
          only(evaluate_rules(masked, kDefault), "gpu-device-time-coverage") &&
              !rule_has_data("gpu-sync-wait", masked));
    WindowSample compute_gap = broken;
    set_count(compute_gap, Counter::GpuDeviceSamplesCompute, 0);
    set_count(compute_gap, Counter::GpuDeviceSamplesGraphics, 2500);
    check("a missing compute stream is caught independently too",
          only(evaluate_rules(compute_gap, kDefault), "gpu-device-time-coverage"));

    WindowSample boundary = healthy();
    set_cost(boundary, Cost::GpuWaitGraphics, 10, 100, 1);
    set_count(boundary, Counter::GpuDeviceSamplesGraphics, 80);
    check("80% coverage at the wait floor is data and quiet",
          rule_has_data("gpu-device-time-coverage", boundary) &&
              evaluate_rules(boundary, kDefault).empty());
    set_count(boundary, Counter::GpuDeviceSamplesGraphics, 79);
    check("79% at the wait floor fires", only(evaluate_rules(boundary, kDefault),
                                            "gpu-device-time-coverage"));
    check("threshold sensitivity does not change the instrument-validity bound",
          only(evaluate_rules(boundary, RuleThresholds::scaled(25)),
               "gpu-device-time-coverage"));
    set_cost(boundary, Cost::GpuWaitGraphics, 10, 99, 1);
    set_count(boundary, Counter::GpuDeviceSamplesGraphics, 0);
    check("a small unmatched population below the wait floor has no coverage data",
          !rule_has_data("gpu-device-time-coverage", boundary) &&
              evaluate_rules(boundary, kDefault).empty());
    check("a GPU-free window is NO DATA, not measured quiet",
          !rule_has_data("gpu-device-time-coverage", healthy()));
    WindowSample extra = fixed;
    set_count(extra, Counter::GpuDeviceSamplesGraphics, 2501);
    check("an extra sample across a window boundary does not imply missing coverage",
          !fired(evaluate_rules(extra, kDefault), "gpu-device-time-coverage"));

    EngineConfig config;
    config.log = nullptr;
    AlarmEngine engine(std::move(config));
    check("one incomplete window is not reported", engine.close_window(broken, 5).empty());
    check("two incomplete windows report the validity alarm",
          only(engine.close_window(broken, 10), "gpu-device-time-coverage"));
    engine.close_window(healthy(), 15);
    check("a no-data window resets its sustain",
          engine.close_window(broken, 20).empty() &&
              engine.times_fired("gpu-device-time-coverage") == 1);
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
    test_gpu_memory_off_device();
    test_unaccounted_draws();
    test_unimplemented_hle_calls();
    test_diagnostic_path_active();
    test_present_path_fallback();
    test_pipeline_cache_thrash();
    test_texture_validation_churn();
    test_host_copy_pressure();
    test_host_copy_per_flip();
    test_present_slot_trouble();
    test_shader_compile();
    test_sampler_seed();
    test_threshold_scaling();
    test_budget_follows_flip_rate();
    test_cost_scope_nesting();
    test_texture_reference_sample();
    test_engine();
    test_engine_breakdowns();
    test_engine_host_copy_per_flip();
    test_engine_host_copy_alternating();
    test_rtt_destination_refused();
    test_color_target_count_ceiling();
    test_gpu_present_stalled();
    test_exact_result_census();
    test_engine_2026_09_29_censuses();
    test_gpu_sync_wait();
    test_gpu_device_time_coverage();
    check("rule_names lists the six phase-2 rules, the three added with phase 3, "
          "gpu-memory-off-device (#3897), the six 2026-09-28 queue rules and the two "
          "2026-09-29 ones (host-copy-per-flip, present-slot-trouble, rtt-destination-refused, "
          "color-target-count-ceiling), gpu-sync-wait (#3948), gpu-present-stalled (#3951) "
          "and gpu-device-time-coverage (#3891)",
          rule_names().size() == 23);
    std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "ok", g_failures);
    return g_failures ? 1 : 0;
}
