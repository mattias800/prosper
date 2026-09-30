// CPU sampler/ledger/window controls. Callback fixtures do not prove CommandProcessor packet
// routing, the actual pending queue, warning suppression wiring or Vulkan/runtime behavior.
#include "gpu/pm4/wait_regmem_sample.hpp"
#include "diagnostics/perf/perf_alarms.hpp"
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

using namespace prosper::gpu;
namespace perf = prosper::diagnostics::perf;
static unsigned failures = 0;
#define CHECK(condition, label) do { \
    if (condition) std::printf("[ok] %s\n", label); \
    else { std::printf("[FAIL] %s\n", label); ++failures; } \
} while (false)

static uint64_t count(perf::Counter counter) {
    return perf::ledger().counters[static_cast<size_t>(counter)].load(std::memory_order_relaxed);
}

static std::string read_file(FILE* file) {
    std::rewind(file);
    std::string text;
    char buffer[4096];
    while (const size_t bytes = std::fread(buffer, 1, sizeof buffer, file))
        text.append(buffer, bytes);
    return text;
}

static std::string summary(perf::AlarmEngine& engine) {
    FILE* file = std::tmpfile();
    if (!file) { CHECK(false, "summary scratch opens"); return {}; }
    engine.write_summary(file);
    std::fflush(file);
    const auto text = read_file(file);
    std::fclose(file);
    return text;
}

static void observe(const WaitRegMemPredicateSample& sample, bool deferred = false) {
    perf::note_wait_regmem_direct_evaluation(
        sample.readable, sample.comparison_supported, sample.satisfied);
    if (!sample.satisfied) perf::note_wait_regmem_direct_false_action(deferred);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: test_wait_regmem_observation <jsonl-output>\n");
        return 2;
    }
    std::puts("CPU sampler/ledger/engine controls; packet route and pending queue UNVERIFIED");
    Pm4Command command;
    command.kind = Pm4Command::Kind::WaitRegMem;
    command.wm_addr = 0x1000;
    command.wm_mask = 0xff;
    command.wm_ref = 5;
    struct Case { uint32_t func; uint64_t raw; bool expected; };
    const Case cases[] = {
        {0, 0, true}, {1, 4, true}, {1, 5, false}, {2, 5, true}, {2, 6, false},
        {3, 5, true}, {3, 6, false}, {4, 6, true}, {4, 5, false},
        {5, 5, true}, {5, 4, false}, {6, 6, true}, {6, 5, false}, {7, 5, false}
    };
    for (const auto& item : cases) {
        command.wm_func = item.func;
        unsigned reads = 0, overlays = 0;
        const auto sample = sample_wait_regmem_predicate(command,
            [&](uint64_t address, uint64_t* value) {
                CHECK(address == command.wm_addr, "sampler uses selected address");
                ++reads; *value = item.raw; return true;
            }, [&](uint64_t, uint64_t*) { ++overlays; return false; });
        CHECK(sample.satisfied == item.expected, "predicate comparison result");
        CHECK(reads == 1 && overlays == 1, "exactly one reader and overlay call");
        CHECK(sample.readable && sample.raw_value == item.raw &&
              sample.effective_value == item.raw && sample.masked_value == item.raw,
              "readable decision retains raw/effective/masked values");
        CHECK(sample.comparison_supported == (item.func <= 6), "supported comparison classification");
    }

    command.wm_func = 3;
    command.wm_mask = 0xff;
    command.wm_ref = 0x105;
    const auto unmasked_reference = sample_wait_regmem_predicate(command,
        [](uint64_t, uint64_t* value) { *value = 0x105; return true; },
        [](uint64_t, uint64_t*) { return false; });
    CHECK(unmasked_reference.masked_value == 5 && !unmasked_reference.satisfied,
          "reference stays unmasked");

    command.wm_ref = 1;
    uint64_t label = 0;
    unsigned overlay_calls = 0;
    const auto overlaid = sample_wait_regmem_predicate(command,
        [&](uint64_t, uint64_t* value) { *value = label; return true; },
        [&](uint64_t, uint64_t* value) { ++overlay_calls; *value = 1; return true; });
    CHECK(overlaid.satisfied && overlaid.raw_value == 0 && overlaid.effective_value == 1,
          "predicate uses overlaid effective value");
    CHECK(overlaid.overlay_touched && overlaid.masked_value == 1 && label == 0 && overlay_calls == 1,
          "overlay callback retains raw label without publication");

    const auto retained = sample_wait_regmem_predicate(command,
        [&](uint64_t, uint64_t* value) { *value = label; return true; },
        [&](uint64_t, uint64_t*) { label = 1; return false; });
    CHECK(!retained.satisfied && retained.raw_value == 0 && retained.effective_value == 0 &&
          retained.masked_value == 0 && label == 1,
          "later fixture change cannot replace retained decision sample");

    command.wm_func = 0;
    overlay_calls = 0;
    const auto unreadable = sample_wait_regmem_predicate(command,
        [](uint64_t, uint64_t*) { return false; },
        [&](uint64_t, uint64_t*) { ++overlay_calls; return false; });
    CHECK(!unreadable.readable && !unreadable.satisfied && unreadable.comparison_supported,
          "unreadable label fails before always comparison");
    CHECK(overlay_calls == 0, "unreadable sample invokes no overlay");
    command.wm_func = 7;
    const auto unsupported = sample_wait_regmem_predicate(command,
        [](uint64_t, uint64_t* value) { *value = 1; return true; },
        [](uint64_t, uint64_t*) { return false; });
    const auto unreadable_unsupported = sample_wait_regmem_predicate(command,
        [](uint64_t, uint64_t*) { return false; },
        [](uint64_t, uint64_t*) { return false; });
    CHECK(unsupported.readable && !unsupported.satisfied && !unsupported.comparison_supported,
          "readable unsupported comparison is explicit");

    const bool disabled = !perf::enabled();
    observe(overlaid);
    observe(retained);
    observe(unreadable);
    observe(unsupported, true);
    observe(unreadable_unsupported, true);
    CHECK(count(perf::Counter::WaitRegMemDirectEvaluations) == (disabled ? 0 : 5),
          "producer counts every direct helper evaluation");
    CHECK(count(perf::Counter::WaitRegMemDirectCompareFalse) == (disabled ? 0 : 1),
          "producer records numeric false separately");
    CHECK(count(perf::Counter::WaitRegMemDirectUnreadable) == (disabled ? 0 : 2) &&
          count(perf::Counter::WaitRegMemDirectUnsupported) == (disabled ? 0 : 1),
          "unreadable takes precedence over unsupported");
    CHECK(count(perf::Counter::WaitRegMemDirectFalseProceed) == (disabled ? 0 : 2) &&
          count(perf::Counter::WaitRegMemDirectFalseDefer) == (disabled ? 0 : 2),
          "false helper actions remain distinct");
    CHECK(overlaid.satisfied && !retained.satisfied && !unreadable.satisfied,
          "sampler policy result survives disabled observation");

    perf::EngineConfig config;
    config.log = nullptr;
    config.window_ns = 100;
    config.jsonl_path = argv[1];
    {
        perf::AlarmEngine engine(config);
        CHECK(summary(engine).empty(), "no closed window preserves no-report contract");
        engine.on_flip(1000, perf::ledger(), 60);
        for (unsigned i = 0; i < 45; ++i) observe(retained);
        engine.on_flip(1100, perf::ledger(), 60);
        const auto text = summary(engine);
        CHECK(text.find("observer=wait-regmem-direct-fold") != std::string::npos &&
              text.find("completed-windows=1 snapshot=relaxed") != std::string::npos,
              "engine prints bounded relaxed observer scope");
        CHECK(text.find(disabled ? "data=NO DATA evaluations=0" :
                                  "data=OBSERVED evaluations=45 compare-false=45") != std::string::npos,
              "window excludes baseline and counts repeated helper observations");
        CHECK(text.find("false-proceed=45 false-defer=0") != std::string::npos || disabled,
              "window preserves repeated proceed action count");
        observe(retained);
        CHECK(summary(engine) == text, "trailing unclosed observation stays out of summary");
    }
    FILE* jsonl = std::fopen(argv[1], "r");
    CHECK(jsonl != nullptr, "JSONL output exists");
    if (jsonl) {
        const auto text = read_file(jsonl);
        std::fclose(jsonl);
        CHECK(text.find(disabled ? "\"wait_regmem_direct_evaluations\":0" :
                                  "\"wait_regmem_direct_evaluations\":45") != std::string::npos,
              "actual JSONL exports direct evaluations");
        CHECK(text.find(disabled ? "\"wait_regmem_direct_data\":\"NO DATA\"" :
                                  "\"wait_regmem_direct_data\":\"OBSERVED\"") != std::string::npos,
              "actual JSONL distinguishes absence from observation");
    }

    config.jsonl_path.clear();
    perf::AlarmEngine quiet(config), partial(config);
    perf::WindowSample window;
    window.seconds = 1;
    quiet.close_window(window, 1);
    CHECK(summary(quiet).find("data=NO DATA evaluations=0") != std::string::npos,
          "zero direct sample population is NO DATA");
    window.counters[static_cast<size_t>(perf::Counter::WaitRegMemDirectCompareFalse)] = 1;
    partial.close_window(window, 1);
    CHECK(summary(partial).find("data=PARTIAL evaluations=0 compare-false=1") != std::string::npos,
          "straddled snapshot model is PARTIAL rather than NO DATA");

    // A real counter-producing worker is still alive at summary. Balanced data cannot turn
    // snapshot=relaxed into a claim of worker quiescence.
    perf::AlarmEngine live(config);
    live.on_flip(2000, perf::ledger(), 60);
    std::mutex mutex;
    std::condition_variable cv;
    bool ready = false, release = false;
    std::thread worker([&] {
        observe(overlaid);
        std::unique_lock lock(mutex);
        ready = true; cv.notify_one();
        cv.wait(lock, [&] { return release; });
    });
    {
        std::unique_lock lock(mutex);
        cv.wait(lock, [&] { return ready; });
    }
    live.on_flip(2100, perf::ledger(), 60);
    const auto live_text = summary(live);
    CHECK(worker.joinable() && live_text.find("snapshot=relaxed") != std::string::npos,
          "still-live producer does not certify quiescence");
    CHECK(live_text.find(disabled ? "data=NO DATA evaluations=0" :
                                   "data=OBSERVED evaluations=1 compare-false=0") != std::string::npos,
          "satisfied helper sample is observed without numeric false");
    {
        std::lock_guard lock(mutex);
        release = true;
    }
    cv.notify_one();
    worker.join();

    std::printf("result: %u failure(s)\n", failures);
    return failures ? 1 : 0;
}
