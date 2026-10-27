#include "diagnostics/perf/arbitrary_placeholder.hpp"
#include "diagnostics/perf/perf_ledger.hpp"
#include "diagnostics/perf/perf_alarm_rules.hpp"
#include "diagnostics/exit_reports.hpp"
#include <cstdio>
#include <mutex>

namespace prosper::diagnostics::perf {
namespace {
struct Inventory {
    std::mutex mutex;
    ArbitraryPlaceholderInventory programs;
};
Inventory& inventory() {
    // The exit callback can run after unrelated static destruction; this observer lives to exit.
    static auto* const value = [] {
        auto* state = new Inventory;
        register_exit_report([state] {
            ArbitraryPlaceholderInventory snapshot;
            {
                std::lock_guard lock(state->mutex);
                snapshot = state->programs;
            }
            const auto text = format_arbitrary_placeholder_summary(snapshot);
            if (!text.empty()) std::fputs(text.c_str(), stderr);
        });
        return state;
    }();
    return *value;
}
}   // namespace

void note_arbitrary_placeholder_check() {
    if (enabled()) add(Counter::ArbitraryPlaceholderChecks);
}

void note_arbitrary_placeholder_admission(uint64_t program, uint64_t fresh, uint64_t loop) {
    if (!enabled() || (!fresh && !loop)) return;
    add(Counter::ArbitraryPlaceholderPrograms);
    add(Counter::ArbitraryPlaceholderReads, fresh + loop);
    bool first;
    {
        auto& state = inventory();
        std::lock_guard lock(state.mutex);
        auto& row = state.programs[program];
        first = row.compilations++ == 0;
        row.fresh_mask_reads += fresh;
        row.loop_carry_reads += loop;
    }
    if (first) {
        const char* identity = program ? "program" : "unidentified-program";
        std::fprintf(stderr,
                     "[perf-alarm] rule=arbitrary-placeholder-admitted %s=0x%llx "
                     "reads=%llu fresh-mask=%llu loop-carry=%llu "
                     "origin=unloaded scope=compiler-read-sites\n",
                     identity, static_cast<unsigned long long>(program),
                     static_cast<unsigned long long>(fresh + loop),
                     static_cast<unsigned long long>(fresh), static_cast<unsigned long long>(loop));
    }
}

std::string format_arbitrary_placeholder_summary(const ArbitraryPlaceholderInventory& programs) {
    std::string out;
    for (const auto& [program, row] : programs) {
        char line[384];
        std::snprintf(line, sizeof line,
                      "[perf-alarm] summary observer=arbitrary-placeholder-admitted "
                      "%s=0x%llx compilations=%llu reads=%llu fresh-mask=%llu loop-carry=%llu "
                      "origin=unloaded scope=compiler-read-sites\n",
                      program ? "program" : "unidentified-program",
                      static_cast<unsigned long long>(program),
                      static_cast<unsigned long long>(row.compilations),
                      static_cast<unsigned long long>(row.fresh_mask_reads + row.loop_carry_reads),
                      static_cast<unsigned long long>(row.fresh_mask_reads),
                      static_cast<unsigned long long>(row.loop_carry_reads));
        out += line;
    }
    return out;
}

void append_arbitrary_placeholder_alarm(std::vector<AlarmFiring>& out, const WindowSample& w) {
    const auto programs = w.count(Counter::ArbitraryPlaceholderPrograms);
    const auto reads = w.count(Counter::ArbitraryPlaceholderReads);
    if (!programs && !reads) return;
    AlarmFiring alarm;
    alarm.rule = "arbitrary-placeholder-admitted";
    alarm.value = static_cast<double>(programs);
    alarm.threshold = 1;
    alarm.unit = "program admissions";
    alarm.detail = "reads=" + std::to_string(reads) + " origin=unloaded scope=compiler-read-sites";
    alarm.hint =
        "the compiler retained consistent arbitrary values for explicitly unloaded roots; "
        "the run summary names program addresses. Counts are compilations and static "
        "read sites, not draws or executed lane reads; missing inputs/spills remain refused";
    out.push_back(std::move(alarm));
}
}   // namespace prosper::diagnostics::perf
