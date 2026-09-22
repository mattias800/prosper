#include "diagnostics/exit_reports.hpp"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace prosper::diagnostics {
namespace {

struct ExitReportRegistry {
    std::mutex mutex;
    std::vector<ExitReport> reports;
    size_t ran = 0;           // reports[0, ran) have been handed to a flush
    bool atexit_registered = false;
};

// Leaked on purpose: the atexit fallback can run after static destructors, and a flush from a
// guest thread can race process teardown. A registry that is never destroyed cannot be used after
// its destruction.
ExitReportRegistry& registry() {
    static ExitReportRegistry* instance = new ExitReportRegistry();
    return *instance;
}

void flush_at_exit() { flush_exit_reports(); }

}  // namespace

void register_exit_report(ExitReport report) {
    if (!report) return;
    ExitReportRegistry& r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    r.reports.push_back(report);
    if (!r.atexit_registered) {
        r.atexit_registered = true;
        if (std::atexit(&flush_at_exit) != 0)
            std::fprintf(stderr, "[exit-reports] std::atexit refused the fallback; end-of-run "
                                 "reports print only on an explicit flush\n");
    }
}

void flush_exit_reports() {
    ExitReportRegistry& r = registry();
    std::vector<ExitReport> due;
    {
        // Claim the pending reports under the lock, run them outside it: a report is ordinary code
        // and must be free to take its own locks (or even register another report) without
        // deadlocking against this one.
        std::lock_guard<std::mutex> lock(r.mutex);
        due.assign(r.reports.begin() + static_cast<std::ptrdiff_t>(r.ran), r.reports.end());
        r.ran = r.reports.size();
    }
    for (ExitReport report : due) report();
    std::fflush(nullptr);
}

}  // namespace prosper::diagnostics
