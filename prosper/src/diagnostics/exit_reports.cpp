#include "diagnostics/exit_reports.hpp"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <utility>
#include <vector>

namespace prosper::diagnostics {
namespace {

struct ExitReportRegistry {
    std::mutex mutex;
    std::vector<ExitReport> reports;
    size_t ran = 0;           // reports[0, ran) have been handed to a flush
    bool atexit_registered = false;
};

// Never destroyed, on purpose: the atexit fallback can run after static destructors, and a flush
// from a guest thread can race process teardown. A registry that is never destroyed cannot be used
// after its destruction. The union member is constructed by the holder's constructor and the
// holder's destructor deliberately does not destroy it -- the standard no-destroy idiom, with no
// heap allocation.
union NeverDestroyedRegistry {
    ExitReportRegistry value;
    NeverDestroyedRegistry() : value() {}
    ~NeverDestroyedRegistry() {}
    NeverDestroyedRegistry(const NeverDestroyedRegistry&) = delete;
    NeverDestroyedRegistry& operator=(const NeverDestroyedRegistry&) = delete;
};

ExitReportRegistry& registry() {
    static NeverDestroyedRegistry holder;
    return holder.value;
}

void flush_at_exit() { flush_exit_reports(); }

}  // namespace

void register_exit_report(ExitReport report) {
    if (!report) return;
    ExitReportRegistry& r = registry();
    std::scoped_lock lock(r.mutex);
    r.reports.push_back(std::move(report));
    if (!r.atexit_registered) {
        r.atexit_registered = true;
        if (std::atexit(&flush_at_exit) != 0)
            std::fputs("[exit-reports] std::atexit refused the fallback; end-of-run reports "
                       "print only on an explicit flush\n", stderr);
    }
}

void flush_exit_reports() {
    ExitReportRegistry& r = registry();
    std::vector<ExitReport> due;
    {
        // Claim the pending reports under the lock, run them outside it: a report is ordinary code
        // and must be free to take its own locks (or even register another report) without
        // deadlocking against this one.
        std::scoped_lock lock(r.mutex);
        due.assign(r.reports.begin() + static_cast<std::ptrdiff_t>(r.ran), r.reports.end());
        r.ran = r.reports.size();
    }
    for (const ExitReport& report : due) report();
    std::fflush(nullptr);
}

}  // namespace prosper::diagnostics
