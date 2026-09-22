// Compute phase attribution (frontends/shared/compute/compute_phase_attribution.hpp), #3461.
//
// The defect: execute_item()'s phase markers defaulted to the start time, so a dispatch that broke
// out early booked its whole duration to cleanup and a compensating NEGATIVE to the interval it
// broke in. Every case below builds its markers BY HAND on a fake clock (milliseconds as integer
// ticks), including the issue's own worked shape, so the arms do not depend on a real dispatch
// failing at a particular point.
#include "shared/compute/compute_phase_attribution.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <optional>

using namespace prosper::frontend;

namespace {

struct FakeClock {
    using duration = std::chrono::milliseconds;
    using rep = duration::rep;
    using period = duration::period;
    using time_point = std::chrono::time_point<FakeClock>;
    static constexpr bool is_steady = true;
};
using TP = FakeClock::time_point;
using Opt = std::optional<TP>;

TP at(long ms) { return TP(std::chrono::milliseconds(ms)); }

int failures = 0;
void check(const char* what, bool ok) {
    std::printf("  %s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}
bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }
double sum(const ComputePhaseMilliseconds& p) {
    return p.setup + p.pipeline + p.dispatch + p.writeback + p.cleanup;
}
bool nonnegative(const ComputePhaseMilliseconds& p) {
    return p.setup >= 0 && p.pipeline >= 0 && p.dispatch >= 0 && p.writeback >= 0 && p.cleanup >= 0;
}

}  // namespace

int main() {
    std::printf("compute phase attribution\n");

    // 1. A complete dispatch: every interval is its marker difference, cleanup from writeback.
    {
        const auto p = attribute_compute_phases<TP>(at(0), at(3), at(5), at(9), at(11), at(11), at(12));
        check("complete: setup 3, pipeline 2, dispatch 4, writeback 2, cleanup 1",
              near(p.setup, 3) && near(p.pipeline, 2) && near(p.dispatch, 4) &&
              near(p.writeback, 2) && near(p.cleanup, 1));
    }

    // 2. THE ISSUE'S SHAPE: setup reached at 10, the break happens in the pipeline phase, and the
    //    item ends at 12. Pre-fix: setup=+10, pipeline=-10, cleanup=+12.
    {
        const auto p = attribute_compute_phases<TP>(at(0), Opt(at(10)), Opt(), Opt(), Opt(), at(11), at(12));
        check("break in pipeline: no negative interval", nonnegative(p));
        check("break in pipeline: cleanup is only the real teardown (1 ms), not the item",
              near(p.cleanup, 1));
        check("break in pipeline: setup keeps its 10 ms", near(p.setup, 10));
        check("break in pipeline: the 1 ms spent before the break is pipeline's", near(p.pipeline, 1));
        check("break in pipeline: phases that never ran are exactly 0",
              near(p.dispatch, 0) && near(p.writeback, 0));
        check("break in pipeline: the parts still sum to the item", near(sum(p), 12));
    }

    // 3. A break before setup completes: all the pre-break time is setup's.
    {
        const auto p = attribute_compute_phases<TP>(at(0), Opt(), Opt(), Opt(), Opt(), at(7), at(8));
        check("break in setup: setup 7, cleanup 1, the rest 0",
              near(p.setup, 7) && near(p.pipeline, 0) && near(p.dispatch, 0) &&
              near(p.writeback, 0) && near(p.cleanup, 1));
    }

    // 4. A break during writeback (e.g. a failed readback): dispatch keeps its GPU wait.
    {
        const auto p = attribute_compute_phases<TP>(at(0), at(1), at(2), at(20), Opt(), at(23), at(24));
        check("break in writeback: dispatch 18, writeback 3, cleanup 1",
              near(p.dispatch, 18) && near(p.writeback, 3) && near(p.cleanup, 1) && nonnegative(p));
    }

    // 5. The cached-fill early success sets pipeline = dispatch = setup: zero-length, not negative.
    {
        const auto p = attribute_compute_phases<TP>(at(0), at(4), at(4), at(4), at(6), at(6), at(7));
        check("cached-fill path: pipeline and dispatch are 0, writeback 2",
              near(p.pipeline, 0) && near(p.dispatch, 0) && near(p.writeback, 2) &&
              near(p.cleanup, 1) && near(sum(p), 7));
    }

    std::printf("%s\n", failures ? "FAILURES PRESENT" : "all passed");
    return failures ? 1 : 0;
}
