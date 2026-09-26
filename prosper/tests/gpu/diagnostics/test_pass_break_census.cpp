// Pass-break classification: which of the grouping loop's four exits ended a pass.
//
// The decision this census exists to inform is expensive to get wrong -- "lengthen the passes" and
// "make a pass cheaper" touch different code and neither is cheap. So the property that matters is
// not that it counts, but that it keeps the per-reason DISTRIBUTION separable: the reason that ends
// the most passes is not necessarily the reason holding the mean down, and a census that pooled
// them would point at the wrong fix while looking authoritative.

#include "gpu/diagnostics/pass_break_census.hpp"

#include <cstdio>
#include <cstring>

using namespace prosper::gpu;

static int failures = 0;
static void check(bool ok, const char* name) {
    printf("%s: %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) failures++;
}

int main() {
    bool names_ok = true;
    for (int i = 0; i < static_cast<int>(PassBreak::Count); i++) {
        const char* n = pass_break_name(static_cast<PassBreak>(i));
        if (!n || !*n || std::strcmp(n, "unknown") == 0) names_ok = false;
    }
    check(names_ok, "every PassBreak has a stable non-placeholder name");

    auto& c = pass_break_census();
    // A reason that fires often on LONG passes, against one that fires rarely on singletons. The
    // census must keep these apart: pooled, the second is invisible, and it is the actionable one.
    for (int i = 0; i < 10; i++) c.note_break(PassBreak::EndOfItems, 50);
    for (int i = 0; i < 3; i++) c.note_break(PassBreak::TargetsChanged, 1);

    check(c.passes(PassBreak::EndOfItems) == 10 && c.draws(PassBreak::EndOfItems) == 500,
          "a high-volume reason keeps its own pass and draw totals");
    check(c.passes(PassBreak::TargetsChanged) == 3 && c.draws(PassBreak::TargetsChanged) == 3,
          "a low-volume singleton reason is not absorbed into the louder one");
    check(c.passes(PassBreak::DepthFeedback) == 0 && c.draws(PassBreak::DepthFeedback) == 0,
          "a reason that never fired reports zero rather than a share of the total");

    // The pooled mean here is 503/13 = 38.7, which would read as "passes are long, nothing to fix"
    // and hide the 1.00-mean reason entirely. Per-reason means are 50.0 and 1.00.
    const double pooled = 503.0 / 13.0;
    const double targets_mean = static_cast<double>(c.draws(PassBreak::TargetsChanged)) /
                                static_cast<double>(c.passes(PassBreak::TargetsChanged));
    check(pooled > 30.0 && targets_mean == 1.0,
          "per-reason means separate a fragmenting reason that the pooled mean would hide");

    printf("%s\n", failures ? "FAILURES" : "ALL PASS");
    return failures ? 1 : 0;
}
