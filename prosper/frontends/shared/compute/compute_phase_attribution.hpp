// Attribution of one compute dispatch's CPU time to its five phases (#3461).
//
// execute_item() walks start -> setup -> pipeline -> dispatch -> writeback, setting a marker at
// each boundary, then leaves its do/while and runs cleanup. A dispatch that fails leaves the loop
// through an early `break`, so the markers after the break point are never reached.
//
// The markers used to default to `start`, and the five intervals were plain differences of
// consecutive markers. That telescopes to the right TOTAL for any marker values, but on a break
// path it put a NEGATIVE value in the interval spanning the break and booked the whole item to
// cleanup: setup=+10, pipeline=-10, cleanup=+12 for a dispatch that failed 10 ms in. Read as a
// breakdown, that says "teardown is the cost" about a dispatch that never reached teardown.
//
// The rule here uses only facts the markers establish:
//   * a phase whose end marker was reached is the difference of its two markers;
//   * the FIRST phase whose end marker was not reached is where the break happened -- the code
//     between its start marker and the next one is that phase's code -- so it runs from its start
//     marker to the loop exit;
//   * every phase after it did not run, and gets exactly 0 -- never a compensating negative;
//   * cleanup runs from the loop exit to the cleanup marker. On a path that reached every marker,
//     the few instructions between the writeback marker and the loop exit are charged to cleanup,
//     which is what the old formula did as well.
// The five values are therefore non-negative for monotonic clocks and still sum exactly to
// cleanup - start.
#pragma once

#include <chrono>
#include <optional>

namespace prosper::frontend {

struct ComputePhaseMilliseconds {
    double setup = 0.0;
    double pipeline = 0.0;
    double dispatch = 0.0;
    double writeback = 0.0;
    double cleanup = 0.0;
};

template <typename TimePoint>
ComputePhaseMilliseconds attribute_compute_phases(TimePoint start,
                                                  const std::optional<TimePoint>& setup,
                                                  const std::optional<TimePoint>& pipeline,
                                                  const std::optional<TimePoint>& dispatch,
                                                  const std::optional<TimePoint>& writeback,
                                                  TimePoint loop_exit, TimePoint cleanup) {
    auto ms = [](TimePoint begin, TimePoint end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };
    ComputePhaseMilliseconds out;
    double* slots[4] = {&out.setup, &out.pipeline, &out.dispatch, &out.writeback};
    const std::optional<TimePoint>* ends[4] = {&setup, &pipeline, &dispatch, &writeback};
    TimePoint previous = start;
    bool truncated = false;
    for (int phase = 0; phase < 4; ++phase) {
        if (!ends[phase]->has_value()) {
            *slots[phase] = ms(previous, loop_exit);
            truncated = true;
            break;
        }
        *slots[phase] = ms(previous, **ends[phase]);
        previous = **ends[phase];
    }
    out.cleanup = ms(truncated ? loop_exit : previous, cleanup);
    return out;
}

}  // namespace prosper::frontend
