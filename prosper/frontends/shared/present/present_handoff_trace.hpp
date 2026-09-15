#pragma once
#include "shared/perf/performance_capture.hpp"
#include <utility>

namespace prosper::frontend {

// Observes existing operations inside an explicitly armed F8 window. No clocks, strings,
// readbacks or event storage on the ordinary path. Publication IDs are not producer versions.
struct PresentHandoffTrace {
    bool active = perf::interactive_performance_capture().present_handoff_timing_active();
    perf::PresentHandoffRecord identity;

    explicit PresentHandoffTrace(uint64_t source_seq = 0, uint64_t publication_id = 0,
                                 int slot = -1) {
        identity.source_seq = source_seq;
        identity.publication_id = publication_id;
        identity.slot = slot;
    }
    uint64_t now() const { return active ? perf::monotonic_now_ns() : 0; }
    void emit(perf::PresentHandoffEvent event, int result = 0, uint64_t begin_ns = 0,
              uint64_t other_seq = 0, uint64_t address = 0) const {
        if (!active) return;
        auto record = identity;
        record.monotonic_ns = now();
        record.begin_ns = begin_ns;
        record.event = event;
        record.result = result;
        record.other_seq = other_seq;
        record.address = address;
        perf::interactive_performance_capture().record_present(std::move(record));
    }
};

} // namespace prosper::frontend
