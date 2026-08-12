// json_writer.hpp — Minimal JSON writer for boot timeline (PROSPER_DIAGNOSTICS build only).
//
// Produces a compact JSON array of boot phase events suitable for
// machine parsing or human review.

#pragma once

#include "../core/types.hpp"
#include <string>
#include <vector>

namespace prosper::diagnostics {

class JsonWriter {
public:
    // Write boot events as JSON array string.
    // Format: [{"phase":"NAME","timestamp_ms":N.n}, ...]
    static std::string write_events(const std::vector<BootEvent>& events);

    // Write single event as JSON object.
    static std::string write_event(const BootEvent& event);
};

} // namespace prosper::diagnostics
