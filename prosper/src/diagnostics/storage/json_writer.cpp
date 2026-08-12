// json_writer.cpp — Minimal JSON writer implementation.

#include "json_writer.hpp"
#include <cstdio>
#include <cmath>

namespace prosper::diagnostics {

std::string JsonWriter::write_event(const BootEvent& event) {
    char buf[128];
    // Use %g for compact floating-point output
    int n = snprintf(buf, sizeof(buf),
        "{\"phase\":\"%s\",\"timestamp_ms\":%.3g}",
        phase_name(event.phase), event.timestamp_ms);
    return std::string(buf, (n > 0 && n < (int)sizeof(buf)) ? n : 0);
}

std::string JsonWriter::write_events(const std::vector<BootEvent>& events) {
    if (events.empty()) return "[]";

    std::string result = "[\n";
    for (size_t i = 0; i < events.size(); ++i) {
        result += "  " + write_event(events[i]);
        if (i + 1 < events.size()) result += ",";
        result += "\n";
    }
    result += "]";
    return result;
}

} // namespace prosper::diagnostics
