#pragma once
// report_bundle.hpp - ADR 0021, static slice: `prosper-app --report <dir>` writes one
// complete, shareable bug report. This header is the versioned bundle writer over a plain
// struct: title identity, build, host, and the switches in force. Perf-alarm summary,
// unimplemented NIDs, log tail, and fault backtraces arrive as later sections (same schema
// version bump); the reader in prosper/tools/report/ already tolerates their absence.
//
// Pure and dependency-free by the game_path.hpp rule: stdlib only, no SDL/Vulkan/dump, so the
// exact bytes a report carries are unit-tested without a window or a title. The JSON escaper
// mirrors tools/screenshot/capture_manifest.cpp's rather than including it: that TU belongs to
// another tool, and this header must stay includable from the app alone. There is no JSON
// library in the tree (same call game_library.hpp makes for param.json).

#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>
#include <string>

#ifdef _WIN32
// Same guard as main.cpp: windows.h must not redefine std::max. This header comes before
// main.cpp's own include, so the guard has to live here too, not just there.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <crt_externs.h>
#else
// Global scope on purpose: a declaration inside the namespace would name
// prosper::frontend::environ, which mislinks (seen on macOS).
extern char** environ;
#endif

namespace prosper::frontend {

// Bundle format version. Bump when a section is added; readers accept any version they know
// and must fail loudly on one they do not, never by guessing.
inline constexpr int kReportBundleSchema = 1;

struct ReportBundle {
    std::string tool = "prosper-app";
    std::string build_revision;          // HEAD the binary was compiled from
    std::string build_source_identity;   // source bytes + toolchain + config at build time
    std::string os;                      // "windows" | "linux" | "macos"
    std::string title_id;                // e.g. "PPSA24651"; "" when no title was booted
    // Every PROSPER_* variable in force, sorted by name. A reader cannot tell from an
    // artifact whether it qualifies as evidence unless the switches travel with it (0023).
    std::map<std::string, std::string> switches;
};

inline std::string report_json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (const unsigned char c : value) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) {
                char escaped[7];
                std::snprintf(escaped, sizeof escaped, "\\u%04x", static_cast<unsigned>(c));
                out += escaped;
            } else {
                out += static_cast<char>(c);
            }
        }
    }
    return out;
}

inline std::string report_bundle_json(const ReportBundle& bundle) {
    std::ostringstream out;
    out << "{\"type\":\"report\",\"schema\":" << kReportBundleSchema
        << ",\"tool\":\"" << report_json_escape(bundle.tool) << "\""
        << ",\"build_revision\":\"" << report_json_escape(bundle.build_revision) << "\""
        << ",\"build_source_identity\":\"" << report_json_escape(bundle.build_source_identity)
        << "\""
        << ",\"os\":\"" << report_json_escape(bundle.os) << "\""
        << ",\"title_id\":\"" << report_json_escape(bundle.title_id) << "\""
        << ",\"switches\":{";
    bool first = true;
    // std::map iterates sorted, so two reports diff cleanly without canonicalization.
    for (const auto& entry : bundle.switches) {
        if (!first) out << ",";
        first = false;
        out << "\"" << report_json_escape(entry.first) << "\":\""
            << report_json_escape(entry.second) << "\"";
    }
    out << "}}";
    return out.str();
}

// The host OS in the bundle's spelling. Mirrors the screenshot manifest's conditions record.
inline std::string report_host_os() {
#if defined(_WIN32)
    return "windows";
#elif defined(__APPLE__)
    return "macos";
#else
    return "linux";
#endif
}

// Every PROSPER_* variable in force, sorted by name. Sampled by the --report writer after
// argument parsing, so flags that set the environment (--record, --display-mode, ...) are
// part of the record instead of silent ingredients of the run. Mirrors the screenshot
// manifest's collector, with the platform lessons baked in from the start.
inline std::map<std::string, std::string> collect_report_switches() {
    std::map<std::string, std::string> out;
    const auto record = [&out](const std::string& entry) {
        // Split at the FIRST '=': values legitimately contain '=' (routes, dims).
        const size_t at = entry.find('=');
        if (at == std::string::npos) return;
        const std::string name = entry.substr(0, at);
        if (name.size() > 8 && name.compare(0, 8, "PROSPER_") == 0)
            out[name] = entry.substr(at + 1);
    };
#ifdef _WIN32
    // GetEnvironmentStringsA returns NUL-separated "NAME=value" entries behind a final NUL.
    // environ is not reliably populated under MinGW, so the block is read instead.
    char* block = GetEnvironmentStringsA();
    if (!block) return out;
    for (char* entry = block; *entry != '\0'; entry += std::strlen(entry) + 1)
        record(entry);
    FreeEnvironmentStringsA(block);
#elif defined(__APPLE__)
    // environ is not directly addressable on Apple targets; _NSGetEnviron is documented.
    for (char** entry = *_NSGetEnviron(); entry && *entry; ++entry) record(*entry);
#else
    for (char** entry = environ; entry && *entry; ++entry) record(*entry);
#endif
    return out;
}

}  // namespace prosper::frontend
