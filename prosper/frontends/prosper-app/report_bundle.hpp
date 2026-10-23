#pragma once
// report_bundle.hpp - ADR 0021, static slice: `prosper-app --report <dir>` writes one
// static, shareable bug report. This header is the versioned bundle writer over a plain
// struct: title identity, build, host, and the switches in force. Perf-alarm summary,
// unimplemented NIDs, log tail, and fault backtraces arrive as later sections (same schema
// version bump); the reader in prosper/tools/report/ already tolerates their absence.
//
// No SDL/Vulkan or guest execution is needed. Environment collection shares the capture
// manifest's UTF-8 and Windows case-normalization contract instead of maintaining a second copy.

#include <cstdio>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>

#include "shared/capture/capture_manifest.hpp"
#include "host/platform/file_replace.hpp"

namespace prosper::frontend {

// Consume the directory only after validation so a missing value cannot swallow another
// option. A relative path starting with '-' can be written with a './' prefix.
inline bool parse_report_directory(int& index, int argc, const char* const* argv,
                                   std::string& directory) {
    if (index + 1 >= argc || argv[index + 1][0] == '\0' || argv[index + 1][0] == '-') return false;
    directory = argv[++index];
    return true;
}

// Bundle format version. Bump when a section is added; readers accept any version they know
// and must fail loudly on one they do not, never by guessing.
inline constexpr int kReportBundleSchema = 1;

struct ReportBundle {
    std::string tool = "prosper-app";
    std::string build_revision;          // HEAD the binary was compiled from
    std::string build_source_identity;   // source bytes + toolchain + config at build time
    std::string os;                      // "windows" | "linux" | "macos"
    std::string title_id;   // e.g. "PPSA24651"; "" when no title identity was supplied
    // PROSPER_* variables, sorted by name. Path-bearing values are visibly redacted in
    // the shareable serialization; this report is not full reproduction evidence.
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

inline bool report_switch_name(const std::string& name) {
    return name.size() > 8 && name.compare(0, 8, "PROSPER_") == 0 &&
           name.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", 8) == std::string::npos;
}

inline bool report_title_id(const std::string& title) {
    return title.empty() || (title.size() == 9 && title.compare(0, 4, "PPSA") == 0 &&
                             title.find_first_not_of("0123456789", 4) == std::string::npos);
}

inline std::string report_bundle_json(const ReportBundle& bundle) {
    std::ostringstream out;
    out << "{\"type\":\"report\",\"schema\":" << kReportBundleSchema << ",\"tool\":\""
        << report_json_escape(bundle.tool) << "\""
        << ",\"build_revision\":\"" << report_json_escape(bundle.build_revision) << "\""
        << ",\"build_source_identity\":\"" << report_json_escape(bundle.build_source_identity)
        << "\""
        << ",\"os\":\"" << report_json_escape(bundle.os) << "\""
        << ",\"title_id\":\"" << report_json_escape(bundle.title_id) << "\""
        << ",\"switches\":{";
    bool first = true;
    // std::map iterates sorted, so two reports diff cleanly without canonicalization.
    for (const auto& entry : bundle.switches) {
        // Runtime switch names are ASCII identifiers; arbitrary environment names can
        // themselves contain private paths and do not identify a supported switch.
        if (!report_switch_name(entry.first)) continue;
        if (!first) out << ",";
        first = false;
        out << "\"" << report_json_escape(entry.first) << "\":\""
            << report_json_escape(entry.second.find_first_of("/\\") == std::string::npos
                                      ? entry.second
                                      : "<redacted-path>")
            << "\"";
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

// PROSPER_* variables in force, sorted by name. Sampled by the --report writer after
// argument parsing, so flags that set the environment (--record, --display-mode, ...) are
// part of the record instead of silent ingredients of the run. Unreadable UTF-8 or failed
// collection refuses the report, rather than looking like an empty configuration.
inline std::optional<std::map<std::string, std::string>> collect_report_switches() {
    return prosper::screenshot::collect_prosper_switches();
}

inline bool write_report_payload(const std::filesystem::path& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    out.put('\n');
    out.close();
    return static_cast<bool>(out);
}

// The writer includes close success. Injection exercises partial writes and late close errors
// against real files without requiring a full filesystem. Only a completed private sibling is
// installed; a failed write or replacement leaves the previous report available for triage.
inline bool write_report_bundle(const std::filesystem::path& directory, const ReportBundle& bundle,
                                std::error_code& error,
                                bool (*write_payload)(const std::filesystem::path&,
                                                      const std::string&) = write_report_payload) {
    const std::string text = report_bundle_json(bundle);
    std::filesystem::create_directories(directory, error);
    if (error) return false;
    static std::atomic<uint64_t> serial{0};
    std::filesystem::path temporary;
    for (unsigned attempt = 0; attempt < 16; ++attempt) {
        temporary = directory /
                    (".report-tmp-" +
                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                     "-" + std::to_string(serial.fetch_add(1)));
        if (std::filesystem::create_directory(temporary, error)) break;
        if (error || attempt == 15) {
            if (!error) error = std::make_error_code(std::errc::file_exists);
            return false;
        }
    }
    struct Cleanup {
        std::filesystem::path directory;
        ~Cleanup() {
            std::error_code ignored;
            std::filesystem::remove_all(directory, ignored);
        }
    } cleanup{temporary};
    const auto payload = temporary / "report.json";
    if (!write_payload(payload, text)) {
        error = std::make_error_code(std::errc::io_error);
        return false;
    }
    prosper::host::replace_file(payload, directory / "report.json", error);
    return !error;
}

}   // namespace prosper::frontend
