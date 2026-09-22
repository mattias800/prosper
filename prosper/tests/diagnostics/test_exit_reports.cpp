// The end-of-run report registry (src/diagnostics/exit_reports.hpp), #3353.
//
// The defect: end-of-run diagnostics were registered with std::atexit, and every frontend and the
// guest's own exit end the process with _Exit/_exit, which skip atexit -- so the reports never
// printed on the runs that produce this project's evidence, and the silence read as a zero.
//
// The subprocess arms are the ones that matter. "noflush" is the POSITIVE CONTROL for the defect,
// built here rather than inferred: a report registered and then `_Exit` WITHOUT a flush must print
// nothing, which proves the child's exit really bypasses atexit. "flush" is the fix: the same exit
// after flush_exit_reports() prints the report. "return" proves the atexit fallback still covers an
// ordinary return from main().
#include "diagnostics/exit_reports.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif

using namespace prosper::diagnostics;

namespace {

// Counters live in functions, not in mutable globals: a report is a plain callable with no
// captures, so it reaches its counter through the function that owns it.
int& failures() { static int n = 0; return n; }
int& a_runs() { static int n = 0; return n; }
int& b_runs() { static int n = 0; return n; }
int& c_runs() { static int n = 0; return n; }

void check(const char* what, bool ok) {
    std::fputs(ok ? "  PASS  " : "  FAIL  ", stdout);
    std::fputs(what, stdout);
    std::fputs("\n", stdout);
    if (!ok) ++failures();
}

bool has(std::string_view text, std::string_view needle) {
    return text.find(needle) != std::string_view::npos;
}

constexpr std::string_view kMarker = "EXIT-REPORT-MARKER-3353";

void report_a() { ++a_runs(); }
void report_b() { ++b_runs(); }
void report_c() { ++c_runs(); }
void report_marker() { std::fputs("EXIT-REPORT-MARKER-3353\n", stderr); }

std::string run_child(const char* self, const char* mode) {
    const std::string command = std::string("\"") + self + "\" " + mode + " 2>&1";
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) return "<popen failed>";
    std::string out;
    std::array<char, 256> buffer{};
    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe)) out += buffer.data();
    pclose(pipe);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) {
        register_exit_report(&report_marker);
        if (!std::strcmp(argv[1], "--child-flush")) { flush_exit_reports(); std::_Exit(0); }
        if (!std::strcmp(argv[1], "--child-noflush")) { std::fflush(nullptr); std::_Exit(0); }
        if (!std::strcmp(argv[1], "--child-twice")) { flush_exit_reports(); return 0; }
        if (!std::strcmp(argv[1], "--child-return")) return 0;
        return 2;
    }

    std::fputs("exit report registry\n", stdout);

    // In-process: each report runs at most once, in order, and a late registration is not lost.
    register_exit_report(&report_a);
    register_exit_report(&report_b);
    register_exit_report(nullptr);   // ignored, not a crash at flush
    flush_exit_reports();
    check("both registered reports ran once", a_runs() == 1 && b_runs() == 1);
    flush_exit_reports();
    check("a second flush reruns nothing", a_runs() == 1 && b_runs() == 1);
    register_exit_report(&report_c);
    flush_exit_reports();
    check("a report registered after a flush runs at the next one",
          c_runs() == 1 && a_runs() == 1 && b_runs() == 1);

    // Subprocesses: the exit paths themselves.
    const std::string noflush = run_child(argv[0], "--child-noflush");
    check("CONTROL: _Exit without a flush loses the report (atexit really is skipped)",
          !has(noflush, kMarker) && !has(noflush, "popen failed"));
    const std::string flushed = run_child(argv[0], "--child-flush");
    check("_Exit after flush_exit_reports() prints the report", has(flushed, kMarker));
    const std::string returned = run_child(argv[0], "--child-return");
    check("an ordinary return from main still prints it (atexit fallback)", has(returned, kMarker));
    const std::string twice = run_child(argv[0], "--child-twice");
    const size_t first = twice.find(kMarker);
    check("an explicit flush followed by the atexit fallback prints it exactly once",
          first != std::string::npos && !has(std::string_view(twice).substr(first + 1), kMarker));

    std::fputs(failures() ? "FAILURES PRESENT\n" : "all passed\n", stdout);
    return failures() ? 1 : 0;
}
