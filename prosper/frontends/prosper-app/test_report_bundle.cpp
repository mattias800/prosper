// ADR 0021, static slice: the versioned bug-report writer. Each TEST pins one behaviour
// of the bytes a report carries, with a message saying why a wrong value is harmful: a
// report whose typetag, revision, or switches lie is worse than no report, because triage
// trusts it without re-running anything.
#include <gtest/gtest.h>

#include <cstdlib>
#include <map>
#include <string>

#include "report_bundle.hpp"

using prosper::frontend::ReportBundle;
using prosper::frontend::report_bundle_json;
using prosper::frontend::report_host_os;
using prosper::frontend::report_json_escape;

TEST(ReportBundle, NamesItsTypeAndSchema) {
    EXPECT_NE(report_bundle_json(ReportBundle{}).find("\"type\":\"report\""),
              std::string::npos)
        << "a reader that cannot see the type cannot route the file";
    EXPECT_NE(report_bundle_json(ReportBundle{}).find("\"schema\":1"), std::string::npos)
        << "no schema, no versioned evolution of the bundle";
}

TEST(ReportBundle, RecordsBuildIdentityAndTitle) {
    ReportBundle bundle;
    bundle.build_revision = std::string(40, 'a');
    bundle.build_source_identity = "0123456789abcdef";
    bundle.os = "linux";
    bundle.title_id = "PPSA00000";
    const std::string line = report_bundle_json(bundle);
    EXPECT_NE(line.find("\"tool\":\"prosper-app\""), std::string::npos)
        << "the producing harness must be named in the artifact";
    EXPECT_NE(line.find("\"build_revision\":\"" + std::string(40, 'a') + "\""),
              std::string::npos)
        << "trap 81: a report must say which build produced it";
    EXPECT_NE(line.find("\"build_source_identity\":\"0123456789abcdef\""), std::string::npos)
        << "HEAD alone does not see dirty or untracked compiled edits";
    EXPECT_NE(line.find("\"os\":\"linux\""), std::string::npos) << "host goes in the bundle";
    EXPECT_NE(line.find("\"title_id\":\"PPSA00000\""), std::string::npos)
        << "the report must name the title it describes";
}

TEST(ReportBundle, SwitchesAreSortedAndEscaped) {
    ReportBundle bundle;
    // Positive control for the sort discriminator: reversed insertion order passes only if
    // the emission sorts.
    bundle.switches = {{"PROSPER_Z_LAST", "1"}, {"PROSPER_A_FIRST", "x\"y\nz"}};
    const std::string line = report_bundle_json(bundle);
    const size_t a = line.find("PROSPER_A_FIRST");
    const size_t z = line.find("PROSPER_Z_LAST");
    ASSERT_NE(a, std::string::npos);
    ASSERT_NE(z, std::string::npos);
    EXPECT_LT(a, z) << "two reports must diff cleanly";
    EXPECT_NE(line.find("x\\\"y\\nz"), std::string::npos)
        << "switch values need the same escaping as every other manifest string";
    EXPECT_NE(report_bundle_json(ReportBundle{}).find("\"switches\":{}"), std::string::npos)
        << "a run with no switches set still writes the record";
}

TEST(ReportBundle, HostOsIsOneOfThreeSpellings) {
    const std::string os = report_host_os();
    EXPECT_TRUE(os == "windows" || os == "linux" || os == "macos")
        << "the os field has exactly three spellings; anything else breaks consumers";
}

TEST(ReportSwitches, CollectorSeesOnlyTheProsperSurface) {
#ifdef _WIN32
    _putenv_s("PROSPER_TEST_REPORT_SLICE_PROBE", "probe-value");
#else
    setenv("PROSPER_TEST_REPORT_SLICE_PROBE", "probe-value", 1);
#endif
    const auto collected = prosper::frontend::collect_report_switches();
#ifdef _WIN32
    _putenv_s("PROSPER_TEST_REPORT_SLICE_PROBE", "");
#else
    unsetenv("PROSPER_TEST_REPORT_SLICE_PROBE");
#endif
    const auto probe = collected.find("PROSPER_TEST_REPORT_SLICE_PROBE");
    ASSERT_NE(probe, collected.end()) << "a set PROSPER_* switch must be collected";
    EXPECT_EQ(probe->second, "probe-value") << "values travel verbatim";
    EXPECT_EQ(collected.find("PATH"), collected.end());
    EXPECT_EQ(collected.find("Path"), collected.end())
        << "the collector leaves non-PROSPER_* environment out of the record";
}
