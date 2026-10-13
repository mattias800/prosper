// ADR 0021, static slice: the versioned bug-report writer. Each TEST pins one behaviour
// of the bytes a report carries, with a message saying why a wrong value is harmful: a
// report whose typetag, revision, or switches lie is worse than no report, because triage
// trusts it without re-running anything.
#include <gtest/gtest.h>

#include <cstdlib>
#include <map>
#include <string>

#include "report_bundle.hpp"

using prosper::frontend::report_bundle_json;
using prosper::frontend::report_host_os;
using prosper::frontend::report_json_escape;
using prosper::frontend::ReportBundle;

TEST(ReportBundle, NamesItsTypeAndSchema) {
    EXPECT_NE(report_bundle_json(ReportBundle{}).find("\"type\":\"report\""), std::string::npos)
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
    EXPECT_NE(line.find("\"build_revision\":\"" + std::string(40, 'a') + "\""), std::string::npos)
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

TEST(ReportBundle, ShareableSwitchValuesNeverContainConfiguredPaths) {
    ReportBundle bundle;
    bundle.switches = {{"PROSPER_GAME_ROOT", "/private/example/games"},
                       {"PROSPER_CAPTURE_DIR", "C:\\example\\captures"},
                       {"PROSPER_GUEST_ARGS", "-logFile=/private/example/log.txt"},
                       {"PROSPER_/private/example", "1"},
                       {"PROSPER_GFXLOG", "1"}};
    const auto line = report_bundle_json(bundle);
    EXPECT_EQ(line.find("/private/example"), std::string::npos);
    EXPECT_EQ(line.find("example"), std::string::npos);
    EXPECT_NE(line.find("<redacted-path>"), std::string::npos);
    EXPECT_NE(line.find("\"PROSPER_GFXLOG\":\"1\""), std::string::npos);
}

TEST(ReportBundle, TitleIdentityRejectsPathsAndMalformedMetadata) {
    EXPECT_TRUE(prosper::frontend::report_title_id(""));
    EXPECT_TRUE(prosper::frontend::report_title_id("PPSA00000"));
    for (const std::string title : {"/private/example", "PPSA0000", "PPSA0000x"})
        EXPECT_FALSE(prosper::frontend::report_title_id(title));
}

TEST(ReportSwitches, CollectorSeesOnlyTheProsperSurface) {
#ifdef _WIN32
    _putenv_s("PROSPER_TEST_REPORT_SLICE_PROBE", "probe-value");
#else
    // NOLINTNEXTLINE(concurrency-mt-unsafe): isolated single-threaded test.
    setenv("PROSPER_TEST_REPORT_SLICE_PROBE", "probe-value", 1);
#endif
    const auto collected = prosper::frontend::collect_report_switches();
#ifdef _WIN32
    _putenv_s("PROSPER_TEST_REPORT_SLICE_PROBE", "");
#else
    // NOLINTNEXTLINE(concurrency-mt-unsafe): isolated single-threaded test.
    unsetenv("PROSPER_TEST_REPORT_SLICE_PROBE");
#endif
    if (!collected) {
        FAIL() << "failed collection is not an empty environment";
        return;
    }
    const auto& switches = *collected;
    const auto probe = switches.find("PROSPER_TEST_REPORT_SLICE_PROBE");
    ASSERT_NE(probe, switches.end()) << "a set PROSPER_* switch must be collected";
    EXPECT_EQ(probe->second, "probe-value") << "values travel verbatim";
    EXPECT_EQ(switches.find("PATH"), switches.end());
    EXPECT_EQ(switches.find("Path"), switches.end())
        << "the collector leaves non-PROSPER_* environment out of the record";
}
