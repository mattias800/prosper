// ADR 0021, static slice: the versioned bug-report writer. Each TEST pins one behaviour
// of the bytes a report carries, with a message saying why a wrong value is harmful: a
// report whose typetag, revision, or switches lie is worse than no report, because triage
// trusts it without re-running anything.
#include <gtest/gtest.h>

#include <stdlib.h>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <map>
#include <string>
#include <system_error>

#include "fixtures/test_scratch.h"
#include "report_bundle.hpp"

using prosper::frontend::report_bundle_json;
using prosper::frontend::report_host_os;
using prosper::frontend::report_json_escape;
using prosper::frontend::ReportBundle;

TEST(ReportArguments, ConsumesOnlyTheDirectoryAndPreservesFollowingOptions) {
    for (const char* path : {"reports", "./-reports", "/tmp/reports", "reports with spaces"}) {
        const char* argv[] = {"prosper-app", "--report", path, "--fps"};
        int index = 1;
        std::string directory;
        ASSERT_TRUE(prosper::frontend::parse_report_directory(index, 4, argv, directory));
        EXPECT_EQ(index, 2);
        EXPECT_EQ(directory, path);
        EXPECT_STREQ(argv[index + 1], "--fps");
    }
}

TEST(ReportArguments, MissingEmptyAndOptionValuesRefuseWithoutConsumingArguments) {
    const char* missing[] = {"prosper-app", "--report"};
    int index = 1;
    std::string directory = "previous";
    EXPECT_FALSE(prosper::frontend::parse_report_directory(index, 2, missing, directory));
    EXPECT_EQ(index, 1);
    EXPECT_EQ(directory, "previous");
    for (const char* value : {"", "--list-games", "--report", "--fps", "-x"}) {
        SCOPED_TRACE(value);
        const char* argv[] = {"prosper-app", "--report", value};
        EXPECT_FALSE(prosper::frontend::parse_report_directory(index, 3, argv, directory));
        EXPECT_EQ(index, 1) << "the next option must remain available to the parser";
        EXPECT_EQ(directory, "previous") << "a refused value must never become an output path";
        index = 1;
        directory = "previous";
    }
}

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

namespace {
std::string read_report_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
class ReportBundleFile : public testing::Test {
protected:
    void SetUp() override {
        directory = prosper_test::test_scratch_path(
            testing::UnitTest::GetInstance()->current_test_info()->name());
        std::filesystem::create_directories(directory);
    }
    void previous_report() {
        std::ofstream(directory / "report.json") << "previous complete report\n";
        std::ofstream(directory / "unrelated.txt") << "unrelated evidence\n";
    }
    void expect_previous_report() {
        EXPECT_EQ(read_report_file(directory / "report.json"), "previous complete report\n");
        EXPECT_EQ(read_report_file(directory / "unrelated.txt"), "unrelated evidence\n");
        EXPECT_EQ(std::distance(std::filesystem::directory_iterator(directory),
                                std::filesystem::directory_iterator()),
                  2)
            << "failed publication must clean only its private temporary";
    }
    std::filesystem::path directory;
};
}   // namespace

TEST_F(ReportBundleFile, CreatesAndReplacesOnlyWithACompleteReport) {
    std::error_code error;
    ReportBundle bundle;
    ASSERT_TRUE(prosper::frontend::write_report_bundle(directory, bundle, error));
    EXPECT_FALSE(error);
    EXPECT_EQ(read_report_file(directory / "report.json"), report_bundle_json(bundle) + "\n");
    bundle.title_id = "PPSA00000";
    ASSERT_TRUE(prosper::frontend::write_report_bundle(directory, bundle, error));
    EXPECT_EQ(read_report_file(directory / "report.json"), report_bundle_json(bundle) + "\n");
    EXPECT_EQ(std::distance(std::filesystem::directory_iterator(directory),
                            std::filesystem::directory_iterator()),
              1);
}

TEST_F(ReportBundleFile, PartialWritePreservesPreviousReport) {
    previous_report();
    std::error_code error;
    const auto partial = [](const std::filesystem::path& path, const std::string&) {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << "partial";
        out.close();
        return false;
    };
    EXPECT_FALSE(prosper::frontend::write_report_bundle(directory, ReportBundle{}, error, partial));
    EXPECT_EQ(error, std::errc::io_error);
    expect_previous_report();
}

TEST_F(ReportBundleFile, CloseFailurePreservesPreviousReport) {
    previous_report();
    std::error_code error;
    const auto failed_close = [](const std::filesystem::path& path, const std::string& text) {
        // Model the writer's late close error after every payload byte reached the temporary.
        EXPECT_TRUE(prosper::frontend::write_report_payload(path, text));
        return false;
    };
    EXPECT_FALSE(
        prosper::frontend::write_report_bundle(directory, ReportBundle{}, error, failed_close));
    EXPECT_EQ(error, std::errc::io_error);
    expect_previous_report();
}

TEST_F(ReportBundleFile, FailedInstallPreservesDestinationAndUnrelatedEvidence) {
    std::filesystem::create_directory(directory / "report.json");
    std::ofstream(directory / "report.json" / "marker") << "retained\n";
    std::ofstream(directory / "unrelated.txt") << "unrelated evidence\n";
    std::error_code error;
    EXPECT_FALSE(prosper::frontend::write_report_bundle(directory, ReportBundle{}, error));
    EXPECT_TRUE(error);
    EXPECT_EQ(read_report_file(directory / "report.json" / "marker"), "retained\n");
    EXPECT_EQ(read_report_file(directory / "unrelated.txt"), "unrelated evidence\n");
    EXPECT_EQ(std::distance(std::filesystem::directory_iterator(directory),
                            std::filesystem::directory_iterator()),
              2);
}

TEST_F(ReportBundleFile, RefusesAFileInPlaceOfTheReportDirectory) {
    const auto output = directory / "not-a-directory";
    std::ofstream(output) << "retained\n";
    std::error_code error;
    EXPECT_FALSE(prosper::frontend::write_report_bundle(output, ReportBundle{}, error));
    EXPECT_TRUE(error);
    EXPECT_EQ(read_report_file(output), "retained\n");
}
