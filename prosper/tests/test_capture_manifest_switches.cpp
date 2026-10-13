// Switch evidence must preserve values and distinguish a failed collection from
// a successful empty map. Otherwise manifests can certify unequal environments.
#include "shared/capture/capture_manifest.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <map>
#include <optional>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {
using namespace prosper::screenshot;
constexpr const char* kProbe = "PROSPER_TEST_MANIFEST_SWITCH_PROBE";

class CaptureSwitches : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef _WIN32
        const DWORD size =
            GetEnvironmentVariableW(L"PROSPER_TEST_MANIFEST_SWITCH_PROBE", nullptr, 0);
        if (size != 0) {
            std::wstring value(size, L'\0');
            ASSERT_EQ(
                GetEnvironmentVariableW(L"PROSPER_TEST_MANIFEST_SWITCH_PROBE", value.data(), size),
                size - 1);
            value.pop_back();
            previous_ = value;
        }
#else
        // NOLINTNEXTLINE(concurrency-mt-unsafe): This isolated CTest process has no environment-writing peer.
        if (const char* value = std::getenv(kProbe)) previous_ = value;
#endif
    }

    void TearDown() override {
#ifdef _WIN32
        EXPECT_TRUE(SetEnvironmentVariableW(L"PROSPER_TEST_MANIFEST_SWITCH_PROBE",
                                            previous_ ? previous_->c_str() : nullptr));
#else
        // NOLINTNEXTLINE(concurrency-mt-unsafe): Restore this isolated CTest process before exit.
        EXPECT_EQ(previous_ ? setenv(kProbe, previous_->c_str(), 1) : unsetenv(kProbe), 0);
#endif
    }

#ifdef _WIN32
    std::optional<std::wstring> previous_;
#else
    std::optional<std::string> previous_;
#endif
};

TEST(CaptureSwitchRecord, SortedEscapedValues) {
    const std::map<std::string, std::string> switches{{"PROSPER_Z_LAST", "1"},
                                                      {"PROSPER_A_FIRST", "x\"y\nz=a"}};
    EXPECT_EQ(manifest_switches_json(switches),
              "{\"type\":\"switches\",\"schema\":1,\"switches\":{"
              "\"PROSPER_A_FIRST\":\"x\\\"y\\nz=a\",\"PROSPER_Z_LAST\":\"1\"}}");
    EXPECT_EQ(manifest_switches_json({}), "{\"type\":\"switches\",\"schema\":1,\"switches\":{}}");
}

TEST(CaptureSwitchRecord, BuildSourceIdentity) {
    const std::string run = manifest_run_json(CaptureRunConfig{});
    EXPECT_NE(run.find("\"build_source_identity\":\""), std::string::npos);
    EXPECT_EQ(run.find("\"build_source_identity\":\"\""), std::string::npos);
}

TEST_F(CaptureSwitches, CollectsCompleteUtf8ValueAndFiltersOtherVariables) {
    const std::string value = "caf\xc3\xa9=\xf0\x9f\x99\x82\"\n";
#ifdef _WIN32
    const wchar_t wide_value[] = {L'c',
                                  L'a',
                                  L'f',
                                  static_cast<wchar_t>(0x00e9),
                                  L'=',
                                  static_cast<wchar_t>(0xd83d),
                                  static_cast<wchar_t>(0xde42),
                                  L'"',
                                  L'\n',
                                  L'\0'};
    ASSERT_TRUE(SetEnvironmentVariableW(L"PROSPER_TEST_MANIFEST_SWITCH_PROBE", wide_value));
#else
    // NOLINTNEXTLINE(concurrency-mt-unsafe): Set the probe before collection in this isolated process.
    ASSERT_EQ(setenv(kProbe, value.c_str(), 1), 0);
#endif
    const auto switches = collect_prosper_switches();
    if (!switches) {
        FAIL() << "Collection must succeed for this valid environment";
        return;
    }
    ASSERT_TRUE(switches->contains(kProbe));
    EXPECT_EQ(switches->at(kProbe), value);
    EXPECT_FALSE(switches->contains("PATH"));
    EXPECT_FALSE(switches->contains("Path"));
    const std::string json = manifest_switches_json(*switches);
    EXPECT_NE(json.find("caf\xc3\xa9=\xf0\x9f\x99\x82\\\"\\n"), std::string::npos);
}

TEST_F(CaptureSwitches, AbsentProbeIsNotInvented) {
#ifdef _WIN32
    ASSERT_TRUE(SetEnvironmentVariableW(L"PROSPER_TEST_MANIFEST_SWITCH_PROBE", nullptr));
#else
    // NOLINTNEXTLINE(concurrency-mt-unsafe): Set the probe before collection in this isolated process.
    ASSERT_EQ(unsetenv(kProbe), 0);
#endif
    const auto switches = collect_prosper_switches();
    if (!switches) {
        FAIL() << "Collection must succeed for this valid environment";
        return;
    }
    EXPECT_FALSE(switches->contains(kProbe));
}

#ifdef _WIN32
TEST_F(CaptureSwitches, WindowsNamesAreCaseInsensitive) {
    ASSERT_TRUE(SetEnvironmentVariableW(L"PROSPER_TEST_MANIFEST_SWITCH_PROBE", nullptr));
    ASSERT_TRUE(SetEnvironmentVariableW(L"prosper_test_manifest_switch_probe", L"case-value"));
    const auto switches = collect_prosper_switches();
    if (!switches) {
        FAIL() << "Collection must succeed for this valid environment";
        return;
    }
    ASSERT_TRUE(switches->contains(kProbe));
    EXPECT_EQ(switches->at(kProbe), "case-value");
}

TEST_F(CaptureSwitches, InvalidUnicodeRefusesCollection) {
    const wchar_t invalid[] = {static_cast<wchar_t>(0xd800), L'\0'};
    ASSERT_TRUE(SetEnvironmentVariableW(L"PROSPER_TEST_MANIFEST_SWITCH_PROBE", invalid));
    EXPECT_FALSE(collect_prosper_switches().has_value());
}
#endif
} // namespace
