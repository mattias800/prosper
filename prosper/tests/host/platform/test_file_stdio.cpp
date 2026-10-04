// #4378: real native stdio keeps host path bytes, append mode and open errors. Long-path producer
// coverage is in test_refused_shader_dump; these contracts do not claim network-share availability.
#include "host/platform/file_stdio.hpp"
#include "fixtures/test_scratch.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;
using namespace prosper::host;

TEST(FileStdio, RealBinaryAndAppendRoundTrip) {
    const auto path = prosper_test::test_scratch_dir() / fs::path(u8"native-\u00e9-\u03bb.bin");
    std::error_code error;
    FILE* output = open_native_file(path, FileOpenMode::WriteBinary, error);
    ASSERT_NE(output, nullptr) << error.message();
    const std::array<unsigned char, 4> bytes{0, 10, 26, 255};
    EXPECT_EQ(std::fwrite(bytes.data(), 1, bytes.size(), output), bytes.size());
    EXPECT_EQ(std::fclose(output), 0);
    FILE* input = open_native_file(path, FileOpenMode::ReadBinary, error);
    ASSERT_NE(input, nullptr) << error.message();
    std::array<unsigned char, 5> actual{};
    EXPECT_EQ(std::fread(actual.data(), 1, actual.size(), input), bytes.size());
    EXPECT_EQ(std::memcmp(actual.data(), bytes.data(), bytes.size()), 0);
    EXPECT_EQ(std::fclose(input), 0);
    const auto text = prosper_test::test_scratch_path("native-stdio.txt");
    for (const char* line : {"first\n", "second\n"}) {
        FILE* append = open_native_file(text, FileOpenMode::AppendText, error);
        ASSERT_NE(append, nullptr) << error.message();
        EXPECT_GE(std::fprintf(append, "%s", line), 0);
        EXPECT_EQ(std::fclose(append), 0);
    }
    FILE* read = open_native_file(text, FileOpenMode::ReadBinary, error);
    ASSERT_NE(read, nullptr) << error.message();
    std::array<char, 64> buffer{};
    const size_t size = std::fread(buffer.data(), 1, buffer.size(), read);
    EXPECT_EQ(std::fclose(read), 0);
    std::string result(buffer.data(), size);
    result.erase(std::remove(result.begin(), result.end(), '\r'), result.end());
    EXPECT_EQ(result, "first\nsecond\n");
}

TEST(FileStdio, MissingFileAndInvalidSpellingExposeErrors) {
    std::error_code error;
    EXPECT_EQ(open_native_file(prosper_test::test_scratch_path("absent/child.bin"),
                               FileOpenMode::WriteBinary, error),
              nullptr);
    EXPECT_TRUE(error) << "a missing parent must not be a successful empty observation";
    EXPECT_FALSE(error.message().empty());
    EXPECT_TRUE(native_file_path({}, error).empty());
    EXPECT_EQ(error, std::make_error_code(std::errc::invalid_argument));
    auto invalid = prosper_test::test_scratch_path("embedded-null").native();
    invalid.push_back(fs::path::value_type(0));
    invalid += fs::path("suffix").native();
    EXPECT_TRUE(native_file_path(fs::path(invalid), error).empty());
    EXPECT_EQ(error, std::make_error_code(std::errc::invalid_argument));
}

TEST(FileStdio, RelativePathFreezesWithoutChangingTheCurrentDirectory) {
    std::error_code error;
    const auto cwd = fs::current_path(error);
    ASSERT_FALSE(error);
    const auto actual = absolute_file_path(fs::path("relative-stdio") / "child.bin", error);
    ASSERT_FALSE(error);
    EXPECT_EQ(actual, cwd / "relative-stdio" / "child.bin");
    EXPECT_EQ(fs::current_path(), cwd);
    EXPECT_TRUE(actual.is_absolute());
}

// Namespace spelling is an OS primitive, not a reason to skip real filesystem tests elsewhere.
#ifdef _WIN32
TEST(FileStdio, ExtendedDriveUncAndAlreadyExtendedSpellings) {
    std::error_code error;
    EXPECT_EQ(native_file_path(fs::path(L"C:/root/../leaf/file.bin"), error).native(),
              L"\\\\?\\C:\\leaf\\file.bin");
    EXPECT_FALSE(error);
    EXPECT_EQ(native_file_path(fs::path(L"\\\\server\\share\\leaf\\file.bin"), error).native(),
              L"\\\\?\\UNC\\server\\share\\leaf\\file.bin");
    EXPECT_FALSE(error);
    const fs::path extended(L"\\\\?\\C:\\literal\\file.bin");
    EXPECT_EQ(native_file_path(extended, error), extended);
    EXPECT_FALSE(error);
    const fs::path extended_namespace(L"\\\\?\\GLOBALROOT\\Device\\native-spelling-only");
    EXPECT_EQ(native_file_path(extended_namespace, error), extended_namespace);
    EXPECT_FALSE(error);   // Pass-through spelling, not an availability or device-IO claim.
    EXPECT_TRUE(native_file_path(fs::path(L"C:relative.bin"), error).empty());
    EXPECT_EQ(error, std::make_error_code(std::errc::operation_not_supported));
    EXPECT_TRUE(native_file_path(fs::path(L"\\\\.\\device"), error).empty());
    EXPECT_EQ(error, std::make_error_code(std::errc::operation_not_supported));
}
#else
TEST(FileStdio, PosixComponentsRetainSymlinkSensitiveSpelling) {
    std::error_code error;
    const auto path = fs::absolute(prosper_test::test_scratch_path("link/../child.bin"));
    EXPECT_EQ(native_file_path(path, error), path);
    EXPECT_FALSE(error);
}
#endif
