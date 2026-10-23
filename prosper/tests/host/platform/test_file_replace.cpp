// A failed install must preserve both the completed source and the previous destination.
#include "host/platform/file_replace.hpp"
#include "fixtures/test_scratch.h"
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <string>
#include <system_error>

namespace {
std::string read_bytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
}   // namespace

TEST(FileReplace, InstallsOverExistingFile) {
    const auto directory = prosper_test::test_scratch_path("replace-existing");
    std::filesystem::create_directory(directory);
    const auto source = directory / "new";
    const auto destination = directory / "old";
    std::ofstream(source) << "complete new bytes";
    std::ofstream(destination) << "old bytes";
    std::error_code error = std::make_error_code(std::errc::io_error);
    prosper::host::replace_file(source, destination, error);
    EXPECT_FALSE(error);
    EXPECT_EQ(read_bytes(destination), "complete new bytes");
    EXPECT_FALSE(std::filesystem::exists(source));
}

TEST(FileReplace, FailedInstallPreservesSourceAndDestination) {
    const auto directory = prosper_test::test_scratch_path("replace-failed");
    std::filesystem::create_directory(directory);
    const auto source = directory / "new";
    const auto destination = directory / "existing-directory";
    std::ofstream(source) << "complete new bytes";
    std::filesystem::create_directory(destination);
    std::ofstream(destination / "marker") << "old bytes";
    std::error_code error;
    prosper::host::replace_file(source, destination, error);
    EXPECT_TRUE(error);
    EXPECT_EQ(read_bytes(source), "complete new bytes");
    EXPECT_EQ(read_bytes(destination / "marker"), "old bytes");
}
