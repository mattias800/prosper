// Optional diagnostic CPU-test-only retention of the ACTUAL emitted module. No default files,
// lowering inputs, guest entry authority or GPU shader identity are supplied by this observer.
#pragma once
#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace prosper::test::fragment_draw {
inline void retain_source(const std::vector<uint32_t>& words, const char* scenario) {
    const char* root = std::getenv("PROSPER_FRAGMENT_DRAW_SPV_DIRECTORY");
    if (words.empty() || !root || !*root) return;
    const auto* test = ::testing::UnitTest::GetInstance()->current_test_info();
    EXPECT_NE(test, nullptr);
    if (!test) return;
    const auto name = std::string(test->test_suite_name()) + "_" + test->name() + "_" + scenario;
    std::error_code error;
    std::filesystem::create_directories(root, error);
    EXPECT_FALSE(error) << name << ": SOURCE directory: " << error.message();
    if (error) return;
    std::ofstream file(std::filesystem::path(root) / (name + ".spv"), std::ios::binary);
    file.write(reinterpret_cast<const char*>(words.data()),
               static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    file.close();
    EXPECT_TRUE(bool(file)) << name << ": actual CPU-test SOURCE retention";
}
} // namespace prosper::test::fragment_draw
