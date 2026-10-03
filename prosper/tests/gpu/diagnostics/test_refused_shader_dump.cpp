// test_refused_shader_dump (#4273) -- a refused shader leaves its evidence by default.
//
// WHAT EACH ARM KILLS:
//   WritesCodeAndIndexOnce      no dump at all (the pre-#4273 default), or one file per sighting
//   DistinctProgramsAreKept     deduplication keyed on the address instead of the code
//   CapBoundsTheDump            an unbounded dump on a title that refuses many programs
#include "gpu/diagnostics/refused_shader_dump.hpp"
#include "fixtures/test_scratch.h"

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace prosper::gpu;
namespace fs = std::filesystem;

namespace {

fs::path fresh_root(const char* name) {
    const fs::path root = prosper_test::test_scratch_dir() / name;
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);
    reset_refused_shader_dump_for_test(root.string());
    return root;
}

size_t bin_files(const fs::path& dir) {
    size_t n = 0;
    for (const auto& e : fs::directory_iterator(dir)) n += e.path().extension() == ".bin";
    return n;
}

}  // namespace

TEST(RefusedShaderDump, WritesCodeAndIndexOnce) {
    const fs::path root = fresh_root("refused-shader-once");
    EXPECT_TRUE(refused_shader_dump_directory().empty()) << "nothing is created before a refusal";
    const std::vector<uint32_t> code = {0xBE800380u, 0xBF810000u};   // s_mov_b32 s0, 0; s_endpgm
    EXPECT_TRUE(note_refused_shader("vs", 0x1234000, code.data(), code.size(), "draw-order=7"));
    EXPECT_FALSE(note_refused_shader("vs", 0x1234000, code.data(), code.size(), "draw-order=8"))
        << "the same program refused again is not dumped again";
    const fs::path dir = refused_shader_dump_directory();
    ASSERT_FALSE(dir.empty());
    EXPECT_EQ(dir.parent_path(), root);
    ASSERT_EQ(bin_files(dir), 1u);
    for (const auto& e : fs::directory_iterator(dir)) {
        if (e.path().extension() != ".bin") continue;
        std::ifstream in(e.path(), std::ios::binary);
        const std::vector<char> bytes((std::istreambuf_iterator<char>(in)), {});
        ASSERT_EQ(bytes.size(), code.size() * sizeof(uint32_t));
        EXPECT_EQ(0, std::memcmp(bytes.data(), code.data(), bytes.size())) << "raw code is kept";
    }
    std::ifstream index(dir / "index.txt");
    const std::string text((std::istreambuf_iterator<char>(index)), {});
    EXPECT_NE(text.find("vs addr=0x1234000 dwords=2"), std::string::npos) << text;
    EXPECT_NE(text.find("draw-order=7"), std::string::npos) << text;
    EXPECT_EQ(text.find("draw-order=8"), std::string::npos) << "one index line per program";
}

TEST(RefusedShaderDump, DistinctProgramsAreKept) {
    fresh_root("refused-shader-distinct");
    const std::vector<uint32_t> a = {0xBE800380u, 0xBF810000u};
    const std::vector<uint32_t> b = {0xBE800381u, 0xBF810000u};
    EXPECT_TRUE(note_refused_shader("ps", 0x2000, a.data(), a.size(), ""));
    EXPECT_TRUE(note_refused_shader("ps", 0x2000, b.data(), b.size(), ""))
        << "new code at a reused address is a different program";
    EXPECT_TRUE(note_refused_shader("cs", 0x2000, a.data(), a.size(), ""))
        << "the same words refused at another stage are recorded for that stage";
    EXPECT_EQ(bin_files(refused_shader_dump_directory()), 3u);
}

TEST(RefusedShaderDump, CapBoundsTheDump) {
    fresh_root("refused-shader-cap");
    std::vector<uint32_t> code = {0, 0xBF810000u};
    size_t written = 0;
    for (uint32_t i = 0; i < kRefusedShaderDumpMaxPrograms + 10; ++i) {
        code[0] = 0xBE800300u + i;
        written += note_refused_shader("cs", 0x3000 + i, code.data(), code.size(), "");
    }
    EXPECT_EQ(written, kRefusedShaderDumpMaxPrograms);
    EXPECT_EQ(bin_files(refused_shader_dump_directory()), kRefusedShaderDumpMaxPrograms);
}
