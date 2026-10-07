// test_refused_shader_dump (#4273) -- a refused shader leaves its evidence by default.
//
// WHAT EACH ARM KILLS:
//   WritesCodeAndIndexOnce      no dump at all (the pre-#4273 default), or one file per sighting
//   DistinctProgramsAreKept     deduplication keyed on the address instead of the code
//   CapBoundsTheDump            an unbounded dump on a title that refuses many programs
//   OwnerMemoBoundsWarmWork     aliases growing metadata or repeats rehashing original code
//   VersionStageAndReset        an address/first-word memo hiding rewritten or another-stage evidence
//   ConcurrentOwnerObservedOnce competing workers all hashing/writing the same original version
//   IndexNamesTranslatorReject  the index naming only the compute-safe coverage census, not the
//                               stage translator's own terminal reject for the program
#include "gpu/diagnostics/refused_shader_dump.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"   // record_terminal_reject_reason
#include "fixtures/test_scratch.h"

#include <gtest/gtest.h>

#include <cstring>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <utility>
#include <string>
#include <vector>
#include <thread>

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

struct OwnedCode {
    explicit OwnedCode(std::vector<uint32_t> words) : words(std::move(words)) {}
    std::vector<uint32_t> words;
    RefusedShaderMemo memo;
};

RefusedShaderSource original(const std::shared_ptr<OwnedCode>& owner) {
    return {{owner, &owner->words}, &owner->memo};
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
    const auto stats = refused_shader_dump_stats();
    EXPECT_EQ(stats.content_records, kRefusedShaderDumpMaxPrograms);
    EXPECT_EQ(stats.hash_evaluations, kRefusedShaderDumpMaxPrograms)
        << "the full guard must precede hashing, including new addresses after capacity";
    EXPECT_EQ(stats.hashed_dwords, kRefusedShaderDumpMaxPrograms * code.size());
    EXPECT_TRUE(refused_shader_dump_full());
}

TEST(RefusedShaderDump, OwnerMemoBoundsWarmWork) {
    fresh_root("refused-shader-owner");
    auto owner = std::make_shared<OwnedCode>(std::vector<uint32_t>{0xBE800380u, 0xBF810000u});
    const RefusedShaderSource source = original(owner);
    EXPECT_FALSE(refused_shader_already_noted("cs", source));
    ASSERT_TRUE(note_refused_shader("cs", 0x4000, source, ""));
    for (uint64_t alias = 0; alias != 4096; ++alias) {
        EXPECT_TRUE(refused_shader_already_noted("cs", source));
        EXPECT_FALSE(note_refused_shader("cs", 0x4000 + alias, source, ""));
    }
    const auto warm = refused_shader_dump_stats();
    EXPECT_EQ(warm.content_records, 1u) << "aliases add no address metadata";
    EXPECT_EQ(warm.hash_evaluations, 1u) << "an already byte-validated owner is not rehashed";
    EXPECT_EQ(warm.hashed_dwords, owner->words.size());
    EXPECT_EQ(bin_files(refused_shader_dump_directory()), 1u);

    // Unowned observations retain content deduplication, not an unbounded quick-address table.
    for (uint64_t alias = 0; alias != 256; ++alias)
        EXPECT_FALSE(note_refused_shader("cs", 0x8000 + alias, owner->words.data(),
                                         owner->words.size(), ""));
    EXPECT_EQ(refused_shader_dump_stats().content_records, 1u);
}

TEST(RefusedShaderDump, VersionStageAndReset) {
    fresh_root("refused-shader-version");
    auto first = std::make_shared<OwnedCode>(std::vector<uint32_t>{0xBE800380u, 0xBF810000u});
    auto changed =
        std::make_shared<OwnedCode>(std::vector<uint32_t>{0xBE800380u, 0xBE810381u, 0xBF810000u});
    ASSERT_EQ(first->words.front(), changed->words.front());
    ASSERT_TRUE(note_refused_shader("ps", 0x5000, original(first), "original"));
    ASSERT_TRUE(note_refused_shader("ps", 0x5000, original(changed), "changed"))
        << "same address and first word do not identify an immutable byte version";
    ASSERT_TRUE(note_refused_shader("vs", 0x5000, original(first), "other-stage"));
    EXPECT_EQ(refused_shader_dump_stats().hash_evaluations, 3u);
    EXPECT_EQ(bin_files(refused_shader_dump_directory()), 3u);
    reset_refused_shader_dump_for_test(
        (prosper_test::test_scratch_dir() / "refused-next-run").string());
    EXPECT_FALSE(refused_shader_already_noted("ps", original(first)));
    EXPECT_TRUE(note_refused_shader("ps", 0x5000, original(first), "next-run"));
    EXPECT_EQ(refused_shader_dump_stats().hash_evaluations, 1u);

    std::weak_ptr<OwnedCode> weak = changed;
    changed.reset();
    EXPECT_TRUE(weak.expired()) << "the dumper must not retain original source owners";
}

TEST(RefusedShaderDump, ConcurrentOwnerObservedOnce) {
    fresh_root("refused-shader-workers");
    auto owner = std::make_shared<OwnedCode>(std::vector<uint32_t>{0xBE800380u, 0xBF810000u});
    std::atomic<size_t> written{0};
    std::vector<std::thread> workers;
    for (size_t i = 0; i != 8; ++i)
        workers.emplace_back([&, i] {
            written.fetch_add(note_refused_shader("vs", 0x6000 + i, original(owner), ""));
        });
    for (auto& worker : workers) worker.join();
    EXPECT_EQ(written.load(), 1u);
    EXPECT_EQ(refused_shader_dump_stats().hash_evaluations, 1u);
    EXPECT_EQ(refused_shader_dump_stats().content_records, 1u);
    EXPECT_EQ(bin_files(refused_shader_dump_directory()), 1u);
}

TEST(RefusedShaderDump, IndexNamesTranslatorReject) {
    const fs::path root = fresh_root("refused-shader-reject-reason");
    const std::vector<uint32_t> code = {0xBE800380u, 0xBF810000u};
    const std::vector<uint32_t> other = {0xBE800381u, 0xBF810000u};
    record_terminal_reject_reason(0x7a5c000ull, "recompile-reject",
                                  "mode=unsupported pc=52 op=0x361\nstage=fragment \"x\"");
    EXPECT_TRUE(note_refused_shader("ps", 0x7a5c000, code.data(), code.size(), "draw-order=1"));
    EXPECT_TRUE(note_refused_shader("ps", 0x7a5d000, other.data(), other.size(), "draw-order=2"));
    std::ifstream index(fs::path(refused_shader_dump_directory()) / "index.txt");
    const std::string text((std::istreambuf_iterator<char>(index)), {});
    EXPECT_NE(text.find("draw-order=1 reject=\"recompile-reject mode=unsupported pc=52 op=0x361 "
                        "stage=fragment  x\"\n"),
              std::string::npos)
        << "the translator's reason is kept on the program's one line, newlines and quotes "
           "flattened so the line stays one parseable record\n"
        << text;
    EXPECT_NE(text.find("draw-order=2\n"), std::string::npos)
        << "a program with no recorded reason carries no reject field\n" << text;
}
