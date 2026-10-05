// test_refused_shader_dump (#4273) -- a refused shader leaves its evidence by default.
//
// WHAT EACH ARM KILLS:
//   WritesCodeAndIndexOnce      no dump at all (the pre-#4273 default), or one file per sighting
//   DistinctProgramsAreKept     deduplication keyed on the address instead of the code
//   CapBoundsTheDump            an unbounded dump on a title that refuses many programs
//   OwnerMemoBoundsWarmWork     aliases growing metadata or repeats rehashing original code
//   VersionStageAndReset        an address/first-word memo hiding rewritten or another-stage evidence
//   ConcurrentOwnerObservedOnce competing workers all hashing/writing the same original version
#include "gpu/diagnostics/refused_shader_dump.hpp"
#include "fixtures/test_scratch.h"
#include "host/platform/file_stdio.hpp"

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
#include <algorithm>

using namespace prosper::gpu;
namespace fs = std::filesystem;

namespace {

fs::path fresh_root(const char* name) {
    const fs::path root = fs::absolute(prosper_test::test_scratch_dir() / name).lexically_normal();
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

fs::path native_path(const fs::path& path) {
    std::error_code error;
    const auto native = prosper::host::native_file_path(path, error);
    EXPECT_FALSE(error) << error.message();
    return native;
}

// Every component stays far below the component limit; only total path length is stressed.
fs::path deep_root(const char* name, size_t minimum) {
    fs::path root = fs::absolute(prosper_test::test_scratch_path(name)).lexically_normal();
    while (root.native().size() < minimum) root /= std::string(40, 'r');
    return root;
}

struct DeepRootCleanup {
    fs::path root;
    ~DeepRootCleanup() {
        std::error_code error;
        const auto native = prosper::host::native_file_path(root, error);
        if (!error && !native.empty()) fs::remove_all(native, error);
    }
};

std::string read_text(const fs::path& path) {
    std::ifstream input(native_path(path), std::ios::binary);
    EXPECT_TRUE(input.is_open());
    return {(std::istreambuf_iterator<char>(input)), {}};
}

fs::path only_bin(const fs::path& directory) {
    fs::path result;
    size_t count = 0;
    for (const auto& entry : fs::directory_iterator(native_path(directory))) {
        if (entry.path().extension() != ".bin") continue;
        EXPECT_TRUE(entry.is_regular_file());
        result = directory / entry.path().filename();
        ++count;
    }
    EXPECT_EQ(count, 1u);
    return result;
}

void expect_original_artifacts(const fs::path& root, const fs::path& expected_root,
                               const std::shared_ptr<OwnedCode>& owner) {
    reset_refused_shader_dump_for_test(root.string());
    EXPECT_TRUE(refused_shader_dump_directory().empty());
    ASSERT_TRUE(note_refused_shader("ps", 0x7000, original(owner), "long-path-proof"));
    ASSERT_FALSE(note_refused_shader("ps", 0x7001, original(owner), "not-a-second-index-line"));
    const fs::path directory = refused_shader_dump_directory();
    ASSERT_FALSE(directory.empty());
    EXPECT_EQ(directory.parent_path(), expected_root);
    const auto bin = only_bin(directory);
    ASSERT_FALSE(bin.empty());
    const auto bytes = read_text(bin);
    ASSERT_EQ(bytes.size(), owner->words.size() * sizeof(uint32_t));
    EXPECT_EQ(std::memcmp(bytes.data(), owner->words.data(), bytes.size()), 0);
    const auto text = read_text(directory / "index.txt");
    EXPECT_EQ(std::count(text.begin(), text.end(), '\n'), 1);
    EXPECT_NE(text.find("ps addr=0x7000 dwords=2"), std::string::npos);
    EXPECT_NE(text.find(bin.filename().string()), std::string::npos);
    EXPECT_NE(text.find("long-path-proof"), std::string::npos);
    EXPECT_EQ(text.find("not-a-second-index-line"), std::string::npos);
    const auto stats = refused_shader_dump_stats();
    EXPECT_EQ(stats.content_records, 1u);
    EXPECT_EQ(stats.hash_evaluations, 1u);
    EXPECT_EQ(stats.hashed_dwords, owner->words.size());
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

TEST(RefusedShaderDump, LongCaptureRootKeepsRawIndexAndOncePerOwnerWork) {
    const auto short_root =
        fs::absolute(prosper_test::test_scratch_path("refused-path-short")).lexically_normal();
    const auto long_root = deep_root("refused-path-long", 340);
    DeepRootCleanup cleanup{long_root};
    auto owner = std::make_shared<OwnedCode>(std::vector<uint32_t>{0xBE800380u, 0xBF810000u});
    expect_original_artifacts(short_root, short_root, owner);
    expect_original_artifacts(long_root, long_root, owner);
    const fs::path directory = refused_shader_dump_directory();
    ASSERT_FALSE(directory.empty());
    EXPECT_GT(directory.native().size(), 260u);
    EXPECT_GT((directory / "index.txt").native().size(), 260u);
    EXPECT_GT(only_bin(directory).native().size(), 260u);
}

TEST(RefusedShaderDump, RelativeLongRootIsResolvedBeforeNativeDirectoryAndFileOperations) {
    const auto cwd = fs::current_path();
    auto root = deep_root("refused-path-relative", 340);
    auto relative = root.lexically_relative(cwd);
    ASSERT_FALSE(relative.empty())
        << "real relative-path IO requires PROSPER_TEST_SCRATCH_DIR on the cwd volume; "
           "configure a same-volume scratch root for this case";
    for (size_t step = 0; relative.native().size() < 320 && step != 8; ++step) {
        root /= std::string(40, 'r');
        relative = root.lexically_relative(cwd);
    }
    ASSERT_FALSE(relative.empty());
    ASSERT_GE(relative.native().size(), 320u) << "bounded long-relative fixture construction";
    ASSERT_FALSE(relative.is_absolute());
    ASSERT_GT(relative.native().size(), 260u);
    DeepRootCleanup cleanup{root};
    auto owner = std::make_shared<OwnedCode>(std::vector<uint32_t>{0xBE800380u, 0xBF810000u});
    expect_original_artifacts(relative, root, owner);
    EXPECT_EQ(fs::current_path(), cwd);
}

TEST(RefusedShaderDump, RawOpenFailureIsActionableAndNotRetriedForTheOwner) {
    fresh_root("refused-path-raw-error");
    auto owner = std::make_shared<OwnedCode>(std::vector<uint32_t>{0xBE800380u, 0xBF810000u});
    ASSERT_TRUE(note_refused_shader("vs", 0x8000, original(owner), "original"));
    const fs::path directory = refused_shader_dump_directory();
    const auto original_bin = only_bin(directory);
    ASSERT_FALSE(original_bin.empty());
    auto name = original_bin.filename().string();
    ASSERT_EQ(name.substr(0, 3), "vs_");
    name.replace(0, 2, "ps");   // Same real hash/address, different observed stage.
    std::error_code error;
    ASSERT_TRUE(fs::create_directory(native_path(directory / name), error)) << error.message();
    testing::internal::CaptureStderr();
    const bool written = note_refused_shader("ps", 0x8000, original(owner), "blocked-raw");
    const auto log = testing::internal::GetCapturedStderr();
    EXPECT_FALSE(written);
    EXPECT_NE(log.find("raw-open failed:"), std::string::npos) << log;
    EXPECT_NE(log.find("error="), std::string::npos) << log;
    EXPECT_NE(log.find("refusal evidence is incomplete"), std::string::npos) << log;
    EXPECT_TRUE(refused_shader_already_noted("ps", original(owner)));
    const auto before = refused_shader_dump_stats();
    testing::internal::CaptureStderr();
    EXPECT_FALSE(note_refused_shader("ps", 0x8001, original(owner), "repeat"));
    EXPECT_TRUE(testing::internal::GetCapturedStderr().empty());
    EXPECT_EQ(refused_shader_dump_stats().hash_evaluations, before.hash_evaluations);
    EXPECT_EQ(refused_shader_dump_stats().content_records, 2u);
    EXPECT_TRUE(fs::is_directory(native_path(directory / name)));
}

TEST(RefusedShaderDump, IndexOpenFailureDoesNotPretendRawEvidenceIsAnIndex) {
    fresh_root("refused-path-index-error");
    auto owner = std::make_shared<OwnedCode>(std::vector<uint32_t>{0xBE800380u, 0xBF810000u});
    ASSERT_TRUE(note_refused_shader("vs", 0x9000, original(owner), "original"));
    const fs::path directory = refused_shader_dump_directory();
    std::error_code error;
    fs::rename(native_path(directory / "index.txt"), native_path(directory / "saved-index.txt"),
               error);
    ASSERT_FALSE(error) << error.message();
    ASSERT_TRUE(fs::create_directory(native_path(directory / "index.txt"), error));
    testing::internal::CaptureStderr();
    const bool written = note_refused_shader("ps", 0x9000, original(owner), "blocked-index");
    const auto log = testing::internal::GetCapturedStderr();
    EXPECT_TRUE(written) << "the established return value describes the raw file, not the index";
    EXPECT_NE(log.find("index-open failed:"), std::string::npos) << log;
    EXPECT_NE(log.find("error="), std::string::npos) << log;
    EXPECT_NE(log.find("refusal evidence is incomplete"), std::string::npos) << log;
    auto raw = fs::path{};
    for (const auto& entry : fs::directory_iterator(native_path(directory))) {
        if (entry.path().filename().string().starts_with("ps_")) raw = entry.path();
    }
    ASSERT_FALSE(raw.empty());
    const auto bytes = read_text(raw);
    ASSERT_EQ(bytes.size(), owner->words.size() * sizeof(uint32_t));
    EXPECT_EQ(std::memcmp(bytes.data(), owner->words.data(), bytes.size()), 0);
    EXPECT_TRUE(refused_shader_already_noted("ps", original(owner)));
    EXPECT_EQ(refused_shader_dump_stats().content_records, 2u);
}

TEST(RefusedShaderDump, DirectoryFailureIsVisibleAndBounded) {
    const auto root = prosper_test::test_scratch_path("refused-path-directory-file");
    {
        std::ofstream file(root);
        ASSERT_TRUE(file.is_open());
        file << "not a directory";
    }
    reset_refused_shader_dump_for_test(root.string());
    auto owner = std::make_shared<OwnedCode>(std::vector<uint32_t>{0xBE800380u, 0xBF810000u});
    testing::internal::CaptureStderr();
    const bool written = note_refused_shader("cs", 0xa000, original(owner), "blocked-root");
    const auto log = testing::internal::GetCapturedStderr();
    EXPECT_FALSE(written);
    EXPECT_NE(log.find("directory-create failed:"), std::string::npos) << log;
    EXPECT_NE(log.find("error="), std::string::npos) << log;
    EXPECT_TRUE(refused_shader_dump_directory().empty());
    testing::internal::CaptureStderr();
    EXPECT_FALSE(note_refused_shader("cs", 0xa001, original(owner), "repeat"));
    EXPECT_TRUE(testing::internal::GetCapturedStderr().empty());
    EXPECT_EQ(refused_shader_dump_stats().content_records, 1u);
    EXPECT_EQ(refused_shader_dump_stats().hash_evaluations, 1u);
    EXPECT_EQ(read_text(root), "not a directory");
}

TEST(RefusedShaderDump, RootResolutionFailureNamesTheConfiguredRootAndIsBounded) {
    const std::string prefix = "refused-invalid-configured-root";
    const std::string root = prefix + std::string(1, '\0') + "not-a-directory-name";
    reset_refused_shader_dump_for_test(root);
    auto owner = std::make_shared<OwnedCode>(std::vector<uint32_t>{0xBE800380u, 0xBF810000u});
    testing::internal::CaptureStderr();
    const bool written = note_refused_shader("ps", 0xb000, original(owner), "invalid-root");
    const auto log = testing::internal::GetCapturedStderr();
    EXPECT_FALSE(written);
    EXPECT_NE(log.find("root-resolve failed: " + prefix + "\\0not-a-directory-name"),
              std::string::npos)
        << log;
    EXPECT_EQ(log.find("directory-create failed:"), std::string::npos) << log;
    EXPECT_EQ(log.find("refused_shaders_"), std::string::npos) << log;
    EXPECT_NE(log.find("error="), std::string::npos) << log;
    EXPECT_TRUE(refused_shader_dump_directory().empty());
    EXPECT_TRUE(refused_shader_already_noted("ps", original(owner)));
    testing::internal::CaptureStderr();
    EXPECT_FALSE(note_refused_shader("ps", 0xb001, original(owner), "repeat"));
    EXPECT_TRUE(testing::internal::GetCapturedStderr().empty());
    EXPECT_EQ(refused_shader_dump_stats().content_records, 1u);
    EXPECT_EQ(refused_shader_dump_stats().hash_evaluations, 1u);
}
