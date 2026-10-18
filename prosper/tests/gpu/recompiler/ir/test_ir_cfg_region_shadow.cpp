// test_ir_cfg_region_shadow -- the step-1 shadow check (region_shadow.hpp): the region tree's
// structured ifs equal detect_forward_ifs' at the pc level, on every committed guest-program
// fixture in tests/data under each stage's admission rules, and on synthetic shapes. One
// hand-built program the two structurers are known to disagree on is the positive control that
// the comparison can see a difference at all.
#include "gpu/recompiler/ir/cfg/region_shadow.hpp"

#include "fixtures/test_data.h"
#include "gpu/recompiler/ir/cfg_streams.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace prosper::gpu::ir::cfg;
using namespace prosper::test::cfg_streams;

namespace {

uint8_t nibble(char c) {
    return static_cast<uint8_t>(c <= '9' ? c - '0' : (c | 0x20) - 'a' + 10);
}

// A tests/data program: `.hex` holds the little-endian bytes as hex text, `.bin` the raw bytes.
std::vector<uint32_t> load_words(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    std::vector<uint8_t> bytes;
    if (path.extension() == ".hex") {
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
        for (size_t i = 0; i + 1 < text.size(); i += 2)
            bytes.push_back(static_cast<uint8_t>(nibble(text[i]) << 4 | nibble(text[i + 1])));
    } else {
        bytes.assign(text.begin(), text.end());
    }
    std::vector<uint32_t> words(bytes.size() / 4);
    for (size_t i = 0; i < words.size(); ++i)
        words[i] = bytes[4 * i] | bytes[4 * i + 1] << 8 | bytes[4 * i + 2] << 16 |
                   static_cast<uint32_t>(bytes[4 * i + 3]) << 24;
    return words;
}

std::vector<std::filesystem::path> guest_program_fixtures() {
    std::vector<std::filesystem::path> out;
    const auto data = prosper::test::tests_root(__FILE__) / "data";
    for (const auto& entry : std::filesystem::directory_iterator(data)) {
        const auto ext = entry.path().extension();
        const auto name = entry.path().filename().string();
        // Every guest-code fixture: the .hex programs and the raw shader .bin files. The
        // il2cpp symbol table and the console oracle folder are not guest code.
        if (ext == ".hex" || (ext == ".bin" && name.find("symtab") == std::string::npos))
            out.push_back(entry.path());
    }
    std::sort(out.begin(), out.end());
    return out;
}

ForwardIfShadow shadow_of(const std::vector<uint32_t>& words, ShadowStage stage) {
    return shadow_forward_ifs(walk(words), stage, words.data(), words.size());
}

TEST(IrRegionShadow, EveryCommittedFixtureAgreesUnderEveryStage) {
    const auto fixtures = guest_program_fixtures();
    ASSERT_GE(fixtures.size(), 9u) << "tests/data lost guest-program fixtures";
    size_t compared = 0;
    for (const auto& path : fixtures) {
        const std::vector<uint32_t> words = load_words(path);
        ASSERT_FALSE(words.empty()) << path;
        for (ShadowStage stage :
             {ShadowStage::Vertex, ShadowStage::Fragment, ShadowStage::Compute}) {
            const ForwardIfShadow shadow = shadow_of(words, stage);
            EXPECT_NE(shadow.verdict, ForwardIfShadow::Verdict::Disagree)
                << path.filename() << " stage " << static_cast<int>(stage) << ": "
                << shadow.difference;
            if (shadow.verdict == ForwardIfShadow::Verdict::Agree) ++compared;
        }
    }
    // Not every fixture is acyclic; at least some must actually be compared.
    EXPECT_GT(compared, 0u);
}

TEST(IrRegionShadow, CategoriesOnSyntheticShapes) {
    const std::vector<uint32_t> linear = {kSMovS0Zero, kEndpgm};
    EXPECT_STREQ(shadow_category(shadow_of(linear, ShadowStage::Fragment)), "agree-linear");

    const std::vector<uint32_t> if_else = {sopp(kVccnz, 2), kVNop, sopp(kBranch, 1), kVNop,
                                           kEndpgm};
    const ForwardIfShadow structured = shadow_of(if_else, ShadowStage::Vertex);
    EXPECT_STREQ(shadow_category(structured), "agree-structured");
    ASSERT_EQ(structured.tree.size(), 1u);
    EXPECT_TRUE(structured.tree[0].has_else);
    EXPECT_EQ(structured.tree[0].merge_pc, 4u);

    const std::vector<uint32_t> refused = {sopp(kExecnz, 1), kVNop, kEndpgm};
    EXPECT_STREQ(shadow_category(shadow_of(refused, ShadowStage::Vertex)), "agree-refused");

    // A counted loop: s_branch back-edge with a forward scc exit.
    const std::vector<uint32_t> loop = {kSMovS0Zero, sopp(kScc0, 1), sopp(kBranch, -3), kEndpgm};
    EXPECT_STREQ(shadow_category(shadow_of(loop, ShadowStage::Fragment)), "loops");
}

// Positive control, built by hand outside any generator: an inner if in the then-arm of an
// if/else targets the ELSE arm. detect_forward_ifs' nesting check only bounds it by the if/else's
// merge and accepts; the region tree bounds it by the then-arm and refuses.
TEST(IrRegionShadow, AnIfIntoTheElseArmIsADisagreement) {
    const std::vector<uint32_t> words = {sopp(kScc0, 3), sopp(kScc1, 3), kVNop,  sopp(kBranch, 2),
                                         kVNop,          kVNop,          kEndpgm};
    const ForwardIfShadow shadow = shadow_of(words, ShadowStage::Fragment);
    EXPECT_EQ(shadow.verdict, ForwardIfShadow::Verdict::Disagree);
    EXPECT_FALSE(shadow.detector_refused);
    EXPECT_TRUE(shadow.tree_unstructured);
    EXPECT_STREQ(shadow_category(shadow), "disagree");
}

}   // namespace
