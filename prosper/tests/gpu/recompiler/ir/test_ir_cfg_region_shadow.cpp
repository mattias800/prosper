// test_ir_cfg_region_shadow -- the shadow check (region_shadow.hpp): the region tree's ifs and
// loops equal detect_forward_ifs', detect_divergent_loops' and detect_counted_loop's at the pc
// level, on every committed guest-program fixture in tests/data under each stage's admission
// rules, and on synthetic shapes. Hand-built programs the structurers are known to disagree on
// are the positive controls that the comparison can see a difference at all.
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

RegionShadow shadow_of(const std::vector<uint32_t>& words, ShadowStage stage) {
    return shadow_regions(walk(words), stage, words.data(), words.size());
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
            const RegionShadow shadow = shadow_of(words, stage);
            EXPECT_NE(shadow.verdict, RegionShadow::Verdict::Disagree)
                << path.filename() << " stage " << static_cast<int>(stage) << ": "
                << shadow.difference;
            if (shadow.verdict == RegionShadow::Verdict::Agree) ++compared;
        }
    }
    EXPECT_EQ(compared, 3 * fixtures.size());
}

TEST(IrRegionShadow, CategoriesOnSyntheticShapes) {
    const std::vector<uint32_t> linear = {kSMovS0Zero, kEndpgm};
    EXPECT_STREQ(shadow_category(shadow_of(linear, ShadowStage::Fragment)), "agree-linear");

    const std::vector<uint32_t> if_else = {sopp(kVccnz, 2), kVNop, sopp(kBranch, 1), kVNop,
                                           kEndpgm};
    const RegionShadow structured = shadow_of(if_else, ShadowStage::Vertex);
    EXPECT_STREQ(shadow_category(structured), "agree-structured");
    ASSERT_EQ(structured.tree.size(), 1u);
    EXPECT_TRUE(structured.tree[0].has_else);
    EXPECT_EQ(structured.tree[0].merge_pc, 4u);

    const std::vector<uint32_t> refused = {sopp(kExecnz, 1), kVNop, kEndpgm};
    EXPECT_STREQ(shadow_category(shadow_of(refused, ShadowStage::Vertex)), "agree-refused");

    // A counted loop: s_branch back-edge with a forward scc exit.
    const std::vector<uint32_t> loop = {kSMovS0Zero, sopp(kScc0, 1), sopp(kBranch, -3), kEndpgm};
    const RegionShadow counted = shadow_of(loop, ShadowStage::Fragment);
    EXPECT_STREQ(shadow_category(counted), "agree-counted");
    ASSERT_EQ(counted.tree_loops.size(), 1u);
    EXPECT_TRUE(counted.tree_loops[0].counted);
    EXPECT_EQ(counted.tree_loops[0].exit_branch_pc, 1u);
}

// Loop shapes the divergent detector admits, each with ifs or loops inside. Every one must agree
// under every stage (a stage that refuses a shape must refuse it on both sides).
TEST(IrRegionShadow, SyntheticLoopShapesAgreeUnderEveryStage) {
    const std::vector<std::vector<uint32_t>> shapes = {
        // execz-exit loop with an s_branch back edge
        {kSMovS0Zero, sopp(kExecz, 2), kVNop, sopp(kBranch, -3), kEndpgm},
        // ... with a scalar if inside
        {kSMovS0Zero, sopp(kExecz, 4), sopp(kScc0, 1), kVNop, kVNop, sopp(kBranch, -5), kEndpgm},
        // nested execz loops
        {kSMovS0Zero, sopp(kExecz, 6), kVNop, sopp(kExecz, 2), kVNop, sopp(kBranch, -3), kVNop,
         sopp(kBranch, -7), kEndpgm},
        // execnz back edge, header exit, an execz break
        {kSMovS0Zero, sopp(kExecz, 4), kVNop, sopp(kExecz, 2), kVNop, sopp(kExecnz, -5), kEndpgm},
        // bottom-tested execnz loop
        {kVNop, kSMovExecZero, sopp(kExecnz, -3), kEndpgm},
        // two sequential loops, then an if after them
        {kSMovS0Zero, sopp(kExecz, 2), kVNop, sopp(kBranch, -3), sopp(kExecz, 2), kVNop,
         sopp(kBranch, -3), sopp(kScc0, 1), kVNop, kEndpgm},
        // a vccz exit: fragment admits it, the other stages refuse it
        {kSMovS0Zero, sopp(kVccz, 2), kVNop, sopp(kBranch, -3), kEndpgm},
        // a break inside an inner if: both refuse
        {kSMovS0Zero, sopp(kExecz, 5), sopp(kScc0, 2), sopp(kExecz, 3), kVNop, kVNop,
         sopp(kExecnz, -6), kEndpgm},
    };
    size_t loops = 0;
    for (size_t i = 0; i < shapes.size(); ++i)
        for (ShadowStage stage :
             {ShadowStage::Vertex, ShadowStage::Fragment, ShadowStage::Compute}) {
            const RegionShadow shadow = shadow_of(shapes[i], stage);
            EXPECT_EQ(shadow.verdict, RegionShadow::Verdict::Agree)
                << "shape " << i << " stage " << static_cast<int>(stage) << ": "
                << shadow.difference;
            if (std::string(shadow_category(shadow)) == "agree-loops") ++loops;
        }
    EXPECT_GE(loops, 15u);   // most shapes are structured loops under most stages
}

// Positive control for the loop comparison: a do-while loop nothing can reach. The counted
// detector scans pcs and claims it; the tree finds loops from the CFG's back edges, sees none, and
// then refuses the forward s_branch that jumps over the dead loop.
TEST(IrRegionShadow, AnUnreachableLoopIsADisagreement) {
    // 0: s_branch -> 3; 1: v_nop; 2: scc1 -> 1; 3: s_endpgm
    const std::vector<uint32_t> words = {sopp(kBranch, 2), kVNop, sopp(kScc1, -2), kEndpgm};
    const RegionShadow shadow = shadow_of(words, ShadowStage::Fragment);
    EXPECT_TRUE(shadow.counted);
    EXPECT_EQ(shadow.verdict, RegionShadow::Verdict::Disagree);
    EXPECT_TRUE(shadow.tree_unstructured);
    EXPECT_NE(shadow.difference.find("unclaimed s_branch"), std::string::npos) << shadow.difference;
}

// Positive control, built by hand outside any generator: an inner if in the then-arm of an
// if/else targets the ELSE arm. detect_forward_ifs' nesting check only bounds it by the if/else's
// merge and accepts; the region tree bounds it by the then-arm and refuses.
TEST(IrRegionShadow, AnIfIntoTheElseArmIsADisagreement) {
    const std::vector<uint32_t> words = {sopp(kScc0, 3), sopp(kScc1, 3), kVNop,  sopp(kBranch, 2),
                                         kVNop,          kVNop,          kEndpgm};
    const RegionShadow shadow = shadow_of(words, ShadowStage::Fragment);
    EXPECT_EQ(shadow.verdict, RegionShadow::Verdict::Disagree);
    EXPECT_FALSE(shadow.detector_refused);
    EXPECT_TRUE(shadow.tree_unstructured);
    EXPECT_STREQ(shadow_category(shadow), "disagree");
}

}   // namespace
