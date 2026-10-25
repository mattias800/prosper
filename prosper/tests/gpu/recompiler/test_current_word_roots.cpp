// #5497: a current rewrite can owe fewer roots than its preserved inactive lanes.
// The independent view must retain broader-cohort checks and missing atomic inputs.
#include "gpu/recompiler/rdna2_current_word_roots.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_predicate_operand_roots.hpp"
#include "gpu/recompiler/rdna2_reg_state.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"

#include <gtest/gtest.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>

namespace {
using namespace prosper::gpu;
constexpr uint64_t old_root = 0x100;
constexpr uint64_t data_root = 0x101;
constexpr uint64_t address_root = 0x102;
const std::set<uint64_t>& read_roots(const RegState& rs, const ScalarSourceMarks& marks,
                                     uint32_t exec) {
    if (const auto* current = current_word_roots(rs, marks, exec)) return *current;
    if (marks.scoped_complete_word && marks.outside_exec &&
        exec_within(rs, exec, marks.outside_exec))
        return marks.scoped_blanket_roots;
    return marks.blanket_roots;
}
ScalarSourceMarks bounded() {
    ScalarSourceMarks marks;
    marks.placeholder = marks.hard = marks.scoped_complete_word = true;
    marks.outside_exec = 10;
    marks.blanket_roots = marks.scoped_blanket_roots = {old_root};
    return marks;
}
struct Frame {
    RegState rs;
    Frame() {
        rs.exec = 20;
        rs.exec_narrowed = true;
        rs.vector_wave_size = 64;
        rs.exec_subset_of[20] = 10;
        rs.exec_subset_of[30] = 20;
        rs.vreg[0] = 100;
        rs.vreg[1] = 101;
        rs.vreg[7] = 107;
        for (int reg = 8; reg < 12; ++reg) rs.sreg[reg] = 200 + reg;
    }
};
Rdna2Inst literal() {
    const uint32_t word = 0x7e000281u;
    return rdna2_decode_one(&word, 1);
}
Rdna2Inst atomic(bool exchange = false) {
    const uint32_t words[] = {exchange ? 0xe0c06000u : 0xe0c86000u, 0x80020007u};
    return rdna2_decode_one(words, 2);
}
void legacy_equal(const ScalarSourceMarks& a, const ScalarSourceMarks& b) {
    EXPECT_EQ(a.placeholder, b.placeholder);
    EXPECT_EQ(a.memory, b.memory);
    EXPECT_EQ(a.hard, b.hard);
    EXPECT_EQ(a.arbitrary, b.arbitrary);
    EXPECT_EQ(a.blanket_roots, b.blanket_roots);
    EXPECT_EQ(a.outside_exec, b.outside_exec);
    EXPECT_EQ(a.scoped_complete_word, b.scoped_complete_word);
    EXPECT_EQ(a.scoped_blanket_roots, b.scoped_blanket_roots);
}
}  // namespace

TEST(CurrentWordRoots, ACurrentDefinitionKeepsTheBroaderLegacyCoverage) {
    Frame f;
    auto result = bounded();
    const auto before = result;
    const auto publication = snapshot_current_word_roots(f.rs, literal(), {}, read_roots);
    ASSERT_EQ(publication.view.exec, 20u);
    ASSERT_TRUE(publication.view.roots.empty());
    EXPECT_FALSE(publication.atomic_result);
    retain_current_word_roots(result, f.rs, publication, read_roots);
    ASSERT_TRUE(result.current_roots);
    EXPECT_TRUE(read_roots(f.rs, result, 20).empty());
    EXPECT_TRUE(read_roots(f.rs, result, 30).empty());
    EXPECT_EQ(read_roots(f.rs, result, 10), std::set<uint64_t>{old_root});
    legacy_equal(result, before);
}

TEST(CurrentWordRoots, AnActualInputRetainsItsOwnCurrentDependencies) {
    Frame f;
    f.rs.vreg_source_marks[1] = bounded();
    auto& input = f.rs.vreg_source_marks[1];
    input.blanket_roots.insert(data_root);
    input.scoped_blanket_roots.insert(data_root);
    input.current_roots =
        std::make_shared<const CurrentWordRoots>(CurrentWordRoots{20, {data_root}});
    const uint32_t word = 0x7e000301u;
    const auto in = rdna2_decode_one(&word, 1);
    const auto publication = snapshot_current_word_roots(f.rs, in, input, read_roots);
    ASSERT_EQ(publication.view.exec, 20u);
    EXPECT_EQ(publication.view.roots, std::set<uint64_t>{data_root});
}

TEST(CurrentWordRoots, AMissingInputCannotBorrowTheDestinationProof) {
    Frame f;
    auto& input = f.rs.vreg_source_marks[1];
    input.placeholder = input.hard = true;
    const uint32_t word = 0x7e000301u;
    const auto in = rdna2_decode_one(&word, 1);
    EXPECT_EQ(snapshot_current_word_roots(f.rs, in, {}, read_roots).view.exec, 0u);
}

TEST(CurrentWordRoots, AViewCannotInventANewRootObligation) {
    Frame f;
    auto result = bounded();
    const auto before = result;
    retain_current_word_roots(result, f.rs, {{20, {data_root}}, false}, read_roots);
    EXPECT_FALSE(result.current_roots);
    legacy_equal(result, before);
}

TEST(CurrentWordRoots, TheCanonicalCompleteWordLifetimeStillGatesTheView) {
    Frame f;
    auto result = bounded();
    result.current_roots = std::make_shared<const CurrentWordRoots>(CurrentWordRoots{20, {}});
    ASSERT_NE(current_word_roots(f.rs, result, 20), nullptr);
    result.scoped_complete_word = false;
    EXPECT_EQ(current_word_roots(f.rs, result, 20), nullptr);
    result.scoped_complete_word = true;
    result.outside_exec = 0;
    EXPECT_EQ(current_word_roots(f.rs, result, 20), nullptr);
}

TEST(CurrentWordRoots, UnsupportedContextAndRawGenerationsKeepTheLegacyView) {
    for (unsigned mutation = 0; mutation < 11; ++mutation) {
        Frame f;
        auto in = literal();
        ScalarSourceMarks source;
        switch (mutation) {
            case 0: f.rs.vector_wave_size = 32; break;
            case 1: f.rs.exec = 0; break;
            case 2: f.rs.exec_narrowed = false; break;
            case 3: f.rs.vector_full_exec = true; break;
            case 4: f.rs.vreg.erase(0); break;
            case 5: in.synthetic_terminator = true; break;
            case 6: in.has_sdwa = true; break;
            case 7: in.has_dpp = true; break;
            case 8: in.has_modifier = true; break;
            case 9: in.words[0] ^= 1u; break;
            case 10: source.placeholder = source.hard = true; break;
        }
        EXPECT_EQ(snapshot_current_word_roots(f.rs, in, source, read_roots).view.exec, 0u)
            << mutation;
    }
}

TEST(CurrentWordRoots, AnImplicitLaneMaskDoesNotBecomeAnExplicitNumericInput) {
    Frame f;
    const uint32_t word = 0x02000101u;
    const auto in = rdna2_decode_one(&word, 1);
    EXPECT_EQ(snapshot_current_word_roots(f.rs, in, {}, read_roots).view.exec, 0u);
}

TEST(CurrentWordRoots, AtomicReturnsRetainAddressAndOldDataObligations) {
    for (bool exchange : {false, true}) {
        Frame f;
        auto& old = f.rs.vreg_source_marks[0];
        old = bounded();
        old.blanket_roots.insert(data_root);
        old.scoped_blanket_roots.insert(data_root);
        old.current_roots =
            std::make_shared<const CurrentWordRoots>(CurrentWordRoots{20, {data_root}});
        auto& address = f.rs.vreg_source_marks[7];
        address.placeholder = true;
        address.blanket_roots = {address_root};
        const auto publication =
            snapshot_current_word_roots(f.rs, atomic(exchange), address, read_roots);
        EXPECT_TRUE(publication.atomic_result);
        EXPECT_EQ(publication.view.exec, 20u);
        EXPECT_EQ(publication.view.roots, (std::set<uint64_t>{data_root, address_root}));
    }
}

TEST(CurrentWordRoots, MissingAtomicDataResourcesAndAddressValuesDoNotPublish) {
    for (unsigned mutation = 0; mutation < 4; ++mutation) {
        Frame f;
        if (mutation == 0) f.rs.sreg_merge_placeholder.insert(9);
        if (mutation == 1) {
            auto& old = f.rs.vreg_source_marks[0];
            old.placeholder = old.hard = true;
        }
        if (mutation == 2) f.rs.vreg.erase(7);
        if (mutation == 3) f.rs.vreg.erase(0);
        const auto publication = snapshot_current_word_roots(f.rs, atomic(), {}, read_roots);
        EXPECT_FALSE(publication.atomic_result) << mutation;
        EXPECT_EQ(publication.view.exec, 0u) << mutation;
    }
}

TEST(CurrentWordRoots, AGenuineScalarSiblingDoesNotResurrectInactiveVectorRoots) {
    Frame f;
    f.rs.sreg[36] = 236;
    auto& vector = f.rs.vreg_source_marks[0];
    vector = bounded();
    vector.current_roots = std::make_shared<const CurrentWordRoots>(CurrentWordRoots{20, {}});
    const uint32_t word = 0x7da80024u;
    const auto in = rdna2_decode_one(&word, 1);
    const auto roots = predicate_operand_roots(f.rs, in, read_roots);
    EXPECT_EQ(roots, (std::optional<std::set<uint64_t>>{std::set<uint64_t>{}}));
    f.rs.unavailable_saved_mask_words.insert(36);
    EXPECT_FALSE(predicate_operand_roots(f.rs, in, read_roots).has_value());
    f.rs.unavailable_saved_mask_words.clear();
    f.rs.sreg_loop_blanket[36] = {data_root};
    EXPECT_FALSE(predicate_operand_roots(f.rs, in, read_roots).has_value());
}

TEST(CurrentWordRoots, TrailingLiteralPacketsRetainTheirActualConstant) {
    Frame f;
    const uint32_t words[] = {0x7e0002ffu, 0x12345678u};
    const auto in = rdna2_decode_one(words, 2);
    ASSERT_TRUE(in.has_literal);
    ASSERT_EQ(in.literal, 0x12345678u);
    EXPECT_EQ(snapshot_current_word_roots(f.rs, in, {}, read_roots).view.exec, 20u);
    auto changed = in;
    changed.src[0].kind = OperandKind::SGPR;
    changed.src[0].value = 124;
    EXPECT_EQ(snapshot_current_word_roots(f.rs, changed, {}, read_roots).view.exec, 0u);
}
TEST(CurrentWordRoots, MeetsAndLaneSelectionsDropTheProducingGenerationView) {
    Frame f;
    auto mark = bounded();
    mark.current_roots = std::make_shared<const CurrentWordRoots>(CurrentWordRoots{20, {}});
    const auto before = mark;
    union_source_marks(mark, {});
    EXPECT_FALSE(mark.current_roots);
    legacy_equal(mark, before);
    f.rs.vreg_source_marks[1] = before;
    const uint32_t word = 0x7e060501u; // s3 = readfirstlane(v1)
    const auto read = rdna2_decode_one(&word, 1);
    const auto selected = vector_lane_read_marks(f.rs, read);
    EXPECT_FALSE(selected.current_roots);
    legacy_equal(selected, before);
}
