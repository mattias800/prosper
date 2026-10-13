// #4623: a verified early-out reads past the first END. Those bytes must survive the immutable
// code key and raw capture, while failed probes cannot lend terminal/executable authority. These
// CPU-only contracts also mutate the tail at one address without clearing either shader cache.
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/execute/shader_source_window.hpp"
#include "gpu/recompiler/rdna2_terminal_tail_span.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

using namespace prosper::gpu;

namespace {
constexpr uint32_t kEnd = 0xbf810000u;
constexpr uint32_t kNop = 0xbf800000u;
constexpr uint32_t kUnknown = 0xffffffffu;
constexpr uint32_t kMove = 0xbe800381u; // s_mov_b32 s0, 1: real scalar effect, not padding

uint32_t conditional(uint32_t opcode, int16_t displacement) {
    return 0xbf800000u | (opcode << 16u) | uint16_t(displacement);
}

Rdna2TerminalTailSpan terminal_span(std::span<const uint32_t> code) {
    std::vector<Rdna2Inst> body;
    rdna2_walk(code.data(), code.size(), body);
    return rdna2_terminal_tail_span(code.data(), code.size(), body);
}

std::vector<uint32_t> fragment_program() {
    return {
        0xbe840380u,              // pc0: genuine s4 = 0
        0xbf068004u,              // pc1: s_cmp_eq_u32 s4, 0 supplies SCC
        conditional(0x04u, 6),    // pc2: s_cbranch_scc0 -> pc9
        0x7e000280u,              // pc3: v_mov_b32 v0, 0
        0xf800180fu,
        0u,          // pc4: exp mrt0 v0,v0,v0,v0
        kEnd,                    // pc6: first END
        kNop,
        kNop,              // pc7/8: unrelated gap, not a target
        kNop,
        kEnd,              // pc9/10: verified terminal target
        kUnknown,                // pc11: must not be retained or read as executable
    };
}

std::vector<uint32_t> fragment(std::span<const uint32_t> code) {
    return recompile_fragment(code.data(), code.size());
}

class TerminalTailRetention : public testing::Test {
protected:
    void SetUp() override {
        clear_shader_recompile_cache();
        clear_shader_analysis_cache();
    }
};
} // namespace

TEST_F(TerminalTailRetention, EveryArchitecturalConditionalRetainsItsImmediateEndTarget) {
    for (uint32_t opcode = 0x04; opcode <= 0x09; ++opcode) {
        SCOPED_TRACE(opcode);
        const std::array code{conditional(opcode, 2), kEnd, kUnknown, kEnd, kUnknown};
        const auto proof = terminal_span(code);
        EXPECT_EQ(proof.retained_dwords, 4u);
        EXPECT_EQ(proof.inspected_dwords, 4u);
        EXPECT_FALSE(proof.source_limit_required);
        EXPECT_EQ(proof.terminating_targets, std::vector<uint32_t>{3});
        EXPECT_EQ(rdna2_recompile_code_span(code.data(), code.size()), 4u);
    }
}

TEST_F(TerminalTailRetention, MultipleTargetsRetainOnlyTheirProvenNopEndSpan) {
    const std::array code{
        conditional(0x04, 3), conditional(0x06, 3), kEnd, kUnknown, kNop, kNop, kEnd, kUnknown};
    const auto proof = terminal_span(code);
    EXPECT_EQ(proof.retained_dwords, 7u);
    EXPECT_EQ(proof.inspected_dwords, 7u);
    EXPECT_EQ(proof.terminating_targets, (std::vector<uint32_t>{4, 5}));
    EXPECT_EQ(shader_source_code_span(code.data(), code.size()), 7u);
    std::vector<Rdna2Inst> inventory;
    uint32_t unavailable = 0;
    EXPECT_TRUE(rdna2_recompile_executable_instructions(code.data(), 7, inventory, unavailable));
    EXPECT_EQ(unavailable, UINT32_MAX);
}

TEST_F(TerminalTailRetention, UnknownAndNonterminalTargetsCannotBorrowALaterEnd) {
    for (uint32_t bad : {kUnknown, kMove, 0xf800180fu}) {
        SCOPED_TRACE(bad);
        std::array code{conditional(0x04, 3), kEnd, kNop, kNop, kNop, kEnd};
        EXPECT_EQ(terminal_span(code).retained_dwords, code.size());
        code[4] = bad;
        const auto proof = terminal_span(code);
        EXPECT_EQ(proof.retained_dwords, 2u);
        EXPECT_EQ(proof.inspected_dwords, 5u);
        EXPECT_FALSE(proof.source_limit_required);
        EXPECT_TRUE(proof.terminating_targets.empty());
        EXPECT_EQ(shader_source_code_span(code.data(), code.size()), 2u);
        std::vector<Rdna2Inst> inventory;
        uint32_t unavailable = 0;
        EXPECT_FALSE(rdna2_recompile_executable_instructions(code.data(), code.size(), inventory,
                                                             unavailable));
    }
}

TEST_F(TerminalTailRetention, MissingAndOpenTargetsDependOnTheActualSourceLimit) {
    const std::array missing{conditional(0x04, 3), kEnd, kNop, kNop};
    const auto absent = terminal_span(missing);
    EXPECT_EQ(absent.retained_dwords, 2u);
    EXPECT_EQ(absent.inspected_dwords, 2u);
    EXPECT_TRUE(absent.source_limit_required);
    EXPECT_TRUE(absent.terminating_targets.empty());
    const std::array open{conditional(0x04, 3), kEnd, kNop, kNop, kNop};
    const auto unterminated = terminal_span(open);
    EXPECT_EQ(unterminated.retained_dwords, 2u);
    EXPECT_EQ(unterminated.inspected_dwords, open.size());
    EXPECT_TRUE(unterminated.source_limit_required);
    EXPECT_TRUE(unterminated.terminating_targets.empty());
}

TEST_F(TerminalTailRetention, UnconditionalAndUnreferencedTrailersDoNotExpandTheContract) {
    const std::array branch{0xbf820002u, kEnd, kUnknown, kEnd};
    EXPECT_EQ(terminal_span(branch).retained_dwords, 2u);
    EXPECT_TRUE(terminal_span(branch).terminating_targets.empty());
    const std::array plain{kNop, kEnd, kNop, kEnd};
    EXPECT_EQ(shader_source_code_span(plain.data(), plain.size()), 2u);
    const std::array no_end{kNop, kUnknown, kEnd};
    EXPECT_EQ(terminal_span(no_end).retained_dwords, 0u);
    std::vector<Rdna2Inst> empty;
    EXPECT_EQ(rdna2_terminal_tail_span(nullptr, 0, empty).retained_dwords, 0u);
}

TEST_F(TerminalTailRetention, ASyntheticOrMismatchedBodyEndCannotProveATrailer) {
    std::array code{conditional(0x04, 2), kEnd, kUnknown, kEnd};
    std::vector<Rdna2Inst> body;
    rdna2_walk(code.data(), code.size(), body);
    ASSERT_EQ(body.size(), 2u);
    EXPECT_EQ(rdna2_terminal_tail_span(code.data(), code.size(), body).retained_dwords, 4u);
    body.back().synthetic_terminator = true;
    EXPECT_EQ(rdna2_terminal_tail_span(code.data(), code.size(), body).retained_dwords, 0u);
    body.back().synthetic_terminator = false;
    code[1] = kUnknown;
    EXPECT_EQ(rdna2_terminal_tail_span(code.data(), code.size(), body).retained_dwords, 0u);
}

TEST_F(TerminalTailRetention, ABodyLiteralWithEndBitsIsNotAnInstructionTarget) {
    std::array code{conditional(0x04, 1), 0xbe8003ffu, kEnd, kEnd};
    // pc1/2 is s_mov_b32 s0,literal. pc2's END-looking operand is not an instruction boundary.
    const auto proof = terminal_span(code);
    EXPECT_TRUE(proof.terminating_targets.empty());
    std::vector<Rdna2Inst> inventory;
    uint32_t unavailable = UINT32_MAX;
    EXPECT_FALSE(
        rdna2_recompile_executable_instructions(code.data(), code.size(), inventory, unavailable));
    EXPECT_EQ(unavailable, 0u);
    code[0] = conditional(0x04, 2);   // paired control targets the genuine instruction at pc3
    EXPECT_TRUE(
        rdna2_recompile_executable_instructions(code.data(), code.size(), inventory, unavailable));
}

TEST_F(TerminalTailRetention, AnEndLookingTailWordDoesNotLicenseAnotherNonterminalTarget) {
    std::array code{
        conditional(0x04, 3), conditional(0x05, 3), kEnd, kNop, 0x7e0002ffu, kEnd, kEnd};
    // pc4/5 is one literal VMOV. The pc5 target's raw END cannot authenticate the pc4 target.
    const auto proof = terminal_span(code);
    EXPECT_EQ(proof.terminating_targets, std::vector<uint32_t>{5});
    ASSERT_EQ(proof.retained_dwords, 6u);
    std::vector<Rdna2Inst> inventory;
    uint32_t unavailable = UINT32_MAX;
    EXPECT_FALSE(rdna2_recompile_executable_instructions(code.data(), proof.retained_dwords,
                                                         inventory, unavailable));
    EXPECT_EQ(unavailable, 0u);
    code[4] = kNop;   // both actual targets now prove NOP/END, on the same retained byte span
    EXPECT_TRUE(rdna2_recompile_executable_instructions(code.data(), proof.retained_dwords,
                                                        inventory, unavailable));
}

TEST_F(TerminalTailRetention, LargeOrNegativeTargetsCannotWrapIntoATerminalAlias) {
    const std::array far{conditional(0x04, INT16_MAX), kEnd};
    const auto unavailable = terminal_span(far);
    EXPECT_EQ(unavailable.retained_dwords, 2u);
    EXPECT_TRUE(unavailable.terminating_targets.empty());
    EXPECT_TRUE(unavailable.source_limit_required);
    const std::array negative{conditional(0x04, INT16_MIN), kEnd};
    EXPECT_TRUE(terminal_span(negative).terminating_targets.empty());
    std::vector<Rdna2Inst> body;
    rdna2_walk(far.data(), far.size(), body);
    ASSERT_EQ(body.size(), 2u);
    body.back().pc = UINT32_MAX;
    EXPECT_EQ(rdna2_terminal_tail_span(far.data(), far.size(), body).retained_dwords, 0u);
}

TEST_F(TerminalTailRetention, WarmTailMutationInvalidatesBothPositiveAndNegativeAnalyses) {
    std::array code{conditional(0x04, 3), kEnd, kNop, kNop, kNop, kEnd, kUnknown};
    const auto positive = acquire_shader_analysis(code.data(), code.size());
    ASSERT_TRUE(positive);
    ASSERT_EQ(shader_analysis_owned_words(positive)->size(), 6u);
    EXPECT_EQ(acquire_shader_analysis(code.data(), code.size()), positive);
    code[5] = kUnknown;
    const auto negative = acquire_shader_analysis(code.data(), code.size());
    ASSERT_TRUE(negative);
    EXPECT_NE(negative, positive);
    EXPECT_EQ(shader_analysis_owned_words(negative)->size(), 2u);
    EXPECT_EQ(acquire_shader_analysis(code.data(), code.size()), negative);
    code[5] = kEnd;
    const auto restored = acquire_shader_analysis(code.data(), code.size());
    ASSERT_TRUE(restored);
    EXPECT_NE(restored, negative);
    EXPECT_EQ(shader_analysis_owned_words(restored)->size(), 6u);
    EXPECT_EQ(shader_analysis_cache_stats().invalidations, 2u);
}

TEST_F(TerminalTailRetention, APreviouslyMissingTargetIsRecheckedWhenTheWindowGrows) {
    const std::array code{conditional(0x04, 3), kEnd, kNop, kNop, kNop, kEnd};
    const auto missing = acquire_shader_analysis(code.data(), 4);
    ASSERT_TRUE(missing);
    EXPECT_EQ(shader_analysis_owned_words(missing)->size(), 2u);
    const auto supplied = acquire_shader_analysis(code.data(), code.size());
    ASSERT_TRUE(supplied);
    EXPECT_NE(supplied, missing);
    EXPECT_EQ(shader_analysis_owned_words(supplied)->size(), code.size());
    EXPECT_EQ(shader_analysis_cache_stats().invalidations, 1u);
}

TEST_F(TerminalTailRetention, TheLiveKeyKeepsTheActualTerminalProof) {
    const auto code = fragment_program();
    const auto direct = fragment(code);
    ASSERT_FALSE(direct.empty());
    const auto cached =
        recompile_graphics_shader_cached(ShaderProgramStage::Fragment, code.data(), code.size());
    EXPECT_EQ(cached, direct);
    const auto analysis = acquire_shader_analysis(code.data(), code.size());
    ASSERT_TRUE(analysis);
    const auto retained = shader_analysis_owned_words(analysis);
    ASSERT_TRUE(retained);
    ASSERT_EQ(retained->size(), 11u);
    EXPECT_EQ(*retained, (std::vector<uint32_t>(code.begin(), code.begin() + 11)));
    EXPECT_EQ(fragment(*retained), direct);

    auto unknown = code;
    unknown[9] = kUnknown;
    EXPECT_TRUE(fragment(unknown).empty());
    auto nonterminal = code;
    nonterminal[9] = kMove;
    EXPECT_TRUE(fragment(nonterminal).empty());
    EXPECT_TRUE(fragment(std::span(code).first(7)).empty());
}

TEST_F(TerminalTailRetention, RawCaptureKeepsTheActualTerminalProofIndependentlyOfTheCache) {
    const auto code = fragment_program();
    const auto direct = fragment(code);
    ASSERT_FALSE(direct.empty());
    const std::vector<uint32_t> expected(code.begin(), code.begin() + 11);
    constexpr uint64_t address = 0x40000;
    DrawItem draw;
    draw.vs = {0x07230203, 1, 2}; // metadata fixture, not an executed vertex module
    draw.fs = direct;
    draw.fs_guest_addr = address;
    draw.vertex_count = draw.raw_draw_count = 3;
    draw.ps.topology = 3;
    draw.ps.color_write_mask = 0xf;
    draw.color0_base = 0x2000;
    draw.color0_width = draw.color0_height = 16;
    draw.command_order = 1;
    GpuCaptureMetadata metadata;
    metadata.width = metadata.height = 16;
    const auto reader = [&](uint64_t at, uint8_t* destination, size_t bytes) -> size_t {
        if (at < address || at - address >= code.size() * sizeof(uint32_t)) return 0;
        const size_t offset = static_cast<size_t>(at - address);
        const size_t count = std::min(bytes, code.size() * sizeof(uint32_t) - offset);
        std::memcpy(destination, reinterpret_cast<const uint8_t*>(code.data()) + offset, count);
        return count;
    };
    GpuCaptureFile capture;
    std::string error;
    ASSERT_TRUE(capture_draw_items({draw}, metadata, reader, capture, error)) << error;
    ASSERT_EQ(capture.draws.size(), 1u);
    const auto index = capture.draws[0].fs_raw_shader_index;
    ASSERT_LT(index, capture.raw_shader_versions.size());
    EXPECT_EQ(capture.raw_shader_versions[index].words, expected);
    EXPECT_EQ(fragment(capture.raw_shader_versions[index].words), direct);
}
