// Cold original-stage facts must never hide a writer behind folded/native reflection, a short
// END/Unknown prefix, synthetic terminator, or forged decoded metadata. These CPU tests grant
// no resource/entry/currentness authority and do not claim GPU or title rendering.
#include "gpu/execute/original_graphics_stage_effects.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "fixtures/fragment_scalar_bank_fixture.hpp"
#include <gtest/gtest.h>

namespace prosper::gpu {
namespace {
constexpr uint32_t end = 0xbf810000;
std::vector<Rdna2Inst> inventory(const std::vector<uint32_t>& words) {
    std::vector<Rdna2Inst> result;
    rdna2_walk(words.data(), words.size(), result);
    return result;
}
OriginalGraphicsStageEffects effects(const std::vector<uint32_t>& words,
                                     ShaderProgramStage stage = ShaderProgramStage::Fragment) {
    return original_graphics_stage_effects(words, inventory(words), stage);
}
uint32_t exp(uint32_t target) {
    return 0xf800180f | (target << 4);
}
uint32_t smem(uint32_t opcode) {
    return 0xf4000000 | (opcode << 18);
}
uint32_t mubuf(uint32_t opcode) {
    return 0xe0000000 | (opcode << 18);
}

TEST(OriginalGraphicsStageEffects, CompleteScalarLoadsRetainExactOriginalOwnerAndWidths) {
    for (const uint32_t opcode : {0u, 1u, 2u, 3u, 4u, 8u, 9u, 10u, 11u, 12u}) {
        const std::vector<uint32_t> words{smem(opcode), 0xfa000000, end};
        const auto result = effects(words);
        EXPECT_TRUE(result.known_read_only()) << "original scalar load opcode=" << opcode;
        EXPECT_EQ(result.source_words, &words);
        EXPECT_EQ(result.consumed_dwords, 3u);
        EXPECT_EQ(result.instruction_count, 2u);
        EXPECT_TRUE(result.guest_memory_reads);
        EXPECT_FALSE(result.attachment_exports);
    }
}

TEST(OriginalGraphicsStageEffects, VertexBufferAndCorrectStageExportsRemainReadOnly) {
    const std::vector<uint32_t> vertex{mubuf(12), 0, exp(12), 0x03020100, exp(32), 0x03020100, end};
    const auto result = effects(vertex, ShaderProgramStage::Vertex);
    EXPECT_TRUE(result.known_read_only()) << result.rejection;
    EXPECT_TRUE(result.guest_memory_reads);
    EXPECT_FALSE(result.attachment_exports);
    const std::vector<uint32_t> fragment{exp(0), 0x03020100, end};
    EXPECT_TRUE(effects(fragment).known_read_only());
    EXPECT_TRUE(effects(fragment).attachment_exports);
    EXPECT_FALSE(effects(fragment, ShaderProgramStage::Vertex).known_read_only());
    const std::vector<uint32_t> position{exp(12), 0x03020100, end};
    EXPECT_FALSE(effects(position).known_read_only());
}

TEST(OriginalGraphicsStageEffects, RegisteredProceduralVertexArithmeticKeepsCompleteReadOnlyFacts) {
    prosper::register_builtin_hle();
    const auto words = prosper::test::fragment_draw::vertex_words();
    const auto* program = prosper::test::scalar_bank::register_original(true, words);
    ASSERT_TRUE(program);
    const auto address = reinterpret_cast<uint64_t>(program->code.data());
    const auto source = registered_graphics_read_source(address);
    ASSERT_TRUE(source.words);
    ASSERT_TRUE(source.decoded);
    ASSERT_TRUE(source.vertex_effects);
    ASSERT_TRUE(source.header_snapshot);
    EXPECT_EQ(*source.words, words) << "the complete unchanged registered original, not a utility";
    EXPECT_EQ(source.header_snapshot->shader_size, words.size() * sizeof(uint32_t));
    EXPECT_EQ(source.vertex_effects->source_words, source.words.get());
    EXPECT_FALSE(source.words.owner_before(source.vertex_effects));
    EXPECT_FALSE(source.vertex_effects.owner_before(source.words));
    EXPECT_TRUE(source.vertex_effects->known_read_only()) << source.vertex_effects->rejection;
    EXPECT_TRUE(source.vertex_effects->architectural_end);
    EXPECT_EQ(source.vertex_effects->consumed_dwords, words.size());
    EXPECT_FALSE(source.vertex_effects->guest_memory_reads);
    EXPECT_FALSE(source.vertex_effects->attachment_exports);
}

TEST(OriginalGraphicsStageEffects, OriginalWriterCannotDisappearBehindReadOnlyMetadata) {
    const std::vector<uint32_t> original{mubuf(12), 0, mubuf(24), 0, end};
    for (const auto stage : {ShaderProgramStage::Vertex, ShaderProgramStage::Fragment}) {
        const auto result = effects(original, stage);
        EXPECT_FALSE(result.known_read_only());
        EXPECT_EQ(result.rejection, "original-stage-memory-effect-not-read-only:pc=2");
    }
    auto forged = inventory(original);
    forged[1].opcode = 12;
    const auto result =
        original_graphics_stage_effects(original, forged, ShaderProgramStage::Fragment);
    EXPECT_FALSE(result.known_read_only());
    EXPECT_EQ(result.rejection, "original-stage-source-inventory-mismatch:pc=2");
}

TEST(OriginalGraphicsStageEffects, LocalAndGlobalDsDoNotBorrowTheOldHelpersDefault) {
    for (const uint32_t header : {0xd8000000u, 0xd8020000u}) {
        const std::vector<uint32_t> words{header, 0, end};
        const auto decoded = inventory(words);
        ASSERT_EQ(decoded.front().fmt, Rdna2Format::DS);
        EXPECT_FALSE(rdna2_instruction_may_write_memory(decoded.front()));
        const auto result = effects(words);
        EXPECT_FALSE(result.known_read_only());
        EXPECT_EQ(result.rejection, "original-stage-ds-gds-effect-unimplemented:pc=0");
    }
}

TEST(OriginalGraphicsStageEffects, PrefetchAndNonzeroWaitDoNotInventGuestDataWrites) {
    // gfx1030/ISA Table72: opcode32 is S_INST_PREFETCH3, not S_CLAUSE (opcode33).
    const std::vector<uint32_t> words{0xbfa00003, smem(8), 0xfa000000, 0xbf8cc07f, end};
    const auto result = effects(words);
    EXPECT_TRUE(result.known_read_only()) << result.rejection;
    EXPECT_EQ(result.instruction_count, 4u);
    EXPECT_TRUE(result.guest_memory_reads);
    const std::vector<uint32_t> reserved_prefetch{0xbfa00000, end};
    EXPECT_FALSE(effects(reserved_prefetch).known_read_only());
}

TEST(OriginalGraphicsStageEffects, EveryControlTransferNeedsAWholeProgramProof) {
    const std::vector<uint32_t> branch{0xbf820002, mubuf(24), 0, end};
    const auto result = effects(branch);
    EXPECT_FALSE(result.known_read_only());
    EXPECT_EQ(result.rejection, "original-stage-control-or-program-effect-unimplemented:pc=0");
    const std::vector<uint32_t> indirect{0xbe802000, end};
    EXPECT_FALSE(effects(indirect).known_read_only());
}

TEST(OriginalGraphicsStageEffects, UnknownAndHiddenPostEndBytesAreNotCompleteness) {
    const std::vector<uint32_t> unknown{0xffffffff, mubuf(24), 0, end};
    const auto first = effects(unknown);
    EXPECT_FALSE(first.known_read_only());
    EXPECT_EQ(first.rejection, "original-stage-unknown-encoding:pc=0");
    const std::vector<uint32_t> tail{end, mubuf(24), 0, end};
    EXPECT_EQ(effects(tail).rejection, "original-stage-original-inventory-incomplete:pc=1");
    const std::vector<uint32_t> no_end{0xbf800000};
    EXPECT_FALSE(effects(no_end).known_read_only());
}

TEST(OriginalGraphicsStageEffects, TruncatedOperandsAndSyntheticOrChangedInventoriesRefuse) {
    for (const std::vector<uint32_t> words : {
             std::vector<uint32_t>{smem(8)},
             std::vector<uint32_t>{0xbe8003ff}, // missing mandatory scalar MOV literal
             std::vector<uint32_t>{0xd5030000, 0x000000ff}, // missing VOP3 literal
             std::vector<uint32_t>{0xf0000006, 0}, // missing three MIMG NSA DWORDs
         })
        EXPECT_EQ(effects(words).rejection, "original-stage-truncated-instruction:pc=0");
    const std::vector<uint32_t> words{0xbf800000, end};
    auto changed = inventory(words);
    changed.back().synthetic_terminator = true;
    EXPECT_FALSE(original_graphics_stage_effects(words, changed, ShaderProgramStage::Fragment)
                     .known_read_only());
    changed = inventory(words);
    changed.back().pc = 0;
    EXPECT_FALSE(original_graphics_stage_effects(words, changed, ShaderProgramStage::Fragment)
                     .known_read_only());
    changed = inventory(words);
    changed.front().words[0] ^= 1;
    EXPECT_EQ(
        original_graphics_stage_effects(words, changed, ShaderProgramStage::Fragment).rejection,
        "original-stage-source-inventory-mismatch:pc=0");
}

TEST(OriginalGraphicsStageEffects, UnsupportedStageAndUnknownRegisterEffectsStayNamed) {
    const std::vector<uint32_t> words{end};
    EXPECT_TRUE(
        effects(words).known_read_only()); // no-op code fact, NOT drawable-stage admission
    EXPECT_EQ(effects(words, ShaderProgramStage::Compute).rejection,
              "original-stage-kind-unimplemented:pc=4294967295");
    const std::vector<uint32_t> unsupported{0xbe807f00, end};
    EXPECT_EQ(effects(unsupported).rejection, "original-stage-register-effect-unimplemented:pc=0");
}
} // namespace
} // namespace prosper::gpu
