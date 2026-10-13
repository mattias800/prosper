// Conditional discard tails must remain owned code and real scalar-CFG edges. A missing edge
// cannot make a saved predicate dead; a complete no-reader tail can. All instructions are synthetic.
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_fragment_terminal_tail.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

using namespace prosper::gpu;

namespace {
constexpr uint32_t kEnd = 0xbf810000u, kNop = 0xbf800000u, kUnknown = 0xffffffffu;
constexpr uint32_t kClearExec = 0xbefe0480u, kEmptyExport = 0xf8001c00u;

constexpr uint32_t branch(uint32_t op, int16_t offset) {
    return 0xbf800000u | (op << 16u) | uint16_t(offset);
}

std::vector<uint32_t> program() {
    return {
        0xbe840380u,   // pc0: genuine s4 = 0
        0xbf068004u,   // pc1: compare supplies SCC
        branch(4, 1),   // pc2: conditional one-arm join at pc4
        0xbe82047eu,   // pc3: s_mov_b64 s[2:3], EXEC, only one predecessor
        0x87987e7eu,   // pc4: s_and_b64 s[24:25], EXEC, EXEC
        branch(4, 5),   // pc5: existing mask-SCC early-out -> pc11
        0xbefe0418u,   // pc6: EXEC = genuine s[24:25]
        0x7e000280u,   // pc7: v0 = 0
        0xf800180fu,  0u,   // pc8/9: enabled color export from defined v0
        kEnd,   // pc10: first END
        kClearExec,   kEmptyExport, 0x12345678u, kEnd,   // pc11..14: terminal discard
        kUnknown,     kNop,   // unreferenced metadata stays out of the owned code
    };
}

Rdna2FragmentTerminalTail proof(std::span<const uint32_t> code) {
    std::vector<Rdna2Inst> body;
    rdna2_walk(code.data(), code.size(), body);
    return rdna2_fragment_terminal_tail(code.data(), code.size(), body);
}

std::vector<uint32_t> compile(std::span<const uint32_t> code, uint64_t address = 0x50860001) {
    return recompile_fragment(code.data(), code.size(), nullptr, nullptr, UINT32_MAX, nullptr,
                              false, {RecompileDiagnosticStage::Fragment, address});
}

class FragmentTerminalTail : public testing::Test {
protected:
    void SetUp() override {
        clear_shader_recompile_cache();
        clear_shader_analysis_cache();
    }
};
}   // namespace

TEST_F(FragmentTerminalTail, ExactDiscardInstructionsCompleteTheDeadPredicateJoin) {
    const auto code = program();
    const auto tail = proof(code);
    ASSERT_EQ(tail.retained_dwords, 15u);
    ASSERT_EQ(tail.instructions.size(), 3u);
    EXPECT_EQ(tail.instructions[0].pc, 11u);
    EXPECT_EQ(tail.instructions[1].pc, 12u);
    EXPECT_EQ(tail.instructions[1].exp_target, 0u);
    EXPECT_EQ(tail.instructions[1].exp_en, 0u);
    EXPECT_TRUE(tail.instructions[1].exp_compr);
    EXPECT_EQ(tail.instructions[1].words[1], 0x12345678u);
    EXPECT_TRUE(tail.instructions.back().is_end);
    const auto direct = compile(code);
    ASSERT_FALSE(direct.empty());
    const auto cached =
        recompile_graphics_shader_cached(ShaderProgramStage::Fragment, code.data(), code.size());
    EXPECT_EQ(cached, direct);
    const auto owned =
        shader_analysis_owned_words(acquire_shader_analysis(code.data(), code.size()));
    ASSERT_TRUE(owned);
    EXPECT_EQ(*owned, (std::vector<uint32_t>(code.begin(), code.begin() + 15)));
    EXPECT_EQ(compile(*owned), direct);
}

TEST_F(FragmentTerminalTail, ExecutableInventoryIncludesActualTerminalInstructions) {
    const auto code = program();
    std::vector<Rdna2Inst> inventory;
    uint32_t unavailable = UINT32_MAX;
    ASSERT_TRUE(rdna2_recompile_executable_instructions(code.data(), 15, inventory, unavailable));
    ASSERT_EQ(inventory.back().pc, 14u);
    EXPECT_TRUE(inventory.back().is_end);
    EXPECT_TRUE(sgpr_dead_at_merge(inventory, 4, 2));
    EXPECT_TRUE(sgpr_dead_at_merge(inventory, 4, 3));
}

TEST_F(FragmentTerminalTail, ActualLowAndHighTailReadsCannotRetireAMissingPredicate) {
    for (uint32_t source : {2u, 3u}) {
        auto code = program();
        code[11] = 0xbe800300u | source;   // real s_mov_b32 s0,s2/s3 before terminal tail
        EXPECT_EQ(proof(code).retained_dwords, 11u);
        const uint64_t address = 0x50860100u + source;
        EXPECT_TRUE(compile(code, address).empty());
        EXPECT_NE(
            last_terminal_reject_reason(address).find("structured-saved-mask-domain-missing-edge"),
            std::string::npos);
        std::vector<Rdna2Inst> full;
        rdna2_walk(code.data(), code.size(), full);
        // Independently inventory the known reader to show this negative is a real read,
        // not solely an absent-edge test of the narrow terminal classifier.
        std::vector<Rdna2Inst> tail;
        rdna2_walk(code.data() + 11, 4, tail);
        for (auto& in : tail) {
            in.pc += 11;
            full.push_back(in);
        }
        ScalarMergeBlocker blocker;
        EXPECT_FALSE(sgpr_dead_at_merge(full, 4, source, ScalarMergeProof::AnyRead, &blocker));
        EXPECT_EQ(blocker.pc, 11u);
        EXPECT_EQ(std::string(blocker.kind), "source-dword");
    }
}

TEST_F(FragmentTerminalTail, MissingUnknownAndTruncatedTailsPreserveTheNamedRefusal) {
    const auto original = program();
    for (size_t count : {11u, 12u, 13u, 14u}) {
        const auto partial = std::span(original).first(count);
        EXPECT_EQ(proof(partial).retained_dwords, 11u);
        EXPECT_TRUE(proof(partial).source_limit_required);
        const uint64_t address = 0x50860200u + count;
        EXPECT_TRUE(compile(partial, address).empty());
        EXPECT_NE(
            last_terminal_reject_reason(address).find("structured-saved-mask-domain-missing-edge"),
            std::string::npos);
    }
    auto unknown = original;
    unknown[11] = kUnknown;
    EXPECT_TRUE(proof(unknown).instructions.empty());
    EXPECT_TRUE(compile(unknown, 0x50860220).empty());
}

TEST_F(FragmentTerminalTail, ScalarEffectsEnabledExportsAndForeignTargetsAreNotDiscardProofs) {
    for (uint32_t replacement : {0xbe800381u, 0xbefe0481u, kUnknown, branch(2, 0), 0xbe801d00u}) {
        auto code = program();
        code[11] = replacement;
        EXPECT_TRUE(proof(code).instructions.empty());
        EXPECT_EQ(proof(code).retained_dwords, 11u);
    }
    for (uint32_t replacement : {kEmptyExport | 1u, kEmptyExport | (8u << 4),
                                 kEmptyExport | (9u << 4), kEmptyExport | (1u << 13)}) {
        auto code = program();
        code[12] = replacement;
        EXPECT_TRUE(proof(code).instructions.empty());
        EXPECT_EQ(proof(code).retained_dwords, 11u);
    }
}

TEST_F(FragmentTerminalTail, UniformBranchesReceiveNoNewLinearization) {
    auto code = program();
    code[4] = 0xbe980380u;   // scalar MOV does not provide the existing mask-SCC early-out shape
    ASSERT_EQ(proof(code).retained_dwords, 15u);
    EXPECT_TRUE(compile(code, 0x50860300).empty());
}

TEST_F(FragmentTerminalTail, ARealTailCannotManufactureALiveSkippedEdgeMask) {
    auto code = program();
    code[6] = 0xbefe0402u;   // restore EXEC from the missing s[2:3], instead of the defined mask
    EXPECT_TRUE(compile(code, 0x50860310).empty());
    EXPECT_NE(
        last_terminal_reject_reason(0x50860310).find("structured-saved-mask-domain-missing-edge"),
        std::string::npos);
}

TEST_F(FragmentTerminalTail, WarmChangesInvalidateEveryInspectedWordAndWindow) {
    auto code = program();
    const auto positive = acquire_shader_analysis(code.data(), code.size());
    ASSERT_TRUE(positive);
    EXPECT_EQ(acquire_shader_analysis(code.data(), code.size()), positive);
    code[14] = kUnknown;
    const auto negative = acquire_shader_analysis(code.data(), code.size());
    ASSERT_TRUE(negative);
    EXPECT_NE(positive, negative);
    EXPECT_EQ(shader_analysis_owned_words(negative)->size(), 11u);
    EXPECT_EQ(acquire_shader_analysis(code.data(), code.size()), negative);
    code[14] = kEnd;
    const auto restored = acquire_shader_analysis(code.data(), code.size());
    EXPECT_NE(restored, negative);
    EXPECT_EQ(shader_analysis_owned_words(restored)->size(), 15u);
    const auto missing = acquire_shader_analysis(code.data(), 13);
    EXPECT_EQ(shader_analysis_owned_words(missing)->size(), 11u);
    EXPECT_NE(acquire_shader_analysis(code.data(), 15), missing);
}

TEST_F(FragmentTerminalTail, OpaqueDisabledExportPayloadStillParticipatesInTheOwnedKey) {
    auto code = program();
    const auto first = acquire_shader_analysis(code.data(), code.size());
    code[13] ^= 1u;
    const auto changed = acquire_shader_analysis(code.data(), code.size());
    ASSERT_TRUE(changed);
    EXPECT_NE(changed, first);
    EXPECT_EQ(shader_analysis_owned_words(changed)->at(13), code[13]);
    EXPECT_FALSE(compile(code).empty());
}

TEST_F(FragmentTerminalTail, OverlappingLiteralTargetsCannotAcquireInstructionAuthority) {
    const std::array code{branch(4, 2), branch(5, 3), kEnd, kClearExec, kEmptyExport, kEnd, kEnd};
    // pc3's EXP includes word5 as payload. A second branch cannot reinterpret word5 as END.
    const auto tail = proof(code);
    EXPECT_TRUE(tail.instructions.empty());
    EXPECT_EQ(tail.retained_dwords, 3u);
}

TEST_F(FragmentTerminalTail, UnconditionalUnreferencedAndSyntheticEndsDoNotSupplyTails) {
    std::array code{branch(2, 1), kEnd, kClearExec, kEmptyExport, 0u, kEnd};
    EXPECT_TRUE(proof(code).instructions.empty());
    EXPECT_EQ(rdna2_recompile_code_span(code.data(), code.size()), 2u);
    code[0] = kNop;
    EXPECT_TRUE(proof(code).instructions.empty());
    std::vector<Rdna2Inst> body;
    rdna2_walk(code.data(), code.size(), body);
    body.back().synthetic_terminator = true;
    EXPECT_EQ(rdna2_fragment_terminal_tail(code.data(), code.size(), body).retained_dwords, 0u);
}

TEST_F(FragmentTerminalTail, ConditionalTargetsShareNopsButCannotWrapOrExceedTheBudget) {
    for (uint32_t opcode = 4; opcode <= 9; ++opcode) {
        const std::array code{branch(opcode, 1), kEnd, kNop, kClearExec, kNop,
                              kEmptyExport,      0u,   kNop, kEnd,       kUnknown};
        EXPECT_EQ(proof(code).retained_dwords, 9u) << opcode;
        EXPECT_EQ(proof(code).instructions.size(), 6u) << opcode;
    }
    const std::array multiple{branch(4, 2), branch(5, 2), kEnd, kNop,
                              kClearExec,   kEmptyExport, 0u,   kEnd};
    EXPECT_EQ(proof(multiple).retained_dwords, multiple.size());
    EXPECT_EQ(proof(multiple).instructions.size(), 4u);
    const std::array far{branch(4, INT16_MAX), kEnd};
    EXPECT_TRUE(proof(far).source_limit_required);
    const std::array negative{branch(4, INT16_MIN), kEnd};
    EXPECT_TRUE(proof(negative).instructions.empty());
    std::vector<uint32_t> long_tail(260, kNop);
    long_tail[0] = branch(4, 1);
    long_tail[1] = kEnd;
    long_tail[256] = kClearExec;
    long_tail[257] = kEmptyExport;
    long_tail[258] = 0;
    long_tail[259] = kEnd;
    EXPECT_EQ(proof(long_tail).retained_dwords, 2u);
    EXPECT_TRUE(proof(long_tail).instructions.empty());
}

TEST_F(FragmentTerminalTail, RawCaptureRetainsExactlyTheBoundedDiscardTail) {
    const auto code = program();
    DrawItem draw;
    draw.vs = {0x07230203, 1, 2};
    draw.fs = compile(code);
    ASSERT_FALSE(draw.fs.empty());
    constexpr uint64_t address = 0x40000;
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
    EXPECT_EQ(capture.raw_shader_versions[index].words,
              (std::vector<uint32_t>(code.begin(), code.begin() + 15)));
    EXPECT_EQ(compile(capture.raw_shader_versions[index].words), draw.fs);
}
