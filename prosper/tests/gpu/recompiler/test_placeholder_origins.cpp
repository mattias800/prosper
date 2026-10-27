// Diagnostic slices distinguish missing hardware inputs from unloaded words and spills.
// Hard public compiler fixtures remain refused; unloaded-only roots now have a separate policy.
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_reg_state.hpp"
#include "gpu/recompiler/rdna2_placeholder_origins.hpp"
#include "gpu/recompiler/rdna2_placeholder_origin_trace.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_launch_vgprs.hpp"
#include <cstdint>
#include <memory>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {
using namespace prosper::gpu;
PlaceholderOriginProgram program(const std::vector<uint32_t>& words) {
    PlaceholderOriginProgram p;
    EXPECT_EQ(rdna2_walk(words.data(), words.size(), p.instructions), words.size());
    p.entry_full_exec = p.vector_profile_known = p.scalar_profile_known = true;
    p.vectors = compute_launch_vgprs(0);
    p.supplied_vectors.insert(0);
    return p;
}
std::string roots(const PlaceholderOriginProgram& p, uint32_t pc, PlaceholderWord word) {
    return format_placeholder_origins(placeholder_roots_before(p, pc, {word}));
}
} // namespace

TEST(PlaceholderOrigins, AnUnloadedWordNamesItsOriginalRegisterThroughARelay) {
    const auto p = program({0x7e0c0308u, 0x7d840d06u, 0xbf810000u});
    EXPECT_EQ(roots(p, 1, {PlaceholderFile::Vector, 6}), "origin=unloaded word=v8 at=entry");
}

TEST(PlaceholderOrigins, ALoadedButUnsuppliedWordIsALaunchInput) {
    auto p = program({0x7e0c0301u, 0x7d840d06u, 0xbf810000u});
    p.vectors = compute_launch_vgprs(1);
    EXPECT_EQ(roots(p, 1, {PlaceholderFile::Vector, 6}), "origin=launch-input word=v1 at=entry");
}

TEST(PlaceholderOrigins, APartialProfileDoesNotInventUnloadedRegisters) {
    auto p = program({0x7d840d06u, 0xbf810000u});
    p.vector_profile_known = false;
    EXPECT_EQ(roots(p, 0, {PlaceholderFile::Vector, 6}), "origin=unknown-launch word=v6 at=entry");
}

TEST(PlaceholderOrigins, AnUntrackedSpillSlotIsNotAnUnloadedVgpr) {
    const auto p = program({0x7d840d06u, 0xbf810000u});
    EXPECT_EQ(roots(p, 0, {PlaceholderFile::Spill, 20, 3}),
              "origin=spill word=spill-v20:3 at=entry");
}

TEST(PlaceholderOrigins, AJoinReportsBothIncomingOrigins) {
    auto p =
        program({0xbf840002u, 0x7e0c0301u, 0xbf820001u, 0x7e0c0214u, 0x7d840d06u, 0xbf810000u});
    p.vectors = compute_launch_vgprs(1);
    const auto text = roots(p, 4, {PlaceholderFile::Vector, 6});
    EXPECT_NE(text.find("origin=launch-input word=v1 at=entry"), std::string::npos);
    EXPECT_NE(text.find("origin=unloaded word=s20 at=entry"), std::string::npos);
    EXPECT_EQ(text.find("merge"), std::string::npos);
}

TEST(PlaceholderOrigins, ALoopIsTracedThroughTheBackedgeToItsInput) {
    // s10 = s60; loop: s10 = s11; s11 = s61; conditional backedge.
    const auto p = program({0xbe8a033cu, 0xbe8a030bu, 0xbe8b033du, 0xbf85fffdu, 0xbf810000u});
    const auto text = roots(p, 3, {PlaceholderFile::Scalar, 10});
    EXPECT_NE(text.find("origin=unloaded word=s11 at=entry"), std::string::npos);
    EXPECT_NE(text.find("origin=unloaded word=s61 at=entry"), std::string::npos);
    EXPECT_EQ(text.find("loop"), std::string::npos);
}

TEST(PlaceholderOrigins, AMissingScalarLaunchInputDoesNotBecomeUnloaded) {
    auto p = program({0x7e0c0214u, 0x7d840d06u, 0xbf810000u});
    p.loaded_scalars.insert(20);
    EXPECT_EQ(roots(p, 1, {PlaceholderFile::Vector, 6}), "origin=launch-input word=s20 at=entry");
}

TEST(PlaceholderOrigins, APublishedNativeInputDoesNotChargeItsEncodedParameterField) {
    auto p = program({0xc81a0002u, 0x7d840d06u, 0xbf810000u});
    p.vectors = compute_launch_vgprs(2);
    p.native_interpolation = true;
    p.observations->native_inputs.insert(0);
    EXPECT_EQ(roots(p, 1, {PlaceholderFile::Vector, 6}), "origin=lost-definition word=v6 at=1");
}

TEST(PlaceholderOrigins, AProvedDescriptorDoesNotHideAMissingVectorAddress) {
    auto p = program({0xe0302000u, 0x80030604u, 0x7d840d06u, 0xbf810000u});
    for (int word = 12; word < 16; ++word) p.loaded_scalars.insert(word);
    p.observations->genuine_resources[0] = true;
    EXPECT_EQ(roots(p, 2, {PlaceholderFile::Vector, 6}), "origin=unloaded word=v4 at=entry");
    p.observations->genuine_resources[0] = false;
    EXPECT_NE(roots(p, 2, {PlaceholderFile::Vector, 6}).find("origin=launch-input word=s12"),
              std::string::npos);
}

TEST(PlaceholderOrigins, AScalarLayoutDoesNotInventSpecialRegisterInitialization) {
    const auto p = program({0x7e0c027cu, 0x7d840d06u, 0xbf810000u});
    EXPECT_EQ(roots(p, 1, {PlaceholderFile::Vector, 6}),
              "origin=unknown-launch word=s124 at=entry");
}

TEST(PlaceholderOrigins, AnUninventoriedImageAddressDoesNotBecomeAnUnloadedWord) {
    auto p = program({0xf0800309u, 0x00820604u, 0x7d840d06u, 0xbf810000u});
    ASSERT_EQ(p.instructions.front().opcode, 0xa0u);
    p.observations->genuine_resources[0] = true;
    auto text = roots(p, 2, {PlaceholderFile::Vector, 6});
    EXPECT_NE(text.find("origin=unmodelled-transfer word=v6 at=0"), std::string::npos);
    EXPECT_EQ(text.find("origin=unloaded"), std::string::npos);
    p.observations->image_addresses[0] = {4, 5};
    text = roots(p, 2, {PlaceholderFile::Vector, 6});
    EXPECT_NE(text.find("origin=unloaded word=v4 at=entry"), std::string::npos);
    EXPECT_NE(text.find("origin=unloaded word=v5 at=entry"), std::string::npos);
    EXPECT_EQ(text.find("origin=unmodelled-transfer"), std::string::npos);
}

TEST(PlaceholderOrigins, ARefusalWithNoMissingInputNamesTheLostDefinition) {
    const auto p = program({0x7e0c0281u, 0x7d840d06u, 0xbf810000u});
    EXPECT_EQ(roots(p, 1, {PlaceholderFile::Vector, 6}), "origin=lost-definition word=v6 at=1");
}

TEST(PlaceholderOrigins, PublicUnloadedScalarReadIsAdmittedWithoutAStaleRefusal) {
    const std::vector<uint32_t> words{0x7e0c023cu, 0x7d840d06u, 0xbf810000u};
    ComputeShaderConfig config;
    config.local_x = config.wave_size = config.native_subgroup_size = 64;
    prosper::test::complete_compute_launch(config);
    constexpr uint64_t address = 0x5497a00100ull;
    EXPECT_FALSE(recompile_compute(words.data(), words.size(), nullptr, config,
                                   {RecompileDiagnosticStage::Compute, address})
                     .empty());
    const auto reason = last_terminal_reject_reason(address);
    EXPECT_TRUE(reason.empty()) << reason;
}

TEST(PlaceholderOrigins, PublicUnknownScalarLaunchRemainsUnknown) {
    const std::vector<uint32_t> words{0x7e0c023cu, 0x7d840d06u, 0xbf810000u};
    ComputeShaderConfig config;
    config.local_x = config.wave_size = config.native_subgroup_size = 64;
    constexpr uint64_t address = 0x5497a00200ull;
    EXPECT_TRUE(recompile_compute(words.data(), words.size(), nullptr, config,
                                  {RecompileDiagnosticStage::Compute, address})
                    .empty());
    const auto reason = last_terminal_reject_reason(address);
    EXPECT_NE(reason.find("fabricated-fresh-mask-source"), std::string::npos) << reason;
    EXPECT_NE(reason.find("origin=unknown-launch word=s60 at=entry"), std::string::npos) << reason;
}

TEST(PlaceholderOrigins, PublicUnsuppliedBarycentricsNameTheLaunchInput) {
    const std::vector<uint32_t> words{0x7e0c0300u, 0x7d840d06u, 0x7e000280u,
                                      0xf800080fu, 0u,          0xbf810000u};
    PixelSystemInputMapping inputs;
    inputs.ena = inputs.addr = 1; // hardware loads perspective sample I/J; this shell does not
    constexpr uint64_t address = 0x5497a00300ull;
    EXPECT_TRUE(recompile_fragment(words.data(), words.size(), nullptr, &inputs, UINT32_MAX,
                                   nullptr, false, {RecompileDiagnosticStage::Fragment, address})
                    .empty());
    const auto reason = last_terminal_reject_reason(address);
    EXPECT_NE(reason.find("fabricated-fresh-mask-source"), std::string::npos) << reason;
    EXPECT_NE(reason.find("origin=launch-input word=v0 at=entry"), std::string::npos) << reason;
}

TEST(PlaceholderOrigins, PublicNativeInputPublicationDoesNotChargeItsEncodedParameterField) {
    const std::vector<uint32_t> words{0xc8020002u, 0x7da40080u, 0xc81a0402u, 0xbefe04c1u,
                                      0x7e1002fau, 0xff080006u, 0x7d841108u, 0xf800080fu,
                                      0x08080808u, 0xbf810000u};
    PixelSystemInputMapping inputs;
    inputs.ena = inputs.addr = 0x87; // v6 is the genuinely unsupplied LINE_STIPPLE word
    constexpr uint64_t address = 0x5497a00700ull;
    EXPECT_TRUE(recompile_fragment(words.data(), words.size(), nullptr, &inputs, UINT32_MAX,
                                   nullptr, false, {RecompileDiagnosticStage::Fragment, address})
                    .empty());
    const auto reason = last_terminal_reject_reason(address);
    EXPECT_NE(reason.find("fabricated-fresh-mask-source"), std::string::npos) << reason;
    EXPECT_NE(reason.find("origin=launch-input word=v6 at=entry"), std::string::npos) << reason;
    EXPECT_EQ(reason.find("origin=launch-input word=v2 at=entry"), std::string::npos) << reason;
}

TEST(PlaceholderOrigins, AWave32MaskWriteDoesNotEraseTheUnwrittenNumericHighWord) {
    auto p = program({0xd4c2000eu, 0x00020100u, 0x7d840d06u, 0xbf810000u});
    p.wave_size = 32;
    p.loaded_scalars.insert(15);
    EXPECT_EQ(roots(p, 2, {PlaceholderFile::Scalar, 15}), "origin=launch-input word=s15 at=entry");
}

TEST(PlaceholderOrigins, ARelativeReadMustVisitTheSelectedWordBeyondItsEncodedBase) {
    auto p = program({0xbefc0381u, 0x7e0c8704u, 0x7d840d06u, 0xbf810000u});
    p.vectors.push_back({5, 0, 0, LaunchVgprState::Unsupplied});
    const auto broad = roots(p, 2, {PlaceholderFile::Vector, 6});
    EXPECT_NE(broad.find("origin=launch-input word=v5 at=entry"), std::string::npos);
    p.observations->relative_words[1] = {5};
    EXPECT_EQ(roots(p, 2, {PlaceholderFile::Vector, 6}), "origin=launch-input word=v5 at=entry");
}

TEST(PlaceholderOrigins, ImplicitVccWritersDoNotInventAnEntryConditionInput) {
    for (const auto word : {0x7d840100u, 0x500c0100u}) {
        // ADDC consumes old VCC; seed it with a genuine comparison first.
        const auto p = program({0x7d840100u, word, 0xbf810000u});
        for (int reg : {106, 107})
            EXPECT_EQ(roots(p, 2, {PlaceholderFile::Scalar, reg}),
                      "origin=lost-definition word=s" + std::to_string(reg) + " at=2");
    }
}

TEST(PlaceholderOrigins, AWorkLimitReportsAnIncompleteTraceInsteadOfUnloaded) {
    const auto p = program({0xbe8a033cu, 0xbe8b030au, 0xbf810000u});
    const detail::PlaceholderOriginTrace trace(p);
    const auto text =
        format_placeholder_origins(trace.before(2, {{PlaceholderFile::Scalar, 11}}, 1));
    EXPECT_EQ(text, "origin=trace-budget word=s10 at=0");
}

TEST(PlaceholderOrigins, AMissingInstructionIsNamedInsteadOfInventingALaunchRoot) {
    const auto p = program({0xbf810000u});
    EXPECT_EQ(roots(p, 99, {PlaceholderFile::Scalar, 10}), "origin=unreachable word=s10 at=99");
}

TEST(PlaceholderOrigins, AnUnrecordedOptionalScalarLaunchWordRemainsUnknown) {
    auto p = program({0x7d840d06u, 0xbf810000u});
    p.loaded_scalars = {0, 1};
    p.supplied_scalars = {0, 1};
    EXPECT_EQ(roots(p, 0, {PlaceholderFile::Scalar, 2}), "origin=unknown-launch word=s2 at=entry");
    EXPECT_EQ(roots(p, 0, {PlaceholderFile::Scalar, 3}), "origin=unloaded word=s3 at=entry");
}

TEST(PlaceholderOrigins, AWave32CarryDoesNotReadTheNumericVccHighWord) {
    auto p = program({0x500c0108u, 0xbf810000u});
    ASSERT_EQ(p.instructions.front().opcode, 0x28u);
    ASSERT_EQ(p.instructions.front().src[0].kind, OperandKind::VGPR);
    ASSERT_EQ(p.instructions.front().src[0].value, 8);
    p.wave_size = 32;
    p.supplied_scalars.insert(106);
    EXPECT_EQ(roots(p, 1, {PlaceholderFile::Vector, 6}), "origin=unloaded word=v8 at=entry");
    RegState rs;
    rs.vector_wave_size = 32;
    rs.vreg[0] = rs.sreg[106] = 1;
    rs.placeholder_origin_program = std::make_shared<PlaceholderOriginProgram>(p);
    EXPECT_EQ(fresh_placeholder_origins(rs, p.instructions.front()),
              "origin=unloaded word=v8 at=entry");
    p.wave_size = 64;
    EXPECT_NE(roots(p, 1, {PlaceholderFile::Vector, 6}).find("origin=unknown-launch word=s107"),
              std::string::npos);
}

TEST(PlaceholderOrigins, ADynamicSpillWriteMayLeaveTheQueriedFixedSlotUnwritten) {
    const auto p = [] {
        auto value = program({0xd7610014u, 1u | (1u << 9u), 0xbf810000u});
        EXPECT_EQ(value.instructions.front().src[1].kind, OperandKind::SGPR);
        value.supplied_scalars.insert(1);
        return value;
    }();
    EXPECT_EQ(roots(p, 2, {PlaceholderFile::Spill, 20, 3}),
              "origin=spill word=spill-v20:3 at=entry");
}

TEST(PlaceholderOrigins, AnUnmatchedLoopEdgeDoesNotInferAnUnloadedRootFromTheHeader) {
    const auto p = program({0xbe8a033cu, 0xbf810000u});
    const RegState rs;
    EXPECT_EQ(loop_placeholder_origins(rs, 0, {10}, &p),
              "origin=unmodelled-transfer word=s10 at=0");
}

TEST(PlaceholderOrigins, AnImplicitLdsReadDoesNotInventAnUnloadedAddressRoot) {
    const auto p = program({0xbefc033cu, 0xdac40000u, 0x06000000u, 0xbf810000u});
    ASSERT_EQ(p.instructions[1].opcode, 0xb1u);
    const auto text = roots(p, 3, {PlaceholderFile::Vector, 6});
    // Legacy memory-state inventory is deliberately not consumed as a new validity proof.
    EXPECT_EQ(text, "origin=unmodelled-transfer word=v6 at=1");
}

TEST(PlaceholderOrigins, AWordBeyondThePhysicalVectorFileIsNotAnUnloadedRegister) {
    const auto p = program({0xbf810000u});
    EXPECT_EQ(roots(p, 0, {PlaceholderFile::Vector, 256}),
              "origin=unmodelled-transfer word=v256 at=entry");
}
